"""Roll out the BC policy, encode what it saw, and cache it — failures included.

SPEC.md §4.2 trains the dynamics model on "LIBERO demos plus policy rollouts (successes *and*
failures — failures are informative for dynamics even though they are excluded as imitation
targets)". That is the opposite of `collect_demos.py`, which refuses to cache an episode whose
replay failed: a demo that no longer succeeds is not expert data, but a *rollout* that fails is
exactly the thing this project studies. The `success` flag rides along in the sidecar so no
downstream consumer has to guess.

These 50 episodes are also the seed of Phase 3's natural failure set (SPEC.md §4.4), so they
are collected once and used twice.

Usage:
    python scripts/collect_rollouts.py --config configs/encoder.yaml --seed 0 \
        --policy-run p1-bc4 --run-id p2-rollouts
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache BC policy rollouts, successes and failures")
    p.add_argument("--config", required=True, type=Path, help="configs/encoder.yaml")
    p.add_argument("--env-config", type=Path, default=REPO_ROOT / "configs" / "env.yaml")
    p.add_argument("--policy-config", type=Path, default=REPO_ROOT / "configs" / "policy.yaml")
    p.add_argument("--policy-run", default="p1-bc4", help="run_id holding policy.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--limit", type=int, default=None, help="only the first N init states")
    p.add_argument("--force", action="store_true", help="re-encode episodes already cached")
    return p.parse_args()


def load_recorded_outcomes(policy_run: str) -> dict[int, dict]:
    """The rollout outcomes `train_policy.py` already measured, for a determinism cross-check."""
    metrics = REPO_ROOT / "results" / "runs" / policy_run / "metrics.jsonl"
    if not metrics.exists():
        return {}
    out = {}
    for line in metrics.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "rollout":
            out[int(rec["init_state_index"])] = rec
    return out


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    env_cfg = yaml.safe_load(args.env_config.read_text())
    pol_yaml = yaml.safe_load(args.policy_config.read_text())
    run_id = args.run_id or f"p2-rollouts-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    # Must precede any MuJoCo import.
    os.environ.setdefault("MUJOCO_GL", env_cfg["render_backend"])

    import torch

    from cfe.data.demos import cache_episode_id, gl_renderer, make_replay_env
    from cfe.encoders import Dinov2Encoder, EncoderConfig
    from cfe.encoders.cache import EncodingCache, Fingerprint
    from cfe.policy.bc import (
        BCPolicy,
        Normalizer,
        PolicyConfig,
        find_target_object_key,
        rollout,
    )
    from scripts.verify_env import load_suite

    data_cfg, runtime_cfg = cfg["data"], cfg["runtime"]
    enc_cfg = EncoderConfig.from_dict(cfg["encoder"])
    ev = pol_yaml["eval"]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    suite = load_suite({"suite": data_cfg["suite"]})
    task = suite.get_task(int(data_cfg["task_id"]))
    init_states = suite.get_task_init_states(int(data_cfg["task_id"]))
    n = len(init_states) if args.limit is None else min(args.limit, len(init_states))

    ckpt_path = REPO_ROOT / "results" / "runs" / args.policy_run / "policy.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"{ckpt_path} not found; run scripts/train_policy.py first")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    pol_cfg = PolicyConfig.from_dict(ck["policy_config"])

    print(
        f"collect_rollouts  run_id={run_id}  seed={args.seed}  MUJOCO_GL={os.environ['MUJOCO_GL']}"
    )
    print(f"suite={data_cfg['suite']}  task={task.name}  rollouts={n}  policy={args.policy_run}")
    print("-" * 78)

    env = make_replay_env(env_cfg, task, args.seed)
    renderer = gl_renderer()
    encoder = Dinov2Encoder(enc_cfg, device=runtime_cfg.get("device", "auto"))
    device = encoder.device
    # Same fingerprint as the demos on purpose: the *source of the actions* does not change
    # what a frame encodes to, and the dynamics model has to read demos and rollouts together.
    fingerprint = Fingerprint.build(
        enc_cfg.hash_payload(), device, renderer=renderer, obs_source="replay"
    )
    cache = EncodingCache(runtime_cfg["cache_dir"], fingerprint)

    policy = BCPolicy(ck["input_dim"], pol_cfg).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    norm = Normalizer.from_state_dict(ck["normalizer"])

    print(f"renderer: {renderer}")
    print(f"device:   {device}")
    print(f"cache:    {cache.root}")
    print("-" * 78)

    recorded = load_recorded_outcomes(args.policy_run)
    records, skipped, mismatches = [], [], []
    n_frames = 0
    t_start = time.time()
    try:
        target_key = find_target_object_key(env.reset().keys(), task.language)
        for i in range(n):
            cid = cache_episode_id(data_cfg["suite"], data_cfg["task_id"], f"rollout_{i}")
            if cache.has(cid) and cache.has_meta(cid) and not args.force:
                skipped.append(cid)
                continue

            t0 = time.time()
            out = rollout(
                env,
                policy,
                encoder,
                norm,
                init_states[i],
                int(ev["max_steps"]),
                device,
                target_key,
                capture=True,
            )
            t_roll = time.time() - t0

            # I5 cross-check against what train_policy.py measured for the same seed.
            prior = recorded.get(i)
            if prior is not None and (
                bool(prior["success"]) != out["success"] or int(prior["steps"]) != out["steps"]
            ):
                mismatches.append(
                    {
                        "init_state_index": i,
                        "recorded": {"success": prior["success"], "steps": prior["steps"]},
                        "now": {"success": out["success"], "steps": out["steps"]},
                    }
                )

            t0 = time.time()
            tokens = encoder.encode(out["frames"])
            t_encode = time.time() - t0

            cache.write(cid, tokens)
            cache.write_meta(
                cid,
                actions=out["actions"],
                proprio=out["proprio"],
                success=np.array(out["success"]),
                init_state_index=np.array(i),
                target_displacement_xy=np.array(out["target_displacement_xy"]),
                other_object_displacement_xy=np.array(out["other_object_displacement_xy"]),
            )
            n_frames += len(out["frames"])
            records.append(
                {
                    "episode_id": cid,
                    "init_state_index": i,
                    "success": out["success"],
                    "steps": out["steps"],
                    "failure_mode": out["failure_mode"],
                    "n_frames": int(len(out["frames"])),
                    "n_actions": int(len(out["actions"])),
                    "rollout_s": round(t_roll, 2),
                    "encode_s": round(t_encode, 2),
                }
            )
            print(
                f"[{i + 1:3d}/{n}] {cid}: success={out['success']!s:5s} steps={out['steps']:3d}  "
                f"tokens={tokens.shape}  roll={t_roll:5.1f}s  encode={t_encode:4.1f}s"
            )
    finally:
        env.close()

    elapsed = time.time() - t_start
    n_success = sum(1 for r in records if r["success"])
    summary = {
        "run_id": run_id,
        "seed": args.seed,
        "suite": data_cfg["suite"],
        "task_id": int(data_cfg["task_id"]),
        "task_name": task.name,
        "policy_run": args.policy_run,
        "encoder_cfg_hash": cache.hash,
        "renderer": renderer,
        "device": str(device),
        "target_object": target_key,
        "n_requested": n,
        "n_cached": len(records),
        "n_skipped_already_cached": len(skipped),
        "n_success": n_success,
        "n_failure": len(records) - n_success,
        "frames_encoded": n_frames,
        "determinism_mismatches_vs_policy_run": mismatches,
        "elapsed_s": round(elapsed, 1),
        "frames_per_s": round(n_frames / elapsed, 2) if elapsed > 0 and n_frames else None,
    }

    run_dir = REPO_ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "encoder.yaml")
    shutil.copyfile(args.env_config, run_dir / "env.yaml")
    shutil.copyfile(args.policy_config, run_dir / "policy.yaml")
    with (run_dir / "metrics.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps({"run_id": run_id, "kind": "episode", **r}) + "\n")
        fh.write(json.dumps({"run_id": run_id, "kind": "summary", **summary}) + "\n")

    print("-" * 78)
    print(json.dumps(summary, indent=2))
    print(f"artifacts: {run_dir}")

    if mismatches:
        # I5: same seed, same config, same outcome. Do not paper over this.
        print(
            f"\n{len(mismatches)} rollout(s) disagree with run {args.policy_run} at the same "
            f"seed. That is a determinism failure, not noise — stop and investigate."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
