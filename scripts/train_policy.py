"""Train the BC failure-generating policy and measure its success rate (SPEC.md §4.3, G1).

The policy is a means to an end. SPEC.md wants it mediocre — 40-70% success — because a
policy that always succeeds produces no failures and one that always fails produces no
recoverable ones. The measured rate is reported whatever it is; it is never tuned upward to
land in the band (CLAUDE.md rule 3).

Usage:
    python scripts/train_policy.py --config configs/policy.yaml --seed 0 --run-id p1-bc
"""

import argparse
import collections
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
    p = argparse.ArgumentParser(description="Train and evaluate the BC policy")
    p.add_argument("--config", required=True, type=Path, help="configs/policy.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--n-rollouts", type=int, default=None, help="override eval rollout count")
    p.add_argument("--skip-eval", action="store_true", help="train only; no sim rollouts")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    enc_path = REPO_ROOT / cfg["encoder_config"]
    env_path = REPO_ROOT / cfg["env_config"]
    enc_yaml = yaml.safe_load(enc_path.read_text())
    env_cfg = yaml.safe_load(env_path.read_text())
    run_id = args.run_id or f"p1-bc-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    os.environ.setdefault("MUJOCO_GL", env_cfg["render_backend"])

    import torch

    from cfe.data.demos import gl_renderer, make_replay_env
    from cfe.encoders import Dinov2Encoder, EncoderConfig
    from cfe.encoders.cache import EncodingCache, Fingerprint
    from cfe.policy.bc import (
        BCPolicy,
        Normalizer,
        PolicyConfig,
        build_arrays,
        chunk_loss,
        find_target_object_key,
        rollout,
    )
    from scripts.verify_env import load_suite

    # I5: explicit seed, deterministic given the same config.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    pol_cfg = PolicyConfig.from_dict(cfg["policy"])
    tr, ev = cfg["train"], cfg["eval"]
    data_cfg, runtime_cfg = enc_yaml["data"], enc_yaml["runtime"]
    enc_cfg = EncoderConfig.from_dict(enc_yaml["encoder"])

    suite = load_suite({"suite": data_cfg["suite"]})
    task = suite.get_task(int(data_cfg["task_id"]))
    init_states = suite.get_task_init_states(int(data_cfg["task_id"]))

    print(f"train_policy  run_id={run_id}  seed={args.seed}  MUJOCO_GL={os.environ['MUJOCO_GL']}")
    print(f"suite={data_cfg['suite']}  task={task.name}")
    print("-" * 78)

    # The env must exist before gl_renderer() can report a context, and the fingerprint must
    # match the one collect_demos.py wrote or the cache refuses to open.
    env = make_replay_env(env_cfg, task, args.seed)
    try:
        renderer = gl_renderer()
        encoder = Dinov2Encoder(enc_cfg, device=runtime_cfg.get("device", "auto"))
        device = encoder.device
        fingerprint = Fingerprint.build(
            enc_cfg.hash_payload(), device, renderer=renderer, obs_source="replay"
        )
        cache = EncodingCache(runtime_cfg["cache_dir"], fingerprint)
        episode_ids = cache.episode_ids()
        if not episode_ids:
            raise RuntimeError(f"cache {cache.root} is empty — run scripts/collect_demos.py first")
        print(f"cache: {cache.root}  ({len(episode_ids)} episodes)")

        # Split by EPISODE, not by sample: consecutive frames within an episode are near
        # duplicates, so a per-sample split would leak the validation set into training.
        rng = np.random.default_rng(args.seed)
        order = rng.permutation(len(episode_ids))
        n_val = max(1, int(round(float(tr["val_fraction"]) * len(episode_ids))))
        val_ids = [episode_ids[i] for i in order[:n_val]]
        train_ids = [episode_ids[i] for i in order[n_val:]]
        print(f"episodes: {len(train_ids)} train / {len(val_ids)} val")

        t0 = time.time()
        train_d = build_arrays(cache, train_ids, pol_cfg)
        val_d = build_arrays(cache, val_ids, pol_cfg)
        print(
            f"samples:  {len(train_d['x'])} train / {len(val_d['x'])} val  "
            f"(built in {time.time() - t0:.1f}s)"
        )

        norm = Normalizer.fit(train_d["x"], train_d["delta"])

        def tensors(d):
            return (
                torch.from_numpy(norm.norm_x(d["x"]).astype(np.float32)).to(device),
                torch.from_numpy(norm.norm_delta(d["delta"]).astype(np.float32)).to(device),
                torch.from_numpy(d["grip"]).to(device),
            )

        xtr, dtr, gtr = tensors(train_d)
        xva, dva, gva = tensors(val_d)

        policy = BCPolicy(xtr.shape[1], pol_cfg).to(device)
        n_params = sum(p.numel() for p in policy.parameters())
        print(
            f"policy:   input_dim={xtr.shape[1]}  chunk={pol_cfg.chunk_size}  params={n_params:,}"
        )
        print("-" * 78)

        opt = torch.optim.AdamW(
            policy.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"])
        )
        bs, epochs = int(tr["batch_size"]), int(tr["epochs"])
        history = []
        for epoch in range(epochs):
            policy.train()
            perm = torch.from_numpy(rng.permutation(len(xtr))).to(device)
            agg = {"loss": 0.0, "delta": 0.0, "gripper": 0.0}
            nb = 0
            for i in range(0, len(perm), bs):
                idx = perm[i : i + bs]
                loss, parts = chunk_loss(
                    policy(xtr[idx]),
                    dtr[idx],
                    gtr[idx],
                    gripper_weight=float(tr["gripper_loss_weight"]),
                    delta_loss=tr["delta_loss"],
                )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                for k in agg:
                    agg[k] += parts[k]
                nb += 1
            policy.eval()
            with torch.no_grad():
                _, vparts = chunk_loss(
                    policy(xva),
                    dva,
                    gva,
                    gripper_weight=float(tr["gripper_loss_weight"]),
                    delta_loss=tr["delta_loss"],
                )
            rec = {
                "epoch": epoch,
                "train_loss": round(agg["loss"] / nb, 5),
                "train_delta": round(agg["delta"] / nb, 5),
                "train_gripper": round(agg["gripper"] / nb, 5),
                "val_loss": round(vparts["loss"], 5),
                "val_delta": round(vparts["delta"], 5),
                "val_gripper": round(vparts["gripper"], 5),
            }
            history.append(rec)
            print(
                f"epoch {epoch:3d}  train {rec['train_loss']:.4f}  val {rec['val_loss']:.4f}  "
                f"(delta {rec['val_delta']:.4f}  grip {rec['val_gripper']:.4f})"
            )

        # ---- evaluation: success comes from the simulator, never from the loss ----
        rollouts = []
        success_rate = None
        if not args.skip_eval:
            n = int(args.n_rollouts or ev["n_rollouts"])
            n = min(n, len(init_states))
            print("-" * 78)
            # The failure-mode breakdown separates the task's target object from distractors;
            # identify it once, and fail loudly rather than mislabel every rollout.
            target_key = find_target_object_key(env.reset().keys(), task.language)
            print(f"evaluating {n} rollouts (max_steps={ev['max_steps']}) target={target_key}")
            t0 = time.time()
            for i in range(n):
                out = rollout(
                    env,
                    policy,
                    encoder,
                    norm,
                    init_states[i],
                    int(ev["max_steps"]),
                    device,
                    target_key,
                )
                rollouts.append({"init_state_index": i, **out})
                print(f"  rollout {i + 1:3d}/{n}: success={out['success']}  steps={out['steps']}")
            success_rate = float(np.mean([r["success"] for r in rollouts]))
            print(f"eval done in {time.time() - t0:.1f}s")
    finally:
        env.close()

    # SPEC.md §4.3 (amended 2026-07-31): the gate is failure supply plus a mode breakdown.
    # `success_bounds` is a wide sanity check, not the old 40-70% band. See CHANGELOG.md.
    lo, hi = (float(x) for x in ev["success_bounds"])
    min_failures = int(ev["min_failures"])
    failures = [r for r in rollouts if not r["success"]]
    failure_modes = collections.Counter(r["failure_mode"] for r in failures)
    enough_failures = len(failures) >= min_failures
    rate_sane = success_rate is not None and lo <= success_rate <= hi
    summary = {
        "run_id": run_id,
        "seed": args.seed,
        "task_name": task.name,
        "encoder_cfg_hash": cache.hash,
        "renderer": renderer,
        "device": str(device),
        "n_train_episodes": len(train_ids),
        "n_val_episodes": len(val_ids),
        "n_train_samples": int(len(train_d["x"])),
        "input_dim": int(xtr.shape[1]),
        "n_params": int(n_params),
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
        "n_rollouts": len(rollouts),
        "success_rate": success_rate,
        "success_bounds": [lo, hi],
        "rate_within_bounds": rate_sane,
        "n_failures": len(failures),
        "min_failures": min_failures,
        "enough_failures": enough_failures,
        "failure_modes": dict(sorted(failure_modes.items())),
    }

    run_dir = REPO_ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "policy.yaml")
    shutil.copyfile(enc_path, run_dir / "encoder.yaml")
    shutil.copyfile(env_path, run_dir / "env.yaml")
    import torch as _torch

    _torch.save(
        {
            "policy": policy.state_dict(),
            "normalizer": norm.state_dict(),
            "policy_config": cfg["policy"],
            "input_dim": int(xtr.shape[1]),
        },
        run_dir / "policy.pt",
    )
    with (run_dir / "metrics.jsonl").open("w") as fh:
        for rec in history:
            fh.write(json.dumps({"run_id": run_id, "kind": "epoch", **rec}) + "\n")
        for r in rollouts:
            fh.write(json.dumps({"run_id": run_id, "kind": "rollout", **r}) + "\n")
        fh.write(json.dumps({"run_id": run_id, "kind": "summary", **summary}) + "\n")

    print("-" * 78)
    print(json.dumps(summary, indent=2))
    print(f"artifacts: {run_dir}")
    if success_rate is not None:
        n_ok = len(rollouts) - len(failures)
        print(f"\nBC success rate {success_rate:.1%}  ({n_ok}/{len(rollouts)})")
        print(f"failures: {len(failures)} (need >= {min_failures})  modes:")
        for mode, count in sorted(failure_modes.items()):
            print(f"  {mode:28s} {count}")
        if not enough_failures:
            print(f"\nToo few failures for SPEC.md §4.4: {len(failures)} < {min_failures}.")
        if not rate_sane:
            print(f"\nRate {success_rate:.1%} is outside the sanity bounds {lo:.0%}-{hi:.0%}.")
        if not (enough_failures and rate_sane):
            print(
                "Reported as measured. Per CLAUDE.md rule 3 nothing here — the policy, "
                "max_steps, or the thresholds — is tuned to make this pass; the number and "
                "its cause are the finding."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
