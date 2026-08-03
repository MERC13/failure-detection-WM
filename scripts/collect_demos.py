"""Materialise demo observations by replaying demos in our simulator, then encode them.

Observations are regenerated rather than read from the HDF5 so that the BC policy's train and
deploy distributions match — see `cfe/data/demos.py` for the measurement that motivates it.

An episode whose replay does not succeed is reported and **not cached**: a demo that no longer
reaches the goal in our simulator is not expert data, and silently training on it would be the
kind of quiet corruption CLAUDE.md warns about.

Usage:
    python scripts/collect_demos.py --config configs/encoder.yaml --seed 0 --run-id p1-demos
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
    p = argparse.ArgumentParser(description="Replay demos, encode, and populate the cache")
    p.add_argument("--config", required=True, type=Path, help="configs/encoder.yaml")
    p.add_argument("--env-config", type=Path, default=REPO_ROOT / "configs" / "env.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--limit", type=int, default=None, help="only the first N episodes")
    p.add_argument("--force", action="store_true", help="re-encode episodes already cached")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    env_cfg = yaml.safe_load(args.env_config.read_text())
    run_id = args.run_id or f"p1-demos-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    # Must precede any MuJoCo import.
    os.environ.setdefault("MUJOCO_GL", env_cfg["render_backend"])

    from cfe.data.demos import (
        cache_episode_id,
        demo_file_path,
        episode_ids,
        gl_renderer,
        make_replay_env,
        read_episode,
        replay_episode,
    )
    from cfe.encoders import Dinov2Encoder, EncoderConfig
    from cfe.encoders.cache import EncodingCache, Fingerprint
    from scripts.verify_env import load_suite

    data_cfg, runtime_cfg = cfg["data"], cfg["runtime"]
    enc_cfg = EncoderConfig.from_dict(cfg["encoder"])

    suite = load_suite({"suite": data_cfg["suite"]})
    task = suite.get_task(int(data_cfg["task_id"]))
    path = demo_file_path(data_cfg["dataset_dir"], data_cfg["suite"], task)
    ids = episode_ids(path)
    if args.limit is not None:
        ids = ids[: args.limit]

    print(f"collect_demos  run_id={run_id}  seed={args.seed}  MUJOCO_GL={os.environ['MUJOCO_GL']}")
    print(f"suite={data_cfg['suite']}  task={task.name}  episodes={len(ids)}")
    print(f"demo file: {path}")
    print("-" * 78)

    env = make_replay_env(env_cfg, task, args.seed)
    renderer = gl_renderer()
    encoder = Dinov2Encoder(enc_cfg, device=runtime_cfg.get("device", "auto"))
    fingerprint = Fingerprint.build(
        enc_cfg.hash_payload(), encoder.device, renderer=renderer, obs_source="replay"
    )
    cache = EncodingCache(runtime_cfg["cache_dir"], fingerprint)
    print(f"renderer: {renderer}")
    print(f"device:   {encoder.device}")
    print(f"cache:    {cache.root}")
    print("-" * 78)

    records, failed, skipped = [], [], []
    t_start = time.time()
    n_frames = 0
    try:
        for i, eid in enumerate(ids):
            # `eid` indexes the HDF5; `cid` names the entry in the flat, task-shared cache.
            cid = cache_episode_id(data_cfg["suite"], data_cfg["task_id"], eid)
            if cache.has(cid) and cache.has_meta(cid) and not args.force:
                skipped.append(cid)
                continue
            ep = read_episode(path, eid)
            t0 = time.time()
            out = replay_episode(env, ep["actions"], ep["states"][0])
            t_replay = time.time() - t0

            if not out["success"]:
                # Reported, not cached, and not tuned away.
                failed.append(cid)
                print(
                    f"[{i + 1:3d}/{len(ids)}] {eid}: REPLAY FAILED "
                    f"(reward_sum={out['reward_sum']:.1f}) — not cached"
                )
                continue

            t0 = time.time()
            tokens = encoder.encode(out["frames"])
            t_encode = time.time() - t0

            cache.write(cid, tokens)
            cache.write_meta(
                cid,
                actions=out["actions"],
                proprio=out["proprio"],
                success=np.array(out["success"]),
                reward_sum=np.array(out["reward_sum"]),
            )
            n_frames += len(out["frames"])
            records.append(
                {
                    "episode_id": cid,
                    "demo_id": eid,
                    "n_frames": int(len(out["frames"])),
                    "n_actions": int(len(out["actions"])),
                    "success": True,
                    "reward_sum": out["reward_sum"],
                    "replay_s": round(t_replay, 2),
                    "encode_s": round(t_encode, 2),
                }
            )
            print(
                f"[{i + 1:3d}/{len(ids)}] {eid}: T={len(out['actions']):3d}  "
                f"tokens={tokens.shape}  replay={t_replay:5.1f}s  encode={t_encode:4.1f}s"
            )
    finally:
        env.close()

    elapsed = time.time() - t_start
    summary = {
        "run_id": run_id,
        "seed": args.seed,
        "suite": data_cfg["suite"],
        "task_id": int(data_cfg["task_id"]),
        "task_name": task.name,
        "encoder_cfg_hash": cache.hash,
        "renderer": renderer,
        "device": str(encoder.device),
        "episodes_total": len(ids),
        "episodes_cached": len(records),
        "episodes_skipped_already_cached": len(skipped),
        "episodes_replay_failed": len(failed),
        "replay_failed_ids": failed,
        "frames_encoded": n_frames,
        "elapsed_s": round(elapsed, 1),
        "frames_per_s": round(n_frames / elapsed, 2) if elapsed > 0 and n_frames else None,
    }

    run_dir = REPO_ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "encoder.yaml")
    shutil.copyfile(args.env_config, run_dir / "env.yaml")
    with (run_dir / "metrics.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps({"run_id": run_id, "kind": "episode", **r}) + "\n")
        fh.write(json.dumps({"run_id": run_id, "kind": "summary", **summary}) + "\n")

    print("-" * 78)
    print(json.dumps(summary, indent=2))
    print(f"artifacts: {run_dir}")

    if failed:
        print(
            f"\n{len(failed)} demo(s) did not reproduce success on replay. These are excluded "
            f"from the cache and are a finding to investigate, not to work around."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
