"""Does goal-latent distance track task success at all? Measured before any model exists.

SPEC.md §4.2 requires the Spearman correlation between end-of-episode goal-latent distance and
true task success as part of G2, and warns: "If rho is near zero the search objective is not a
proxy for anything." That is a property of the *representation*, not of the dynamics model, so
it can and should be measured first — it costs a few seconds and it decides which latent
reduction the dynamics model should operate on.

Two things are swept, because both change the space the search will actually work in:

  - the spatial reduction (`cls`, `mean`, `grid2`, `grid4`, `full`)
  - raw vs LayerNormed latents, since SPEC.md §4.2 trains on LayerNormed latents

`z_goal` follows SPEC.md §5's fallback: the minimum distance over the successful demos'
terminal latents. §5's primary reading — "a successful demo for the same task *and initial
state*" — has no referent here, because Phase 1 verified that no demo init state matches any
benchmark init state, so no rollout has a same-init-state demo to compare against. See
PROGRESS.md.

Usage:
    python scripts/probe_representation.py --config configs/encoder.yaml --seed 0 \
        --run-id p2-probe
"""

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cfe.dynamics.latents import goal_distance, layer_norm, spearman  # noqa: E402
from cfe.encoders.dinov2 import REDUCTIONS, EncoderConfig, reduce_latents  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Goal-distance vs success, before any model")
    p.add_argument("--config", required=True, type=Path, help="configs/encoder.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    run_id = args.run_id or f"p2-probe-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    np.random.seed(args.seed)

    from cfe.data.demos import gl_renderer
    from cfe.encoders.cache import EncodingCache, Fingerprint

    data_cfg, runtime_cfg = cfg["data"], cfg["runtime"]
    enc_cfg = EncoderConfig.from_dict(cfg["encoder"])

    # The cache fingerprint needs the renderer string, which needs a GL context. Reading a
    # cache written elsewhere must still be a hard error, so this is not faked.
    import os

    env_cfg = yaml.safe_load((REPO_ROOT / "configs" / "env.yaml").read_text())
    os.environ.setdefault("MUJOCO_GL", env_cfg["render_backend"])
    from cfe.data.demos import make_replay_env
    from scripts.verify_env import load_suite

    suite = load_suite({"suite": data_cfg["suite"]})
    task = suite.get_task(int(data_cfg["task_id"]))
    env = make_replay_env(env_cfg, task, args.seed)
    try:
        renderer = gl_renderer()
    finally:
        env.close()

    import torch

    device_kind = "cuda" if torch.cuda.is_available() else "cpu"
    fp = Fingerprint.build(enc_cfg.hash_payload(), device_kind, renderer, obs_source="replay")
    cache = EncodingCache(runtime_cfg["cache_dir"], fp)

    # Anchored classification, not a substring test: Phase 3's injected failures are named after
    # the demo they corrupt and an `"_demo_" in e` test would put corrupted episodes into the
    # goal set. See cfe/data/demos.py:episode_kind.
    from cfe.data.demos import DEMO, ROLLOUT, episode_kind

    ids = cache.episode_ids()
    demo_ids = [e for e in ids if episode_kind(e) == DEMO]
    roll_ids = [e for e in ids if episode_kind(e) == ROLLOUT]
    if not demo_ids or not roll_ids:
        raise RuntimeError(
            f"need both demos and rollouts in {cache.root}; "
            f"found {len(demo_ids)} demos, {len(roll_ids)} rollouts"
        )

    print(f"probe_representation  run_id={run_id}  seed={args.seed}")
    print(f"cache={cache.root}")
    print(f"{len(demo_ids)} demo episodes (goal set), {len(roll_ids)} rollout episodes")
    print("-" * 78)

    # Terminal frame of every episode. Only the last row of each (T, N, D) file is touched.
    demo_end = np.stack([np.asarray(cache.read(e)[-1], dtype=np.float32) for e in demo_ids])
    roll_end = np.stack([np.asarray(cache.read(e)[-1], dtype=np.float32) for e in roll_ids])
    success = np.array([bool(cache.read_meta(e)["success"]) for e in roll_ids], dtype=np.float64)
    print(f"rollouts: {int(success.sum())} success / {int((1 - success).sum())} failure")
    print("-" * 78)

    rows = []
    for mode in REDUCTIONS:
        g_raw = reduce_latents(demo_end, mode)
        r_raw = reduce_latents(roll_end, mode)
        for space, g, r in (
            ("raw", g_raw, r_raw),
            ("layernorm", layer_norm(g_raw), layer_norm(r_raw)),
        ):
            d = np.array([goal_distance(r[i], g) for i in range(len(r))])
            # Distance should be LOWER for successes, so a working representation gives a
            # NEGATIVE rho against the success flag.
            rho = spearman(d, success)
            row = {
                "reduction": mode,
                "space": space,
                "tokens_per_frame": int(r.shape[1]),
                "spearman_rho": round(rho, 4),
                "abs_rho": round(abs(rho), 4),
                "sign_correct": rho < 0,
                "mean_dist_success": round(float(d[success == 1].mean()), 4),
                "mean_dist_failure": round(float(d[success == 0].mean()), 4),
            }
            rows.append(row)
            print(
                f"{mode:6s} {space:10s} R={row['tokens_per_frame']:4d}  "
                f"rho={rho:+.4f}  {'ok  ' if rho < 0 else 'WRONG SIGN'}  "
                f"d_success={row['mean_dist_success']:8.3f}  "
                f"d_failure={row['mean_dist_failure']:8.3f}"
            )

    usable = [r for r in rows if r["sign_correct"]]
    best = max(usable, key=lambda r: r["abs_rho"]) if usable else None

    summary = {
        "run_id": run_id,
        "seed": args.seed,
        "encoder_cfg_hash": cache.hash,
        "renderer": renderer,
        "n_goal_episodes": len(demo_ids),
        "n_rollout_episodes": len(roll_ids),
        "n_success": int(success.sum()),
        "n_failure": int((1 - success).sum()),
        "goal_rule": "min L2 to any successful demo terminal latent (SPEC.md §5 fallback)",
        "best": best,
    }

    run_dir = REPO_ROOT / "results" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, run_dir / "encoder.yaml")
    with (run_dir / "metrics.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps({"run_id": run_id, "kind": "reduction", **r}) + "\n")
        fh.write(json.dumps({"run_id": run_id, "kind": "summary", **summary}) + "\n")

    print("-" * 78)
    print(json.dumps(summary, indent=2))
    print(f"artifacts: {run_dir}")

    if best is None:
        print(
            "\nNo reduction gives a correctly-signed correlation. SPEC.md §4.2: the search "
            "objective is not a proxy for anything. This is a hard stop, not a tuning problem."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
