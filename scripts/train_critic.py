"""Train the deployment critic checkpoint for SPEC.md §5's D slot (cfe/search/critic.py).

`critic-screen-v1` (PROGRESS.md 2026-08-10) established, via grouped leave-one-episode-out, that
this exact recipe — logistic regression on pooled `z_end`, fit on already-replayed outcomes —
clears the goal-distance objective's ranking accuracy (0.711 vs 0.6211-0.6731 single model,
0.6680 3-seed ensemble). That LOEO run is the generalisation estimate for what this script now
trains: it fits the checkpoint on ALL available labelled candidates rather than holding a slice
back, the same way `train_dynamics.py` fits its final checkpoint after G2's diagnostics have
already characterised the architecture, not on a fresh held-out split re-measured here.

**No new simulation.** Every label is a real replayed outcome already on disk; this script never
imports the simulator.

Usage:
    python scripts/train_critic.py --config configs/critic.yaml --seed 0 --run-id p35-critic
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the deployment critic checkpoint")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "critic.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    run_id = args.run_id or f"critic-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    import torch

    from cfe.data.failure_set import failure_set_dir, load_failure_set
    from cfe.search.critic import (
        load_labelled_candidates,
        rollout_features,
        save_critic,
        train_critic,
    )
    from scripts.run_search import build_problem, load_dynamics_only
    from scripts.train_dynamics import open_cache

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    banks = list(cfg["data"]["banks"])
    dyn_run = str(cfg["dyn_run"])
    c_cfg = cfg["critic"]

    print(f"train_critic  run_id={run_id}  seed={args.seed}  device={device}")
    print(f"banks={banks}  dyn_run={dyn_run}  hidden={c_cfg['hidden']}")
    print("-" * 78)

    candidates = load_labelled_candidates(REPO_ROOT, banks)
    if not candidates:
        raise RuntimeError(f"no labelled candidates joined from {banks}; check critic.yaml:data")
    n_pos = sum(1 for c in candidates if c["success"])
    n_fail = len(candidates) - n_pos
    print(f"labelled candidates: {len(candidates)}  (success {n_pos}, fail {n_fail})")

    cache, *_ = open_cache(cfg, args.seed)
    model, dyn_cfg, _ = load_dynamics_only(dyn_run, device)
    recs = {r.failure_id: r for r in load_failure_set(failure_set_dir(REPO_ROOT))}

    problems = {}
    for c in candidates:
        fid = c["failure_id"]
        if fid not in problems:
            problems[fid] = build_problem(cache, recs[fid], dyn_cfg, device)
    x_all, y_all, episode_ids = rollout_features(candidates, problems, model, device)
    n_episodes = len(set(episode_ids.tolist()))
    print(f"unique episodes: {n_episodes}  feature dim: {x_all.shape[1]}")

    # Trained on 100% of the labelled data — LOEO in critic-screen-v1 already estimated how this
    # recipe generalises; that estimate is what this checkpoint inherits, not re-derived here on
    # a smaller held-out slice (see the module docstring).
    mean = x_all.mean(axis=0)
    std = x_all.std(axis=0) + 1e-6
    critic = train_critic(
        (x_all - mean) / std,
        y_all,
        hidden=int(c_cfg["hidden"]),
        epochs=int(c_cfg["epochs"]),
        lr=float(c_cfg["lr"]),
        weight_decay=float(c_cfg["weight_decay"]),
        device=device,
        seed=args.seed,
    )

    out_dir = REPO_ROOT / "results" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, out_dir / "critic.yaml")
    save_critic(
        out_dir / "critic.pt",
        critic,
        mean,
        std,
        extra={
            "banks": banks,
            "dyn_run": dyn_run,
            "seed": args.seed,
            "n_candidates": len(candidates),
            "n_episodes": n_episodes,
        },
    )

    summary = {
        "run_id": run_id,
        "kind": "summary",
        "seed": args.seed,
        "banks": banks,
        "dyn_run": dyn_run,
        "hidden": int(c_cfg["hidden"]),
        "n_candidates": len(candidates),
        "n_success": n_pos,
        "n_episodes": n_episodes,
        "feature_dim": int(x_all.shape[1]),
        "generalisation_estimate": "see critic-screen-v1 (PROGRESS.md 2026-08-10): grouped "
        "leave-one-episode-out on this exact recipe measured pairwise_accuracy 0.711, "
        "rank_corr -0.269, pooled_auroc 0.689 -- this checkpoint is trained on all labelled "
        "data and was not independently re-evaluated on a held-out split.",
    }
    with (out_dir / "metrics.jsonl").open("w") as fh:
        fh.write(json.dumps(summary, default=str) + "\n")

    print("-" * 78)
    print(json.dumps(summary, indent=2))
    print(f"artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
