"""Does a LEARNED critic rank candidate edits better than goal-distance? No new simulation.

PROGRESS.md 2026-08-10 (probe-window replay) confirmed, across three independent window-quality
regimes now (divergence heuristic, LOEO probe, ground truth), that stage 2 — not stage 1 — is
the persistent bottleneck: `search` does not beat `random_perturb` even at a good window, and
loses outright to it on `mistimed_close` and `wrong_target`. CLAUDE.md's "Q as critic" idea,
analysed and left as the cheap in-scope option during the Q-learning discussion, is the natural
next thing to screen: replace (or challenge) the objective's hand-written
`D(z_end, z_goal)` term with a **learned** `f(z_end) -> P(success)`, fit by supervised learning
(Monte Carlo — the episode's real replayed outcome is the label, no bootstrapping, no Bellman
target) on candidates this project has *already paid to replay*.

This is exactly `diagnose_objective_quality.py`'s screen (within-episode pairwise ranking
accuracy against real replayed outcomes, **no simulator**), applied to a different scoring
function. It reuses that module's `ranking_quality` rather than reimplementing it, and
`cfe/search/critic.py`'s `train_critic` / data-loading helpers — the same code
`scripts/train_critic.py` uses to fit the deployed checkpoint, so this screen's LOEO estimate is
of the exact recipe that gets deployed, not a lookalike.

**Data.** Every already-replayed bank with both `edits.jsonl` and a completed `run_replay.py`
pass qualifies. Default banks are the three at budget K=8 (`p3-bank-replay`,
`p3-oracle56-replay`, `p35-probe-window-replay`) so the headline number is comparable to the
established single/ensemble baselines (0.6211 / 0.6680, both measured on `p3-oracle56-replay`
alone). `p3-replay-RD` (the full K-grid) can be added via `--banks` but is not default, since
mixing K muddies the comparison to those established numbers.

**Evaluation.** Grouped leave-one-episode-out, exactly as the ceiling test's Part 2/3: the
critic never sees a held-out episode's own candidates while scoring it, so this is a fair
estimate of a trained component, not an oracle peek.

**Still diagnostic only** (CLAUDE.md rule 2): a ranking-quality screen is not a replay success
rate. Only a variant that clears the existing objective's ranking accuracy by a real margin
earns the next replay run — screening exists so that a variant which doesn't move this number
never costs simulator time.

Usage:
    python scripts/diagnose_critic_quality.py --run-id critic-screen-v1
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BANKS = ["p3-bank-replay", "p3-oracle56-replay", "p35-probe-window-replay"]

NOT_A_RESULT = (
    "ranking quality of a LEARNED critic against recorded replay outcomes; never a replay "
    "success rate (CLAUDE.md rule 2). The critic is trained and screened here with NO new "
    "simulation, only outcomes this project already paid to replay. Clearing the existing "
    "objective's ranking accuracy is necessary but not sufficient to justify a new search+"
    "replay run — that promotion is a separate step."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Screen a learned P(success) critic, no simulator")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "search.yaml")
    p.add_argument("--banks", nargs="+", default=DEFAULT_BANKS, help="run_replay.py run_id(s)")
    p.add_argument("--dyn-run", default="p2-dyn-res", help="model of record; matches G3")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--hidden", type=int, default=0, help="0 = logistic regression, the default")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    return p.parse_args()


def main() -> int:  # noqa: PLR0915 — a linear diagnostic pipeline reads better than split-up
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    run_id = args.run_id or f"critic-screen-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    import torch

    from cfe.data.failure_set import failure_set_dir, load_failure_set
    from cfe.search.critic import load_labelled_candidates, rollout_features, train_critic
    from scripts.diagnose_fault_localization import auroc
    from scripts.diagnose_objective_quality import ranking_quality
    from scripts.run_search import build_problem, load_dynamics_only
    from scripts.train_dynamics import open_cache

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"diagnose_critic_quality  run_id={run_id}  seed={args.seed}  device={device}")
    print(f"banks={args.banks}  dyn_run={args.dyn_run}")
    print(f"!! {NOT_A_RESULT}")
    print("-" * 78)

    candidates = load_labelled_candidates(REPO_ROOT, args.banks)
    if not candidates:
        raise RuntimeError(f"no labelled candidates joined from {args.banks}; check --banks")
    n_pos = sum(1 for c in candidates if c["success"])
    n_fail = len(candidates) - n_pos
    print(f"labelled candidates: {len(candidates)}  (success {n_pos}, fail {n_fail})")

    cache, *_ = open_cache({"encoder_config": cfg["encoder_config"], **cfg}, args.seed)
    model, dyn_cfg, _ = load_dynamics_only(args.dyn_run, device)
    recs = {r.failure_id: r for r in load_failure_set(failure_set_dir(REPO_ROOT))}

    # -- roll every candidate to z_end once; this is the only GPU cost, no simulator involved --
    problems = {c["failure_id"]: None for c in candidates}
    for fid in problems:
        problems[fid] = build_problem(cache, recs[fid], dyn_cfg, device)
    x_all, y_all, episode_ids = rollout_features(candidates, problems, model, device)
    group_keys = [f"{c['failure_id']}@{c['window_start']}" for c in candidates]
    unique_episodes = sorted(set(episode_ids.tolist()))
    print(f"unique episodes: {len(unique_episodes)}  feature dim: {x_all.shape[1]}")

    # -- grouped leave-one-episode-out, exactly the ceiling test's discipline --
    held_probs = np.empty(len(candidates), dtype=np.float64)
    for gi, fid in enumerate(unique_episodes):
        train_mask = episode_ids != fid
        held_mask = episode_ids == fid
        mu = x_all[train_mask].mean(axis=0)
        sd = x_all[train_mask].std(axis=0) + 1e-6
        critic = train_critic(
            (x_all[train_mask] - mu) / sd,
            y_all[train_mask],
            hidden=args.hidden,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            seed=args.seed,
        )
        with torch.no_grad():
            x_held = torch.as_tensor((x_all[held_mask] - mu) / sd, dtype=torch.float32).to(device)
            held_probs[held_mask] = torch.sigmoid(critic(x_held)).cpu().numpy()
        if (gi + 1) % 10 == 0 or gi + 1 == len(unique_episodes):
            print(f"  [{gi + 1:3d}/{len(unique_episodes)}] LOEO folds trained")

    pooled_auroc = auroc(held_probs, y_all)

    # `ranking_quality` wants LOWER score = more likely to succeed (goal-distance convention,
    # see diagnose_objective_quality.py). The critic outputs P(success), so negate it.
    scored = [
        {"episode": gk, "score": -float(p), "success": bool(y)}
        for gk, p, y in zip(group_keys, held_probs, y_all, strict=True)
    ]
    quality = ranking_quality(scored)

    by_cause_probs = defaultdict(list)
    by_cause_labels = defaultdict(list)
    for c, p, y in zip(candidates, held_probs, y_all, strict=True):
        by_cause_probs[c["cause"]].append(p)
        by_cause_labels[c["cause"]].append(y)
    per_cause_auroc = {
        cause: auroc(np.array(by_cause_probs[cause]), np.array(by_cause_labels[cause]))
        for cause in sorted(by_cause_probs)
    }

    summary = {
        "run_id": run_id,
        "kind": "summary",
        "diagnostic": True,
        "not_a_result": NOT_A_RESULT,
        "seed": args.seed,
        "banks": args.banks,
        "dyn_run": args.dyn_run,
        "n_candidates": len(candidates),
        "n_success": n_pos,
        "n_episodes": len(unique_episodes),
        "quality": quality,
        "pooled_auroc": pooled_auroc,
        "per_cause_auroc": per_cause_auroc,
        "reference": "goal-distance objective, same ranking-quality screen, on p3-oracle56-"
        "replay: single model pairwise_accuracy 0.6211-0.6731, 3-seed ensemble 0.6680 "
        "(PROGRESS.md 2026-08-06)",
    }

    out_dir = REPO_ROOT / "results" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, out_dir / "search.yaml")
    with (out_dir / "metrics.jsonl").open("w") as fh:
        for c, p, y in zip(candidates, held_probs, y_all, strict=True):
            row = {
                "run_id": run_id,
                "kind": "scored",
                "failure_id": c["failure_id"],
                "cause": c["cause"],
                "arm": c["arm"],
                "window_start": c["window_start"],
                "window_source": c.get("window_source"),
                "critic_prob": float(p),
                "success": bool(y),
            }
            fh.write(json.dumps(row, default=str) + "\n")
        fh.write(json.dumps(summary, default=str) + "\n")

    print("-" * 78)
    print(json.dumps({k: v for k, v in summary.items() if k != "not_a_result"}, indent=2))
    print(f"artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
