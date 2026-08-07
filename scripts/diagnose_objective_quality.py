"""Score a search objective against already-replayed edits. **No simulator.**

Phase 3 spent ~1.6 h of simulation to evaluate a single objective, which is why only two were
ever tried. This scores any objective against a *frozen* set of (edit → real outcome) pairs that
were already replayed, so a variant costs minutes of GPU instead of hours of CPU rendering. Only
variants that improve the screen earn simulator time.

**The two numbers, and why these two.** Both are computed *within episode*, because pooling
across episodes conflates "this episode was easy" with "this edit was good" — the pooled
correlation on the current data even flips sign between two runs, which is what a confound looks
like.

- **pairwise ranking accuracy** — over pairs of candidates for the same episode whose outcomes
  differ, how often the objective ranked the successful one better. 0.50 is a coin flip.
  Established baseline on the `p3-oracle56` edits: **0.673** (35/52, binomial p = 0.0088).
- **within-episode rank correlation** — of the within-group *ranks* of the objective against the
  ranks of the outcome. Ranks, not raw values: a raw-residual correlation moves when the
  objective is rescaled (the same data scored raw vs standardised gives −0.342 vs −0.237), and a
  screen whose number depends on the scale of what it is screening cannot compare objectives.
  Established baseline for **this** definition: **−0.2003**. The −0.342 quoted elsewhere is the
  raw-residual variant and is not what this reports.

Pairs with equal outcomes are **excluded, not scored as ties**. In the current data 116 of 168
pairs are uninformative that way, so folding them in would drag any accuracy toward 0.5 and hide
whatever differences exist between objectives.

**This is a ranking diagnostic, not a result.** It never touches replay success as a headline
number — it uses recorded outcomes as *labels* to ask whether the objective orders edits
correctly. CLAUDE.md rule 2 still holds: latent distance is never reported as a success measure.

Usage:
    python scripts/diagnose_objective_quality.py --bank p3-oracle56-replay --objective single
    python scripts/diagnose_objective_quality.py --bank p3-bank-replay --objective ensemble \
        --members p2-dyn-res p2-dyn
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score a search objective against replayed edits")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "search.yaml")
    p.add_argument(
        "--bank",
        nargs="+",
        required=True,
        help="run_replay.py run_id(s) whose replayed edits form the labelled set",
    )
    p.add_argument("--objective", choices=("single", "ensemble"), default="single")
    p.add_argument(
        "--members",
        nargs="+",
        default=["p2-dyn-res"],
        help="dyn_run id(s); one for --objective single, N for ensemble",
    )
    p.add_argument("--arms", nargs="*", default=None, help="restrict to these arms")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def ranking_quality(scored: list[dict]) -> dict:
    """Within-episode pairwise accuracy and Spearman. `scored` rows need episode/score/success.

    Lower `score` must mean "more likely to succeed", matching goal distance.
    """
    from cfe.dynamics.latents import spearman

    by_ep = defaultdict(list)
    for r in scored:
        by_ep[r["episode"]].append(r)

    from cfe.dynamics.latents import ranks

    concordant = discordant = excluded_ties = 0
    dres, yres = [], []
    n_eps_used = 0
    for rows in by_ep.values():
        if len(rows) < 2:
            continue
        n_eps_used += 1
        s = np.array([r["score"] for r in rows], dtype=np.float64)
        y = np.array([float(r["success"]) for r in rows], dtype=np.float64)
        # Correlate the WITHIN-GROUP RANKS, not the raw values. A raw-residual correlation is
        # sensitive to each group's scale, so it moves when the objective is rescaled — the
        # ensemble standardises per population, and the same data scored raw vs standardised
        # gave -0.342 vs -0.237. Ranking quality must not depend on that; on ranks it does not,
        # and the statistic then agrees with the pairwise accuracy about what it is measuring.
        sr, yr = ranks(s), ranks(y)
        dres += list(sr - sr.mean())
        yres += list(yr - yr.mean())
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if y[i] == y[j]:
                    excluded_ties += 1
                elif (s[i] < s[j]) == (y[i] > y[j]):
                    concordant += 1
                else:
                    discordant += 1

    informative = concordant + discordant
    return {
        "pairwise_accuracy": round(concordant / informative, 4) if informative else None,
        "n_informative_pairs": informative,
        "n_concordant": concordant,
        "n_discordant": discordant,
        "n_excluded_equal_outcome_pairs": excluded_ties,
        "within_episode_rank_corr": round(spearman(np.array(dres), np.array(yres)), 4)
        if dres
        else None,
        "n_episodes": n_eps_used,
        "n_candidates": len(scored),
        "reading": "accuracy 0.50 = coin flip; rank_corr should be NEGATIVE (lower predicted "
        "distance -> success). Single-model (p2-dyn-res) baseline on the p3-oracle56 search "
        "edits: accuracy 0.6731, rank_corr -0.2003. Both are invariant to monotone rescaling "
        "within a group, which a raw-residual correlation is not.",
    }


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    env_cfg = yaml.safe_load((REPO_ROOT / cfg["env_config"]).read_text())
    os.environ.setdefault("MUJOCO_GL", env_cfg["render_backend"])
    run_id = args.run_id or f"p35-objq-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    import torch

    from cfe.data.failure_set import failure_set_dir, load_failure_set
    from cfe.search.ensemble import evaluate_candidates_ensemble, member_agreement
    from scripts.run_search import build_problem, load_dynamics_only
    from scripts.train_dynamics import open_cache

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.objective == "single" and len(args.members) != 1:
        raise ValueError("--objective single takes exactly one --members entry")

    # Outcomes from the replay run(s); actions from the matching search run's edits file.
    outcomes, edits = {}, {}
    for bank in args.bank:
        mpath = REPO_ROOT / "results" / "runs" / bank / "metrics.jsonl"
        epath = REPO_ROOT / "results" / "runs" / bank / "edits.jsonl"
        if not mpath.exists() or not epath.exists():
            raise FileNotFoundError(f"{bank} needs both metrics.jsonl and edits.jsonl")
        for line in mpath.read_text().splitlines():
            r = json.loads(line)
            if r.get("kind") == "replay":
                outcomes[(bank, r["failure_id"], r["arm"], r["lam"], r.get("window_source"))] = r[
                    "success"
                ]
        for line in epath.read_text().splitlines():
            e = json.loads(line)
            edits[(bank, e["failure_id"], e["arm"], e["lam"], e.get("window_source"))] = e

    keys = sorted(k for k in edits if k in outcomes and "actions" in edits[k])
    if args.arms:
        keys = [k for k in keys if k[2] in set(args.arms)]
    if not keys:
        raise RuntimeError("no replayed edits matched; check --bank and --arms")

    print(f"diagnose_objective_quality  run_id={run_id}  device={device}")
    print(f"bank={args.bank}  objective={args.objective}  members={args.members}")
    print(f"labelled edits: {len(keys)}")
    print("!! ranking diagnostic only; no number here is a replay success rate")
    print("-" * 78)

    cache, *_ = open_cache({"encoder_config": cfg["encoder_config"], **cfg}, args.seed)
    recs = {r.failure_id: r for r in load_failure_set(failure_set_dir(REPO_ROOT))}

    loaded = [load_dynamics_only(m, device) for m in args.members]
    models = [m for m, _, _ in loaded]
    dyn_cfgs = [c for _, c, _ in loaded]

    # Candidates are scored in GROUPS of (episode, window_start). Standardisation is relative to
    # a population, so scoring one candidate at a time would make every score 0; and comparing
    # candidates across different windows is not meaningful anyway, because they have different
    # baselines. The group is therefore the unit of both the standardisation and the ranking.
    groups: dict[tuple, list] = defaultdict(list)
    for key in keys:
        groups[(key[1], int(edits[key]["window_start"]))].append(key)

    scored, per_group_raw = [], []
    problems: dict[str, list] = {}
    for (fid, wstart), gkeys in sorted(groups.items()):
        if fid not in problems:
            problems[fid] = [build_problem(cache, recs[fid], c, device) for c in dyn_cfgs]
        cand = np.stack([np.asarray(edits[k]["actions"], dtype=np.float64) for k in gkeys])
        out = evaluate_candidates_ensemble(models, problems[fid], wstart, cand, lam=0.0)
        per_group_raw.append(out["goal_distance_per_member"])
        for k, score, member_scores in zip(
            gkeys, out["goal_distance"], out["goal_distance_per_member"].T, strict=True
        ):
            scored.append(
                {
                    "episode": f"{fid}@{wstart}",
                    "failure_id": fid,
                    "window_start": wstart,
                    "arm": k[2],
                    "lam": k[3],
                    "window_source": k[4],
                    "score": float(score),
                    "per_member": [float(x) for x in member_scores],
                    "success": bool(outcomes[k]),
                }
            )

    agreement = member_agreement(np.concatenate(per_group_raw, axis=1)) if len(models) > 1 else None

    quality = ranking_quality(scored)

    summary = {
        "run_id": run_id,
        "kind": "summary",
        "diagnostic": True,
        "not_a_result": "ranking quality of a search objective against recorded outcomes; "
        "never a replay success rate (CLAUDE.md rule 2)",
        "bank": args.bank,
        "objective": args.objective,
        "members": args.members,
        "arms": sorted({r["arm"] for r in scored}),
        "quality": quality,
        "member_agreement": agreement,
    }

    out_dir = REPO_ROOT / "results" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, out_dir / "search.yaml")
    with (out_dir / "metrics.jsonl").open("w") as fh:
        for r in scored:
            fh.write(json.dumps({"run_id": run_id, "kind": "scored", **r}) + "\n")
        fh.write(json.dumps(summary, default=str) + "\n")

    print(json.dumps({k: v for k, v in summary.items() if k != "not_a_result"}, indent=2))
    print(f"artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
