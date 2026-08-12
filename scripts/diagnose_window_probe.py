"""Does a LEARNED window replace the true one, end to end? (Phase 3.5 ceiling-test follow-up.)

`diagnose_fault_localization.py` (`ceiling-test-v2`) found that a probe trained on `z_t` alone,
generalising across episodes via leave-one-episode-out CV, localises three of four injected
causes far better than either the trained model's stage-1 heuristic (mean IoU 0.035) or a
pointwise diff against the true clean counterpart (mean IoU 0.051) — see PROGRESS.md
2026-08-10. That is a localisation finding; it says nothing yet about whether a better window
actually produces a better *replayed* edit. This answers that.

For each episode, this forces stage 2 (CEM) to the window `ceiling-test-v2`'s probe predicted
for it — critically, the LOEO prediction, i.e. the window predicted by a probe that never saw
*this* episode's own ground-truth label, only the other 55 episodes'. That is a materially
different kind of access than `diagnose_window_oracle.py`'s forced ground-truth window: it is
the same generalisation structure the trained dynamics model or the BC policy already rely on,
not a peek at this episode's own answer. It is still run here as an **ungated diagnostic**,
tagged `PROBE_WINDOW` and deliberately absent from `configs/search.yaml:experiments` so
`verify_g3.py:gated_rows` cannot pick it up — promoting a stage-1 change into the registered
method is a SPEC.md §5 amendment and is the user's call (CLAUDE.md), independent of what this
run shows.

Three numbers matter, all at K=8 (the ceiling test's own window length):
  - `no_edit` / `expert_oracle` at this window are NOT re-run: they were already measured at the
    ground-truth window (`p3-replay-RD`: 0.40 / 1.00) and at stage1_top1 (the divergence
    heuristic's window). The probe's window is a third point on that same axis.
  - `search` vs `random_perturb` here is the load-bearing comparison — same shape as every
    other stage-2 test in this project.

Usage:
    python scripts/diagnose_window_probe.py --probe-run ceiling-test-v2 --run-id p35-probe-window
    python scripts/run_replay.py --search-run p35-probe-window --run-id p35-probe-window-replay
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

#: Deliberately not a key in `configs/search.yaml:experiments`, so `verify_g3.py:gated_rows`
#: cannot select these rows into any criterion — same convention as diagnose_window_oracle.py.
EXPERIMENT = "PROBE_WINDOW"

NOT_A_RESULT = (
    "the edit window is a leave-one-episode-out probe's PREDICTION ({probe_run}), not the "
    "ground truth: the probe never saw this episode's own label, only the other episodes' in "
    "that run. That is structurally closer to a trained pipeline stage than to the "
    "ground-truth-window diagnostics elsewhere in Phase 3.5 — but it has not been registered as "
    "SPEC.md §5's stage 1 (that is the user's call), has not been tested on an unseen CAUSE, "
    "and this run is diagnostic only. Never reportable as an RSR result for C1 until promoted."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 with the window forced to the probe's window")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "search.yaml")
    p.add_argument("--experiment-model", default="R", help="which checkpoint; R = model of record")
    p.add_argument(
        "--probe-run", default="ceiling-test-v2", help="diagnose_fault_localization.py run_id"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--budget-k", type=int, default=8, help="matches the ceiling test's window length"
    )
    p.add_argument("--n-failures", type=int, default=None)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument(
        "--exclude-run",
        default=None,
        help="skip any failure_id already covered by this run_id's edits.jsonl — for testing "
        "on genuinely NEW episodes after the failure set has grown (e.g. more-data experiments)",
    )
    return p.parse_args()


def covered_failure_ids(run_id: str) -> set[str]:
    path = REPO_ROOT / "results" / "runs" / run_id / "edits.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    lines = (line for line in path.read_text().splitlines() if line.strip())
    return {json.loads(line)["failure_id"] for line in lines}


def load_probe_windows(probe_run: str) -> dict[str, tuple[int, int] | None]:
    """`{failure_id: (start, end) or None}`, from a fault-localization run's Part 2 rows."""
    path = REPO_ROOT / "results" / "runs" / probe_run / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run diagnose_fault_localization.py first")
    out: dict[str, tuple[int, int] | None] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("kind") != "part2_episode":
            continue
        w = r.get("found_window")
        out[r["failure_id"]] = (int(w[0]), int(w[1])) if w is not None else None
    if not out:
        raise ValueError(f"{path} has no part2_episode rows; wrong --probe-run?")
    return out


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    enc_path = REPO_ROOT / cfg["encoder_config"]
    env_path = REPO_ROOT / cfg["env_config"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"p35-probe-window-{stamp}"
    exp = cfg["experiments"][args.experiment_model]

    import torch

    from cfe.data.failure_set import INJECTED, failure_set_dir, load_failure_set
    from cfe.search.arms import RANDOM_PERTURB, SEARCH, propose_baseline, realised_magnitude
    from cfe.search.cem import CEMConfig, cem_search
    from scripts.run_search import (
        build_problem,
        cell_seed,
        describe_edit,
        load_models,
        select_failures,
    )
    from scripts.train_dynamics import open_cache

    s_cfg = cfg["search"]
    r_cfg = cfg["replay"]
    lambdas = [float(x) for x in s_cfg["objective"]["lambdas"]]
    sparsity_eps = [float(e) for e in s_cfg["sparsity_eps"]]
    cem_cfg = CEMConfig.from_dict(s_cfg["cem"])
    k = int(args.budget_k)
    task_id = int(args.task_id if args.task_id is not None else r_cfg["g3_task_id"])
    n_failures = int(args.n_failures or r_cfg["g3_n_failures"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    not_a_result = NOT_A_RESULT.format(probe_run=args.probe_run)
    print(f"diagnose_window_probe  run_id={run_id}  seed={args.seed}  device={device}")
    print(f"model={exp['dyn_run']}  probe_run={args.probe_run}  K={k}  lambdas={lambdas}")
    print(f"!! {not_a_result}")
    print("-" * 78)

    probe_windows = load_probe_windows(args.probe_run)
    models, dyn_cfg, _ = load_models(exp["dyn_run"], device)
    cache, renderer, _, _ = open_cache({"encoder_config": cfg["encoder_config"], **cfg}, args.seed)

    records = load_failure_set(failure_set_dir(REPO_ROOT), source=INJECTED, task_id=task_id)
    if args.exclude_run:
        already = covered_failure_ids(args.exclude_run)
        before = len(records)
        records = [r for r in records if r.failure_id not in already]
        print(f"--exclude-run {args.exclude_run}: {before} -> {len(records)} candidate episodes")
    # The same selection G3 used, so this is directly comparable episode by episode to
    # p3-replay-RD (stage1_top1, ground_truth) and to p3-oracle56-replay (ground_truth, n=56).
    # ceiling-test-v2 covered all 56, so every G3-selected episode has a probe prediction.
    chosen = select_failures(records, n_failures, bool(r_cfg["g3_stratify_by_cause"]))

    rows, skipped = [], []
    for i, rec in enumerate(chosen):
        problem = build_problem(cache, rec, dyn_cfg, device)
        original = problem.actions

        w_pred = probe_windows.get(rec.failure_id)
        if w_pred is None:
            skipped.append({"failure_id": rec.failure_id, "reason": "no probe prediction"})
            print(f"  [{i + 1:2d}] {rec.failure_id[-34:]}: SKIPPED, no probe prediction")
            continue
        w = int(w_pred[0])

        valid = problem.valid_window_starts(k)
        if valid.size == 0 or w < valid[0] or w > valid[-1]:
            skipped.append(
                {
                    "failure_id": rec.failure_id,
                    "reason": "probe-predicted window outside the valid range",
                    "probe_window_start": w,
                    "valid_range": [int(valid[0]), int(valid[-1])] if valid.size else None,
                }
            )
            print(f"  [{i + 1:2d}] {rec.failure_id[-34:]}: SKIPPED, window {w} out of range")
            continue

        for lam in lambdas:
            res = cem_search(
                models["dynamics"],
                problem,
                w,
                k,
                lam,
                cem_cfg,
                np.random.default_rng(cell_seed(args.seed, rec.failure_id, SEARCH, k, lam)),
            )
            from cfe.eval.replay_protocol import Edit

            edit = Edit(window_start=res.window_start, actions=res.actions)
            scale = realised_magnitude(original, edit)
            rp_seed = cell_seed(args.seed, rec.failure_id, RANDOM_PERTURB, k, lam)
            rp = propose_baseline(
                RANDOM_PERTURB,
                original_actions=original,
                window_start=w,
                budget_k=k,
                rng=np.random.default_rng(rp_seed),
                magnitude_scale=scale,
            )
            for arm, e, extra in (
                (
                    SEARCH,
                    edit,
                    {
                        "search": {
                            "objective": res.objective,
                            "goal_distance": res.goal_distance,
                            "penalty": res.penalty,
                            "baseline_objective": res.baseline_objective,
                            "is_identity": res.is_identity,
                            "n_forward_passes": res.n_forward_passes,
                        }
                    },
                ),
                (RANDOM_PERTURB, rp, {"magnitude_scale_requested": float(scale)}),
            ):
                rows.append(
                    {
                        "run_id": run_id,
                        "experiment": EXPERIMENT,
                        "diagnostic": True,
                        "not_a_result": not_a_result,
                        "dyn_run": exp["dyn_run"],
                        "probe_run": args.probe_run,
                        "failure_id": rec.failure_id,
                        "episode_id": rec.episode_id,
                        "task_id": rec.task_id,
                        "cause": rec.cause,
                        "truth_window": list(rec.edit_window),
                        "probe_window": list(w_pred),
                        "arm": arm,
                        "budget_k": k,
                        "lam": lam,
                        "variant": "primary",
                        "window_source": "probe_predicted",
                        "window_start": int(e.window_start),
                        "window_end": int(e.window_end),
                        "actions": np.asarray(e.actions).tolist(),
                        "not_applicable": None,
                        **describe_edit(original, e, sparsity_eps),
                        **extra,
                    }
                )
        print(
            f"  [{i + 1:2d}/{len(chosen)}] {rec.failure_id[-34:]}: window forced to probe's {w} "
            f"(truth {rec.edit_window}), {len(lambdas)} lambdas"
        )

    out_dir = REPO_ROOT / "results" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, out_dir / "search.yaml")
    shutil.copyfile(enc_path, out_dir / "encoder.yaml")
    shutil.copyfile(env_path, out_dir / "env.yaml")
    shutil.copyfile(REPO_ROOT / cfg["dynamics_config"], out_dir / "dynamics.yaml")
    with (out_dir / "edits.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")

    summary = {
        "run_id": run_id,
        "kind": "summary",
        "experiment": EXPERIMENT,
        "diagnostic": True,
        "not_a_result": not_a_result,
        "seed": args.seed,
        "dyn_run": exp["dyn_run"],
        "probe_run": args.probe_run,
        "budget_k": k,
        "lambdas": lambdas,
        "n_failures": len(chosen),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "n_edits": len(rows),
        "renderer": renderer,
        "encoder_cfg_hash": cache.hash,
        "reference": "compare against p3-replay-RD @ K=8: no_edit@ground_truth 0.40, "
        "expert_oracle@ground_truth 1.00, search@stage1_top1(divergence heuristic) 0.00-0.10; "
        "and p3-oracle56-replay: search@ground_truth 0.375-0.536",
    }
    with (out_dir / "metrics.jsonl").open("w") as fh:
        fh.write(json.dumps(summary, default=str) + "\n")

    print("-" * 78)
    print(json.dumps(summary, indent=2, default=str))
    print(f"artifacts: {out_dir}")
    print(f"next: scripts/run_replay.py --search-run {run_id} --run-id {run_id}-replay")
    return 0


if __name__ == "__main__":
    sys.exit(main())
