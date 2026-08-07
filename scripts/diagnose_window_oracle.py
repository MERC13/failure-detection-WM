"""Does stage 2 work when stage 1 is taken out of the loop? (Phase 3 post-G3 diagnostic.)

G3 FAILed and the headroom table says both stages lose about half the signal: an ideal edit is
worth +0.60 at the injection window and +0.30 at the window stage 1 picked, and the search
achieved 0.00-0.10. That leaves one question unanswered — *given the right window*, can the CEM
find an edit that beats a magnitude-matched random perturbation?

This forces the window to the ground-truth injection point and runs stage 2 there. It isolates
stage 2 completely:

- if the search beats `random_perturb` here, stage 2 works and the stage-1 criterion is the
  thing worth rebuilding;
- if it does not, stage 2 is weak too, and SPEC.md §0's stated fallback applies — "a negative
  result about world-model fidelity ... should be written up as such".

**THIS IS ORACLE-ASSISTED AND ITS OUTPUT IS NOT A RESULT.** The ground-truth window is
information the method does not have at deployment: it comes from the injection record, not
from anything the world model computed. Every row is stamped `diagnostic: true` and carries the
experiment tag `ORACLE`, which is deliberately absent from `configs/search.yaml:experiments`, so
`verify_g3.py:gated_rows` structurally cannot pick these rows up. Nothing here may be reported
as an RSR figure for SPEC.md's C1.

`no_edit` and `expert_oracle` at this window are **not re-run**: `p3-replay-RD` already measured
them at K=8 (0.40 and 1.00 respectively) and replay is deterministic, so re-measuring would burn
20 minutes to reproduce two known numbers. `action_ignoring_dynamics` is also skipped — the twin
proposes the identity edit by construction, so its RSR here would equal `no_edit`'s.

Usage:
    python scripts/diagnose_window_oracle.py --run-id p3-oracle
    python scripts/run_replay.py --search-run p3-oracle --run-id p3-oracle-replay
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
#: cannot select these rows into any criterion.
EXPERIMENT = "ORACLE"

NOT_A_RESULT = (
    "oracle-assisted: the edit window is the ground-truth injection window, which the method "
    "does not have at deployment. Diagnostic only; never reportable as an RSR result for C1."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 with the window forced to the injection")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "search.yaml")
    p.add_argument("--experiment-model", default="R", help="which checkpoint; R = model of record")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--budget-k", type=int, default=8, help="K=8 has the largest headroom (+0.60)")
    p.add_argument("--n-failures", type=int, default=None)
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument(
        "--mode",
        choices=("lambda-sweep", "bank"),
        default="lambda-sweep",
        help="lambda-sweep: search vs random at each lambda. bank: a diverse labelled "
        "candidate set for scripts/diagnose_objective_quality.py, and the exploitation test",
    )
    p.add_argument("--bank-lam", type=float, default=0.01, help="lambda used for bank mode")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    enc_path = REPO_ROOT / cfg["encoder_config"]
    env_path = REPO_ROOT / cfg["env_config"]
    run_id = args.run_id or f"p3-oracle-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
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

    print(f"diagnose_window_oracle  run_id={run_id}  seed={args.seed}  device={device}")
    print(f"model={exp['dyn_run']}  K={k}  lambdas={lambdas}")
    print(f"!! {NOT_A_RESULT}")
    print("-" * 78)

    models, dyn_cfg, _ = load_models(exp["dyn_run"], device)
    cache, renderer, _, _ = open_cache({"encoder_config": cfg["encoder_config"], **cfg}, args.seed)

    records = load_failure_set(failure_set_dir(REPO_ROOT), source=INJECTED, task_id=task_id)
    # The same selection G3 used, so the two are directly comparable episode by episode.
    chosen = select_failures(records, n_failures, bool(r_cfg["g3_stratify_by_cause"]))

    rows, skipped = [], []
    for i, rec in enumerate(chosen):
        problem = build_problem(cache, rec, dyn_cfg, device)
        original = problem.actions
        w = int(rec.edit_window[0])

        # The forced window must still satisfy the objective's own preconditions: a real
        # context behind it and at least one whole chunk ahead. An episode that cannot supply
        # both is reported, not silently shifted to a window that works.
        valid = problem.valid_window_starts(k)
        if valid.size == 0 or w < valid[0] or w > valid[-1]:
            skipped.append(
                {
                    "failure_id": rec.failure_id,
                    "reason": "ground-truth window outside the valid range",
                    "truth_window_start": w,
                    "valid_range": [int(valid[0]), int(valid[-1])] if valid.size else None,
                }
            )
            print(f"  [{i + 1:2d}] {rec.failure_id[-34:]}: SKIPPED, window {w} out of range")
            continue

        if args.mode == "bank":
            # Six candidates spanning a wide range of optimisation pressure and quality. The
            # spread is the point: the screen in diagnose_objective_quality.py only learns from
            # pairs whose OUTCOMES DIFFER, and the lambda-sweep bank yielded just 52 informative
            # pairs out of 168 candidates - too few to separate two objectives. Diverse
            # candidates produce far more outcome variety per episode.
            #
            # It is also the model-exploitation test: CEM-best, the elite mean, and iteration
            # 1's best are the same search at decreasing optimisation pressure. If the
            # less-optimised ones replay BETTER, the search is exploiting model error.
            lam = float(args.bank_lam)
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

            best = Edit(window_start=w, actions=res.actions)
            scale = realised_magnitude(original, best)
            entries = [
                (
                    "no_edit",
                    propose_baseline(
                        "no_edit",
                        original_actions=original,
                        window_start=w,
                        budget_k=k,
                        rng=np.random.default_rng(0),
                    ),
                    "identity",
                ),
                ("search", best, "cem_best"),
                ("search", Edit(window_start=w, actions=res.elite_mean_actions), "cem_elite_mean"),
                ("search", Edit(window_start=w, actions=res.first_iter_actions), "cem_iter1_best"),
            ]
            for tag_i in (0, 1):
                entries.append(
                    (
                        "random_perturb",
                        propose_baseline(
                            RANDOM_PERTURB,
                            original_actions=original,
                            window_start=w,
                            budget_k=k,
                            rng=np.random.default_rng(
                                cell_seed(args.seed, rec.failure_id, RANDOM_PERTURB, k, lam, tag_i)
                            ),
                            magnitude_scale=scale,
                        ),
                        f"random_seed{tag_i}",
                    )
                )
            for arm, e, variant_tag in entries:
                rows.append(
                    {
                        "run_id": run_id,
                        "experiment": EXPERIMENT,
                        "diagnostic": True,
                        "not_a_result": NOT_A_RESULT,
                        "dyn_run": exp["dyn_run"],
                        "failure_id": rec.failure_id,
                        "episode_id": rec.episode_id,
                        "task_id": rec.task_id,
                        "cause": rec.cause,
                        "truth_window": list(rec.edit_window),
                        "arm": arm,
                        "budget_k": k,
                        # `lam` distinguishes bank entries that share an arm; without it the
                        # (failure_id, arm, lam) key collides and run_replay dedupes them away.
                        "lam": lam,
                        "variant": "primary",
                        "window_source": variant_tag,
                        "candidate": variant_tag,
                        "window_start": int(e.window_start),
                        "window_end": int(e.window_end),
                        "actions": np.asarray(e.actions).tolist(),
                        "not_applicable": None,
                        **describe_edit(original, e, sparsity_eps),
                    }
                )
            print(f"  [{i + 1:2d}/{len(chosen)}] {rec.failure_id[-34:]}: 6 bank candidates at {w}")
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
            rp = propose_baseline(
                RANDOM_PERTURB,
                original_actions=original,
                window_start=w,
                budget_k=k,
                rng=np.random.default_rng(
                    cell_seed(args.seed, rec.failure_id, RANDOM_PERTURB, k, lam)
                ),
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
                        "not_a_result": NOT_A_RESULT,
                        "dyn_run": exp["dyn_run"],
                        "failure_id": rec.failure_id,
                        "episode_id": rec.episode_id,
                        "task_id": rec.task_id,
                        "cause": rec.cause,
                        "truth_window": list(rec.edit_window),
                        "arm": arm,
                        "budget_k": k,
                        "lam": lam,
                        "variant": "primary",
                        "window_source": "ground_truth_forced",
                        "window_start": int(e.window_start),
                        "window_end": int(e.window_end),
                        "actions": np.asarray(e.actions).tolist(),
                        "not_applicable": None,
                        **describe_edit(original, e, sparsity_eps),
                        **extra,
                    }
                )
        print(
            f"  [{i + 1:2d}/{len(chosen)}] {rec.failure_id[-34:]}: window forced to {w} "
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
        "not_a_result": NOT_A_RESULT,
        "seed": args.seed,
        "dyn_run": exp["dyn_run"],
        "budget_k": k,
        "lambdas": lambdas,
        "n_failures": len(chosen),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "n_edits": len(rows),
        "identity_edits": sum(1 for r in rows if r["arm"] == SEARCH and r["search"]["is_identity"]),
        "renderer": renderer,
        "encoder_cfg_hash": cache.hash,
        "reference": "compare against p3-replay-RD @ K=8: no_edit@ground_truth 0.40, "
        "expert_oracle@ground_truth 1.00, search@stage1_top1 0.00-0.10",
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
