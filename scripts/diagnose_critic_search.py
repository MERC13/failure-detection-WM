"""Does the learned critic, wired into real stage-2 optimisation, beat random_perturb?

`critic-screen-v1` (PROGRESS.md 2026-08-10) cleared the goal-distance objective's ranking
accuracy (0.711 vs 0.6211-0.6731) on already-replayed data, with no new simulation.
`scripts/train_critic.py` then fit the deployment checkpoint (`p35-critic`) on all of it. This
is the step that was withheld until the user chose it explicitly (2026-08-10, "Replace the
objective"): swap SPEC.md §5's `D(z_end, z_goal)` for `1 - P(success | z_end)` **inside real
CEM optimisation**, and replay the result. A ranking-accuracy screen is not evidence this
particular thing (an optimum found by driving CEM to the argmin of a learned function) will
replay well — this project has already measured that a screen clearing chance does not
guarantee a stage-2 win, since the *original* objective's 62-67% accuracy did not beat
`random_perturb` either. This is the only way to find out.

**Isolates the objective, not the window.** Stage 1 here is the SAME divergence heuristic
`run_search.py` uses — untouched — so this differs from `p3-replay-RD`'s `stage1_top1` numbers
in exactly one respect: what stage 2 optimises against. Combining this with the LOEO probe's
window (2026-08-10) is a separate experiment for later, once each intervention has been judged
on its own; testing both changes in one run would make it impossible to attribute a result to
either.

**Tagged `CRITIC_OBJECTIVE`, absent from `configs/search.yaml:experiments`** — same convention
as `ORACLE` and `PROBE_WINDOW`, so `verify_g3.py:gated_rows` cannot pick this up. Promoting a
critic-based stage 2 into the registered method is a SPEC.md §5 change and remains the user's
call regardless of what this run shows.

Usage:
    python scripts/diagnose_critic_search.py --run-id p35-critic-search
    python scripts/run_replay.py --search-run p35-critic-search --run-id p35-critic-search-replay
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

#: Deliberately not a key in `configs/search.yaml:experiments` — see the module docstring.
EXPERIMENT = "CRITIC_OBJECTIVE"

NOT_A_RESULT = (
    "stage 2 optimises against a LEARNED critic (results/runs/{critic_run}/critic.pt), not the "
    "goal-distance objective SPEC.md §5 specifies and not yet a registered experiment. Stage 1 "
    "(window selection) is unchanged from the standard divergence heuristic, so this isolates "
    "the objective swap. These ARE real replayed outcomes (I2 holds) but this cannot be "
    "reported as a C1 result or as SPEC.md §5's method until the user promotes it."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real CEM optimisation against the learned critic")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "search.yaml")
    p.add_argument(
        "--experiment-model", default="R", help="dynamics checkpoint; R = model of record"
    )
    p.add_argument("--critic-run", default="p35-critic", help="train_critic.py run_id")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--budget-k", type=int, default=8)
    p.add_argument("--n-failures", type=int, default=None)
    p.add_argument("--task-id", type=int, default=None)
    return p.parse_args()


def main() -> int:  # noqa: PLR0915 — a linear pipeline reads better than split-up stages
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    enc_path = REPO_ROOT / cfg["encoder_config"]
    env_path = REPO_ROOT / cfg["env_config"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"p35-critic-search-{stamp}"
    exp = cfg["experiments"][args.experiment_model]

    import torch

    from cfe.data.failure_set import INJECTED, failure_set_dir, load_failure_set
    from cfe.search.arms import RANDOM_PERTURB, SEARCH, propose_baseline, realised_magnitude
    from cfe.search.cem import CEMConfig, cem_search
    from cfe.search.critic import DynamicsAndCritic, evaluate_candidates_critic, load_critic
    from cfe.search.windows import per_step_divergence, rank_windows
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
    w_cfg = s_cfg["windows"]
    cem_cfg = CEMConfig.from_dict(s_cfg["cem"])
    k = int(args.budget_k)
    task_id = int(args.task_id if args.task_id is not None else r_cfg["g3_task_id"])
    n_failures = int(args.n_failures or r_cfg["g3_n_failures"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    not_a_result = NOT_A_RESULT.format(critic_run=args.critic_run)
    print(f"diagnose_critic_search  run_id={run_id}  seed={args.seed}  device={device}")
    print(f"dyn_run={exp['dyn_run']}  critic_run={args.critic_run}  K={k}  lambdas={lambdas}")
    print(f"!! {not_a_result}")
    print("-" * 78)

    models, dyn_cfg, _ = load_models(exp["dyn_run"], device)
    critic_path = REPO_ROOT / "results" / "runs" / args.critic_run / "critic.pt"
    critic, feat_mean, feat_std, critic_ck = load_critic(critic_path, device)
    if critic_ck["dyn_run"] != exp["dyn_run"]:
        raise ValueError(
            f"{args.critic_run} was trained rolling against dyn_run={critic_ck['dyn_run']!r}, "
            f"but this run uses {exp['dyn_run']!r} — the critic's features would not match "
            f"what it was fit on"
        )
    bundle = DynamicsAndCritic(
        dynamics=models["dynamics"],
        critic=critic,
        feature_mean=torch.from_numpy(feat_mean).to(device),
        feature_std=torch.from_numpy(feat_std).to(device),
    )
    cache, renderer, _, _ = open_cache({"encoder_config": cfg["encoder_config"], **cfg}, args.seed)

    records = load_failure_set(failure_set_dir(REPO_ROOT), source=INJECTED, task_id=task_id)
    chosen = select_failures(records, n_failures, bool(r_cfg["g3_stratify_by_cause"]))

    rows, skipped = [], []
    for i, rec in enumerate(chosen):
        problem = build_problem(cache, rec, dyn_cfg, device)
        original = problem.actions

        # Stage 1: the standard divergence heuristic, untouched — see the module docstring.
        div = per_step_divergence(models["dynamics"], problem, int(w_cfg["score_horizon"]))
        ranked = rank_windows(div, problem, k, int(w_cfg["stride"]), 1)
        if not ranked:
            skipped.append({"failure_id": rec.failure_id, "reason": "no valid stage-1 window"})
            print(f"  [{i + 1:2d}] {rec.failure_id[-34:]}: SKIPPED, no valid window")
            continue
        w = ranked[0]["window_start"]

        for lam in lambdas:
            res = cem_search(
                bundle,
                problem,
                w,
                k,
                lam,
                cem_cfg,
                np.random.default_rng(cell_seed(args.seed, rec.failure_id, SEARCH, k, lam)),
                evaluate_fn=evaluate_candidates_critic,
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
                            # NOTE: "goal_distance" here is 1 - P(success), not a latent
                            # distance — see evaluate_candidates_critic's docstring.
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
                        "critic_run": args.critic_run,
                        "failure_id": rec.failure_id,
                        "episode_id": rec.episode_id,
                        "task_id": rec.task_id,
                        "cause": rec.cause,
                        "truth_window": list(rec.edit_window),
                        "arm": arm,
                        "budget_k": k,
                        "lam": lam,
                        "variant": "primary",
                        "window_source": "stage1_top1",
                        "window_start": int(e.window_start),
                        "window_end": int(e.window_end),
                        "actions": np.asarray(e.actions).tolist(),
                        "not_applicable": None,
                        **describe_edit(original, e, sparsity_eps),
                        **extra,
                    }
                )
        n_lam = len(lambdas)
        print(f"  [{i + 1:2d}/{len(chosen)}] {rec.failure_id[-34:]}: window {w}, {n_lam} lambdas")

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
        "critic_run": args.critic_run,
        "budget_k": k,
        "lambdas": lambdas,
        "n_failures": len(chosen),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "n_edits": len(rows),
        "renderer": renderer,
        "encoder_cfg_hash": cache.hash,
        "reference": "compare against p3-replay-RD @ K=8: search@stage1_top1 (goal-distance "
        "objective) 0.00-0.10; and p35-probe-window-replay: search@probe_window 0.21-0.36",
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
