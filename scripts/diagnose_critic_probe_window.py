"""The last cheap combination: probe window + critic objective, together.

Two interventions were each tried alone today (PROGRESS.md 2026-08-10) and neither closed the
search-vs-random gap:

- The probe's window (LOEO, `ceiling-test-v2`) raised absolute RSR a lot (0.00-0.10 -> 0.21-0.36)
  using the ORIGINAL goal-distance objective, but `search` still didn't beat `random_perturb`.
- The critic objective, at the STANDARD divergence-heuristic window, stayed in the same low
  range as the original objective (0.02-0.05) — a null result that a bad window's small headroom
  cannot rule anything in or out on.

Neither test could see whether the critic does better than goal-distance *specifically where
there is headroom for a difference to show up*. This combines both: each episode forced to its
own LOEO probe-predicted window (never that episode's own label — see
`diagnose_window_probe.py`), stage 2 scored by the critic (see `diagnose_critic_search.py`).

If this still doesn't beat `random_perturb`, the honest reading is that the negative result is
about argmin-style CEM optimisation against ANY imperfect learned scorer — not about which
window or which objective it is handed — and SPEC.md §0's stated fallback applies: write up the
negative result rather than keep permuting the same two knobs.

**Tagged `CRITIC_PROBE_WINDOW`, absent from `configs/search.yaml:experiments`.** Real replayed
outcomes (I2 holds), not a registered method, not reportable as C1 until the user promotes it —
same status as every other diagnostic run today.

Usage:
    python scripts/diagnose_critic_probe_window.py --run-id p35-critic-probe-window
    python scripts/run_replay.py --search-run p35-critic-probe-window \
        --run-id p35-critic-probe-window-replay
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

EXPERIMENT = "CRITIC_PROBE_WINDOW"

NOT_A_RESULT = (
    "BOTH interventions at once: the window is a leave-one-episode-out probe prediction "
    "({probe_run}, never this episode's own label) and stage 2 optimises against a learned "
    "critic ({critic_run}/critic.pt), not the goal-distance objective SPEC.md §5 "
    "specifies. Neither is a registered method. Real replayed outcomes (I2 holds), but not "
    "reportable as a C1 result until the user promotes either or both."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe window + critic objective, combined")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "search.yaml")
    p.add_argument(
        "--experiment-model", default="R", help="dynamics checkpoint; R = model of record"
    )
    p.add_argument(
        "--probe-run", default="ceiling-test-v2", help="diagnose_fault_localization.py run_id"
    )
    p.add_argument("--critic-run", default="p35-critic", help="train_critic.py run_id")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--budget-k", type=int, default=8)
    p.add_argument("--n-failures", type=int, default=None)
    p.add_argument("--task-id", type=int, default=None)
    return p.parse_args()


def load_probe_windows(probe_run: str) -> dict[str, tuple[int, int] | None]:
    """Identical to `diagnose_window_probe.py`'s helper of the same name."""
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


def main() -> int:  # noqa: PLR0915 — a linear pipeline reads better than split-up stages
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    enc_path = REPO_ROOT / cfg["encoder_config"]
    env_path = REPO_ROOT / cfg["env_config"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"p35-critic-probe-window-{stamp}"
    exp = cfg["experiments"][args.experiment_model]

    import torch

    from cfe.data.failure_set import INJECTED, failure_set_dir, load_failure_set
    from cfe.search.arms import RANDOM_PERTURB, SEARCH, propose_baseline, realised_magnitude
    from cfe.search.cem import CEMConfig, cem_search
    from cfe.search.critic import DynamicsAndCritic, evaluate_candidates_critic, load_critic
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

    not_a_result = NOT_A_RESULT.format(probe_run=args.probe_run, critic_run=args.critic_run)
    print(f"diagnose_critic_probe_window  run_id={run_id}  seed={args.seed}  device={device}")
    print(f"dyn_run={exp['dyn_run']}  probe_run={args.probe_run}  critic_run={args.critic_run}")
    print(f"K={k}  lambdas={lambdas}")
    print(f"!! {not_a_result}")
    print("-" * 78)

    probe_windows = load_probe_windows(args.probe_run)
    models, dyn_cfg, _ = load_models(exp["dyn_run"], device)
    critic_path = REPO_ROOT / "results" / "runs" / args.critic_run / "critic.pt"
    critic, feat_mean, feat_std, critic_ck = load_critic(critic_path, device)
    if critic_ck["dyn_run"] != exp["dyn_run"]:
        raise ValueError(
            f"{args.critic_run} was trained rolling against dyn_run={critic_ck['dyn_run']!r}, "
            f"but this run uses {exp['dyn_run']!r}"
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
                }
            )
            print(f"  [{i + 1:2d}] {rec.failure_id[-34:]}: SKIPPED, window {w} out of range")
            continue

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
                            # 1 - P(success), not a latent distance — see
                            # evaluate_candidates_critic's docstring.
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
                        "critic_run": args.critic_run,
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
        "probe_run": args.probe_run,
        "critic_run": args.critic_run,
        "budget_k": k,
        "lambdas": lambdas,
        "n_failures": len(chosen),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "n_edits": len(rows),
        "renderer": renderer,
        "encoder_cfg_hash": cache.hash,
        "reference": "p35-probe-window-replay (goal-distance @ probe window): search 0.21-0.36; "
        "p35-critic-search-replay (critic @ stage1_top1): search 0.02-0.05",
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
