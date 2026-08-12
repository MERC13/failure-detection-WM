"""The fully deployable pipeline: probe window + bank-mode CEM + critic re-rank. No oracle
information anywhere.

2026-08-10's re-rank finding was real (bootstrap CI, McNemar cross-check) but oracle-assisted in
one respect: it selected among candidates generated at the *ground-truth* fault window
(`p3-bank-replay`), which a deployed method never has. This closes that gap: each episode's
window comes from the LOEO probe (never that episode's own label —
`diagnose_fault_localization.py`), not the injection record, and stage 2 still just picks among
CEM's own candidates rather than being driven to convergence against a possibly-exploitable
objective.

**Cheaper than a naive bank-mode run.** A deployed method only ever executes ONE candidate per
episode — there is no reason to replay all four. This generates the four bank candidates
(`no_edit`, `cem_best`, `cem_elite_mean`, `cem_iter1_best`) via the STANDARD goal-distance CEM
(unchanged — the point of re-rank-only is that optimisation itself is untouched), scores all
four with the critic (no simulator), selects the argmax, and replays only that one candidate
plus a magnitude-matched `random_perturb` — half the replay cost of even `p35-probe-window-replay`
at a single lambda, and a small fraction of a full bank-mode-at-every-cell run.

**Tagged `CRITIC_RERANK_PROBE_WINDOW`, absent from `configs/search.yaml:experiments`.** Real
replayed outcomes (I2 holds), but a single lambda (`--bank-lam`, matching every prior bank-mode
run today) rather than a swept grid, and not a registered method — same status as everything
else this session.

Usage:
    python scripts/diagnose_critic_rerank_probe_window.py --run-id p35-critic-rerank-probe-window
    python scripts/run_replay.py --search-run p35-critic-rerank-probe-window \
        --run-id p35-critic-rerank-probe-window-replay
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

EXPERIMENT = "CRITIC_RERANK_PROBE_WINDOW"

NOT_A_RESULT = (
    "NO oracle information: the window is a leave-one-episode-out probe prediction "
    "({probe_run}, never this episode's own label) and the executed candidate is chosen by a "
    "critic ({critic_run}/critic.pt) among four candidates the STANDARD goal-distance CEM "
    "produced at that window — optimisation itself is unchanged from every other run this "
    "session. Single lambda ({bank_lam}), not a swept grid, and not a registered SPEC.md §5 "
    "method. Real replayed outcomes (I2 holds), but not reportable as a C1 result until "
    "promoted and swept."
)

#: Deterministic tie-breaking (I5) -- see PROGRESS.md 2026-08-10's re-rank entry for why a
#: Python set + max(..., key=...) is NOT safe here: cem_best ties bit-identically with identity
#: whenever CEM's optimum IS the unedited window, and set iteration order is hash-seed dependent.
BANK_TAGS = ["cem_best", "cem_elite_mean", "cem_iter1_best", "identity"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe window + bank-mode CEM + critic re-rank")
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
    p.add_argument(
        "--bank-lam", type=float, default=0.01, help="matches every prior bank-mode run today"
    )
    p.add_argument("--n-failures", type=int, default=None)
    p.add_argument("--task-id", type=int, default=None)
    return p.parse_args()


def load_probe_windows(probe_run: str) -> dict[str, tuple[int, int] | None]:
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


def pick_best(scores: dict[str, float]) -> str:
    """argmin 1-P(success) i.e. argmax P(success), ties broken by BANK_TAGS's fixed order."""
    return min(BANK_TAGS, key=lambda t: (scores[t], BANK_TAGS.index(t)))


def main() -> int:  # noqa: PLR0915 — a linear pipeline reads better than split-up stages
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    enc_path = REPO_ROOT / cfg["encoder_config"]
    env_path = REPO_ROOT / cfg["env_config"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"p35-critic-rerank-probe-window-{stamp}"
    exp = cfg["experiments"][args.experiment_model]

    import torch

    from cfe.data.failure_set import INJECTED, failure_set_dir, load_failure_set
    from cfe.eval.replay_protocol import Edit
    from cfe.search.arms import NO_EDIT, RANDOM_PERTURB, propose_baseline, realised_magnitude
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
    sparsity_eps = [float(e) for e in s_cfg["sparsity_eps"]]
    cem_cfg = CEMConfig.from_dict(s_cfg["cem"])
    k = int(args.budget_k)
    bank_lam = float(args.bank_lam)
    task_id = int(args.task_id if args.task_id is not None else r_cfg["g3_task_id"])
    n_failures = int(args.n_failures or r_cfg["g3_n_failures"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    not_a_result = NOT_A_RESULT.format(
        probe_run=args.probe_run, critic_run=args.critic_run, bank_lam=bank_lam
    )
    print(f"diagnose_critic_rerank_probe_window  run_id={run_id}  seed={args.seed}")
    print(f"device={device}  dyn_run={exp['dyn_run']}")
    print(f"probe_run={args.probe_run}  critic_run={args.critic_run}  K={k}  bank_lam={bank_lam}")
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
    # The failure set grew (56 -> 160 injected, PROGRESS.md 2026-08-10's more-data experiment)
    # after ceiling-test-v2 was built, so it only has predictions for the original 56. Restrict
    # to those BEFORE stratified selection, or select_failures silently picks a different,
    # partially-uncovered 56 from the larger pool -- not comparable to p35-probe-window-replay.
    covered = [r for r in records if r.failure_id in probe_windows]
    print(f"records covered by --probe-run {args.probe_run}: {len(covered)}/{len(records)}")
    chosen = select_failures(covered, n_failures, bool(r_cfg["g3_stratify_by_cause"]))

    rows, skipped, diverged = [], [], 0
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
            skipped.append({"failure_id": rec.failure_id, "reason": "window outside valid range"})
            print(f"  [{i + 1:2d}] {rec.failure_id[-34:]}: SKIPPED, window {w} out of range")
            continue

        # -- generate the 4 bank candidates with the STANDARD (unchanged) CEM --
        res = cem_search(
            models["dynamics"],
            problem,
            w,
            k,
            bank_lam,
            cem_cfg,
            np.random.default_rng(cell_seed(args.seed, rec.failure_id, "search", k, bank_lam)),
        )
        no_edit_edit = propose_baseline(
            NO_EDIT,
            original_actions=original,
            window_start=w,
            budget_k=k,
            rng=np.random.default_rng(0),
        )
        candidates = {
            "cem_best": res.actions,
            "cem_elite_mean": res.elite_mean_actions,
            "cem_iter1_best": res.first_iter_actions,
            "identity": no_edit_edit.actions,
        }

        # -- score all 4 with the critic, no simulator --
        batch = np.stack([candidates[t] for t in BANK_TAGS]).astype(np.float32)
        out = evaluate_candidates_critic(bundle, problem, w, batch, lam=0.0)
        scores = dict(zip(BANK_TAGS, out["goal_distance"], strict=True))  # 1 - P(success)
        best_tag = pick_best(scores)
        if best_tag != "cem_best":
            diverged += 1
        selected = Edit(window_start=w, actions=candidates[best_tag])

        scale = realised_magnitude(original, selected)
        rp_seed = cell_seed(args.seed, rec.failure_id, RANDOM_PERTURB, k, bank_lam)
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
                "search",
                selected,
                {
                    "selected_variant": best_tag,
                    "critic_scores": {t: float(s) for t, s in scores.items()},
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
                    "arm": arm,
                    "budget_k": k,
                    "lam": bank_lam,
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
            f"  [{i + 1:2d}/{len(chosen)}] {rec.failure_id[-34:]}: window {w}, selected={best_tag}"
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
        "critic_run": args.critic_run,
        "budget_k": k,
        "bank_lam": bank_lam,
        "n_failures": len(chosen),
        "n_skipped": len(skipped),
        "skipped": skipped,
        "n_edits": len(rows),
        "n_diverged_from_cem_best": diverged,
        "renderer": renderer,
        "encoder_cfg_hash": cache.hash,
        "reference": "p35-probe-window-replay (always cem_best @ probe window): search "
        "0.21-0.36; oracle-window re-rank (2026-08-10, p3-bank-replay analysis): 0.643 vs "
        "random 0.38-0.39",
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
