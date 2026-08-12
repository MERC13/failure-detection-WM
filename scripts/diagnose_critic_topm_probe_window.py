"""Top-M probe windows + bank-mode CEM + critic re-rank. No oracle information anywhere.

SPEC.md §5 always specified stage 1 as "rank candidates... keep the top `M` for stage 2" — every
probe-window pipeline this session used only the single best-predicted window (M=1), which
`diagnose_fault_localization.py`'s original `top_window` simplified down to when first built, not
a deliberate departure. `top_m_windows` (added alongside it) restores the M>1 version.

**Why this might matter when M=1 didn't.** Screened for free before any of this ran
(PROGRESS.md 2026-08-11): mean IoU averaged over the single top-1 window is 0.263, but the BEST
of the top-3 windows per episode averages 0.309 (median 0.4 vs 0.252) — the true fault window is
often captured *somewhere* in the top-3 even when the top-1 pick misses it. This tests whether a
fully deployable selection mechanism (the critic, no oracle access to which of the 3 is actually
good) can find that better window when it's there.

**Mechanism.** For each episode, generate the four bank candidates
(`no_edit`/`cem_best`/`cem_elite_mean`/`cem_iter1_best`) via the STANDARD unchanged CEM at EACH
of the M probe-predicted windows (M x 4 candidates total), critic-score all of them, and select
the single GLOBAL best by critic score — across windows as well as candidate types. Still only
one candidate replayed per episode, plus a magnitude-matched `random_perturb` at that same
selected window.

**Tagged `CRITIC_TOPM_PROBE_WINDOW`, absent from `configs/search.yaml:experiments`.**

Usage:
    python scripts/diagnose_critic_topm_probe_window.py --run-id p35-critic-topm-probe-window
    python scripts/run_replay.py --search-run p35-critic-topm-probe-window \
        --run-id p35-critic-topm-probe-window-replay
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

EXPERIMENT = "CRITIC_TOPM_PROBE_WINDOW"

NOT_A_RESULT = (
    "NO oracle information: the M candidate windows are leave-one-episode-out probe "
    "predictions ({probe_run}, never this episode's own label); the executed candidate is "
    "chosen by a critic ({critic_run}/critic.pt) among the M x 4 candidates the STANDARD "
    "goal-distance CEM produced across all M windows -- optimisation itself is unchanged, only "
    "how many windows it is given. Single lambda ({bank_lam}), not a swept grid, and not a "
    "registered SPEC.md §5 method. Real replayed outcomes (I2 holds), but not reportable as a "
    "C1 result until promoted and swept."
)

#: Deterministic tie-breaking (I5) -- same reasoning as every prior rerank script this session.
CANDIDATE_TAGS = ["cem_best", "cem_elite_mean", "cem_iter1_best", "identity"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Top-M probe windows + bank-mode CEM + critic re-rank")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "search.yaml")
    p.add_argument(
        "--experiment-model", default="R", help="dynamics checkpoint; R = model of record"
    )
    p.add_argument(
        "--probe-run",
        default="ceiling-test-v6-topm3",
        help="a diagnose_fault_localization.py run built with --top-m > 1",
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


def load_probe_windows_m(probe_run: str) -> dict[str, list[tuple[int, int]]]:
    """`{failure_id: [(start, end), ...]}` from a `--top-m > 1` fault-localization run."""
    path = REPO_ROOT / "results" / "runs" / probe_run / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run diagnose_fault_localization.py first")
    out: dict[str, list[tuple[int, int]]] = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get("kind") != "part2_episode":
            continue
        fws = r.get("found_windows")
        if fws:
            out[r["failure_id"]] = [(int(a), int(b)) for a, b in fws]
        elif r.get("found_window"):
            out[r["failure_id"]] = [tuple(r["found_window"])]
    if not out:
        raise ValueError(f"{path} has no usable window predictions; wrong --probe-run?")
    return out


def pick_best(scores: dict[tuple[int, str], float]) -> tuple[int, str]:
    """argmin 1-P(success) over `(window_rank, candidate_tag) -> score`, ties broken by
    (earlier window rank, CANDIDATE_TAGS order)."""
    keys = sorted(scores, key=lambda k: (k[0], CANDIDATE_TAGS.index(k[1])))
    return min(keys, key=lambda k: scores[k])


def main() -> int:  # noqa: PLR0915 — a linear pipeline reads better than split-up stages
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    enc_path = REPO_ROOT / cfg["encoder_config"]
    env_path = REPO_ROOT / cfg["env_config"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"p35-critic-topm-probe-window-{stamp}"
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
    print(f"diagnose_critic_topm_probe_window  run_id={run_id}  seed={args.seed}")
    print(f"device={device}  dyn_run={exp['dyn_run']}")
    print(f"probe_run={args.probe_run}  critic_run={args.critic_run}  K={k}  bank_lam={bank_lam}")
    print(f"!! {not_a_result}")
    print("-" * 78)

    probe_windows_m = load_probe_windows_m(args.probe_run)
    m_counts = {len(v) for v in probe_windows_m.values()}
    print(f"windows per episode: {sorted(m_counts)}")
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
    covered = [r for r in records if r.failure_id in probe_windows_m]
    print(f"records covered by --probe-run {args.probe_run}: {len(covered)}/{len(records)}")
    chosen = select_failures(covered, n_failures, bool(r_cfg["g3_stratify_by_cause"]))

    rows, skipped = [], []
    n_selected_from_window_idx: dict[int, int] = {}
    for i, rec in enumerate(chosen):
        problem = build_problem(cache, rec, dyn_cfg, device)
        original = problem.actions
        valid = problem.valid_window_starts(k)

        windows = probe_windows_m[rec.failure_id]
        all_scores: dict[tuple[int, int, str], float] = {}
        all_actions: dict[tuple[int, int, str], np.ndarray] = {}
        usable_windows = []
        for w_idx, (w, _) in enumerate(windows):
            if valid.size == 0 or w < valid[0] or w > valid[-1]:
                continue
            usable_windows.append(w_idx)
            res = cem_search(
                models["dynamics"],
                problem,
                w,
                k,
                bank_lam,
                cem_cfg,
                np.random.default_rng(
                    cell_seed(args.seed, rec.failure_id, "search", k, bank_lam, w_idx)
                ),
            )
            no_edit_edit = propose_baseline(
                NO_EDIT,
                original_actions=original,
                window_start=w,
                budget_k=k,
                rng=np.random.default_rng(0),
            )
            cand_for_w = {
                "cem_best": res.actions,
                "cem_elite_mean": res.elite_mean_actions,
                "cem_iter1_best": res.first_iter_actions,
                "identity": no_edit_edit.actions,
            }
            batch = np.stack([cand_for_w[t] for t in CANDIDATE_TAGS]).astype(np.float32)
            out = evaluate_candidates_critic(bundle, problem, w, batch, lam=0.0)
            for t, s in zip(CANDIDATE_TAGS, out["goal_distance"], strict=True):
                all_scores[(w_idx, w, t)] = float(s)
                all_actions[(w_idx, w, t)] = cand_for_w[t]

        if not usable_windows:
            skipped.append({"failure_id": rec.failure_id, "reason": "no usable window"})
            print(f"  [{i + 1:2d}] {rec.failure_id[-34:]}: SKIPPED, no usable window")
            continue

        best_key = pick_best({(k0, k2): v for (k0, k1, k2), v in all_scores.items()})
        # best_key is (w_idx, tag); recover the matching (w_idx, w, tag) triple.
        best_w_idx, best_tag = best_key
        best_full_key = next(kk for kk in all_scores if kk[0] == best_w_idx and kk[2] == best_tag)
        best_w = best_full_key[1]
        selected = Edit(window_start=best_w, actions=all_actions[best_full_key])
        n_selected_from_window_idx[best_w_idx] = n_selected_from_window_idx.get(best_w_idx, 0) + 1

        scale = realised_magnitude(original, selected)
        rp_seed = cell_seed(args.seed, rec.failure_id, RANDOM_PERTURB, k, bank_lam)
        rp = propose_baseline(
            RANDOM_PERTURB,
            original_actions=original,
            window_start=best_w,
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
                    "selected_window_rank": int(best_w_idx),
                    "n_windows_considered": len(usable_windows),
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
                    "window_source": "probe_predicted_topm",
                    "window_start": int(e.window_start),
                    "window_end": int(e.window_end),
                    "actions": np.asarray(e.actions).tolist(),
                    "not_applicable": None,
                    **describe_edit(original, e, sparsity_eps),
                    **extra,
                }
            )
        print(
            f"  [{i + 1:2d}/{len(chosen)}] {rec.failure_id[-34:]}: "
            f"window_rank={best_w_idx} window={best_w} selected={best_tag}"
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
        "selected_window_rank_histogram": {
            str(k_): v for k_, v in sorted(n_selected_from_window_idx.items())
        },
        "renderer": renderer,
        "encoder_cfg_hash": cache.hash,
        "reference": "p35-critic-rerank-probe-window-v3-replay (M=1 window): search 0.304, "
        "random 0.268, diff +0.036 (does not exclude 0)",
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
