"""Run the counterfactual search and write every arm's proposed edit (SPEC.md §5).

Split from `run_replay.py` on purpose. The search is GPU-cheap and re-runnable in minutes; the
replay is CPU simulation at ~9 steps/s and takes hours. Emitting a plain `edits.jsonl` in
between means the expensive half can be resumed, re-run, or inspected without touching the
search, and it makes the **magnitude match auditable without a simulator**: every row records
the scale `random_perturb` was asked for and the magnitude it realised, computed with the
frozen `edit_magnitude`.

**Nothing this script prints is a result.** `objective`, `goal_distance` and the window
divergences are search diagnostics; CLAUDE.md rule 2 and SPEC.md I2 make replay the only
ground truth. The word "success" does not appear in this file's output.

Which window each arm is given, and why:

- `search` and `action_ignoring_dynamics` each run **both stages** against their own model, so
  the twin picks its own window. Isolating action-conditioning means isolating it in stage 1
  too; letting the twin inherit the trained model's window would leave half the comparison
  unmade.
- `random_perturb` is given the *search's* chosen window and the search's realised magnitude.
  That is deliberately the harder comparison — see `cfe/search/arms.py`.
- `no_edit` and `expert_oracle` are evaluated at **two** windows, both lambda-independent, and
  each row says which in `window_source`. `stage1_top1` is the window the search chose, so
  `no_edit` there is the control for the policy handover. `ground_truth` is the first K steps of
  the injected window, and it is the only `expert_oracle` that is an upper bound at all:
  measured 2026-08-04, stage 1's window overlaps the injected one with IoU 0.0, so
  `expert_oracle` at `stage1_top1` replays steps the injection never touched and comes out
  byte-identical to `no_edit`. Reporting that as the ceiling would say these failures are
  unrescuable when the experiment never tried to rescue them.
- The open-loop `no_edit` replays the whole original sequence, so it depends on neither the
  window nor K, and is emitted once per episode instead of eight times.

Usage:
    python scripts/run_search.py --config configs/search.yaml --experiment R --seed 0 \
        --run-id p3-search-R
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Counterfactual search; emits edits.jsonl")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "search.yaml")
    p.add_argument("--experiment", default="R", help="key in configs/search.yaml:experiments")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--n-failures", type=int, default=None, help="override replay.g3_n_failures")
    p.add_argument("--task-id", type=int, default=None)
    p.add_argument("--budgets", type=int, nargs="*", default=None, help="override the K grid")
    p.add_argument("--lambdas", type=float, nargs="*", default=None)
    return p.parse_args()


def cell_seed(base: int, *parts) -> int:
    """A stable per-cell seed, so one cell's RNG draw cannot depend on how many ran before it.

    Deriving from a hash of the cell's identity rather than advancing one global stream means a
    resumed or partially re-run job reproduces the same edits (I5).
    """
    import hashlib

    key = "|".join(str(p) for p in parts).encode()
    return (base + int(hashlib.sha256(key).hexdigest()[:8], 16)) % (2**31 - 1)


def select_failures(records: list, n: int, stratify: bool) -> list:
    """Pick `n` failures, round-robin over causes so no cause dominates the skeleton run."""
    if not stratify or n >= len(records):
        return records[:n]
    by_cause: dict[str, list] = {}
    for r in records:
        by_cause.setdefault(r.cause or "natural", []).append(r)
    out, i = [], 0
    while len(out) < n and any(by_cause.values()):
        for cause in sorted(by_cause):
            if len(out) >= n:
                break
            if i < len(by_cause[cause]):
                out.append(by_cause[cause][i])
        i += 1
    return sorted(out, key=lambda r: r.failure_id)[:n]


def load_models(dyn_run: str, device):
    """The trained model and its action-ignoring twin, from one `train_dynamics.py` checkpoint."""
    import torch

    from cfe.dynamics.model import DynamicsConfig, LatentDynamics

    path = REPO_ROOT / "results" / "runs" / dyn_run / "dynamics.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/train_dynamics.py first")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = DynamicsConfig.from_dict(ck["dynamics_config"])
    models = {}
    for tag, ignore in (("dynamics", False), ("action_ignoring", True)):
        m = LatentDynamics(cfg, ignore_actions=ignore).to(device)
        m.load_state_dict(ck[tag])
        models[tag] = m.eval()
    return models, cfg, ck


def build_problem(cache, record, dyn_cfg, device):
    """A `SearchProblem` for one failure: its latents, its actions, and its goal latents.

    `z_goal` is the terminal latent of the **uncorrupted source demo**, which for an injected
    failure shares this episode's initial state by construction — exactly what SPEC.md §5 asks
    for. A natural failure has no such demo, and this raises rather than substituting one.
    """
    import torch

    from cfe.dynamics.data import load_episode
    from cfe.search.objective import SearchProblem

    ep = load_episode(cache, record.episode_id, dyn_cfg.reduction)
    if record.expert_episode_id is None:
        raise ValueError(
            f"{record.failure_id}: no expert episode, so no goal latent from the same initial "
            f"state (SPEC.md §5). Natural failures need the multi-demo minimum; that path is "
            f"Phase 4 and is not reached by G3."
        )
    expert = load_episode(cache, record.expert_episode_id, dyn_cfg.reduction)
    z_goal = torch.from_numpy(expert.z[-1:].astype(np.float32))

    return SearchProblem(
        episode_id=record.episode_id,
        z=torch.from_numpy(ep.z.astype(np.float32)).to(device),
        actions=np.asarray(record.actions, dtype=np.float64),
        z_goal=z_goal.to(device),
        k_context=dyn_cfg.k_context,
        horizon=dyn_cfg.horizon,
    )


def describe_edit(original: np.ndarray, edit, sparsity_eps: list[float]) -> dict:
    """Everything SPEC.md §6 reports about the edit itself, from the frozen metrics."""
    from cfe.eval.metrics import edit_magnitude, edit_sparsity, gripper_flips

    window = original[edit.window_start : edit.window_end]
    mag = edit_magnitude(window, edit.actions)
    return {
        "realised_mean_l2_ee": mag["mean_l2_ee"],
        "realised_max_l2_ee": mag["max_l2_ee"],
        "gripper_flips": gripper_flips(window, edit.actions)["gripper_flips"],
        "sparsity": {
            str(e): edit_sparsity(window, edit.actions, e)["n_steps_changed_ee"]
            for e in sparsity_eps
        },
    }


def main() -> int:  # noqa: PLR0915 — a linear pipeline reads better than split-up stages
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    enc_path = REPO_ROOT / cfg["encoder_config"]
    env_path = REPO_ROOT / cfg["env_config"]
    env_cfg = yaml.safe_load(env_path.read_text())
    run_id = args.run_id or f"p3-search-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    if args.experiment not in cfg["experiments"]:
        raise ValueError(f"unknown experiment {args.experiment!r}; have {list(cfg['experiments'])}")
    exp = cfg["experiments"][args.experiment]

    os.environ.setdefault("MUJOCO_GL", env_cfg["render_backend"])

    import torch

    from cfe.data.failure_set import INJECTED, failure_set_dir, load_failure_set
    from cfe.search.arms import (
        ACTION_IGNORING,
        EXPERT_ORACLE,
        NO_EDIT,
        RANDOM_PERTURB,
        SEARCH,
        NotApplicable,
        assert_arms_cover_baselines,
        edit_from_result,
        propose_baseline,
        realised_magnitude,
        run_search_arm,
    )
    from cfe.search.cem import CEMConfig
    from cfe.search.windows import per_step_divergence, rank_windows, window_recall_of_truth
    from scripts.train_dynamics import open_cache

    s_cfg = cfg["search"]
    r_cfg = cfg["replay"]
    arms = list(r_cfg["arms"])
    assert_arms_cover_baselines(arms)

    budgets = args.budgets or [int(k) for k in s_cfg["objective"]["budgets"]]
    lambdas = args.lambdas or [float(x) for x in s_cfg["objective"]["lambdas"]]
    sparsity_eps = [float(e) for e in s_cfg["sparsity_eps"]]
    w_cfg = s_cfg["windows"]
    cem_cfg = CEMConfig.from_dict(s_cfg["cem"])
    task_id = int(args.task_id if args.task_id is not None else r_cfg["g3_task_id"])
    n_failures = int(args.n_failures or r_cfg["g3_n_failures"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"run_search  run_id={run_id}  experiment={args.experiment}  seed={args.seed}")
    print(f"dyn_run={exp['dyn_run']}  device={device}")
    print(f"K={budgets}  lambda={lambdas}  arms={arms}")
    print("-" * 78)

    models, dyn_cfg, ck = load_models(exp["dyn_run"], device)
    cache, renderer, _, _ = open_cache({"encoder_config": cfg["encoder_config"], **cfg}, args.seed)

    records = load_failure_set(failure_set_dir(REPO_ROOT), source=INJECTED, task_id=task_id)
    if not records:
        raise RuntimeError(
            f"no injected failures for task {task_id}; run scripts/build_failure_set.py first"
        )
    chosen = select_failures(records, n_failures, bool(r_cfg["g3_stratify_by_cause"]))
    print(f"failure set: {len(records)} injected on task {task_id}; using {len(chosen)}")
    for r in chosen:
        print(f"  {r.failure_id}  cause={r.cause}  T={r.n_steps}  truth={r.edit_window}")
    print("-" * 78)

    rows, diagnostics = [], []
    t_start = time.time()
    n_fwd_total = 0

    # The uncorrupted source demo's actions, once per episode. `expert_oracle` needs them at
    # every (K, window), and reading the HDF5 inside those loops re-opened the file 8x per
    # episode and re-printed LIBERO's banner each time.
    from cfe.data.demos import demo_file_path, read_episode
    from scripts.verify_env import load_suite

    dataset_dir = yaml.safe_load(enc_path.read_text())["data"]["dataset_dir"]

    def expert_actions_for(rec) -> np.ndarray | None:
        if rec.expert_episode_id is None:
            return None
        suite = load_suite({"suite": rec.suite})
        task = suite.get_task(rec.task_id)
        local = rec.expert_episode_id.split(f"{rec.suite}_t{rec.task_id}_", 1)[1]
        path = demo_file_path(dataset_dir, rec.suite, task)
        return np.asarray(read_episode(path, local)["actions"], dtype=np.float64)

    for r_i, rec in enumerate(chosen):
        problem = build_problem(cache, rec, dyn_cfg, device)
        original = problem.actions
        expert = expert_actions_for(rec)

        # One divergence pass per model per episode; every K reads the same array.
        div = {
            tag: per_step_divergence(models[tag], problem, int(w_cfg["score_horizon"]))
            for tag in ("dynamics", "action_ignoring")
        }
        diagnostics.append(
            {
                "kind": "episode",
                "failure_id": rec.failure_id,
                "cause": rec.cause,
                "n_steps": rec.n_steps,
                "truth_window": list(rec.edit_window),
                "divergence_mean": float(np.nanmean(div["dynamics"])),
                "divergence_max": float(np.nanmax(div["dynamics"])),
                "divergence_argmax": int(np.nanargmax(div["dynamics"])),
            }
        )

        for k in budgets:
            # Window-level controls, evaluated once per (episode, K): lambda-independent.
            base_ranked = rank_windows(div["dynamics"], problem, k, int(w_cfg["stride"]), 1)
            if not base_ranked:
                print(f"  {rec.failure_id} K={k}: no valid window; skipped and reported")
                diagnostics.append({"kind": "skipped", "failure_id": rec.failure_id, "budget_k": k})
                continue
            base_w = base_ranked[0]["window_start"]

            # Two windows for the two window-level controls, because they answer different
            # questions and the first smoke run showed the difference is not academic:
            #
            #   stage1_top1  — the window the search actually chose. Paired with the search
            #                  arms, so `no_edit` here is the control for the policy handover.
            #   ground_truth — the first K steps of the *injected* window. This is the only
            #                  `expert_oracle` that is an upper bound: measured on 2026-08-04,
            #                  stage 1's window overlaps the injected one with IoU 0.0, so
            #                  `expert_oracle` at stage1_top1 replays actions the injection never
            #                  touched and is byte-identical to `no_edit`. Reporting that as the
            #                  ceiling would say these failures are unrescuable when the
            #                  experiment never tried to rescue them.
            windows = {"stage1_top1": base_w}
            truth_w = rec.edit_window[0] if rec.edit_window is not None else None
            if (
                truth_w is not None
                and truth_w >= dyn_cfg.k_context - 1
                and truth_w + k <= len(original)
            ):
                windows["ground_truth"] = int(truth_w)

            for arm in (NO_EDIT, EXPERT_ORACLE):
                if arm not in arms:
                    continue
                for w_source, w_start in windows.items():
                    try:
                        edit = propose_baseline(
                            arm,
                            original_actions=original,
                            window_start=w_start,
                            budget_k=k,
                            rng=np.random.default_rng(
                                cell_seed(args.seed, rec.failure_id, arm, k, w_source)
                            ),
                            expert_actions=expert if arm == EXPERT_ORACLE else None,
                            cause=rec.cause,
                        )
                    except NotApplicable as exc:
                        rows.append(
                            _row(
                                run_id,
                                args.experiment,
                                exp["dyn_run"],
                                rec,
                                arm,
                                k,
                                None,
                                "primary",
                                None,
                                not_applicable=str(exc),
                                window_source=w_source,
                            )
                        )
                        continue
                    rows.append(
                        _row(
                            run_id,
                            args.experiment,
                            exp["dyn_run"],
                            rec,
                            arm,
                            k,
                            None,
                            "primary",
                            edit,
                            describe=describe_edit(original, edit, sparsity_eps),
                            window_source=w_source,
                        )
                    )

            # The open-loop `no_edit` — §6's replay-bug detector — replays the whole original
            # action sequence, so it does not depend on the window or on K. Emitted once per
            # episode rather than 4x2 times, which would cost ~40 minutes of simulation to
            # re-measure the same trajectory.
            if (
                NO_EDIT in arms
                and k == budgets[0]
                and NO_EDIT in r_cfg.get("both_variants_for", [])
            ):
                rows.append(
                    _row(
                        run_id,
                        args.experiment,
                        exp["dyn_run"],
                        rec,
                        NO_EDIT,
                        k,
                        None,
                        "open_loop",
                        propose_baseline(
                            NO_EDIT,
                            original_actions=original,
                            window_start=base_w,
                            budget_k=k,
                            rng=np.random.default_rng(0),
                        ),
                        describe={"realised_mean_l2_ee": 0.0, "realised_max_l2_ee": 0.0},
                        window_source="stage1_top1",
                    )
                )

            for lam in lambdas:
                # -- the two search arms, each with its own model and its own window --
                results = {}
                for arm, tag in ((SEARCH, "dynamics"), (ACTION_IGNORING, "action_ignoring")):
                    if arm not in arms:
                        continue
                    res, _ = run_search_arm(
                        models[tag],
                        problem,
                        k,
                        lam,
                        cem_cfg,
                        np.random.default_rng(cell_seed(args.seed, rec.failure_id, arm, k, lam)),
                        stride=int(w_cfg["stride"]),
                        top_m=int(w_cfg["top_m"]),
                        score_horizon=int(w_cfg["score_horizon"]),
                        divergence=div[tag],
                    )
                    if res is None:
                        continue
                    results[arm] = res
                    n_fwd_total += res.n_forward_passes
                    edit = edit_from_result(res)
                    ranked = rank_windows(
                        div[tag], problem, k, int(w_cfg["stride"]), int(w_cfg["top_m"])
                    )
                    rows.append(
                        _row(
                            run_id,
                            args.experiment,
                            exp["dyn_run"],
                            rec,
                            arm,
                            k,
                            lam,
                            "primary",
                            edit,
                            describe=describe_edit(original, edit, sparsity_eps),
                            search={
                                "objective": res.objective,
                                "goal_distance": res.goal_distance,
                                "penalty": res.penalty,
                                "baseline_objective": res.baseline_objective,
                                "is_identity": res.is_identity,
                                "n_forward_passes": res.n_forward_passes,
                            },
                            window_iou=window_recall_of_truth(ranked, rec.edit_window),
                        )
                    )

                # -- random_perturb, matched to the search's window AND magnitude --
                if RANDOM_PERTURB in arms and SEARCH in results:
                    src = results[SEARCH]
                    src_edit = edit_from_result(src)
                    scale = realised_magnitude(original, src_edit)
                    rp = propose_baseline(
                        RANDOM_PERTURB,
                        original_actions=original,
                        window_start=src.window_start,
                        budget_k=k,
                        rng=np.random.default_rng(
                            cell_seed(args.seed, rec.failure_id, RANDOM_PERTURB, k, lam)
                        ),
                        magnitude_scale=scale,
                    )
                    rows.append(
                        _row(
                            run_id,
                            args.experiment,
                            exp["dyn_run"],
                            rec,
                            RANDOM_PERTURB,
                            k,
                            lam,
                            "primary",
                            rp,
                            describe=describe_edit(original, rp, sparsity_eps),
                            magnitude_scale_requested=scale,
                        )
                    )

        print(
            f"[{r_i + 1:2d}/{len(chosen)}] {rec.failure_id}: "
            f"{sum(1 for x in rows if x['failure_id'] == rec.failure_id)} edits"
        )

    elapsed = time.time() - t_start
    summary = {
        "run_id": run_id,
        "kind": "summary",
        "experiment": args.experiment,
        "dyn_run": exp["dyn_run"],
        "gated": bool(exp.get("gated", False)),
        "seed": args.seed,
        "task_id": task_id,
        "device": str(device),
        "renderer": renderer,
        "encoder_cfg_hash": cache.hash,
        "reduction": dyn_cfg.reduction,
        "residual": dyn_cfg.residual,
        "n_failures": len(chosen),
        "failure_ids": [r.failure_id for r in chosen],
        "budgets": budgets,
        "lambdas": lambdas,
        "arms": arms,
        "n_edits": len(rows),
        "n_replays_required": len(rows),
        "search_forward_passes_total": n_fwd_total,
        "search_wall_clock_s": round(elapsed, 1),
        "dyn_val_episode_ids": ck["val_episode_ids"],
        "note": "search diagnostics only; no number here is a result (CLAUDE.md rule 2)",
    }

    out_dir = REPO_ROOT / "results" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, out_dir / "search.yaml")
    shutil.copyfile(enc_path, out_dir / "encoder.yaml")
    shutil.copyfile(env_path, out_dir / "env.yaml")
    shutil.copyfile(REPO_ROOT / cfg["dynamics_config"], out_dir / "dynamics.yaml")
    with (out_dir / "edits.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    with (out_dir / "metrics.jsonl").open("w") as fh:
        for d in diagnostics:
            fh.write(json.dumps({"run_id": run_id, **d}, default=str) + "\n")
        fh.write(json.dumps(summary, default=str) + "\n")

    print("-" * 78)
    print(json.dumps({k: v for k, v in summary.items() if k != "dyn_val_episode_ids"}, indent=2))
    print(f"artifacts: {out_dir}")
    print(f"next: scripts/run_replay.py --search-run {run_id}")
    return 0


def _row(
    run_id,
    experiment,
    dyn_run,
    rec,
    arm,
    budget_k,
    lam,
    variant,
    edit,
    *,
    describe=None,
    search=None,
    window_iou=None,
    magnitude_scale_requested=None,
    not_applicable=None,
    window_source="search",
) -> dict:
    """One replayable edit. `lam=None` marks a lambda-independent arm, not a missing value."""
    row = {
        "run_id": run_id,
        "experiment": experiment,
        "dyn_run": dyn_run,
        "failure_id": rec.failure_id,
        "episode_id": rec.episode_id,
        "task_id": rec.task_id,
        "cause": rec.cause,
        "truth_window": list(rec.edit_window) if rec.edit_window is not None else None,
        "arm": arm,
        "budget_k": int(budget_k),
        "lam": lam,
        "variant": variant,
        "window_source": window_source,
        "not_applicable": not_applicable,
    }
    if edit is not None:
        row |= {
            "window_start": int(edit.window_start),
            "window_end": int(edit.window_end),
            "actions": np.asarray(edit.actions).tolist(),
        }
    if describe is not None:
        row |= describe
    if search is not None:
        row["search"] = search
    if window_iou is not None:
        row["window_iou_vs_truth"] = window_iou
    if magnitude_scale_requested is not None:
        row["magnitude_scale_requested"] = float(magnitude_scale_requested)
    return row


if __name__ == "__main__":
    sys.exit(main())
