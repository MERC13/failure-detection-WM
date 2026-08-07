"""G3 gate (SPEC.md §7).

"end-to-end skeleton on one task, 10 failures: search -> replay -> RSR reported with
`random_perturb` and `no_edit`. **RSR must exceed `random_perturb` by a margin that survives a
bootstrap CI, on injected failures.**"

Every criterion is read from `configs/search.yaml:gate`, where it was pre-registered before any
search was run and before any replay number existed — the same discipline
`configs/dynamics.yaml` used for G2. Picking the bar after seeing the number is the failure
CLAUDE.md rule 3 describes, and this project has already amended a gate after a failure twice.

SPEC.md §7 is explicit that a FAIL is a result: "the honest move is to spend the remaining time
on why the world model is insufficient. That is a real result and a better project than a
fabricated positive."

Usage:
    python scripts/verify_g3.py --run p3-replay-R
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

from scripts.verify_env import Result  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G3 gate")
    p.add_argument("--run", required=True, help="run_replay.py run_id")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "search.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--skip-determinism",
        action="store_true",
        help="skip the search re-run check (needs the encoding cache and a GPU)",
    )
    return p.parse_args()


def load_replays(run: str) -> tuple[list[dict], dict]:
    path = REPO_ROOT / "results" / "runs" / run / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/run_replay.py first")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    replays = [r for r in rows if r.get("kind") == "replay"]
    summaries = [r for r in rows if r.get("kind") == "summary"]
    if not replays:
        raise RuntimeError(f"{path} holds no replay records")
    return replays, (summaries[-1] if summaries else {})


def gated_rows(replays: list[dict], cfg: dict) -> list[dict]:
    """The rows the gate is scored on: the gated experiment, primary variant, injected only."""
    gated = {k for k, v in cfg["experiments"].items() if v.get("gated")}
    return [
        r
        for r in replays
        if r.get("experiment") in gated
        and r.get("variant") == "primary"
        and r.get("cause") is not None
    ]


def paired_arms(rows: list[dict], arm_a: str, arm_b: str, budget_k: int, lam: float):
    """Success flags for two arms over the episodes they *both* cover, in a fixed order.

    `bootstrap_rsr_difference` is paired by episode and raises on a length mismatch. Taking the
    intersection rather than zipping whatever is present means an episode missing from one arm
    cannot silently shift the pairing and turn a comparison into noise.
    """

    def index(arm):
        return {
            r["failure_id"]: bool(r["success"])
            for r in rows
            if r["arm"] == arm and r["budget_k"] == budget_k and r["lam"] == lam
        }

    a, b = index(arm_a), index(arm_b)
    shared = sorted(set(a) & set(b))
    return [a[f] for f in shared], [b[f] for f in shared], shared


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------


def check_beats_random_perturb(rows: list[dict], gate: dict) -> Result:
    """Criterion 1. The margin must survive a paired bootstrap CI at every lambda.

    Every lambda, not the best one, because SPEC.md §6 says outright: "A method that wins only
    at one λ has not won." Reporting the winning lambda alone would be exactly the single tuned
    number I4 forbids.
    """
    from cfe.eval.metrics import bootstrap_rsr_difference

    g = gate["beats_random_perturb"]
    k = int(g["budget_k"])
    lambdas = sorted({r["lam"] for r in rows if r["arm"] == "search" and r["lam"] is not None})
    if not lambdas:
        return Result("beats_random_perturb", False, {"reason": "no search rows with a lambda"})

    per_lambda, ok = {}, True
    for lam in lambdas:
        t, c, shared = paired_arms(rows, "search", "random_perturb", k, lam)
        if not shared:
            per_lambda[str(lam)] = {"reason": "no episodes cover both arms"}
            ok = False
            continue
        out = bootstrap_rsr_difference(
            t, c, n_boot=int(g["n_bootstrap"]), seed=0, alpha=float(g["alpha"])
        )
        passed = out["excludes_zero"] and out["ci_low"] > 0
        per_lambda[str(lam)] = {
            "rsr_search": out["rsr_treatment"],
            "rsr_random_perturb": out["rsr_control"],
            "mean_difference": round(out["mean_difference"], 4),
            "ci_low": round(out["ci_low"], 4),
            "ci_high": round(out["ci_high"], 4),
            "excludes_zero": out["excludes_zero"],
            "n_episodes": out["n_episodes"],
            "passed": passed,
        }
        if bool(g.get("require_all_lambdas", True)):
            ok = ok and passed
        else:
            ok = ok or passed
    return Result(
        "beats_random_perturb",
        ok,
        {
            "budget_k": k,
            "require_all_lambdas": bool(g.get("require_all_lambdas", True)),
            "criterion": "paired bootstrap CI on (search - random_perturb) excludes zero, ci_low>0",
            "per_lambda": per_lambda,
        },
    )


def check_no_edit_open_loop(replays: list[dict], gate: dict) -> Result:
    """Criterion 2. §6's replay-bug detector.

    Only the OPEN-LOOP variant replays the original action sequence. The primary variant hands
    control to the BC policy after the window, which never happened to the source episode, so a
    non-zero there means the policy rescued it — informative, but not a protocol bug. Scoring
    the wrong variant here would either hide a real bug or condemn a real result.
    """
    from cfe.eval.metrics import replay_success_rate

    rows = [r for r in replays if r["arm"] == "no_edit" and r["variant"] == "open_loop"]
    if not rows:
        return Result(
            "no_edit_open_loop_is_zero",
            False,
            {
                "reason": "no open-loop no_edit rows. This is §6's only control for replay bugs; "
                "a table without it cannot detect a broken protocol."
            },
        )
    results = [_as_result(r) for r in rows]
    rsr = replay_success_rate(results)
    limit = float(gate["max_no_edit_open_loop_rsr"])
    return Result(
        "no_edit_open_loop_is_zero",
        rsr["replay_success_rate"] <= limit,
        {
            "replay_success_rate": rsr["replay_success_rate"],
            "n_success": rsr["n_success"],
            "n": rsr["n"],
            "max_allowed": limit,
            "successes": [r["failure_id"] for r in rows if r["success"]],
            "meaning": "non-zero means replaying the unedited actions of a measured failure "
            "now succeeds, i.e. the protocol is broken. Hard stop, not a tolerance.",
        },
    )


def check_all_baselines_present(replays: list[dict], cfg: dict, gate: dict) -> Result:
    """Criterion 3. §6: "Every table reports these or it does not ship."

    Scored against the **edits** file rather than the replays, because a baseline that raised
    `NotApplicable` has no replay row and is nonetheless reported. SPEC.md is explicit that
    `expert_oracle` is undefined for natural failures and that the gap is reported rather than
    filled with a nearest-neighbour demo, so scoring it as missing would punish the correct
    behaviour and push a future run toward substituting an approximation.
    """
    from cfe.search.arms import required_baseline_names

    required = required_baseline_names()
    edits_path = REPO_ROOT / "results" / "runs" / cfg["_run"] / "edits.jsonl"
    if not edits_path.exists():
        return Result(
            "all_baselines_present",
            False,
            {"reason": f"{edits_path} not found; cannot tell which arms were proposed"},
        )
    edits = [json.loads(x) for x in edits_path.read_text().splitlines() if x.strip()]

    present = {e["arm"] for e in edits}
    missing_overall = sorted(required - present)

    # Per cell of each experiment's grid, so a baseline cannot be present in aggregate while
    # absent from the cells a curve is actually drawn from.
    cells = defaultdict(set)
    for e in edits:
        cells[(e["experiment"], e["budget_k"])].add(e["arm"])
    incomplete = {
        f"{exp}/K={k}": sorted(required - arms)
        for (exp, k), arms in sorted(cells.items())
        if required - arms
    }
    not_applicable = [
        {k: e.get(k) for k in ("experiment", "failure_id", "arm", "budget_k", "not_applicable")}
        for e in edits
        if e.get("not_applicable") is not None
    ]
    ok = not missing_overall and not incomplete
    if not bool(gate["require_all_baselines"]):
        ok = not missing_overall
    return Result(
        "all_baselines_present",
        ok,
        {
            "required": sorted(required),
            "present": sorted(present),
            "missing_overall": missing_overall,
            "cells_missing_a_baseline": incomplete,
            "n_cells": len(cells),
            "n_reported_not_applicable": len(not_applicable),
            "not_applicable": not_applicable[:5],
        },
    )


def check_magnitude_match(rows: list[dict], gate: dict) -> Result:
    """Criterion 4. `random_perturb` matched on magnitude, not only on budget K.

    SESSION_HANDOFF: "This is the single easiest way to fake a positive result in this project —
    an under-scaled random baseline makes the search look good against nothing."
    """
    tol = float(gate["magnitude_match_tol"])
    by_cell = defaultdict(dict)
    for r in rows:
        if r["arm"] in ("search", "random_perturb") and r.get("realised_mean_l2_ee") is not None:
            key = (r["failure_id"], r["budget_k"], r["lam"])
            by_cell[key][r["arm"]] = r
    diffs, worst, n_pairs = [], None, 0
    for key, arms in sorted(by_cell.items()):
        if "search" not in arms or "random_perturb" not in arms:
            continue
        s = float(arms["search"]["realised_mean_l2_ee"])
        p = float(arms["random_perturb"]["realised_mean_l2_ee"])
        d = abs(s - p)
        diffs.append(d)
        n_pairs += 1
        if worst is None or d > worst["abs_diff"]:
            worst = {
                "cell": list(key),
                "search_mean_l2_ee": s,
                "random_perturb_mean_l2_ee": p,
                "abs_diff": d,
            }
    if not diffs:
        return Result("random_perturb_magnitude_matched", False, {"reason": "no matched pairs"})
    return Result(
        "random_perturb_magnitude_matched",
        max(diffs) <= tol,
        {
            "n_pairs": n_pairs,
            "max_abs_diff": round(float(max(diffs)), 8),
            "mean_abs_diff": round(float(np.mean(diffs)), 8),
            "tolerance": tol,
            "worst_cell": worst,
            "note": "a non-zero difference is expected only where clipping at the action bounds "
            "pulled a realised magnitude below the requested scale",
        },
    )


def check_search_deterministic(run: str, seed: int) -> Result:
    """Criterion 5 (I5). Re-runs the search for one recorded cell and demands the same edit.

    One cell, not the whole grid: determinism is a property of the code path, and re-running
    every cell would make the gate as expensive as the thing it checks.
    """
    import torch

    from cfe.search.arms import SEARCH, edit_from_result, run_search_arm
    from cfe.search.cem import CEMConfig
    from scripts.run_search import build_problem, cell_seed, load_models
    from scripts.train_dynamics import open_cache

    run_dir = REPO_ROOT / "results" / "runs" / run
    cfg = yaml.safe_load((run_dir / "search.yaml").read_text())
    rows = [json.loads(x) for x in (run_dir / "edits.jsonl").read_text().splitlines()]
    target = next((r for r in rows if r["arm"] == SEARCH), None)
    if target is None:
        return Result("search_deterministic", False, {"reason": "no search rows to re-run"})

    from cfe.data.failure_set import failure_set_dir, load_failure_set

    records = {r.failure_id: r for r in load_failure_set(failure_set_dir(REPO_ROOT))}
    rec = records[target["failure_id"]]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, dyn_cfg, _ = load_models(target["dyn_run"], device)
    cache, *_ = open_cache(cfg, seed)
    problem = build_problem(cache, rec, dyn_cfg, device)
    w = cfg["search"]["windows"]

    res, _ = run_search_arm(
        models["dynamics"],
        problem,
        int(target["budget_k"]),
        float(target["lam"]),
        CEMConfig.from_dict(cfg["search"]["cem"]),
        np.random.default_rng(
            cell_seed(seed, rec.failure_id, SEARCH, int(target["budget_k"]), float(target["lam"]))
        ),
        stride=int(w["stride"]),
        top_m=int(w["top_m"]),
        score_horizon=int(w["score_horizon"]),
    )
    edit = edit_from_result(res)
    recorded = np.asarray(target["actions"], dtype=np.float64)
    identical = edit.window_start == target["window_start"] and np.array_equal(
        edit.actions, recorded
    )
    return Result(
        "search_deterministic",
        bool(identical),
        {
            "cell": {k: target[k] for k in ("failure_id", "arm", "budget_k", "lam")},
            "window_start_recorded": target["window_start"],
            "window_start_rerun": edit.window_start,
            "actions_bit_identical": bool(np.array_equal(edit.actions, recorded)),
            "max_abs_diff": float(np.abs(edit.actions - recorded).max())
            if edit.actions.shape == recorded.shape
            else None,
            "comparison": "np.array_equal, bit-exact",
        },
    )


def check_config_snapshot(run: str) -> Result:
    """Criterion 6. CLAUDE.md rule 4: a number without its config is an orphan."""
    run_dir = REPO_ROOT / "results" / "runs" / run
    wanted = [
        "search.yaml",
        "encoder.yaml",
        "env.yaml",
        "dynamics.yaml",
        "edits.jsonl",
        "metrics.jsonl",
    ]
    present = {f: (run_dir / f).exists() for f in wanted}
    return Result(
        "config_snapshot_present", all(present.values()), {"run_dir": str(run_dir)} | present
    )


# ---------------------------------------------------------------------------
# Reported, not gated
# ---------------------------------------------------------------------------


def _as_result(row: dict):
    from cfe.eval.replay_protocol import ReplayResult

    return ReplayResult(
        success=bool(row["success"]),
        variant=row["variant"],
        window_start=int(row["window_start"]),
        window_end=int(row["window_end"]),
        steps_executed=int(row["steps_executed"]),
        scripted_steps=int(row["scripted_steps"]),
        policy_steps=int(row["policy_steps"]),
        timed_out=bool(row["timed_out"]),
        max_steps_total=int(row["max_steps_total"]),
        policy_never_ran=bool(row["policy_never_ran"]),
    )


def rsr_tables(replays: list[dict]) -> dict:
    """The full RSR curve per experiment and arm — I4's "never a single tuned number"."""
    from cfe.eval.metrics import replay_success_rate, rsr_curve

    out: dict = {}
    by_exp_arm = defaultdict(list)
    for r in replays:
        if r.get("variant") != "primary":
            continue
        by_exp_arm[(r["experiment"], r["arm"], r.get("window_source"))].append(r)

    for (exp, arm, wsrc), rows in sorted(by_exp_arm.items(), key=lambda kv: str(kv[0])):
        key = f"{exp}/{arm}" + (f"@{wsrc}" if wsrc and wsrc != "search" else "")
        if any(r["lam"] is not None for r in rows):
            recs = [
                {"budget_k": r["budget_k"], "lam": r["lam"], "result": _as_result(r)}
                for r in rows
                if r["lam"] is not None
            ]
            curve = rsr_curve(recs)
            out[key] = {
                str(k): {str(lam): v["replay_success_rate"] for lam, v in by_lam.items()}
                for k, by_lam in curve["curve"].items()
            }
        else:
            # lambda-independent arms: a K-only curve, stated as such rather than duplicated
            # across lambdas as though it had been measured there.
            by_k = defaultdict(list)
            for r in rows:
                by_k[r["budget_k"]].append(_as_result(r))
            out[key] = {
                str(k): replay_success_rate(v)["replay_success_rate"]
                for k, v in sorted(by_k.items())
            }
    return out


def diagnostics(replays: list[dict], gate: dict) -> dict:
    """Everything §6 reports alongside the gate, plus the recoverability warning."""
    from cfe.eval.metrics import cause_top1_accuracy, replay_success_rate, search_cost

    primary = [r for r in replays if r.get("variant") == "primary"]
    oracle_truth = [
        r
        for r in primary
        if r["arm"] == "expert_oracle" and r.get("window_source") == "ground_truth"
    ]
    oracle_rsr = (
        replay_success_rate([_as_result(r) for r in oracle_truth])["replay_success_rate"]
        if oracle_truth
        else None
    )
    ious = [
        r["window_iou_vs_truth"]["top1_iou"]
        for r in primary
        if isinstance(r.get("window_iou_vs_truth"), dict)
        and r["window_iou_vs_truth"].get("top1_iou") is not None
    ]
    n_identity = sum(
        1 for r in primary if isinstance(r.get("search"), dict) and r["search"].get("is_identity")
    )
    n_search = sum(1 for r in primary if isinstance(r.get("search"), dict))

    # Cause identification: stage 1 does not predict a tag, so the predicted cause is None for
    # every episode. Reported as measured — a top-1 accuracy of 0 against labelled truth is the
    # honest reading of "the window ranking does not identify the cause", not a missing metric.
    labelled = [r for r in primary if r["arm"] == "search" and r.get("cause")]
    cause = cause_top1_accuracy([None] * len(labelled), [r["cause"] for r in labelled])

    total_s = sum(float(r.get("replay_s") or 0.0) for r in replays)
    n_cf = max(
        1, len({(r["experiment"], r["failure_id"], r["budget_k"], r["lam"]) for r in primary})
    )
    fwd = sum(
        int(r["search"].get("n_forward_passes", 0))
        for r in primary
        if isinstance(r.get("search"), dict)
    )

    return {
        "expert_oracle_rsr_at_ground_truth_window": oracle_rsr,
        "expert_oracle_warning": (
            None
            if oracle_rsr is None or oracle_rsr >= float(gate["warn_expert_oracle_rsr_below"])
            else "expert_oracle is near zero: no edit of any budget rescues these episodes, so "
            "the gated comparison is between two ways of failing. Read criterion 1 in that light."
        ),
        "no_edit_primary_rsr": _arm_rsr(primary, "no_edit"),
        "search_identity_edit_fraction": round(n_identity / n_search, 4) if n_search else None,
        "window_iou_vs_truth": {
            "mean": round(float(np.mean(ious)), 4) if ious else None,
            "max": round(float(np.max(ious)), 4) if ious else None,
            "n": len(ious),
        },
        "cause_top1_accuracy": cause,
        "search_cost": search_cost(total_s, fwd, n_cf),
        "edit_magnitude_mean_l2_ee_by_arm": _mag_by_arm(primary),
    }


def _arm_rsr(rows: list[dict], arm: str):
    from cfe.eval.metrics import replay_success_rate

    sel = [r for r in rows if r["arm"] == arm]
    return replay_success_rate([_as_result(r) for r in sel])["replay_success_rate"] if sel else None


def _mag_by_arm(rows: list[dict]) -> dict:
    by_arm = defaultdict(list)
    for r in rows:
        if r.get("realised_mean_l2_ee") is not None:
            by_arm[r["arm"]].append(float(r["realised_mean_l2_ee"]))
    return {a: round(float(np.mean(v)), 5) for a, v in sorted(by_arm.items())}


# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    cfg["_run"] = args.run
    gate = cfg["gate"]
    run_id = args.run_id or f"g3-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    env_cfg = yaml.safe_load((REPO_ROOT / cfg["env_config"]).read_text())
    os.environ.setdefault("MUJOCO_GL", env_cfg["render_backend"])

    replays, summary = load_replays(args.run)
    rows = gated_rows(replays, cfg)

    print(f"G3 gate  run_id={run_id}  replay_run={args.run}")
    print(f"pre-registered criteria: {json.dumps(gate)}")
    print(f"replays: {len(replays)}  gated (experiment R, primary, injected): {len(rows)}")
    if not summary.get("complete", True):
        print("WARNING: the replay run reports itself INCOMPLETE; numbers below are partial")
    print("-" * 78)

    results = [
        check_beats_random_perturb(rows, gate),
        check_no_edit_open_loop(replays, gate),
        check_all_baselines_present(replays, cfg, gate),
        check_magnitude_match(rows, gate),
        check_config_snapshot(args.run),
    ]
    if bool(gate["require_search_deterministic"]) and not args.skip_determinism:
        results.insert(4, check_search_deterministic(args.run, args.seed))

    tables = rsr_tables(replays)
    diag = diagnostics(replays, gate)

    for r in results:
        print(r.line())
    print("-" * 78)
    print("RSR curves (primary variant; I4 — never collapse these to one number):")
    print(json.dumps(tables, indent=2, default=str))
    print("-" * 78)
    print("reported, NOT gated:")
    print(json.dumps(diag, indent=2, default=str))

    out_dir = REPO_ROOT / "results" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, out_dir / "search.yaml")
    with (out_dir / "metrics.jsonl").open("w") as fh:
        for r in results:
            fh.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "gate": "G3",
                        "check": r.name,
                        "passed": r.passed,
                        "seed": args.seed,
                        "replay_run": args.run,
                        "detail": r.detail,
                    },
                    default=str,
                )
                + "\n"
            )
        fh.write(
            json.dumps(
                {
                    "run_id": run_id,
                    "kind": "diagnostic",
                    "rsr_curves": tables,
                    "reported_not_gated": diag,
                    "replay_summary": summary,
                },
                default=str,
            )
            + "\n"
        )

    failed = [r.name for r in results if not r.passed]
    print("-" * 78)
    print(
        f"G3: {'PASS' if not failed else 'FAIL'} "
        f"({len(results) - len(failed)}/{len(results)} checks"
        + (f"; failed: {', '.join(failed)}" if failed else "")
        + ")"
    )
    print(f"artifacts: {out_dir}")
    if failed:
        print(
            "\nA failing gate is information (CLAUDE.md rule 3). Do not relax a criterion to "
            "make this pass; bring the measurement and decide."
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
