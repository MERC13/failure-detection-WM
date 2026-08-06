"""G3's scoring logic, on hand-built replay records (SPEC.md §7, §8).

The gate reads a few hundred JSON rows and decides whether the project's central claim holds.
Everything it does to those rows is pure, so it can be checked against inputs whose answer is
known — which is the only way to find out that a criterion is scoring the wrong subset *before*
five hours of simulation produce a number nobody can interpret.

The subset selection matters as much as the arithmetic: `beats_random_perturb` is defined on the
gated experiment, primary variant, injected failures only, and each of those filters exists to
exclude something that would otherwise contaminate it.
"""

import numpy as np
import pytest
import yaml

from scripts.verify_g3 import (
    check_beats_random_perturb,
    check_magnitude_match,
    check_no_edit_open_loop,
    gated_rows,
    paired_arms,
    rsr_tables,
)
from tests.conftest import REPO_ROOT

GATE = yaml.safe_load((REPO_ROOT / "configs" / "search.yaml").read_text())["gate"]
CFG = yaml.safe_load((REPO_ROOT / "configs" / "search.yaml").read_text())


def _row(**over) -> dict:
    base = {
        "kind": "replay",
        "experiment": "R",
        "failure_id": "ep0",
        "arm": "search",
        "budget_k": 8,
        "lam": 0.1,
        "variant": "primary",
        "window_source": "search",
        "cause": "grasp_offset",
        "success": True,
        "window_start": 10,
        "window_end": 18,
        "steps_executed": 200,
        "scripted_steps": 18,
        "policy_steps": 182,
        "timed_out": False,
        "max_steps_total": 300,
        "policy_never_ran": False,
        "realised_mean_l2_ee": 0.5,
    }
    return base | over


def _arm_rows(arm: str, successes: list[bool], **over) -> list[dict]:
    return [_row(arm=arm, failure_id=f"ep{i}", success=s, **over) for i, s in enumerate(successes)]


# --- subset selection -------------------------------------------------------


def test_gated_rows_keeps_only_the_gated_experiment():
    rows = [_row(experiment="R"), _row(experiment="D")]
    assert [r["experiment"] for r in gated_rows(rows, CFG)] == ["R"]


def test_gated_rows_drops_the_open_loop_variant():
    """The gate is on the primary variant; §6's secondary is logged alongside, never instead."""
    rows = [_row(variant="primary"), _row(variant="open_loop")]
    assert len(gated_rows(rows, CFG)) == 1


def test_gated_rows_drops_natural_failures():
    """SPEC.md §7 scopes the G3 comparison to injected failures."""
    rows = [_row(cause="grasp_offset"), _row(cause=None)]
    assert len(gated_rows(rows, CFG)) == 1


def test_paired_arms_intersects_rather_than_zipping():
    """A missing episode must not shift the pairing and turn the comparison into noise."""
    rows = [
        *_arm_rows("search", [True, True, True]),
        *_arm_rows("random_perturb", [False, False]),  # only ep0, ep1
    ]
    t, c, shared = paired_arms(rows, "search", "random_perturb", 8, 0.1)
    assert shared == ["ep0", "ep1"]
    assert t == [True, True]
    assert c == [False, False]


# --- criterion 1 ------------------------------------------------------------


def _grid(search: list[bool], random: list[bool], lambdas=(0.01, 0.1, 1.0)) -> list[dict]:
    rows = []
    for lam in lambdas:
        rows += _arm_rows("search", search, lam=lam)
        rows += _arm_rows("random_perturb", random, lam=lam)
    return rows


def test_a_clear_margin_at_every_lambda_passes():
    res = check_beats_random_perturb(_grid([True] * 10, [False] * 10), GATE)
    assert res.passed
    for lam in ("0.01", "0.1", "1.0"):
        assert res.detail["per_lambda"][lam]["passed"]
        assert res.detail["per_lambda"][lam]["ci_low"] > 0


def test_no_margin_fails():
    res = check_beats_random_perturb(
        _grid([True] * 5 + [False] * 5, [True] * 5 + [False] * 5), GATE
    )
    assert not res.passed


def test_winning_at_one_lambda_only_is_not_winning():
    """SPEC.md §6, verbatim: "A method that wins only at one λ has not won"."""
    rows = []
    rows += _arm_rows("search", [True] * 10, lam=0.01)
    rows += _arm_rows("random_perturb", [False] * 10, lam=0.01)
    for lam in (0.1, 1.0):
        rows += _arm_rows("search", [False] * 10, lam=lam)
        rows += _arm_rows("random_perturb", [False] * 10, lam=lam)

    res = check_beats_random_perturb(rows, GATE)
    assert not res.passed
    assert res.detail["per_lambda"]["0.01"]["passed"]
    assert not res.detail["per_lambda"]["0.1"]["passed"]


def test_the_criterion_is_scored_at_the_configured_budget():
    """Rows at another K must not be able to carry the gate."""
    rows = _grid([True] * 10, [False] * 10)
    for r in rows:
        r["budget_k"] = 1  # nothing at K=8 any more
    res = check_beats_random_perturb(rows, GATE)
    assert not res.passed


def test_a_search_that_loses_narrowly_does_not_pass_on_the_point_estimate():
    """8/10 vs 6/10 is a positive mean difference; the CI is what decides."""
    res = check_beats_random_perturb(
        _grid([True] * 8 + [False] * 2, [True] * 6 + [False] * 4), GATE
    )
    assert res.detail["per_lambda"]["0.1"]["mean_difference"] > 0
    assert res.passed == all(v["passed"] for v in res.detail["per_lambda"].values())


# --- criterion 2 ------------------------------------------------------------


def test_open_loop_no_edit_at_zero_passes():
    rows = _arm_rows("no_edit", [False] * 10, variant="open_loop", lam=None)
    assert check_no_edit_open_loop(rows, GATE).passed


def test_any_open_loop_no_edit_success_is_a_hard_stop():
    """Replaying the unedited actions of a measured failure cannot succeed unless replay is
    broken. Not a tolerance to widen."""
    rows = _arm_rows("no_edit", [False] * 9 + [True], variant="open_loop", lam=None)
    res = check_no_edit_open_loop(rows, GATE)
    assert not res.passed
    assert res.detail["successes"] == ["ep9"]


def test_a_missing_open_loop_control_fails_rather_than_being_skipped():
    """A table without §6's replay-bug detector cannot detect a broken protocol."""
    res = check_no_edit_open_loop(_arm_rows("no_edit", [False] * 5), GATE)
    assert not res.passed
    assert "only control for replay bugs" in res.detail["reason"]


def test_the_primary_variant_is_not_scored_as_the_replay_bug_control():
    """A primary no_edit success means the policy rescued it, not that replay is broken."""
    rows = [
        *_arm_rows("no_edit", [True] * 5, variant="primary", lam=None),
        *_arm_rows("no_edit", [False] * 5, variant="open_loop", lam=None),
    ]
    assert check_no_edit_open_loop(rows, GATE).passed


# --- criterion 4 ------------------------------------------------------------


def test_matched_magnitudes_pass():
    rows = [
        _row(arm="search", realised_mean_l2_ee=0.5123456),
        _row(arm="random_perturb", realised_mean_l2_ee=0.5123456),
    ]
    res = check_magnitude_match(rows, GATE)
    assert res.passed
    assert res.detail["max_abs_diff"] == 0.0


def test_an_under_scaled_random_baseline_is_caught():
    """SESSION_HANDOFF: the single easiest way to fake a positive result in this project."""
    rows = [
        _row(arm="search", realised_mean_l2_ee=0.50),
        _row(arm="random_perturb", realised_mean_l2_ee=0.05),
    ]
    res = check_magnitude_match(rows, GATE)
    assert not res.passed
    assert res.detail["worst_cell"]["abs_diff"] == pytest.approx(0.45)


def test_magnitude_pairs_are_matched_within_a_cell_not_across_the_grid():
    """Averaging across cells would let a large mismatch hide behind an exact match elsewhere."""
    rows = [
        _row(arm="search", budget_k=1, realised_mean_l2_ee=0.5),
        _row(arm="random_perturb", budget_k=1, realised_mean_l2_ee=0.5),
        _row(arm="search", budget_k=8, realised_mean_l2_ee=0.5),
        _row(arm="random_perturb", budget_k=8, realised_mean_l2_ee=0.1),
    ]
    res = check_magnitude_match(rows, GATE)
    assert not res.passed
    assert res.detail["worst_cell"]["cell"][1] == 8


# --- reported tables --------------------------------------------------------


def test_rsr_tables_keep_the_full_k_by_lambda_grid():
    """I4: never collapsed to one number."""
    rows = []
    for k in (1, 8):
        for lam in (0.01, 1.0):
            rows += _arm_rows("search", [True, False], budget_k=k, lam=lam)
    out = rsr_tables(rows)
    assert set(out["R/search"]) == {"1", "8"}
    assert set(out["R/search"]["1"]) == {"0.01", "1.0"}
    assert out["R/search"]["8"]["1.0"] == pytest.approx(0.5)


def test_lambda_independent_arms_get_a_k_only_curve():
    """They were never measured per lambda; duplicating them across the grid would say they
    were."""
    rows = _arm_rows("no_edit", [False, False], lam=None, window_source="ground_truth")
    out = rsr_tables(rows)
    assert out["R/no_edit@ground_truth"] == {"8": 0.0}


def test_the_two_window_sources_are_reported_separately():
    """expert_oracle at stage 1's window is a different quantity from the upper bound at the
    injected window, and averaging them would hide that the first is often just no_edit."""
    rows = [
        *_arm_rows("expert_oracle", [False, False], lam=None, window_source="stage1_top1"),
        *_arm_rows("expert_oracle", [True, True], lam=None, window_source="ground_truth"),
    ]
    out = rsr_tables(rows)
    assert out["R/expert_oracle@stage1_top1"] == {"8": 0.0}
    assert out["R/expert_oracle@ground_truth"] == {"8": 1.0}


def test_experiments_are_tabulated_separately():
    rows = [
        *_arm_rows("search", [True, True], experiment="R"),
        *_arm_rows("search", [False, False], experiment="D"),
    ]
    out = rsr_tables(rows)
    assert out["R/search"]["8"]["0.1"] == 1.0
    assert out["D/search"]["8"]["0.1"] == 0.0


def test_the_bootstrap_is_reproducible_at_a_fixed_seed():
    """I5 reaches the gate too: two scorings of the same rows must agree exactly."""
    rows = _grid([True] * 7 + [False] * 3, [True] * 2 + [False] * 8)
    a = check_beats_random_perturb(rows, GATE)
    b = check_beats_random_perturb(rows, GATE)
    assert a.detail["per_lambda"] == b.detail["per_lambda"]
    assert np.isfinite(a.detail["per_lambda"]["0.1"]["ci_low"])


# --- criterion 3 ------------------------------------------------------------


def _write_edits(tmp_path, rows):
    import json

    run_dir = tmp_path / "results" / "runs" / "fake"
    run_dir.mkdir(parents=True)
    (run_dir / "edits.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return run_dir


def test_all_baselines_present_passes_on_a_complete_grid(monkeypatch, tmp_path):
    import scripts.verify_g3 as g3

    rows = [
        _row(arm=a, budget_k=k, experiment=e)
        for a in (
            "search",
            "random_perturb",
            "no_edit",
            "expert_oracle",
            "action_ignoring_dynamics",
        )
        for k in (1, 8)
        for e in ("R", "D")
    ]
    _write_edits(tmp_path, rows)
    monkeypatch.setattr(g3, "REPO_ROOT", tmp_path)
    res = g3.check_all_baselines_present([], CFG | {"_run": "fake"}, GATE)
    assert res.passed
    assert res.detail["n_cells"] == 4


def test_a_cell_missing_a_baseline_fails(monkeypatch, tmp_path):
    import scripts.verify_g3 as g3

    rows = [_row(arm=a, budget_k=8) for a in ("search", "random_perturb", "no_edit")]
    _write_edits(tmp_path, rows)
    monkeypatch.setattr(g3, "REPO_ROOT", tmp_path)
    res = g3.check_all_baselines_present([], CFG | {"_run": "fake"}, GATE)
    assert not res.passed
    assert "expert_oracle" in res.detail["missing_overall"]


def test_a_not_applicable_baseline_still_counts_as_reported(monkeypatch, tmp_path):
    """expert_oracle is undefined for natural failures; SPEC.md says report the gap, and
    scoring it as missing would push a future run toward substituting an approximation."""
    import scripts.verify_g3 as g3

    rows = [
        _row(arm=a, budget_k=8)
        for a in ("search", "random_perturb", "no_edit", "action_ignoring_dynamics")
    ]
    rows.append(
        _row(arm="expert_oracle", budget_k=8, cause=None, not_applicable="no expert for natural")
    )
    _write_edits(tmp_path, rows)
    monkeypatch.setattr(g3, "REPO_ROOT", tmp_path)
    res = g3.check_all_baselines_present([], CFG | {"_run": "fake"}, GATE)
    assert res.passed
    assert res.detail["n_reported_not_applicable"] == 1
