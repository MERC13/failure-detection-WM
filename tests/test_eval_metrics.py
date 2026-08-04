"""SPEC.md §8: "metric functions on hand-constructed inputs with known answers".

Every expected value here is computed by hand and written as a literal. Deriving an expectation
from the code under test would make these tests agree with a bug, which for a frozen module is
the worst possible outcome — the numbers would be wrong forever and consistently.
"""

import numpy as np
import pytest

from cfe.eval.metrics import (
    NotApplicable,
    bootstrap_rsr_difference,
    cause_top1_accuracy,
    edit_magnitude,
    edit_sparsity,
    expert_oracle,
    gripper_flips,
    no_edit,
    random_perturb,
    replay_success_rate,
    rsr_curve,
    search_cost,
    wilson_interval,
    window_iou,
)
from cfe.eval.replay_protocol import Edit, ReplayResult


def _result(success: bool) -> ReplayResult:
    return ReplayResult(
        success=success,
        variant="primary",
        window_start=0,
        window_end=4,
        steps_executed=100,
        scripted_steps=4,
        policy_steps=96,
        timed_out=False,
        max_steps_total=300,
        policy_never_ran=False,
    )


# --- Wilson interval --------------------------------------------------------


def test_wilson_reproduces_the_documented_g1_interval():
    """36/50 is the measured BC rate; [0.583, 0.825] is recorded in CHANGELOG.md.

    An independent number this module must agree with, rather than one it defines.
    """
    lo, hi = wilson_interval(36, 50)
    assert round(lo, 3) == 0.583
    assert round(hi, 3) == 0.825


def test_wilson_stays_in_bounds_at_zero_successes():
    """`no_edit` is expected at ~0%, which is exactly where the normal approximation goes
    negative. Wilson must not."""
    lo, hi = wilson_interval(0, 50)
    assert lo >= 0.0
    assert hi < 0.08


def test_wilson_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(51, 50)
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_replay_success_rate_counts_only_the_success_flag():
    out = replay_success_rate([_result(True), _result(False), _result(True), _result(True)])
    assert out["n_success"] == 3
    assert out["n"] == 4
    assert out["replay_success_rate"] == 0.75


# --- edit descriptors -------------------------------------------------------


def _window_with_known_deltas() -> tuple[np.ndarray, np.ndarray]:
    """8 steps; step 2 moves by L2 5.0 (3-4-5 triangle), step 5 by 1.0, the rest by 0."""
    original = np.zeros((8, 7))
    edited = original.copy()
    edited[2, 0] = 3.0
    edited[2, 1] = 4.0
    edited[5, 2] = 1.0
    return original, edited


def test_edit_magnitude_hand_computed():
    original, edited = _window_with_known_deltas()
    out = edit_magnitude(original, edited)
    assert out["max_l2_ee"] == 5.0
    assert out["mean_l2_ee"] == pytest.approx(6.0 / 8)  # (5 + 1) over 8 steps
    assert out["n_steps"] == 8


def test_edit_sparsity_two_of_eight_steps_changed():
    original, edited = _window_with_known_deltas()
    out = edit_sparsity(original, edited, eps=0.5)
    assert out["n_steps_changed_ee"] == 2
    assert out["n_steps"] == 8
    assert out["fraction_changed_ee"] == 0.25


def test_edit_sparsity_eps_is_strict_and_excludes_the_boundary():
    original, edited = _window_with_known_deltas()
    # eps exactly equal to the smaller delta: `> eps` must exclude it, leaving only the 5.0 step.
    assert edit_sparsity(original, edited, eps=1.0)["n_steps_changed_ee"] == 1


def test_gripper_never_enters_the_l2_and_is_counted_separately():
    """The CLAUDE.md convention, asserted rather than trusted: a pure gripper edit must have
    zero reported magnitude and a non-zero flip count."""
    original = np.zeros((5, 7))
    edited = original.copy()
    edited[[1, 3, 4], 6] = 1.0

    assert edit_magnitude(original, edited)["mean_l2_ee"] == 0.0
    assert edit_magnitude(original, edited)["max_l2_ee"] == 0.0
    assert edit_sparsity(original, edited, eps=0.0)["n_steps_changed_ee"] == 0
    assert gripper_flips(original, edited)["gripper_flips"] == 3


def test_edit_descriptors_reject_mismatched_windows():
    with pytest.raises(ValueError):
        edit_magnitude(np.zeros((4, 7)), np.zeros((5, 7)))
    with pytest.raises(ValueError):
        edit_magnitude(np.zeros((4, 6)), np.zeros((4, 6)))


# --- cause identification ---------------------------------------------------


def test_window_iou_hand_computed():
    # [10,20) vs [15,30): intersection 5, union 10 + 15 - 5 = 20.
    assert window_iou((10, 20), (15, 30)) == 0.25
    assert window_iou((0, 10), (0, 10)) == 1.0
    assert window_iou((0, 5), (5, 10)) == 0.0  # touching, not overlapping


def test_cause_accuracy_excludes_natural_failures_instead_of_scoring_them_wrong():
    """CLAUDE.md: `cause` is None for natural failures and they are ~half the dataset. Counting
    them as misses would make the metric a function of the natural/injected mix."""
    predicted = ["grasp_offset", "early_release", "grasp_offset", "wrong_target"]
    truth = ["grasp_offset", "grasp_offset", None, "wrong_target"]
    out = cause_top1_accuracy(predicted, truth)
    assert out["n_scored"] == 3
    assert out["n_excluded_unlabelled"] == 1
    assert out["n_correct"] == 2
    assert out["top1_accuracy"] == pytest.approx(2 / 3)


def test_cause_accuracy_does_not_crash_when_every_truth_is_natural():
    out = cause_top1_accuracy([None, "grasp_offset"], [None, None])
    assert out["top1_accuracy"] is None
    assert out["n_scored"] == 0
    assert out["n_excluded_unlabelled"] == 2


# --- comparison and cost ----------------------------------------------------


def test_bootstrap_difference_is_paired_and_signed():
    treatment = [True] * 8 + [False] * 2
    control = [False] * 10
    out = bootstrap_rsr_difference(treatment, control, n_boot=2000, seed=0)
    assert out["rsr_treatment"] == 0.8
    assert out["rsr_control"] == 0.0
    assert out["mean_difference"] == pytest.approx(0.8)
    assert out["excludes_zero"] is True
    assert out["ci_low"] > 0


def test_bootstrap_difference_finds_no_effect_when_arms_are_identical():
    arm = [True, False, True, False, True, False]
    out = bootstrap_rsr_difference(arm, arm, n_boot=2000, seed=0)
    assert out["mean_difference"] == 0.0
    assert out["excludes_zero"] is False


def test_bootstrap_difference_refuses_unpaired_arms():
    with pytest.raises(ValueError, match="paired"):
        bootstrap_rsr_difference([True, False], [True, False, True])


def test_bootstrap_difference_is_deterministic_under_a_seed():
    a, b = [True, False, True, True], [False, False, True, False]
    assert bootstrap_rsr_difference(a, b, n_boot=500, seed=7) == bootstrap_rsr_difference(
        a, b, n_boot=500, seed=7
    )


def test_search_cost_is_per_counterfactual():
    out = search_cost(wall_clock_s=120.0, n_forward_passes=6000, n_counterfactuals=10)
    assert out["wall_clock_s_per_counterfactual"] == 12.0
    assert out["forward_passes_per_counterfactual"] == 600.0


# --- I4: the budget curve ---------------------------------------------------


def test_rsr_curve_nests_by_budget_and_lambda_and_flags_off_grid_budgets():
    records = [
        {"budget_k": 1, "lam": 0.1, "result": _result(False)},
        {"budget_k": 1, "lam": 0.1, "result": _result(True)},
        {"budget_k": 8, "lam": 0.1, "result": _result(True)},
        {"budget_k": 3, "lam": 0.1, "result": _result(True)},  # 3 is not on SPEC's grid
    ]
    out = rsr_curve(records)
    assert out["curve"][1][0.1]["replay_success_rate"] == 0.5
    assert out["curve"][8][0.1]["replay_success_rate"] == 1.0
    assert out["off_grid_budgets"] == [3]
    # I4: there is deliberately no collapsed headline number to reach for.
    assert "overall" not in out


# --- baselines --------------------------------------------------------------


def test_no_edit_returns_the_original_actions_unchanged():
    rng = np.random.default_rng(0)
    original = rng.standard_normal((20, 7))
    edit = no_edit(original, window_start=4, budget_k=8)
    assert isinstance(edit, Edit)
    assert edit.budget_k == 8
    assert edit.window_end == 12
    assert np.array_equal(edit.actions, original[4:12])
    assert edit_magnitude(original[4:12], edit.actions)["max_l2_ee"] == 0.0


def test_random_perturb_matches_the_budget_and_leaves_the_gripper_alone():
    rng = np.random.default_rng(0)
    original = np.zeros((20, 7))
    original[:, 6] = 1.0
    edit = random_perturb(original, 4, 8, rng, magnitude_scale=0.1)
    assert edit.budget_k == 8
    assert np.array_equal(edit.actions[:, 6], original[4:12, 6])
    assert edit_magnitude(original[4:12], edit.actions)["mean_l2_ee"] > 0


def test_random_perturb_realises_exactly_the_requested_l2():
    """§6's baseline must be matched on magnitude, not merely on budget.

    Scaling six standard normals by `s` would give a per-step L2 of `s·√6 ≈ 2.45 s`, making the
    decisive baseline 2.45x stronger than the search it is compared against — with nothing
    downstream to reveal it. The realised magnitude must equal the requested one.
    """
    rng = np.random.default_rng(0)
    original = np.zeros((20, 7))  # far from the clip bounds, so nothing is truncated
    for scale in (0.05, 0.2, 0.5):
        edit = random_perturb(original, 0, 12, rng, magnitude_scale=scale)
        out = edit_magnitude(original[0:12], edit.actions)
        assert out["mean_l2_ee"] == pytest.approx(scale, rel=1e-9)
        assert out["max_l2_ee"] == pytest.approx(scale, rel=1e-9)


def test_random_perturb_rejects_a_negative_magnitude():
    with pytest.raises(ValueError, match="magnitude_scale"):
        random_perturb(np.zeros((10, 7)), 0, 4, np.random.default_rng(0), magnitude_scale=-1.0)


def test_random_perturb_clips_into_the_valid_action_range():
    rng = np.random.default_rng(0)
    edit = random_perturb(np.zeros((10, 7)), 0, 10, rng, magnitude_scale=50.0)
    assert edit.actions[:, :6].min() >= -1.0
    assert edit.actions[:, :6].max() <= 1.0


def test_random_perturb_has_no_default_magnitude():
    """Matching K but not magnitude would make §6's decisive baseline weak by construction."""
    with pytest.raises(TypeError):
        random_perturb(np.zeros((10, 7)), 0, 4, np.random.default_rng(0))


def test_expert_oracle_refuses_natural_failures_rather_than_approximating():
    """No demo shares a natural failure's init state (Phase 1), so there is no upper bound to
    compute and an approximate one would not be an upper bound at all."""
    with pytest.raises(NotApplicable):
        expert_oracle(np.zeros((20, 7)), 0, 4, cause=None)


def test_expert_oracle_returns_the_expert_window_for_an_injected_failure():
    rng = np.random.default_rng(1)
    expert = rng.standard_normal((20, 7))
    edit = expert_oracle(expert, 4, 8, cause="grasp_offset")
    assert np.array_equal(edit.actions, expert[4:12])


def test_action_ignoring_dynamics_is_declared_but_not_implementable_yet():
    from cfe.eval.metrics import BASELINES, action_ignoring_dynamics

    assert set(BASELINES) == {
        "no_edit",
        "random_perturb",
        "expert_oracle",
        "action_ignoring_dynamics",
    }
    with pytest.raises(NotImplementedError, match="cfe/search/"):
        action_ignoring_dynamics()
