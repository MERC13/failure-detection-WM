"""Stage 1 window selection (SPEC.md §5).

The ranking is a heuristic — "rank candidates by per-step latent divergence" — so it is tested
against a case where the answer is known by construction: an episode that follows the toy
dynamics exactly except for one step where the world jumps. The model is exact, so every step
but that one has zero divergence and the correct ranking is unambiguous.
"""

import numpy as np
import pytest
import torch

from cfe.search.objective import SearchProblem
from cfe.search.windows import (
    exhaustive_starts,
    per_step_divergence,
    rank_windows,
    rank_windows_exhaustive,
    window_recall_of_truth,
    window_scores,
)
from tests.test_search_cem import HORIZON, K_CONTEXT, D, R, ToyDynamics

JUMP_AT = 21  # the observed latent departs from the dynamics starting at this frame


def _problem_with_a_jump(n_steps: int = 40, bump: float = 5.0) -> SearchProblem:
    """Consistent with the toy dynamics everywhere except a single unexplained jump.

    The model can predict every transition but the one into `JUMP_AT`, so divergence is exactly
    zero except at the four window starts whose horizon straddles it — and it is largest at the
    start whose whole horizon lies past the jump.
    """
    rng = np.random.default_rng(0)
    actions = np.zeros((n_steps, 7))
    actions[:, :3] = rng.uniform(-0.2, 0.2, size=(n_steps, 3))
    actions[:, 6] = -1.0

    z = np.zeros((n_steps + 1, R, D), dtype=np.float32)
    for t in range(n_steps):
        z[t + 1] = z[t] + actions[t, :3]
    z[JUMP_AT:] += bump

    return SearchProblem(
        episode_id="jump",
        z=torch.from_numpy(z),
        actions=actions,
        z_goal=torch.zeros(1, R, D),
        k_context=K_CONTEXT,
        horizon=HORIZON,
    )


def test_divergence_is_zero_except_around_the_jump():
    problem = _problem_with_a_jump()
    d = per_step_divergence(ToyDynamics(), problem, score_horizon=HORIZON)

    # Starts whose horizon [t+1, t+4] straddles the jump are the only ones that can see it.
    straddling = set(range(JUMP_AT - HORIZON, JUMP_AT))
    for t, v in enumerate(d):
        if np.isnan(v):
            continue
        if t in straddling:
            assert v > 0.0, f"step {t} straddles the jump but scored {v}"
        else:
            assert v == pytest.approx(0.0, abs=1e-5), f"step {t} scored {v}"


def test_the_top_window_is_the_one_that_saw_the_jump():
    problem = _problem_with_a_jump()
    d = per_step_divergence(ToyDynamics(), problem, score_horizon=HORIZON)
    ranked = rank_windows(d, problem, budget_k=1, stride=1, top_m=3)
    # The start whose entire horizon lies past the jump sees the largest discrepancy.
    assert ranked[0]["window_start"] == JUMP_AT - 1


def test_unscored_steps_are_nan_not_zero():
    """Zero is a legitimate divergence; filling with it would rank unscored steps as best."""
    problem = _problem_with_a_jump()
    d = per_step_divergence(ToyDynamics(), problem, score_horizon=HORIZON)
    assert np.isnan(d[: K_CONTEXT - 1]).all()
    assert np.isnan(d[-HORIZON + 1 :]).all()


def test_a_window_containing_an_unscored_step_is_dropped():
    d = np.array([np.nan, 1.0, 2.0, 3.0])
    scores = window_scores(d, budget_k=2, valid_starts=np.array([0, 1, 2]))
    assert np.isnan(scores[0])
    assert scores[1] == pytest.approx(1.5)


def test_stride_ranking_is_a_subset_of_exhaustive_starts():
    problem = _problem_with_a_jump()
    d = per_step_divergence(ToyDynamics(), problem, score_horizon=HORIZON)
    every = set(exhaustive_starts(problem, budget_k=4).tolist())
    strided = {c["window_start"] for c in rank_windows(d, problem, 4, stride=4, top_m=5)}
    assert strided <= every


def test_exhaustive_ranking_is_at_least_as_good_as_strided():
    """SPEC.md §5's correctness check: the stride must not be beating the full enumeration."""
    problem = _problem_with_a_jump()
    d = per_step_divergence(ToyDynamics(), problem, score_horizon=HORIZON)
    best_stride = rank_windows(d, problem, 4, stride=4, top_m=1)[0]["divergence"]
    best_full = rank_windows_exhaustive(d, problem, 4, top_m=1)[0]["divergence"]
    assert best_full >= best_stride


def test_ties_break_on_the_earlier_start():
    """Determinism (I5): equal scores must not depend on numpy's sort internals.

    The divergence array is handed in directly rather than measured, because float32 roundoff
    in the toy rollout makes "equal" scores differ around 1e-8 and the tie this asserts on
    would not be a tie at all.
    """
    problem = _problem_with_a_jump()
    d = np.zeros(problem.n_steps)  # every window scores identically
    ranked = rank_windows(d, problem, budget_k=2, stride=1, top_m=3)
    assert [c["window_start"] for c in ranked] == [K_CONTEXT - 1, K_CONTEXT, K_CONTEXT + 1]


def test_window_recall_reports_none_for_natural_failures():
    """Natural failures have no ground-truth window; a number here would be fabricated."""
    out = window_recall_of_truth([{"window_start": 3, "window_end": 7}], truth=None)
    assert out["best_iou"] is None


def test_window_recall_uses_the_frozen_iou():
    ranked = [{"window_start": 10, "window_end": 14}, {"window_start": 0, "window_end": 4}]
    out = window_recall_of_truth(ranked, truth=(12, 16))
    assert out["top1_iou"] == pytest.approx(2 / 6)
    assert out["best_iou"] == pytest.approx(2 / 6)


def test_score_horizon_must_be_a_multiple_of_the_model_horizon():
    problem = _problem_with_a_jump()
    with pytest.raises(ValueError, match="not a multiple"):
        per_step_divergence(ToyDynamics(), problem, score_horizon=HORIZON + 1)
