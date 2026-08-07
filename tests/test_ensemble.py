"""The ensembled search objective (SPEC.md §5, amended).

The ensemble exists because Phase 3's measurements said the objective's signal is buried in
model-specific noise. Combining members is only sound if it does not quietly change what a
single member computes, and if it is genuinely insensitive to the members' arbitrary output
scales — a raw mean would be dominated by whichever member emits larger numbers, which is a
property of its training, not of the task.

`goal_distance` from an ensemble is a **ranking quantity, not a distance**: standardisation is
relative to the candidate population it was scored with. These pin that it behaves like one.
"""

import numpy as np
import pytest

from cfe.search.ensemble import (
    combine_member_distances,
    evaluate_candidates_ensemble,
    member_agreement,
    standardise,
)
from cfe.search.objective import evaluate_candidates
from tests.test_search_cem import ToyDynamics, _problem, _with_goal


def _setup(seed: int = 0):
    problem, actions = _problem()
    problem = _with_goal(problem, 8, np.array([0.2, -0.1, 0.1]))
    rng = np.random.default_rng(seed)
    cand = np.repeat(actions[8:12][None], 6, axis=0)
    cand[:, :, :6] += rng.uniform(-0.3, 0.3, size=(6, 4, 6))
    return problem, np.clip(cand, -1.0, 1.0)


# --- standardise ------------------------------------------------------------


def test_standardise_is_zero_mean_unit_sd():
    x = standardise(np.array([1.0, 2.0, 3.0, 10.0]))
    assert x.mean() == pytest.approx(0.0, abs=1e-12)
    assert x.std() == pytest.approx(1.0, abs=1e-12)


def test_a_degenerate_population_becomes_zeros_not_nan():
    """A NaN here would propagate through the mean and poison every other member's ranking."""
    out = standardise(np.array([5.0, 5.0, 5.0]))
    assert np.all(out == 0.0)
    assert np.isfinite(out).all()


# --- equivalence and invariance ---------------------------------------------


def test_one_member_ranks_identically_to_the_single_model_objective():
    """Ensembling must not change what a single model says about which edit is better."""
    problem, cand = _setup()
    single = evaluate_candidates(ToyDynamics(), problem, 8, cand, lam=0.0)["goal_distance"]
    ens = evaluate_candidates_ensemble([ToyDynamics()], problem, 8, cand, lam=0.0)["goal_distance"]
    assert np.array_equal(np.argsort(single), np.argsort(ens))


def test_the_combination_is_invariant_to_a_members_output_scale():
    """A raw mean would be dominated by whichever member emits the largest numbers, which is a
    property of that member's training, not of the task."""
    raw = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 30.0, 20.0, 40.0]])
    scaled = raw.copy()
    scaled[1] = scaled[1] * 1000.0 + 500.0  # affine: same ranking, wildly different scale
    assert combine_member_distances(raw) == pytest.approx(combine_member_distances(scaled))


def test_the_combination_gives_every_member_equal_weight():
    """Two members that rank oppositely must cancel, not be decided by the louder one."""
    raw = np.array([[1.0, 2.0, 3.0], [300.0, 200.0, 100.0]])
    assert combine_member_distances(raw) == pytest.approx(np.zeros(3), abs=1e-12)


def test_the_combination_rejects_a_non_matrix():
    with pytest.raises(ValueError, match=r"\(M, P\)"):
        combine_member_distances(np.array([1.0, 2.0, 3.0]))


def test_per_member_distances_are_returned_unstandardised():
    """The screen re-groups candidates itself, so it needs raw values, not standardised ones."""
    problem, cand = _setup()
    out = evaluate_candidates_ensemble([ToyDynamics(), ToyDynamics()], problem, 8, cand, lam=0.0)
    raw = out["goal_distance_per_member"]
    assert raw.shape == (2, len(cand))
    # Raw goal distances are positive magnitudes; standardised ones would be zero-mean.
    assert (raw > 0).all()
    assert abs(raw[0].mean()) > 1e-6


def test_the_penalty_is_added_in_its_own_units():
    """Only D is standardised; λ·Σ||Δa||² keeps its scale or λ would mean something different."""
    problem, cand = _setup()
    z = evaluate_candidates_ensemble([ToyDynamics()], problem, 8, cand, lam=0.0)
    p = evaluate_candidates_ensemble([ToyDynamics()], problem, 8, cand, lam=2.5)
    assert p["objective"] == pytest.approx(z["objective"] + 2.5 * z["penalty"], abs=1e-9)


def test_an_empty_ensemble_is_refused():
    problem, cand = _setup()
    with pytest.raises(ValueError, match="at least one member"):
        evaluate_candidates_ensemble([], problem, 8, cand, lam=0.0)


def test_mismatched_models_and_problems_are_refused():
    problem, cand = _setup()
    with pytest.raises(ValueError, match="but 2 problems"):
        evaluate_candidates_ensemble([ToyDynamics()], [problem, problem], 8, cand, lam=0.0)


# --- member agreement -------------------------------------------------------


def test_identical_members_agree_perfectly():
    raw = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert member_agreement(raw)["mean_pairwise_spearman"] == pytest.approx(1.0)


def test_opposed_members_disagree():
    raw = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    assert member_agreement(raw)["mean_pairwise_spearman"] == pytest.approx(-1.0)


def test_agreement_reports_every_pair():
    raw = np.random.default_rng(0).standard_normal((4, 20))
    out = member_agreement(raw)
    assert out["n_members"] == 4
    assert len(out["pairwise"]) == 6  # 4 choose 2


# --- the screening statistic -------------------------------------------------


def test_ranking_quality_on_a_hand_worked_example():
    """Two episodes, known answer, including the tie handling. In the real data 116 of 168
    pairs are ties, so mishandling them would swamp the metric."""
    from scripts.diagnose_objective_quality import ranking_quality

    scored = [
        # episode A: the lower score succeeded -> concordant
        {"episode": "A", "score": 0.1, "success": True},
        {"episode": "A", "score": 0.9, "success": False},
        # episode B: the higher score succeeded -> discordant
        {"episode": "B", "score": 0.1, "success": False},
        {"episode": "B", "score": 0.9, "success": True},
        # episode B also has a same-outcome pair with the first -> excluded, not scored
        {"episode": "B", "score": 0.5, "success": True},
    ]
    q = ranking_quality(scored)
    assert q["n_concordant"] == 1
    assert q["n_discordant"] == 2
    assert q["n_excluded_equal_outcome_pairs"] == 1
    assert q["pairwise_accuracy"] == pytest.approx(1 / 3, abs=1e-4)  # reported rounded to 4dp
    assert q["n_episodes"] == 2


def test_a_perfect_objective_scores_one_and_a_reversed_one_scores_zero():
    from scripts.diagnose_objective_quality import ranking_quality

    good = [
        {"episode": "A", "score": s, "success": y}
        for s, y in ((0.1, True), (0.2, True), (0.9, False))
    ]
    assert ranking_quality(good)["pairwise_accuracy"] == pytest.approx(1.0)
    bad = [{**r, "score": -r["score"]} for r in good]
    assert ranking_quality(bad)["pairwise_accuracy"] == pytest.approx(0.0)


def test_an_episode_with_one_candidate_contributes_nothing():
    from scripts.diagnose_objective_quality import ranking_quality

    q = ranking_quality([{"episode": "solo", "score": 0.5, "success": True}])
    assert q["n_episodes"] == 0
    assert q["pairwise_accuracy"] is None
