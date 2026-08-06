"""Arm dispatch and the magnitude-matching obligation (SPEC.md §6).

SESSION_HANDOFF names magnitude matching as "the single easiest way to fake a positive result
in this project": `random_perturb` matched only on budget `K` and left free on perturbation size
is weak by construction, and the search then wins against nothing. The frozen module cannot
enforce that the caller passes the search's *realised* magnitude — these tests are the
caller-side half.
"""

import numpy as np
import pytest

from cfe.eval.metrics import BASELINES, edit_magnitude
from cfe.eval.replay_protocol import Edit
from cfe.search.arms import (
    ACTION_IGNORING,
    EXPERT_ORACLE,
    NO_EDIT,
    RANDOM_PERTURB,
    SEARCH,
    NotApplicable,
    assert_arms_cover_baselines,
    propose_baseline,
    realised_magnitude,
    required_baseline_names,
)


def _episode(n: int = 40, seed: int = 0) -> np.ndarray:
    """Actions well inside the bounds, so nothing clips and the match can be exact."""
    rng = np.random.default_rng(seed)
    a = np.zeros((n, 7))
    a[:, :6] = rng.uniform(-0.1, 0.1, size=(n, 6))
    a[:, 6] = -1.0
    return a


def _search_edit(original: np.ndarray, start: int, k: int, delta: float = 0.05) -> Edit:
    edited = original[start : start + k].copy()
    edited[:, :6] += delta
    return Edit(window_start=start, actions=edited)


# --- coverage of §6's four baselines ----------------------------------------


def test_the_required_names_come_from_the_frozen_module():
    """A baseline added to the frozen BASELINES can never be silently missing from a table."""
    assert required_baseline_names() == frozenset(BASELINES)
    assert required_baseline_names() == {
        NO_EDIT,
        RANDOM_PERTURB,
        EXPERT_ORACLE,
        ACTION_IGNORING,
    }


def test_a_table_missing_a_baseline_is_refused():
    with pytest.raises(ValueError, match="omit required"):
        assert_arms_cover_baselines([SEARCH, RANDOM_PERTURB, NO_EDIT])


def test_the_configured_arms_cover_every_baseline():
    import yaml

    from tests.conftest import REPO_ROOT

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "search.yaml").read_text())
    assert_arms_cover_baselines(cfg["replay"]["arms"])


# --- the magnitude match ----------------------------------------------------


@pytest.mark.parametrize("budget_k", [1, 2, 4, 8])
def test_random_perturb_realises_exactly_the_search_magnitude(budget_k):
    original = _episode()
    edit = _search_edit(original, start=10, k=budget_k)
    scale = realised_magnitude(original, edit)

    rp = propose_baseline(
        RANDOM_PERTURB,
        original_actions=original,
        window_start=10,
        budget_k=budget_k,
        rng=np.random.default_rng(0),
        magnitude_scale=scale,
    )
    got = edit_magnitude(original[10 : 10 + budget_k], rp.actions)["mean_l2_ee"]
    assert got == pytest.approx(scale, abs=1e-9)


def test_realised_magnitude_is_the_frozen_metric_not_a_recomputation():
    original = _episode()
    edit = _search_edit(original, start=5, k=4, delta=0.03)
    want = edit_magnitude(original[5:9], edit.actions)["mean_l2_ee"]
    assert realised_magnitude(original, edit) == pytest.approx(want, abs=0.0)


def test_random_perturb_without_a_magnitude_is_refused():
    """No default, on purpose: an unmatched baseline is the failure mode §6 warns about."""
    with pytest.raises(ValueError, match="magnitude_scale"):
        propose_baseline(
            RANDOM_PERTURB,
            original_actions=_episode(),
            window_start=0,
            budget_k=4,
            rng=np.random.default_rng(0),
        )


def test_random_perturb_leaves_the_gripper_alone():
    original = _episode()
    rp = propose_baseline(
        RANDOM_PERTURB,
        original_actions=original,
        window_start=3,
        budget_k=4,
        rng=np.random.default_rng(0),
        magnitude_scale=0.1,
    )
    assert np.array_equal(rp.actions[:, 6], original[3:7, 6])


def test_an_identity_search_edit_makes_random_perturb_the_identity_too():
    """A degenerate cell must be visible, not averaged in as though it compared something."""
    original = _episode()
    identity = Edit(window_start=6, actions=original[6:10])
    scale = realised_magnitude(original, identity)
    assert scale == 0.0

    rp = propose_baseline(
        RANDOM_PERTURB,
        original_actions=original,
        window_start=6,
        budget_k=4,
        rng=np.random.default_rng(0),
        magnitude_scale=scale,
    )
    assert np.array_equal(rp.actions, original[6:10])


# --- the other baselines ----------------------------------------------------


def test_no_edit_reproduces_the_original_window_exactly():
    original = _episode()
    e = propose_baseline(
        NO_EDIT,
        original_actions=original,
        window_start=7,
        budget_k=8,
        rng=np.random.default_rng(0),
    )
    assert np.array_equal(e.actions, original[7:15])


def test_expert_oracle_is_not_applicable_to_natural_failures():
    """No demo shares a natural failure's init state, so there is no upper bound to compute."""
    with pytest.raises(NotApplicable):
        propose_baseline(
            EXPERT_ORACLE,
            original_actions=_episode(),
            window_start=4,
            budget_k=4,
            rng=np.random.default_rng(0),
            expert_actions=_episode(seed=1),
            cause=None,
        )


def test_expert_oracle_returns_the_uncorrupted_demo_window():
    expert = _episode(seed=1)
    e = propose_baseline(
        EXPERT_ORACLE,
        original_actions=_episode(),
        window_start=4,
        budget_k=4,
        rng=np.random.default_rng(0),
        expert_actions=expert,
        cause="grasp_offset",
    )
    assert np.array_equal(e.actions, expert[4:8])


def test_search_arms_are_not_dispatched_as_baselines():
    for arm in (SEARCH, ACTION_IGNORING):
        with pytest.raises(ValueError, match="search arm"):
            propose_baseline(
                arm,
                original_actions=_episode(),
                window_start=0,
                budget_k=4,
                rng=np.random.default_rng(0),
            )


def test_the_frozen_action_ignoring_stub_still_refuses_to_be_called():
    """cfe/eval/ is frozen; this module registers nothing into it. Asserted so a future
    monkeypatch would break a test rather than change frozen behaviour silently."""
    with pytest.raises(NotImplementedError):
        BASELINES[ACTION_IGNORING]()


# --- the action-ignoring twin's analytic degeneracy -------------------------


def test_the_action_ignoring_twin_can_only_propose_the_identity():
    """Its z_end does not vary with the candidate, so J is minimised at the warm start for every
    lambda >= 0. If a real run shows this arm editing, the twin is seeing actions somehow."""
    import torch

    from cfe.search.cem import cem_search
    from tests.test_search_cem import ToyDynamics, _cfg, _problem, _with_goal

    class BlindToy(ToyDynamics):
        """The twin's behaviour: actions zeroed inside forward."""

        def rollout(self, context, actions):
            return super().rollout(context, torch.zeros_like(actions))

    problem, original = _problem()
    problem = _with_goal(problem, 8, np.array([0.2, -0.1, 0.1]))
    for lam in (0.0, 0.01, 1.0):
        res = cem_search(
            BlindToy(),
            problem,
            8,
            budget_k=4,
            lam=lam,
            cfg=_cfg(n_iters=5),
            rng=np.random.default_rng(0),
        )
        assert res.is_identity, f"twin proposed an edit at lam={lam}"
        assert np.array_equal(res.actions, original[8:12])


def test_the_action_ignoring_twin_is_insensitive_to_the_candidate():
    """The harness check: perturbing the candidate must not move the twin's goal distance."""
    import torch

    from cfe.search.objective import evaluate_candidates
    from tests.test_search_cem import ToyDynamics, _problem, _with_goal

    class BlindToy(ToyDynamics):
        def rollout(self, context, actions):
            return super().rollout(context, torch.zeros_like(actions))

    problem, original = _problem()
    problem = _with_goal(problem, 8, np.array([0.2, -0.1, 0.1]))
    cand = np.stack([original[8:12], original[8:12] + 0.5])
    out = evaluate_candidates(BlindToy(), problem, 8, cand, lam=0.0)
    assert out["goal_distance"][0] == pytest.approx(out["goal_distance"][1], abs=1e-6)
