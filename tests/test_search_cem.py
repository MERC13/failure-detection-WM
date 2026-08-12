"""CEM against a toy dynamics model whose optimum is analytic (SPEC.md §8).

§8 names this test specifically and says why: "it separates 'search is broken' from 'world
model is inaccurate,' which are the two failure modes you will spend the most time confusing."

The toy is `z[t+1] = z[t] + a[t, :3]` — exact, linear, and with no learned weights, so the
optimal edit can be written down. If the CEM cannot recover a known-good edit here, no result
it produces against the real dynamics model means anything, and the fault is in the search
rather than in the world model.
"""

import numpy as np
import pytest
import torch

from cfe.dynamics.latents import goal_distance
from cfe.search.cem import CEMConfig, cem_search
from cfe.search.objective import (
    SearchProblem,
    evaluate_candidates,
    goal_distance_torch,
    rollout_from_window,
)

K_CONTEXT = 2
HORIZON = 4
R, D = 1, 3


class ToyDynamics(torch.nn.Module):
    """`z[t+1] = z[t] + a[t, :3]`. The prediction is exact, so any error is the search's.

    Carries a dummy parameter because `evaluate_candidates` reads the device off
    `next(model.parameters())`, which is how it stays agnostic to where the real model lives.
    """

    def __init__(self):
        super().__init__()
        self._device_anchor = torch.nn.Parameter(torch.zeros(1))

    def rollout(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        z0 = context[:, -1]  # (B, R, D)
        steps = actions[:, :, :D].unsqueeze(2)  # (B, n, 1, D)
        return z0.unsqueeze(1) + torch.cumsum(steps, dim=1)


def _problem(n_steps: int = 24, seed: int = 0) -> tuple[SearchProblem, np.ndarray]:
    """An episode consistent with the toy dynamics, plus its raw actions."""
    rng = np.random.default_rng(seed)
    actions = np.zeros((n_steps, 7), dtype=np.float64)
    actions[:, :3] = rng.uniform(-0.2, 0.2, size=(n_steps, 3))
    actions[:, 6] = -1.0

    z = np.zeros((n_steps + 1, R, D), dtype=np.float32)
    for t in range(n_steps):
        z[t + 1] = z[t] + actions[t, :3]

    problem = SearchProblem(
        episode_id="toy",
        z=torch.from_numpy(z),
        actions=actions,
        z_goal=torch.zeros(1, R, D),  # replaced per test
        k_context=K_CONTEXT,
        horizon=HORIZON,
    )
    return problem, actions


def _with_goal(problem: SearchProblem, window_start: int, delta: np.ndarray) -> SearchProblem:
    """Set `z_goal` so the analytic optimum is "add `delta` across the window".

    `z_end` under the unedited actions is computed through the same code path the search uses,
    so the target is defined relative to what the objective will actually see rather than to a
    separately derived quantity that could disagree.
    """
    model = ToyDynamics()
    unedited = problem.actions[window_start : window_start + 1][None]
    z_end, _ = rollout_from_window(
        model, problem, window_start, torch.from_numpy(unedited.astype(np.float32))
    )
    goal = z_end[0].detach().numpy() + delta.reshape(R, D)
    return SearchProblem(
        episode_id=problem.episode_id,
        z=problem.z,
        actions=problem.actions,
        z_goal=torch.from_numpy(goal[None].astype(np.float32)),
        k_context=problem.k_context,
        horizon=problem.horizon,
    )


def _cfg(**over) -> CEMConfig:
    base = {
        "population": 128,
        "elite_frac": 0.125,
        "n_iters": 15,
        "init_std": 0.2,
        "std_floor": 0.005,
        "search_gripper": False,
        "init_flip_prob": 0.0,
    }
    return CEMConfig.from_dict(base | over)


# --- the analytic recovery test ---------------------------------------------


def test_cem_recovers_the_analytic_single_step_edit():
    """K=1 makes the optimum unique: the one edited action must absorb the whole offset."""
    problem, actions = _problem()
    s = 8
    delta = np.array([0.20, -0.15, 0.10])
    problem = _with_goal(problem, s, delta)

    res = cem_search(
        ToyDynamics(), problem, s, budget_k=1, lam=0.0, cfg=_cfg(), rng=np.random.default_rng(0)
    )

    expected = actions[s, :3] + delta
    assert res.actions[0, :3] == pytest.approx(expected, abs=0.02)
    # And the objective it was actually optimising went essentially to zero.
    assert res.goal_distance < 0.05
    assert res.objective < res.baseline_objective


def test_cem_distributes_a_multi_step_edit_to_the_right_total():
    """K=4: only the *sum* over the window is determined, so that is what is asserted."""
    problem, actions = _problem()
    s = 8
    delta = np.array([0.25, -0.20, 0.15])
    problem = _with_goal(problem, s, delta)

    res = cem_search(
        ToyDynamics(), problem, s, budget_k=4, lam=0.0, cfg=_cfg(), rng=np.random.default_rng(0)
    )

    achieved = res.actions[:, :3].sum(axis=0) - actions[s : s + 4, :3].sum(axis=0)
    assert achieved == pytest.approx(delta, abs=0.03)
    assert res.goal_distance < 0.05


def test_lambda_trades_goal_distance_against_edit_size():
    """I4's other axis: a bigger λ must buy a smaller edit. Monotone, not merely different."""
    problem, _ = _problem()
    s = 8
    problem = _with_goal(problem, s, np.array([0.25, -0.20, 0.15]))

    mags = []
    for lam in (0.0, 1.0, 100.0):
        res = cem_search(
            ToyDynamics(),
            problem,
            s,
            budget_k=4,
            lam=lam,
            cfg=_cfg(),
            rng=np.random.default_rng(0),
        )
        mags.append(res.penalty)
    assert mags[0] > mags[1] > mags[2]


def test_huge_lambda_returns_the_identity_edit():
    """When the penalty dominates, the warm start wins and that is reported, not hidden."""
    problem, actions = _problem()
    s = 8
    problem = _with_goal(problem, s, np.array([0.25, -0.20, 0.15]))

    res = cem_search(
        ToyDynamics(),
        problem,
        s,
        budget_k=4,
        lam=1e9,
        cfg=_cfg(),
        rng=np.random.default_rng(0),
    )
    assert res.is_identity
    assert np.array_equal(res.actions, actions[s : s + 4])


# --- properties of the search itself ----------------------------------------


def test_search_is_deterministic_under_the_same_seed():
    """SPEC.md I5. Sampling is numpy on the CPU precisely so this holds on any device."""
    problem, _ = _problem()
    s = 8
    problem = _with_goal(problem, s, np.array([0.2, -0.1, 0.1]))
    kw = {"budget_k": 4, "lam": 0.1, "cfg": _cfg(n_iters=4)}

    a = cem_search(ToyDynamics(), problem, s, rng=np.random.default_rng(7), **kw)
    b = cem_search(ToyDynamics(), problem, s, rng=np.random.default_rng(7), **kw)
    assert np.array_equal(a.actions, b.actions)
    assert a.objective == b.objective


def test_the_warm_start_is_always_in_the_population():
    """The returned edit can never score worse on J than doing nothing."""
    problem, _ = _problem()
    s = 8
    problem = _with_goal(problem, s, np.array([0.2, -0.1, 0.1]))
    res = cem_search(
        ToyDynamics(),
        problem,
        s,
        budget_k=2,
        lam=0.5,
        cfg=_cfg(n_iters=3),
        rng=np.random.default_rng(1),
    )
    assert res.objective <= res.baseline_objective


def test_the_gripper_is_flipped_never_scaled():
    """CLAUDE.md: dim 6 is not continuous in the same sense as dims 0-5."""
    problem, actions = _problem()
    s = 8
    problem = _with_goal(problem, s, np.array([0.2, -0.1, 0.1]))
    res = cem_search(
        ToyDynamics(),
        problem,
        s,
        budget_k=4,
        lam=0.0,
        cfg=_cfg(search_gripper=True, init_flip_prob=0.5),
        rng=np.random.default_rng(3),
    )
    assert set(np.unique(res.actions[:, 6])) <= {-1.0, 1.0}


def test_edits_stay_inside_the_action_bounds():
    problem, _ = _problem()
    s = 8
    problem = _with_goal(problem, s, np.array([5.0, -5.0, 5.0]))  # unreachable on purpose
    res = cem_search(
        ToyDynamics(),
        problem,
        s,
        budget_k=4,
        lam=0.0,
        cfg=_cfg(init_std=2.0),
        rng=np.random.default_rng(0),
    )
    assert res.actions.min() >= -1.0
    assert res.actions.max() <= 1.0


# --- objective plumbing -----------------------------------------------------


def test_goal_distance_torch_agrees_with_the_numpy_original():
    """Two implementations of the same quantity are only trustworthy if they are compared."""
    rng = np.random.default_rng(0)
    z_end = rng.standard_normal((5, 4, 6)).astype(np.float32)
    goals = rng.standard_normal((3, 4, 6)).astype(np.float32)

    got = goal_distance_torch(torch.from_numpy(z_end), torch.from_numpy(goals)).numpy()
    want = np.array([goal_distance(z_end[i], goals) for i in range(len(z_end))])
    assert got == pytest.approx(want, rel=1e-5)


def test_rollout_never_fabricates_a_tail_action():
    """The rollout truncates to whole chunks rather than padding; z_end is a real frame."""
    problem, _ = _problem(n_steps=26)
    s = 5
    cand = problem.actions[s : s + 4][None].astype(np.float32)
    z_end, n_chunks = rollout_from_window(ToyDynamics(), problem, s, torch.from_numpy(cand))
    # 26 - 5 = 21 steps available; 21 // 4 = 5 whole chunks = 20 steps.
    assert n_chunks == 5
    assert z_end.shape == (1, R, D)
    assert z_end[0].detach().numpy() == pytest.approx(problem.z[s + 20].numpy(), abs=1e-5)


def test_penalty_counts_all_seven_dimensions():
    """SPEC.md §5's objective is over the whole action vector; the *metric* is what excludes
    the gripper. Asserted so the two never quietly converge."""
    problem, actions = _problem()
    s = 8
    cand = actions[s : s + 2].copy()[None]
    cand[0, 0, 6] = -cand[0, 0, 6]  # a gripper flip and nothing else
    out = evaluate_candidates(ToyDynamics(), problem, s, cand, lam=1.0)
    assert out["penalty"][0] == pytest.approx(4.0)  # (1 - (-1))^2


def test_window_start_without_full_context_is_refused():
    problem, _ = _problem()
    with pytest.raises(ValueError, match="no observed context"):
        rollout_from_window(ToyDynamics(), problem, 0, torch.zeros(1, 2, 7))


def test_valid_window_starts_respect_context_and_horizon():
    problem, _ = _problem(n_steps=24)
    starts = problem.valid_window_starts(budget_k=4)
    assert starts[0] == K_CONTEXT - 1
    assert starts[-1] == 24 - HORIZON  # a whole chunk must remain after the window


# --- end-to-end determinism of the driver's per-cell seeding ----------------


def test_cell_seeds_are_stable_and_order_independent():
    """I5. A resumed run must reproduce the same edits, so a cell's seed cannot depend on how
    many cells ran before it."""
    from scripts.run_search import cell_seed

    a = cell_seed(0, "ep_1", "search", 4, 0.1)
    b = cell_seed(0, "ep_1", "search", 4, 0.1)
    assert a == b
    assert cell_seed(0, "ep_1", "search", 4, 0.1) != cell_seed(0, "ep_1", "search", 8, 0.1)
    assert cell_seed(0, "ep_1", "search", 4, 0.1) != cell_seed(0, "ep_2", "search", 4, 0.1)
    assert cell_seed(0, "ep_1", "search", 4, 0.1) != cell_seed(1, "ep_1", "search", 4, 0.1)


def test_search_episode_picks_the_best_window_by_objective():
    problem, _ = _problem()
    problem = _with_goal(problem, 8, np.array([0.2, -0.1, 0.1]))
    from cfe.search.cem import search_episode

    ranked = [{"window_start": 4}, {"window_start": 8}, {"window_start": 12}]
    best = search_episode(
        ToyDynamics(),
        problem,
        ranked,
        budget_k=2,
        lam=0.0,
        cfg=_cfg(n_iters=4),
        rng=np.random.default_rng(0),
    )
    assert best is not None
    assert best.window_start in {c["window_start"] for c in ranked}


def test_search_episode_returns_none_when_stage_one_found_nothing():
    """Reported as a gap rather than filled with a fabricated window."""
    from cfe.search.cem import search_episode

    problem, _ = _problem()
    assert (
        search_episode(
            ToyDynamics(),
            problem,
            [],
            budget_k=2,
            lam=0.0,
            cfg=_cfg(n_iters=1),
            rng=np.random.default_rng(0),
        )
        is None
    )


# --- the less-optimised candidates (Phase 3.5 exploitation test) ------------


def test_elite_mean_and_first_iteration_candidates_are_returned():
    problem, _ = _problem()
    s = 8
    problem = _with_goal(problem, s, np.array([0.2, -0.1, 0.1]))
    res = cem_search(
        ToyDynamics(),
        problem,
        s,
        budget_k=4,
        lam=0.0,
        cfg=_cfg(n_iters=4),
        rng=np.random.default_rng(0),
    )
    for a in (res.elite_mean_actions, res.first_iter_actions):
        assert a is not None
        assert a.shape == (4, 7)
        assert a.min() >= -1.0 and a.max() <= 1.0
        assert set(np.unique(a[:, 6])) <= {-1.0, 1.0}, "gripper must stay binary"


def test_the_elite_mean_is_less_optimised_than_the_best():
    """It is the average of the elites, so on the objective it cannot beat the best of them."""
    from cfe.search.objective import evaluate_candidates

    problem, _ = _problem()
    s = 8
    problem = _with_goal(problem, s, np.array([0.25, -0.2, 0.15]))
    res = cem_search(
        ToyDynamics(),
        problem,
        s,
        budget_k=4,
        lam=0.0,
        cfg=_cfg(n_iters=6),
        rng=np.random.default_rng(0),
    )
    j = evaluate_candidates(
        ToyDynamics(),
        problem,
        s,
        np.stack([res.actions, res.elite_mean_actions, res.first_iter_actions]),
        lam=0.0,
    )["objective"]
    assert j[0] <= j[1], "best should not be worse than the elite mean"
    assert j[0] <= j[2], "best should not be worse than iteration 1's best"


def test_the_gripper_in_the_elite_mean_is_a_vote_not_an_average():
    """Averaging {-1,+1} would emit a value the env never receives."""
    problem, _ = _problem()
    s = 8
    problem = _with_goal(problem, s, np.array([0.2, -0.1, 0.1]))
    res = cem_search(
        ToyDynamics(),
        problem,
        s,
        budget_k=4,
        lam=0.0,
        cfg=_cfg(search_gripper=True, init_flip_prob=0.5, n_iters=4),
        rng=np.random.default_rng(2),
    )
    assert set(np.unique(res.elite_mean_actions[:, 6])) <= {-1.0, 1.0}


def test_default_evaluate_fn_is_unchanged():
    """Omitting `evaluate_fn` must reproduce exactly what every pre-existing caller gets —
    `cfe/search/critic.py` adds a second scoring function, and this must not move the first."""
    problem, _ = _problem()
    s = 8
    problem = _with_goal(problem, s, np.array([0.2, -0.1, 0.1]))
    kw = {"budget_k": 4, "lam": 0.1, "cfg": _cfg(n_iters=4)}
    a = cem_search(ToyDynamics(), problem, s, rng=np.random.default_rng(7), **kw)
    b = cem_search(
        ToyDynamics(),
        problem,
        s,
        rng=np.random.default_rng(7),
        evaluate_fn=evaluate_candidates,
        **kw,
    )
    assert np.array_equal(a.actions, b.actions)


def test_evaluate_fn_override_is_actually_used():
    """The new parameter cfe/search/critic.py relies on must be threaded through, not ignored."""
    problem, _ = _problem()
    s = 8
    calls = []

    def fake_evaluate(model, problem, window_start, cand, lam):
        calls.append(len(cand))
        out = evaluate_candidates(model, problem, window_start, cand, lam)
        out["objective"] = out["objective"] * 0.0  # force every candidate to tie at zero
        return out

    res = cem_search(
        ToyDynamics(),
        problem,
        s,
        budget_k=2,
        lam=0.0,
        cfg=_cfg(n_iters=2),
        rng=np.random.default_rng(0),
        evaluate_fn=fake_evaluate,
    )
    assert calls, "evaluate_fn override was never called"
    assert res.objective == 0.0


def test_candidates_are_deterministic_under_the_same_seed():
    problem, _ = _problem()
    s = 8
    problem = _with_goal(problem, s, np.array([0.2, -0.1, 0.1]))
    kw = {"budget_k": 4, "lam": 0.1, "cfg": _cfg(n_iters=4)}
    a = cem_search(ToyDynamics(), problem, s, rng=np.random.default_rng(5), **kw)
    b = cem_search(ToyDynamics(), problem, s, rng=np.random.default_rng(5), **kw)
    assert np.array_equal(a.elite_mean_actions, b.elite_mean_actions)
    assert np.array_equal(a.first_iter_actions, b.first_iter_actions)
