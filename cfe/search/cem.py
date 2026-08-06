"""Stage 2 of SPEC.md §5: CEM over deltas within the chosen window.

"Warm-start at the executed actions. ... subject to at most `K` edited timesteps (enforced
structurally by the window, not by a penalty)." The structural enforcement is `Edit` in the
frozen `cfe/eval/replay_protocol.py`: the window *is* the budget, so there is nowhere to put a
longer edit. Nothing here needs a sparsity penalty to hold `K`.

SPEC.md §5 is explicit that joint sparse optimisation is not attempted: "it is harder to
implement, harder to interpret, and not needed."

Two properties worth stating because they change how a result reads:

- **The unedited window is in every population.** The warm start is re-inserted at each
  iteration, so the returned edit can never score worse on `J` than doing nothing. When the
  original wins outright the result is flagged `is_identity`, which matters downstream: a
  search whose edit is the identity makes `random_perturb` — matched on the search's realised
  magnitude — the identity too, and the cell compares nothing. That has to be visible, not
  averaged in.
- **All sampling is numpy on the CPU with an explicit `Generator`.** Same seed, same edits,
  regardless of device (SPEC.md I5). Sampling in torch on CUDA would make the edits
  device-dependent, and G2 already measured that GPU kernels are not batch-invariant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cfe.search.objective import SearchProblem, evaluate_candidates

ACTION_DIM = 7
N_DELTA = 6
GRIPPER_DIM = 6
ACTION_LOW, ACTION_HIGH = -1.0, 1.0


@dataclass(frozen=True)
class CEMConfig:
    population: int
    elite_frac: float
    n_iters: int
    init_std: float
    std_floor: float
    search_gripper: bool
    init_flip_prob: float

    @classmethod
    def from_dict(cls, d: dict) -> CEMConfig:
        return cls(
            population=int(d["population"]),
            elite_frac=float(d["elite_frac"]),
            n_iters=int(d["n_iters"]),
            init_std=float(d["init_std"]),
            std_floor=float(d["std_floor"]),
            search_gripper=bool(d["search_gripper"]),
            init_flip_prob=float(d["init_flip_prob"]),
        )

    @property
    def n_elite(self) -> int:
        # At least 2: refitting a std from a single elite gives zero spread and the search
        # collapses on iteration 1.
        return max(2, int(round(self.population * self.elite_frac)))


@dataclass(frozen=True)
class CEMResult:
    """The best edit found for one (episode, window, K, lambda) cell.

    `objective`, `goal_distance` and `penalty` are **search diagnostics, not results**
    (CLAUDE.md rule 2). Only the replay of `actions` produces a number that may be reported.
    """

    window_start: int
    budget_k: int
    lam: float
    actions: np.ndarray  # (K, 7)
    objective: float
    goal_distance: float
    penalty: float
    baseline_objective: float  # J of the unedited window, for reference
    n_forward_passes: int
    is_identity: bool
    iterations: list[dict]


def cem_search(
    model,
    problem: SearchProblem,
    window_start: int,
    budget_k: int,
    lam: float,
    cfg: CEMConfig,
    rng: np.random.Generator,
) -> CEMResult:
    """Cross-entropy method over the `K x 7` window, warm-started at the executed actions."""
    original = problem.actions[window_start : window_start + budget_k].astype(np.float64)
    if original.shape != (budget_k, ACTION_DIM):
        raise ValueError(
            f"window [{window_start}, {window_start + budget_k}) runs past the episode's "
            f"{problem.n_steps} actions"
        )

    mean = original[:, :N_DELTA].copy()
    std = np.full((budget_k, N_DELTA), float(cfg.init_std))
    flip_prob = np.full(budget_k, float(cfg.init_flip_prob) if cfg.search_gripper else 0.0)

    best = original.copy()
    best_j = np.inf
    best_dist = np.nan
    best_pen = np.nan
    baseline_j = np.nan
    n_fwd = 0
    history: list[dict] = []

    for it in range(cfg.n_iters):
        cand = _sample(original, mean, std, flip_prob, cfg.population, rng)
        out = evaluate_candidates(model, problem, window_start, cand, lam)
        n_fwd += out["n_forward_passes"]
        j = out["objective"]
        # Candidate 0 is the warm start, so its J is the do-nothing reference every iteration.
        baseline_j = float(j[0])

        order = np.argsort(j, kind="stable")
        elite = order[: cfg.n_elite]
        if j[order[0]] < best_j:
            best_j = float(j[order[0]])
            best = cand[order[0]].copy()
            best_dist = float(out["goal_distance"][order[0]])
            best_pen = float(out["penalty"][order[0]])

        history.append(
            {
                "iter": it,
                "best_objective": float(j[order[0]]),
                "mean_objective": float(j.mean()),
                "elite_mean_objective": float(j[elite].mean()),
                "baseline_objective": baseline_j,
                "mean_std": float(std.mean()),
            }
        )

        # Refit. `ddof=0` because this is the elite population's own spread, not an estimate of
        # a wider population's.
        elite_actions = cand[elite]
        mean = elite_actions[:, :, :N_DELTA].mean(axis=0)
        std = np.maximum(elite_actions[:, :, :N_DELTA].std(axis=0), cfg.std_floor)
        if cfg.search_gripper:
            flipped = elite_actions[:, :, GRIPPER_DIM] != original[None, :, GRIPPER_DIM]
            flip_prob = flipped.mean(axis=0)

    return CEMResult(
        window_start=int(window_start),
        budget_k=int(budget_k),
        lam=float(lam),
        actions=best,
        objective=best_j,
        goal_distance=best_dist,
        penalty=best_pen,
        baseline_objective=baseline_j,
        n_forward_passes=n_fwd,
        is_identity=bool(np.array_equal(best, original)),
        iterations=history,
    )


def _sample(
    original: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    flip_prob: np.ndarray,
    population: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """`(P, K, 7)` candidates. Row 0 is always the unedited window; see the module docstring."""
    k = original.shape[0]
    cand = np.empty((population, k, ACTION_DIM), dtype=np.float64)

    deltas = rng.standard_normal(size=(population, k, N_DELTA)) * std[None]
    cand[:, :, :N_DELTA] = np.clip(mean[None] + deltas, ACTION_LOW, ACTION_HIGH)

    # The gripper is flipped, never scaled: it takes exactly {-1, +1} and CLAUDE.md forbids
    # treating dim 6 as continuous like dims 0-5.
    flips = rng.random(size=(population, k)) < flip_prob[None]
    grip = np.where(flips, -original[None, :, GRIPPER_DIM], original[None, :, GRIPPER_DIM])
    cand[:, :, GRIPPER_DIM] = grip

    cand[0] = original
    return cand


def search_episode(
    model,
    problem: SearchProblem,
    ranked_windows: list[dict],
    budget_k: int,
    lam: float,
    cfg: CEMConfig,
    rng: np.random.Generator,
) -> CEMResult | None:
    """Run stage 2 on each stage-1 candidate window and keep the best by `J`.

    Returns `None` when stage 1 produced no usable window — for instance an episode too short
    to leave a whole chunk after the window. Reported as a gap rather than filled with a
    fabricated window.
    """
    best: CEMResult | None = None
    for cand in ranked_windows:
        res = cem_search(model, problem, cand["window_start"], budget_k, lam, cfg, rng)
        if best is None or res.objective < best.objective:
            best = res
    return best
