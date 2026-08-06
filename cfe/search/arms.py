"""Arm dispatch: the search plus SPEC.md §6's four mandatory baselines.

§6: "Every table reports these or it does not ship." `assert_arms_cover_baselines` reads the
names straight out of the frozen `cfe.eval.metrics.BASELINES`, so a baseline added there can
never be quietly missing from a table here.

**`cfe.eval.metrics.BASELINES` is read, never mutated.** Its `action_ignoring_dynamics` entry
raises `NotImplementedError` with instructions to "implement it in cfe/search/ (Phase 3) and
register it here". The implementation lives below; registering it by writing into the frozen
module's globals at import time would change the behaviour of a frozen file at a distance, so
this module keeps its own dispatch table and uses the frozen dict only as the authority on
*which* names must be covered.

Three baselines are called straight through to the frozen implementations rather than
reimplemented, `no_edit` most of all: §6 wants it to traverse byte-identical machinery to a
real edit, which is the only reason it detects protocol bugs.

**Every baseline is given the search's chosen window.** That is deliberately the harder
comparison. `random_perturb` at a randomly chosen window would be controlling for stage 1 and
stage 2 at once and would be easy to beat; handing it the window the world model picked
isolates the question §6 actually asks — do the world model's *deltas* beat random deltas of
the same size in the same place. If the search cannot win that, it has not won.
"""

from __future__ import annotations

import numpy as np

from cfe.eval.metrics import (
    BASELINES,
    NotApplicable,
    edit_magnitude,
    expert_oracle,
    no_edit,
    random_perturb,
)
from cfe.eval.replay_protocol import Edit
from cfe.search.cem import CEMConfig, CEMResult, search_episode
from cfe.search.objective import SearchProblem
from cfe.search.windows import per_step_divergence, rank_windows

SEARCH = "search"
RANDOM_PERTURB = "random_perturb"
NO_EDIT = "no_edit"
EXPERT_ORACLE = "expert_oracle"
ACTION_IGNORING = "action_ignoring_dynamics"

#: Arms that are the search run against some dynamics model, and therefore need stage 1 + 2.
SEARCH_ARMS = (SEARCH, ACTION_IGNORING)


def required_baseline_names() -> frozenset[str]:
    """The four names SPEC.md §6 requires, taken from the frozen module."""
    return frozenset(BASELINES)


def assert_arms_cover_baselines(arms) -> None:
    missing = required_baseline_names() - set(arms)
    if missing:
        raise ValueError(
            f"arms {sorted(arms)} omit required SPEC.md §6 baselines {sorted(missing)}. "
            f"'Every table reports these or it does not ship.'"
        )


def run_search_arm(
    model,
    problem: SearchProblem,
    budget_k: int,
    lam: float,
    cem_cfg: CEMConfig,
    rng: np.random.Generator,
    *,
    stride: int,
    top_m: int,
    score_horizon: int,
    divergence: np.ndarray | None = None,
) -> tuple[CEMResult | None, np.ndarray]:
    """Both stages against one dynamics model. Returns the best edit and the divergence array.

    `action_ignoring_dynamics` is this function with the twin passed as `model` — the *whole*
    search, window selection included, run on the model that never sees actions. Isolating
    action-conditioning means isolating it in both stages; letting the twin inherit the
    action-conditioned model's window would leave half the comparison unmade.

    **The twin's edit is analytically the identity, and that is the point.** `LatentDynamics`
    zeroes actions inside `forward` when `ignore_actions` is set, so `z_end` does not vary with
    the candidate and `J(a') = const + lam * sum||a' - a||^2` is minimised at `a' = a` for every
    `lam >= 0`. The measured arm should come out as "no edit", every cell, and a run where it
    does not means the twin is seeing actions through some path it should not. Reported as a
    measurement rather than short-circuited: §6 asks what this baseline *does* under replay, and
    an arm that is skipped because its outcome is predictable stops being a check on anything.

    The divergence array is returned so the caller can reuse it across budgets: it depends on
    the model and the episode, not on `K` or `lambda`.
    """
    if divergence is None:
        divergence = per_step_divergence(model, problem, score_horizon)
    ranked = rank_windows(divergence, problem, budget_k, stride, top_m)
    if not ranked:
        return None, divergence
    return search_episode(model, problem, ranked, budget_k, lam, cem_cfg, rng), divergence


def realised_magnitude(original_actions: np.ndarray, edit: Edit) -> float:
    """The search's realised `mean_l2_ee` over dims 0-5, from the frozen metric.

    This is what `random_perturb` must be matched to. Taken from `edit_magnitude` rather than
    recomputed so the matched quantity is *by construction* the same quantity the table reports.
    """
    original = np.asarray(original_actions, dtype=np.float64)
    window = original[edit.window_start : edit.window_end]
    return float(edit_magnitude(window, edit.actions)["mean_l2_ee"])


def propose_baseline(
    arm: str,
    *,
    original_actions: np.ndarray,
    window_start: int,
    budget_k: int,
    rng: np.random.Generator,
    magnitude_scale: float | None = None,
    expert_actions: np.ndarray | None = None,
    cause: str | None = None,
) -> Edit:
    """Dispatch to the frozen baseline implementations.

    Raises `NotApplicable` where the frozen module does — notably `expert_oracle` on natural
    failures. That is not an error to handle away: Phase 1 verified no demo shares a natural
    failure's initial state, so there is no upper bound to compute and substituting a
    nearest-neighbour demo would report an approximation as a bound.
    """
    if arm == NO_EDIT:
        return no_edit(original_actions, window_start, budget_k)
    if arm == RANDOM_PERTURB:
        if magnitude_scale is None:
            raise ValueError(
                "random_perturb needs the search's realised mean_l2_ee as magnitude_scale. "
                "Matching only K leaves the perturbation size free, which makes §6's decisive "
                "baseline weak by construction — the easiest way to fake a positive here."
            )
        return random_perturb(
            original_actions, window_start, budget_k, rng, magnitude_scale=magnitude_scale
        )
    if arm == EXPERT_ORACLE:
        return expert_oracle(expert_actions, window_start, budget_k, cause)
    if arm in SEARCH_ARMS:
        raise ValueError(f"{arm!r} is a search arm; use run_search_arm, not propose_baseline")
    raise ValueError(f"unknown arm {arm!r}")


def edit_from_result(result: CEMResult) -> Edit:
    """Wrap a CEM result in the frozen `Edit`, which enforces the budget structurally."""
    return Edit(window_start=result.window_start, actions=result.actions)


__all__ = [
    "ACTION_IGNORING",
    "EXPERT_ORACLE",
    "NO_EDIT",
    "RANDOM_PERTURB",
    "SEARCH",
    "SEARCH_ARMS",
    "NotApplicable",
    "assert_arms_cover_baselines",
    "edit_from_result",
    "propose_baseline",
    "realised_magnitude",
    "required_baseline_names",
    "run_search_arm",
]
