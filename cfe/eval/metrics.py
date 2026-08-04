"""Reported metrics (SPEC.md §6). **FROZEN after Phase 2** — see `cfe/eval/__init__.py`.

Every function here is pure: arrays and result lists in, numbers out. No env, no torch, no
model. That is what makes §8's hand-constructed test cases exact rather than approximate, and
it keeps the instrument independent of anything it measures.

**The gripper dimension never enters an L2.** SPEC.md §5's search objective is
`λ · Σ||a'_t − a_t||²` over the whole action vector, and that stays as written *for the
objective*. But CLAUDE.md records that dim 6 is not continuous in the same sense as dims 0-5,
so averaging it into a reported magnitude would add a near-binary quantity to a continuous one
and call the sum a distance. Reported edit magnitude and sparsity are therefore computed over
**dims 0-5 only**, with gripper changes reported separately as a flip count. The divergence
between the objective and the metric is deliberate and is flagged at both sites.

**I4 lives in `rsr_curve`.** SPEC.md §6: "RSR is reported as a curve over edit budget
`K ∈ {1, 2, 4, 8}` and over `λ` on a fixed grid. Never a single tuned number." `rsr_curve`
returns the nested table and deliberately provides no "overall" key — collapsing it is the
caller's decision to make visibly, not a convenience this module offers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

import numpy as np

from cfe.eval.replay_protocol import ACTION_DIM, Edit, ReplayResult

#: Dims 0-5 are the end-effector delta; dim 6 is the gripper. See the module docstring.
EE_SLICE = slice(0, 6)
GRIPPER_DIM = 6

#: SPEC.md §6's edit-budget grid. Named here so a result reported at some other K is visibly
#: off-grid rather than quietly incomparable.
BUDGET_GRID = (1, 2, 4, 8)


# ---------------------------------------------------------------------------
# Replay success rate — the primary metric
# ---------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Wilson rather than the normal approximation because RSR is expected near 0 for `no_edit`
    and the normal interval is badly wrong (and can leave [0, 1]) at the extremes, which is
    exactly where the baseline that catches replay bugs lives.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= successes <= n:
        raise ValueError(f"successes {successes} outside [0, {n}]")
    denom = n + z * z
    center = (successes + z * z / 2) / denom
    half = (z / denom) * np.sqrt(successes * (n - successes) / n + z * z / 4)
    return float(center - half), float(center + half)


def replay_success_rate(results: Sequence[ReplayResult]) -> dict:
    """SPEC.md §6's primary metric. Reads `ReplayResult.success` and nothing else (I2)."""
    if len(results) == 0:
        raise ValueError("replay_success_rate needs at least one result")
    successes = sum(1 for r in results if r.success)
    n = len(results)
    lo, hi = wilson_interval(successes, n)
    return {
        "replay_success_rate": successes / n,
        "n_success": successes,
        "n": n,
        "ci_low": lo,
        "ci_high": hi,
        "ci_method": "wilson_95",
    }


def rsr_curve(records: Iterable[dict]) -> dict:
    """RSR over the edit-budget × λ grid (I4). Never collapse this to one number.

    Each record needs `budget_k`, `lam`, and `result`. Returns
    `{budget_k: {lam: replay_success_rate(...)}}` plus the off-grid budgets encountered, so a
    curve reported at a K the spec does not name is visible rather than silently averaged in.
    """
    grouped: dict[int, dict[float, list[ReplayResult]]] = {}
    for rec in records:
        k, lam = int(rec["budget_k"]), float(rec["lam"])
        grouped.setdefault(k, {}).setdefault(lam, []).append(rec["result"])
    if not grouped:
        raise ValueError("rsr_curve needs at least one record")
    curve = {
        k: {lam: replay_success_rate(rs) for lam, rs in sorted(by_lam.items())}
        for k, by_lam in sorted(grouped.items())
    }
    return {
        "curve": curve,
        "budgets": sorted(grouped),
        "off_grid_budgets": sorted(k for k in grouped if k not in BUDGET_GRID),
        "lambdas": sorted({lam for by_lam in grouped.values() for lam in by_lam}),
    }


# ---------------------------------------------------------------------------
# Edit descriptors
# ---------------------------------------------------------------------------


def _as_window(a: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != ACTION_DIM:
        raise ValueError(f"{name} must be (W, {ACTION_DIM}), got {arr.shape}")
    return arr


def _ee_step_norms(original: np.ndarray, edited: np.ndarray) -> np.ndarray:
    o, e = _as_window(original, "original"), _as_window(edited, "edited")
    if o.shape != e.shape:
        raise ValueError(f"window shapes differ: {o.shape} vs {e.shape}")
    return np.linalg.norm(e[:, EE_SLICE] - o[:, EE_SLICE], axis=1)


def edit_magnitude(original: np.ndarray, edited: np.ndarray) -> dict:
    """SPEC.md §6: mean and max L2 of the per-step delta. **Dims 0-5 only.**"""
    norms = _ee_step_norms(original, edited)
    return {
        "mean_l2_ee": float(norms.mean()) if norms.size else 0.0,
        "max_l2_ee": float(norms.max()) if norms.size else 0.0,
        "n_steps": int(norms.size),
        "dims": "0-5 (gripper excluded; see module docstring)",
    }


def edit_sparsity(original: np.ndarray, edited: np.ndarray, eps: float) -> dict:
    """SPEC.md §6: timesteps whose action changed by more than `eps`. **Dims 0-5 only.**

    `eps` is a swept config field, never tuned to a favourable value (CLAUDE.md rule 3).
    """
    if eps < 0:
        raise ValueError(f"eps must be >= 0, got {eps}")
    norms = _ee_step_norms(original, edited)
    changed = int(np.count_nonzero(norms > eps))
    return {
        "n_steps_changed_ee": changed,
        "n_steps": int(norms.size),
        "fraction_changed_ee": float(changed / norms.size) if norms.size else 0.0,
        "eps": float(eps),
    }


def gripper_flips(original: np.ndarray, edited: np.ndarray) -> dict:
    """Gripper changes, counted rather than normed. See the module docstring for why."""
    o, e = _as_window(original, "original"), _as_window(edited, "edited")
    if o.shape != e.shape:
        raise ValueError(f"window shapes differ: {o.shape} vs {e.shape}")
    flips = int(np.count_nonzero(e[:, GRIPPER_DIM] != o[:, GRIPPER_DIM]))
    return {"gripper_flips": flips, "n_steps": int(o.shape[0])}


# ---------------------------------------------------------------------------
# Cause identification (injected failures only)
# ---------------------------------------------------------------------------


def window_iou(found: tuple[int, int], truth: tuple[int, int]) -> float:
    """IoU of two half-open step intervals `[start, end)`. Injected failures only."""
    for name, w in (("found", found), ("truth", truth)):
        if w[1] < w[0]:
            raise ValueError(f"{name} window {w} has end before start")
    lo = max(found[0], truth[0])
    hi = min(found[1], truth[1])
    inter = max(0, hi - lo)
    union = (found[1] - found[0]) + (truth[1] - truth[0]) - inter
    return float(inter / union) if union > 0 else 0.0


def cause_top1_accuracy(predicted: Sequence[str | None], truth: Sequence[str | None]) -> dict:
    """Top-1 accuracy of the predicted cause tag, over labelled episodes only.

    **Natural failures carry `cause = None`** and are roughly half the failure set (CLAUDE.md;
    SPEC.md §4.4). They have no ground-truth cause, so they are excluded and counted rather
    than scored as wrong — scoring an unlabelled episode as a miss would make the metric a
    function of the natural/injected mix instead of of explanation quality.
    """
    if len(predicted) != len(truth):
        raise ValueError(f"predicted has {len(predicted)} entries, truth has {len(truth)}")
    labelled = [(p, t) for p, t in zip(predicted, truth, strict=True) if t is not None]
    n_excluded = len(truth) - len(labelled)
    if not labelled:
        return {
            "top1_accuracy": None,
            "n_scored": 0,
            "n_excluded_unlabelled": n_excluded,
            "reason": "no injected failures present; cause accuracy is undefined here",
        }
    correct = sum(1 for p, t in labelled if p == t)
    return {
        "top1_accuracy": correct / len(labelled),
        "n_correct": correct,
        "n_scored": len(labelled),
        "n_excluded_unlabelled": n_excluded,
    }


# ---------------------------------------------------------------------------
# Search cost and comparison
# ---------------------------------------------------------------------------


def search_cost(wall_clock_s: float, n_forward_passes: int, n_counterfactuals: int) -> dict:
    """SPEC.md §6: wall-clock and dynamics-model forward passes per counterfactual."""
    if n_counterfactuals <= 0:
        raise ValueError(f"n_counterfactuals must be positive, got {n_counterfactuals}")
    return {
        "wall_clock_s_total": float(wall_clock_s),
        "wall_clock_s_per_counterfactual": float(wall_clock_s) / n_counterfactuals,
        "forward_passes_total": int(n_forward_passes),
        "forward_passes_per_counterfactual": n_forward_passes / n_counterfactuals,
        "n_counterfactuals": int(n_counterfactuals),
    }


def bootstrap_rsr_difference(
    treatment: Sequence[bool],
    control: Sequence[bool],
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """CI for `RSR(treatment) − RSR(control)`, resampling **episodes**, paired.

    This is what G3 reads: "RSR must exceed `random_perturb` by a margin that survives a
    bootstrap CI". Paired because both arms are replayed on the same episodes from the same
    init states — the per-episode difference removes the variance from some episodes simply
    being easier, which is most of the spread. Element `i` of both arguments must be the same
    episode; an unpaired comparison here would overstate the interval.
    """
    t = np.asarray(treatment, dtype=bool)
    c = np.asarray(control, dtype=bool)
    if t.shape != c.shape:
        raise ValueError(
            f"treatment has {t.shape} entries and control {c.shape}; the comparison is paired "
            "by episode, so both arms must cover the same episodes in the same order"
        )
    if t.size == 0:
        raise ValueError("bootstrap_rsr_difference needs at least one episode")
    d = t.astype(np.float64) - c.astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "rsr_treatment": float(t.mean()),
        "rsr_control": float(c.mean()),
        "mean_difference": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_episodes": int(t.size),
        "n_boot": int(n_boot),
    }


# ---------------------------------------------------------------------------
# The four mandatory baselines (SPEC.md §6)
# ---------------------------------------------------------------------------


class NotApplicable(Exception):
    """This baseline is undefined for this episode. Never substitute an approximation."""


class Baseline(Protocol):
    """Proposes an edit for one episode at a given budget.

    Written now, before any search exists, so Phase 3 implements the search against a fixed
    interface rather than the reverse.
    """

    def propose(self, episode: dict, budget_k: int, rng: np.random.Generator) -> Edit: ...


def no_edit(original_actions: np.ndarray, window_start: int, budget_k: int) -> Edit:
    """§6: replay the original actions. Should be ~0% by construction; catches replay bugs.

    **If this does not come out near zero, replay is broken and that is a hard stop.** It is
    not a number to tune around — detecting exactly that is why §6 makes it mandatory.
    """
    from cfe.eval.replay_protocol import identity_edit

    return identity_edit(original_actions, window_start, budget_k)


def random_perturb(
    original_actions: np.ndarray,
    window_start: int,
    budget_k: int,
    rng: np.random.Generator,
    magnitude_scale: float,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> Edit:
    """§6: random deltas at the **same edit budget**. The baseline that decides if this is real.

    `magnitude_scale` is **the target per-step L2 of the delta**, in the same units and over the
    same dims 0-5 that `edit_magnitude` reports, so it can be passed straight from the search's
    realised `edit_magnitude(...)["mean_l2_ee"]` and the match is exact by construction. Each
    step gets a delta of exactly that norm in a uniformly random direction on the 6-sphere.

    Scaling six independent standard normals by `s` would give a per-step L2 of `s·√6 ≈ 2.45 s`,
    not `s` — so the obvious implementation makes this baseline 2.45x stronger than the
    magnitude it was asked to match, in a way nothing downstream would reveal. Normalising the
    direction removes the dimension-count dependence entirely.

    `magnitude_scale` is required and has no default **on purpose**. Matching only `K` while
    leaving the perturbation size free would make this baseline weak by construction — tiny
    random deltas fail trivially and the search would "win" against nothing. §6 calls this the
    baseline that decides whether the project is real, so it must be matched on budget *and*
    magnitude. That the scale comes from the search's realised edits is a caller obligation this
    frozen module cannot enforce, and it is the single easiest way to fake a positive result
    here. Stated so it cannot be forgotten quietly.

    Only dims 0-5 are perturbed; the gripper is left alone, consistent with the reported
    metrics. Actions are clipped to the env's valid range so the baseline lives in the same
    action space the search does — which means the *realised* magnitude can come in under
    `magnitude_scale` when the original action is already near a bound. Callers comparing arms
    should report `edit_magnitude` of what was actually replayed, not the requested scale.
    """
    if magnitude_scale < 0:
        raise ValueError(f"magnitude_scale must be >= 0, got {magnitude_scale}")
    original = np.asarray(original_actions, dtype=np.float64)
    edited = original[window_start : window_start + budget_k].copy()
    if edited.shape[0] != budget_k:
        raise ValueError(
            f"window [{window_start}, {window_start + budget_k}) runs past the episode's "
            f"{len(original)} actions"
        )
    direction = rng.standard_normal(size=(budget_k, 6))
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # measure-zero, but a zero-norm row would divide by zero
    delta = direction / norms * magnitude_scale
    edited[:, EE_SLICE] = np.clip(edited[:, EE_SLICE] + delta, action_low, action_high)
    return Edit(window_start=window_start, actions=edited)


def expert_oracle(
    expert_actions: np.ndarray | None,
    window_start: int,
    budget_k: int,
    cause: str | None,
) -> Edit:
    """§6: replay the successful demo's actions from the window. Upper bound.

    **Defined for injected failures only.** SPEC.md §5 wants "a successful demo for the same
    task *and initial state*", and Phase 1 verified that no demo init state matches any
    benchmark init state — so for a natural failure (`cause is None`) there is no expert
    trajectory from the same state and none can be manufactured. Raises `NotApplicable` rather
    than substituting a nearest-neighbour demo: an approximate expert is not an upper bound,
    and CLAUDE.md is explicit that approximations here are reported, not improvised.

    For an injected failure the expert is the original uncorrupted episode, which exists by
    construction.
    """
    if cause is None:
        raise NotApplicable(
            "expert_oracle is undefined for natural failures: no successful demo shares this "
            "episode's initial state (SPEC.md §5 / §6 open gap, see PROGRESS.md)"
        )
    if expert_actions is None:
        raise NotApplicable(f"no expert trajectory supplied for injected cause {cause!r}")
    expert = np.asarray(expert_actions, dtype=np.float64)
    if window_start + budget_k > len(expert):
        raise ValueError(
            f"window [{window_start}, {window_start + budget_k}) runs past the expert's "
            f"{len(expert)} actions"
        )
    return Edit(window_start=window_start, actions=expert[window_start : window_start + budget_k])


def action_ignoring_dynamics(*_args, **_kwargs) -> Edit:
    """§6: run the search using the actions-zeroed dynamics model. Isolates action-conditioning.

    Interface only. This baseline *is* the search, run against the trained twin
    (`action_ignoring` in a `train_dynamics.py` checkpoint), so it cannot be implemented before
    `cfe/search/` exists. Declared here so Phase 3 cannot ship a table without it — §6 says
    every table reports all four or it does not ship.
    """
    raise NotImplementedError(
        "action_ignoring_dynamics is the search run with the action-ignoring twin; implement it "
        "in cfe/search/ (Phase 3) and register it here. The twin weights are already trained "
        "and live under the 'action_ignoring' key of a train_dynamics.py checkpoint."
    )


#: SPEC.md §6: "Every table reports these or it does not ship."
BASELINES = {
    "no_edit": no_edit,
    "random_perturb": random_perturb,
    "expert_oracle": expert_oracle,
    "action_ignoring_dynamics": action_ignoring_dynamics,
}
