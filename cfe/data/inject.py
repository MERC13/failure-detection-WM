"""Corrupting successful demos in known ways, to make failures with ground-truth causes.

SPEC.md §4.4 needs two kinds of failure. Natural ones come free from BC rollouts but carry no
cause label. These are the other kind: take a demo that succeeds, break it in a way we chose,
and we know both *what* went wrong and *which timesteps* were responsible. That ground-truth
edit window is what makes SPEC.md §6's cause-identification metrics computable at all.

Three properties every injection here holds to, because each one is load-bearing downstream:

- **Action-sequence length is preserved.** The uncorrupted source demo is then a valid
  `expert_actions` for `cfe.eval.metrics.expert_oracle` at the *same indices*, which is the only
  reason that upper bound is computable. An injection that inserted or deleted steps would
  silently misalign the oracle by however many steps it moved.
- **Dim 6 stays in {-1, +1}.** CLAUDE.md: the gripper is not continuous in the same sense as
  dims 0-5. Injections that touch it flip it, never scale it.
- **Offsets are specified in metres**, not action units, and converted through a calibration
  measured from that task's own demos (`metres_per_unit_action`). "Shift the approach by δ in
  x/y" in SPEC.md §4.4 is a statement about the world, and a number in action units cannot be
  checked against anything. The realised offset is verified afterwards against the *replayed*
  proprioception, so the injection is confirmed by an independent measurement rather than
  assumed to have worked.

Measured on the 48 cached `libero_object` task-0 demos (2026-08-04): every demo opens the
gripper at t=0, closes once around t=42 and releases once around t=128. 44 have exactly one
close/release pair; 4 regrasp. `gripper_phases` is written for both.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTION_DIM = 7
GRIPPER_DIM = 6
N_DELTA = 6

#: The `cause` tags of SPEC.md §4.4. Natural failures carry `None`, not a member of this tuple.
CAUSES = ("grasp_offset", "mistimed_close", "early_release", "wrong_target")

#: Env action bounds. Injections clip to these, so a requested offset can be realised smaller
#: than asked for; `Injection.realised_offset_m` records what actually went in.
ACTION_LOW, ACTION_HIGH = -1.0, 1.0


@dataclass(frozen=True)
class GripperPhases:
    """Where the grasp and the release happen in a demo's action sequence.

    `close_idx` is the first step commanding a closed gripper; `release_idx` is the *last*
    reopening after it. Taking the last rather than the first matters for the 4-in-48 demos
    that regrasp: their first reopening is a mid-episode fumble, not the place-and-release, and
    an `early_release` injection anchored there would corrupt the wrong phase of the episode.
    """

    close_idx: int
    release_idx: int
    n_transitions: int

    @property
    def regrasped(self) -> bool:
        return self.n_transitions > 2


def gripper_phases(actions: np.ndarray) -> GripperPhases:
    """Locate the grasp and release in `(T, 7)` actions.

    Raises rather than guessing. A demo without a clean close-then-open structure is not one
    these injections describe, and silently anchoring an injection at step 0 would produce a
    "failure" whose recorded cause is a fiction.
    """
    a = _as_actions(actions, "actions")
    g = a[:, GRIPPER_DIM]
    # The raw values, not their sign: `np.sign` maps 0.3 to 1.0, so signing first would accept
    # a continuous gripper and then locate the phases as though it were binary. The demos are
    # strictly {-1, +1} (PROGRESS.md surprise 10) and anything else is a different dataset.
    if not np.all(np.isin(g, (-1.0, 1.0))):
        offending = np.unique(g[~np.isin(g, (-1.0, 1.0))])[:5]
        raise ValueError(
            f"gripper dimension must be strictly -1 or +1 in demo actions; saw {offending}"
        )

    changes = np.flatnonzero(g[1:] != g[:-1]) + 1
    closes = [int(i) for i in changes if g[i] > 0]
    opens = [int(i) for i in changes if g[i] < 0]
    if not closes or not opens:
        raise ValueError(
            f"expected at least one close and one release; found {len(closes)} close(s) "
            f"and {len(opens)} release(s)"
        )
    close_idx = closes[0]
    after = [i for i in opens if i > close_idx]
    if not after:
        raise ValueError(f"no gripper release after the close at step {close_idx}")
    return GripperPhases(close_idx, after[-1], len(changes))


def metres_per_unit_action(actions: np.ndarray, eef_pos: np.ndarray) -> dict:
    """Calibrate translation action units against end-effector travel, per axis.

    `actions (T, 7)` and `eef_pos (T+1, 3)` from the same replayed episode: `eef_pos[t+1] -
    eef_pos[t]` is what `actions[t]` moved the arm. Fit is least-squares through the origin —
    a zero action must mean zero travel, so an intercept would be fitting noise a physical
    model has no room for.

    Returns the per-axis slope and the correlation, because a slope without its correlation is
    a number one cannot judge. Measured on task 0: 0.0099 / 0.0130 / 0.0121 m per unit, with
    correlations 0.90 / 0.99 / 0.98.
    """
    a = _as_actions(actions, "actions")[:, :3]
    p = np.asarray(eef_pos, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"eef_pos must be (T+1, 3), got {p.shape}")
    if len(p) != len(a) + 1:
        raise ValueError(f"{len(p)} eef positions for {len(a)} actions; expected T+1 for T")

    d = np.diff(p, axis=0)
    slope = np.empty(3)
    corr = np.empty(3)
    for i in range(3):
        denom = float(a[:, i] @ a[:, i])
        slope[i] = float(a[:, i] @ d[:, i]) / denom if denom > 0 else 0.0
        corr[i] = float(np.corrcoef(a[:, i], d[:, i])[0, 1])
    return {
        "m_per_unit": slope,
        "corr": corr,
        "n_steps": int(len(a)),
        "fit": "least squares through the origin",
    }


@dataclass(frozen=True)
class Injection:
    """A corrupted action sequence plus everything needed to score the counterfactual against it.

    `window` is the **ground-truth edit window**, half-open `[start, end)`, in the same
    convention as `cfe.eval.metrics.window_iou`. It is the set of steps this injection actually
    changed, not the set it was asked to change: if clipping at the action bounds left a step
    untouched, that step is not in the window.
    """

    actions: np.ndarray  # (T, 7)
    window: tuple[int, int]
    cause: str
    params: dict

    def __post_init__(self) -> None:
        a = np.array(self.actions, dtype=np.float64)
        _as_actions(a, "injected actions")
        if not np.all(np.isin(np.sign(a[:, GRIPPER_DIM]), (-1.0, 1.0))):
            raise ValueError("injection left the gripper dimension outside {-1, +1}")
        if self.cause not in CAUSES:
            raise ValueError(f"unknown cause {self.cause!r}; expected one of {CAUSES}")
        s, e = self.window
        if not 0 <= s <= e <= len(a):
            raise ValueError(f"window {self.window} outside [0, {len(a)}]")
        a.flags.writeable = False
        object.__setattr__(self, "actions", a)


# ---------------------------------------------------------------------------
# The four causes of SPEC.md §4.4
# ---------------------------------------------------------------------------


def grasp_offset(
    actions: np.ndarray,
    phases: GripperPhases,
    delta_m: np.ndarray,
    window_steps: int,
    m_per_unit: np.ndarray,
) -> Injection:
    """§4.4: "shift the approach by δ in x/y before grasp".

    The shift is spread evenly over the `window_steps` immediately before the close, so the arm
    arrives at the grasp pose displaced by `delta_m` and stays displaced — the offset is not
    compensated afterwards, which is what makes it a grasp *offset* rather than a detour.

    Spreading rather than applying it in one step is deliberate: a single-step correction of
    several centimetres saturates the action bound and the OSC controller would not track it,
    so the realised offset would silently be a fraction of the requested one.
    """
    return _apply_xy_offset(
        actions,
        phases,
        np.asarray(delta_m, float),
        window_steps,
        m_per_unit,
        cause="grasp_offset",
        extra={},
    )


def wrong_target(
    actions: np.ndarray,
    phases: GripperPhases,
    target_xy: np.ndarray,
    distractor_xy: np.ndarray,
    fraction: float,
    window_steps: int,
    m_per_unit: np.ndarray,
) -> Injection:
    """§4.4: "retarget to a distractor object".

    Implemented as an approach offset pointed at a specific distractor, `fraction` of the way
    from the target to it. **This is mechanically the same operation as `grasp_offset`**, with
    a direction chosen by the scene rather than arbitrarily and a magnitude an order larger.
    The two causes are distinguishable by direction and size but they are not independent
    mechanisms, and pretending otherwise would overstate how diverse the failure set is.
    Recorded here, in PROGRESS.md, and in anything that reports cause accuracy.
    """
    t = np.asarray(target_xy, dtype=np.float64)[:2]
    d = np.asarray(distractor_xy, dtype=np.float64)[:2]
    return _apply_xy_offset(
        actions,
        phases,
        (d - t) * float(fraction),
        window_steps,
        m_per_unit,
        cause="wrong_target",
        extra={
            "target_xy": t.tolist(),
            "distractor_xy": d.tolist(),
            "fraction": float(fraction),
            "separation_m": float(np.linalg.norm(d - t)),
        },
    )


def mistimed_close(actions: np.ndarray, phases: GripperPhases, shift: int) -> Injection:
    """§4.4: "close the gripper N steps early/late". Negative `shift` is early, positive late.

    Only the steps between the old and new close instants change, and they change by a flip,
    never a scale. The release is left where it was: moving both would be a phase shift of the
    whole grasp rather than a mistimed close.
    """
    a = _as_actions(actions, "actions").copy()
    if shift == 0:
        raise ValueError("shift must be non-zero; a zero shift is not an injection")
    new_close = int(np.clip(phases.close_idx + shift, 1, phases.release_idx - 1))
    if new_close == phases.close_idx:
        raise ValueError(
            f"shift {shift} leaves the close at step {phases.close_idx} after clipping to "
            f"[1, {phases.release_idx - 1}]; this would produce an uncorrupted episode"
        )
    lo, hi = min(new_close, phases.close_idx), max(new_close, phases.close_idx)
    # Early close: the window is commanded shut where it used to be open, and vice versa.
    a[lo:hi, GRIPPER_DIM] = 1.0 if new_close < phases.close_idx else -1.0
    return Injection(
        actions=a,
        window=(lo, hi),
        cause="mistimed_close",
        params={
            "shift": int(shift),
            "original_close_idx": int(phases.close_idx),
            "new_close_idx": new_close,
        },
    )


def early_release(actions: np.ndarray, phases: GripperPhases, steps: int) -> Injection:
    """§4.4: "open the gripper before the place pose".

    `steps` counts back from the demo's own release. Clipped so the release cannot land at or
    before the close — an "early release" that opens before the grasp is a failure to grasp,
    which is a different cause and would be mislabelled.
    """
    a = _as_actions(actions, "actions").copy()
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    start = int(np.clip(phases.release_idx - steps, phases.close_idx + 1, phases.release_idx - 1))
    if start >= phases.release_idx:
        raise ValueError(
            f"the grasp at {phases.close_idx} and release at {phases.release_idx} leave no room "
            f"to release {steps} steps early"
        )
    a[start : phases.release_idx, GRIPPER_DIM] = -1.0
    return Injection(
        actions=a,
        window=(start, int(phases.release_idx)),
        cause="early_release",
        params={
            "steps": int(steps),
            "realised_steps": int(phases.release_idx - start),
            "original_release_idx": int(phases.release_idx),
        },
    )


# ---------------------------------------------------------------------------


def _apply_xy_offset(
    actions: np.ndarray,
    phases: GripperPhases,
    delta_m: np.ndarray,
    window_steps: int,
    m_per_unit: np.ndarray,
    *,
    cause: str,
    extra: dict,
) -> Injection:
    """Shared body of `grasp_offset` and `wrong_target`. See both for the semantics."""
    a = _as_actions(actions, "actions").copy()
    delta_m = np.asarray(delta_m, dtype=np.float64)[:2]
    scale = np.asarray(m_per_unit, dtype=np.float64)[:2]
    if window_steps <= 0:
        raise ValueError(f"window_steps must be positive, got {window_steps}")
    if np.any(scale <= 0):
        raise ValueError(f"m_per_unit must be positive in x and y, got {scale}")

    end = int(phases.close_idx)
    start = int(max(0, end - window_steps))
    if start >= end:
        raise ValueError(f"no approach window before the close at step {end}")

    per_step = delta_m / (scale * (end - start))
    before = a[start:end, :2].copy()
    a[start:end, :2] = np.clip(before + per_step, ACTION_LOW, ACTION_HIGH)
    # What actually went in, after the bounds had their say. Requested and realised are both
    # recorded because they can differ and only one of them is true.
    realised_m = (a[start:end, :2] - before).sum(axis=0) * scale
    changed = np.flatnonzero(np.any(a[start:end, :2] != before, axis=1))

    return Injection(
        actions=a,
        # The window is the steps that actually moved, not the steps we aimed at.
        window=(start + int(changed[0]), start + int(changed[-1]) + 1)
        if changed.size
        else (start, start),
        cause=cause,
        params={
            "requested_delta_m": delta_m.tolist(),
            "realised_delta_m": realised_m.tolist(),
            "window_steps": int(end - start),
            "per_step_action_offset": per_step.tolist(),
            "m_per_unit_xy": scale.tolist(),
            "clipped": bool(changed.size < (end - start)),
            **extra,
        },
    )


def _as_actions(a: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != ACTION_DIM:
        raise ValueError(f"{name} must be (T, {ACTION_DIM}), got {arr.shape}")
    return arr


def realised_offset_from_proprio(
    baseline_eef: np.ndarray, injected_eef: np.ndarray, at_step: int
) -> dict:
    """Independent check that an offset injection did what it claimed, from replayed proprio.

    CLAUDE.md: distrust a new measurement until something independent agrees with it. The
    injection computes its offset from a calibration; this reads the *actual* end-effector
    separation between the clean and corrupted replays at the grasp instant. If the two
    disagree badly, the calibration is wrong and the `cause` labels are not describing what
    happened.
    """
    b = np.asarray(baseline_eef, dtype=np.float64)
    j = np.asarray(injected_eef, dtype=np.float64)
    i = int(min(at_step, len(b) - 1, len(j) - 1))
    sep = j[i, :2] - b[i, :2]
    return {
        "at_step": i,
        "separation_xy_m": sep.tolist(),
        "separation_norm_m": float(np.linalg.norm(sep)),
    }
