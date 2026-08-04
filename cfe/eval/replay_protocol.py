"""The replay protocol (SPEC.md §6). **FROZEN after Phase 2** — see `cfe/eval/__init__.py`.

Per SPEC.md I2 this module is the *only* source of ground truth in the project. Every reported
success number comes from executing actions here; latent goal distance is a search objective and
never appears in this file.

**Primary variant**, §6 verbatim:

1. reset env to the exact original init state (I1)
2. execute original actions up to window start
3. execute the *edited* actions through the window
4. hand control to the BC policy closed-loop for the remainder
5. record task success

**Secondary variant** is the same without step 4.

Three decisions are baked in here rather than left to callers, because each of them moves the
reported number and a knob that moves the number must not be settable per-run:

- **`max_steps_total` is counted from the episode's first step**, so an edited episode gets
  exactly the horizon the original had and a late edit cannot buy extra time. CLAUDE.md is
  explicit that this knob moves the success rate as surely as retraining does. Default 300,
  matching the `eval.max_steps` that produced the measured 72% BC rate, so RSR stays comparable
  to it.
- **Termination is `done or env.check_success()`**, matching what `cfe/policy/bc.py:rollout`
  did when the 72% was measured. A different stopping rule would silently make the two
  incomparable.
- **This module does not import `cfe/policy/bc.py`.** That file is not frozen; if the protocol
  imported it, a later edit there would change frozen replay behaviour and every number derived
  from it without touching a frozen file. The policy arrives as a plain callable instead, and
  the ~30 lines of stepping duplicated from `rollout` are the price of the freeze meaning
  something.

`ReplayResult` deliberately carries no failure-mode classification and no object displacements.
Both need object-key heuristics and a contact threshold that live in `cfe/policy/bc.py`, and
copying a threshold into a frozen file means the two can disagree silently forever. Callers that
want them pass `obs_hook` and accumulate whatever they like.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

ACTION_DIM = 7

#: Total env steps allowed per replayed episode, counted from the episode's first step.
#: Matches `eval.max_steps` from the run that measured the 72% BC success rate.
DEFAULT_MAX_STEPS_TOTAL = 300

PRIMARY = "primary"
OPEN_LOOP = "open_loop"


@dataclass(frozen=True, eq=False)
class Edit:
    """An edit to an episode's actions: replace `actions` starting at `window_start`.

    The edit budget `K` is the window length, enforced *structurally* — there is nowhere to put
    a longer edit — which is what SPEC.md §5 means by "enforced structurally by the window, not
    by a penalty".

    `actions` is **copied and made read-only** on construction. `np.asarray` returns a view when
    handed a float64 array, so without the copy `identity_edit` would return a window into the
    caller's episode and mutating that episode would silently change an edit that had already
    been replayed. "Frozen" has to mean the contents too, or a replayed result stops
    corresponding to the edit it is filed under.

    `eq=False` with an explicit `__eq__` because the generated one compares numpy arrays inside
    a tuple, which raises `ValueError: truth value of an array is ambiguous` rather than
    answering. Defining `__eq__` also makes `Edit` unhashable, which is honest — it holds an
    array.
    """

    window_start: int
    actions: np.ndarray  # (W, 7)

    def __post_init__(self) -> None:
        a = np.array(self.actions, dtype=np.float64)  # always a copy; see the class docstring
        if a.ndim != 2 or a.shape[1] != ACTION_DIM:
            raise ValueError(f"edit actions must be (W, {ACTION_DIM}), got {a.shape}")
        if self.window_start < 0:
            raise ValueError(f"window_start must be >= 0, got {self.window_start}")
        a.flags.writeable = False
        object.__setattr__(self, "actions", a)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edit):
            return NotImplemented
        return self.window_start == other.window_start and np.array_equal(
            self.actions, other.actions
        )

    @property
    def budget_k(self) -> int:
        """Number of edited timesteps. `K` in SPEC.md §6's RSR-vs-budget curve."""
        return int(self.actions.shape[0])

    @property
    def window_end(self) -> int:
        return self.window_start + self.budget_k


@dataclass(frozen=True)
class ReplayResult:
    """What the simulator did. `success` is the only ground truth in the project (I2)."""

    success: bool
    variant: str
    window_start: int
    window_end: int
    steps_executed: int
    scripted_steps: int
    policy_steps: int
    timed_out: bool
    max_steps_total: int
    #: True when the window ended at or past the step budget, so step 4 got zero steps. Recorded
    #: rather than clamped: a "primary" replay that never ran the policy is not the primary
    #: variant, and silently reporting it as one would misattribute the result.
    policy_never_ran: bool


def identity_edit(original_actions: np.ndarray, window_start: int, budget_k: int) -> Edit:
    """The `no_edit` baseline's edit: the original actions, unchanged, over the same window.

    Exists so `no_edit` traverses byte-identical machinery to a real edit. A `no_edit` that
    took a shortcut would not catch protocol bugs, which is the only reason §6 requires it.
    """
    original = np.asarray(original_actions, dtype=np.float64)
    _validate_window(original, window_start, budget_k)
    return Edit(window_start=window_start, actions=original[window_start : window_start + budget_k])


def _validate_window(original: np.ndarray, window_start: int, budget_k: int) -> None:
    if original.ndim != 2 or original.shape[1] != ACTION_DIM:
        raise ValueError(f"original actions must be (T, {ACTION_DIM}), got {original.shape}")
    if window_start + budget_k > len(original):
        raise ValueError(
            f"window [{window_start}, {window_start + budget_k}) runs past the episode's "
            f"{len(original)} actions; the edit would have no original actions to replace"
        )


def _run(
    env,
    init_state: np.ndarray,
    original_actions: np.ndarray,
    edit: Edit,
    *,
    variant: str,
    policy_fn: Callable[[dict], np.ndarray] | None,
    max_steps_total: int,
    obs_hook: Callable[[dict], None] | None,
) -> ReplayResult:
    """The one stepping core. Both variants go through it; they differ only after the window.

    Sharing it is not tidiness. `no_edit` is §6's control for replay bugs, and a control that
    runs different code from the thing it controls detects nothing.
    """
    original = np.asarray(original_actions, dtype=np.float64)
    _validate_window(original, edit.window_start, edit.budget_k)
    if variant not in (PRIMARY, OPEN_LOOP):
        raise ValueError(f"unknown variant {variant!r}; expected {PRIMARY!r} or {OPEN_LOOP!r}")
    if variant == PRIMARY and policy_fn is None:
        raise ValueError("the primary variant hands control to the policy; policy_fn is required")

    # Step 1 — exact state reset (I1). A silent failure here invalidates every number, so an
    # env that will not materialise observations is an error, never a warning.
    env.reset()
    obs = env.set_init_state(init_state)
    if obs is None:
        raise RuntimeError("set_init_state returned None; cannot replay from the exact state")
    if obs_hook is not None:
        obs_hook(obs)

    # Steps 2-3, plus the tail for the open-loop variant. §6's "edited actions to the end" is
    # read as *the edited action sequence* -- original before the window, edited within it,
    # original after -- executed with no policy resume. The other reading, extending the edit
    # itself to episode end, is not an edit of bounded budget at all, so it cannot be what a
    # budget-swept metric means. Recorded in PROGRESS.md as an interpretation.
    pieces = [original[: edit.window_start], edit.actions]
    if variant == OPEN_LOOP:
        pieces.append(original[edit.window_end :])
    scripted = np.concatenate(pieces, axis=0)[:max_steps_total]

    steps = 0
    done = False
    for action in scripted:
        obs, _reward, done, _info = env.step(np.asarray(action, dtype=np.float64).copy())
        steps += 1
        if obs_hook is not None:
            obs_hook(obs)
        if done or env.check_success():
            break
    scripted_steps = steps

    # Step 4 — closed-loop policy for whatever budget the original episode had left.
    policy_steps = 0
    policy_never_ran = variant == PRIMARY and edit.window_end >= max_steps_total
    if variant == PRIMARY and not done and not env.check_success():
        while steps < max_steps_total:
            proposed = np.atleast_2d(np.asarray(policy_fn(obs), dtype=np.float64))
            if proposed.shape[1] != ACTION_DIM:
                raise ValueError(
                    f"policy_fn returned shape {proposed.shape}; expected (7,) or (C, 7)"
                )
            for action in proposed:
                if steps >= max_steps_total:
                    break
                obs, _reward, done, _info = env.step(action.copy())
                steps += 1
                policy_steps += 1
                if obs_hook is not None:
                    obs_hook(obs)
                if done or env.check_success():
                    break
            if done or env.check_success():
                break

    # Step 5 — the simulator's verdict, and the only one that counts.
    return ReplayResult(
        success=bool(env.check_success()),
        variant=variant,
        window_start=edit.window_start,
        window_end=edit.window_end,
        steps_executed=steps,
        scripted_steps=scripted_steps,
        policy_steps=policy_steps,
        timed_out=steps >= max_steps_total,
        max_steps_total=max_steps_total,
        policy_never_ran=policy_never_ran,
    )


def replay_primary(
    env,
    init_state: np.ndarray,
    original_actions: np.ndarray,
    edit: Edit,
    policy_fn: Callable[[dict], np.ndarray],
    *,
    max_steps_total: int = DEFAULT_MAX_STEPS_TOTAL,
    obs_hook: Callable[[dict], None] | None = None,
) -> ReplayResult:
    """SPEC.md §6's primary variant. This is what "would this edit have rescued it" means.

    `policy_fn` maps an observation to either one action `(7,)` or an action chunk `(C, 7)`;
    chunked policies are the normal case here and the protocol stays agnostic to which.
    """
    return _run(
        env,
        init_state,
        original_actions,
        edit,
        variant=PRIMARY,
        policy_fn=policy_fn,
        max_steps_total=max_steps_total,
        obs_hook=obs_hook,
    )


def replay_open_loop(
    env,
    init_state: np.ndarray,
    original_actions: np.ndarray,
    edit: Edit,
    *,
    max_steps_total: int = DEFAULT_MAX_STEPS_TOTAL,
    obs_hook: Callable[[dict], None] | None = None,
) -> ReplayResult:
    """§6's secondary variant: no policy resume. Logged alongside, never instead."""
    return _run(
        env,
        init_state,
        original_actions,
        edit,
        variant=OPEN_LOOP,
        policy_fn=None,
        max_steps_total=max_steps_total,
        obs_hook=obs_hook,
    )
