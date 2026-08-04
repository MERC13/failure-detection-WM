"""Tests for the frozen replay protocol (SPEC.md §6, §8).

The fast tests drive a fake env so the protocol's *logic* — step accounting, which actions run
in which order, where the policy takes over — is checked exactly and without the 9 steps/s
simulator. The slow tests then check the one thing a fake env cannot: that replaying unedited
actions through this protocol reproduces the real episode's recorded success flag.
"""

import numpy as np
import pytest

from cfe.eval.replay_protocol import (
    DEFAULT_MAX_STEPS_TOTAL,
    Edit,
    identity_edit,
    replay_open_loop,
    replay_primary,
)


class FakeEnv:
    """Records every action it is stepped with. Succeeds once `succeed_after` steps have run."""

    def __init__(self, succeed_after: int | None = None, init_obs_none: bool = False):
        self.executed: list[np.ndarray] = []
        self.resets = 0
        self.init_state = None
        self.succeed_after = succeed_after
        self.init_obs_none = init_obs_none

    def reset(self):
        self.resets += 1
        self.executed = []

    def set_init_state(self, state):
        self.init_state = np.asarray(state).copy()
        return None if self.init_obs_none else self._obs()

    def _obs(self):
        return {"step": len(self.executed)}

    def step(self, action):
        self.executed.append(np.asarray(action, dtype=np.float64).copy())
        return self._obs(), 0.0, False, {}

    def check_success(self):
        return self.succeed_after is not None and len(self.executed) >= self.succeed_after


def _episode(t: int = 20) -> np.ndarray:
    """Distinct, identifiable actions so a misordered replay is visible, not plausible."""
    return np.arange(t * 7, dtype=np.float64).reshape(t, 7)


def _constant_policy(value: float = -1.0):
    return lambda _obs: np.full(7, value)


# --- I1: exact reset --------------------------------------------------------


def test_replay_resets_and_sets_the_exact_init_state():
    env = FakeEnv()
    original = _episode()
    init = np.array([1.5, 2.5, 3.5])
    replay_open_loop(env, init, original, identity_edit(original, 0, 4), max_steps_total=20)
    assert env.resets == 1
    assert np.array_equal(env.init_state, init)


def test_a_none_observation_from_set_init_state_is_an_error_not_a_warning():
    """I1 silently failing invalidates every number downstream, so it must raise."""
    env = FakeEnv(init_obs_none=True)
    original = _episode()
    with pytest.raises(RuntimeError, match="set_init_state"):
        replay_open_loop(env, np.zeros(3), original, identity_edit(original, 0, 4))


# --- the two variants -------------------------------------------------------


def test_open_loop_with_an_identity_edit_executes_exactly_the_original_actions():
    """The `no_edit` control, at the level the fake env can check exactly."""
    env = FakeEnv()
    original = _episode(12)
    res = replay_open_loop(
        env,
        np.zeros(3),
        original,
        identity_edit(original, 4, 4),
        max_steps_total=DEFAULT_MAX_STEPS_TOTAL,
    )
    assert np.array_equal(np.stack(env.executed), original)
    assert res.policy_steps == 0
    assert res.variant == "open_loop"


def test_primary_runs_original_then_edit_then_hands_over_to_the_policy():
    env = FakeEnv()
    original = _episode(12)
    edit = Edit(window_start=4, actions=np.full((4, 7), 99.0))
    res = replay_primary(
        env, np.zeros(3), original, edit, _constant_policy(-1.0), max_steps_total=20
    )

    executed = np.stack(env.executed)
    assert np.array_equal(executed[:4], original[:4])  # step 2
    assert np.array_equal(executed[4:8], edit.actions)  # step 3
    assert np.all(executed[8:] == -1.0)  # step 4
    assert res.scripted_steps == 8
    assert res.policy_steps == 12
    assert res.steps_executed == 20
    assert res.timed_out is True


def test_primary_does_not_replay_the_original_tail_the_way_open_loop_does():
    """The variants must actually differ after the window, or the secondary is not secondary."""
    original = _episode(12)
    edit = Edit(window_start=4, actions=np.full((4, 7), 99.0))

    primary_env, open_env = FakeEnv(), FakeEnv()
    replay_primary(primary_env, np.zeros(3), original, edit, _constant_policy(), max_steps_total=12)
    replay_open_loop(open_env, np.zeros(3), original, edit, max_steps_total=12)

    assert np.array_equal(np.stack(open_env.executed)[8:], original[8:])
    assert np.all(np.stack(primary_env.executed)[8:] == -1.0)


# --- step budget: counted from episode start --------------------------------


def test_the_policy_budget_is_what_the_original_episode_had_left():
    """A late edit must not buy extra time: budget is `max_steps_total - window_end`."""
    for window_start, expected_policy_steps in ((0, 16), (8, 8), (12, 4)):
        env = FakeEnv()
        original = _episode(20)
        edit = identity_edit(original, window_start, 4)
        res = replay_primary(
            env, np.zeros(3), original, edit, _constant_policy(), max_steps_total=20
        )
        assert res.policy_steps == expected_policy_steps
        assert res.steps_executed == 20


def test_a_window_at_the_budget_records_that_the_policy_never_ran():
    """Recorded rather than clamped: a primary replay with zero policy steps is not the
    primary variant and must not be reported as one."""
    env = FakeEnv()
    original = _episode(20)
    res = replay_primary(
        env,
        np.zeros(3),
        original,
        identity_edit(original, 16, 4),
        _constant_policy(),
        max_steps_total=20,
    )
    assert res.policy_never_ran is True
    assert res.policy_steps == 0


# --- termination ------------------------------------------------------------


def test_replay_stops_as_soon_as_the_simulator_reports_success():
    env = FakeEnv(succeed_after=6)
    original = _episode(20)
    res = replay_primary(
        env,
        np.zeros(3),
        original,
        identity_edit(original, 0, 4),
        _constant_policy(),
        max_steps_total=20,
    )
    assert res.success is True
    assert res.steps_executed == 6
    assert res.timed_out is False


def test_success_comes_from_the_simulator_not_from_the_step_count():
    env = FakeEnv(succeed_after=None)
    original = _episode(20)
    res = replay_primary(
        env,
        np.zeros(3),
        original,
        identity_edit(original, 0, 4),
        _constant_policy(),
        max_steps_total=20,
    )
    assert res.success is False


# --- policy interface -------------------------------------------------------


def test_policy_may_return_a_single_action_or_a_chunk():
    original = _episode(12)
    edit = identity_edit(original, 0, 4)

    single, chunked = FakeEnv(), FakeEnv()
    r1 = replay_primary(
        single, np.zeros(3), original, edit, lambda _o: np.full(7, -1.0), max_steps_total=12
    )
    r2 = replay_primary(
        chunked, np.zeros(3), original, edit, lambda _o: np.full((3, 7), -1.0), max_steps_total=12
    )
    assert r1.steps_executed == r2.steps_executed == 12
    assert np.array_equal(np.stack(single.executed), np.stack(chunked.executed))


def test_a_chunked_policy_does_not_overrun_the_budget():
    """The chunk that straddles the cap must be truncated, not completed."""
    env = FakeEnv()
    original = _episode(12)
    res = replay_primary(
        env,
        np.zeros(3),
        original,
        identity_edit(original, 0, 4),
        lambda _o: np.full((5, 7), -1.0),
        max_steps_total=10,
    )
    assert res.steps_executed == 10
    assert len(env.executed) == 10


def test_a_misshapen_policy_output_is_rejected():
    env = FakeEnv()
    original = _episode(12)
    with pytest.raises(ValueError, match="expected"):
        replay_primary(
            env,
            np.zeros(3),
            original,
            identity_edit(original, 0, 4),
            lambda _o: np.zeros(5),
            max_steps_total=12,
        )


def test_primary_requires_a_policy():
    from cfe.eval.replay_protocol import PRIMARY, _run

    env = FakeEnv()
    original = _episode(12)
    with pytest.raises(ValueError, match="policy_fn is required"):
        _run(
            env,
            np.zeros(3),
            original,
            identity_edit(original, 0, 4),
            variant=PRIMARY,
            policy_fn=None,
            max_steps_total=12,
            obs_hook=None,
        )


# --- validation -------------------------------------------------------------


def test_edit_rejects_a_wrong_action_width():
    with pytest.raises(ValueError, match=r"\(W, 7\)"):
        Edit(window_start=0, actions=np.zeros((4, 6)))


def test_edit_rejects_a_negative_window_start():
    with pytest.raises(ValueError, match="window_start"):
        Edit(window_start=-1, actions=np.zeros((4, 7)))


def test_a_window_past_the_end_of_the_episode_is_rejected():
    original = _episode(10)
    with pytest.raises(ValueError, match="runs past"):
        identity_edit(original, 8, 4)


def test_budget_k_and_window_end_follow_from_the_edit():
    edit = Edit(window_start=5, actions=np.zeros((8, 7)))
    assert edit.budget_k == 8
    assert edit.window_end == 13


def test_obs_hook_sees_the_initial_observation_and_every_step():
    env = FakeEnv()
    original = _episode(12)
    seen: list[dict] = []
    replay_open_loop(
        env,
        np.zeros(3),
        original,
        identity_edit(original, 0, 4),
        max_steps_total=12,
        obs_hook=seen.append,
    )
    assert len(seen) == 13  # T+1: the reset observation plus one per step


# --- I5: determinism --------------------------------------------------------


def test_replay_is_deterministic_given_the_same_inputs():
    original = _episode(20)
    edit = identity_edit(original, 4, 4)
    results = [
        replay_primary(
            FakeEnv(), np.zeros(3), original, edit, _constant_policy(), max_steps_total=20
        )
        for _ in range(2)
    ]
    assert results[0] == results[1]


# --- the simulator (slow) ---------------------------------------------------


@pytest.mark.slow
def test_no_edit_agrees_with_the_phase_1_replay_path(cfg, seed, task_and_init_states):
    """SPEC.md §8 and §6's `no_edit` baseline, against the real simulator.

    Pins this frozen protocol against `cfe/data/demos.py:replay_episode`, which Phase 1 already
    established and which G1 depends on. An identity edit replayed open-loop *is* that replay,
    so the two must agree on the success flag; if they diverge, one of them is wrong and every
    downstream number is meaningless. That is a hard stop, not a tolerance to widen.

    Note the two differ deliberately in one respect: this protocol stops at the first
    `check_success()`, while `replay_episode` runs every action. They agree here because
    LIBERO's success is a terminal-state predicate on these episodes. A divergence would mean
    success is transient on this task, which Phase 3's search would need to know about.
    """
    from cfe.data.demos import (
        demo_file_path,
        episode_ids,
        make_replay_env,
        read_episode,
        replay_episode,
    )

    task, _ = task_and_init_states
    path = demo_file_path(cfg["dataset_dir"], cfg["suite"], task)
    ids = episode_ids(path)[:3]

    env = make_replay_env(cfg, task, seed)
    try:
        for eid in ids:
            ep = read_episode(path, eid)
            actions = np.asarray(ep["actions"], dtype=np.float64)
            init_state = ep["states"][0]

            trusted = replay_episode(env, actions, init_state)
            res = replay_open_loop(
                env,
                init_state,
                actions,
                identity_edit(actions, 0, len(actions)),
                max_steps_total=len(actions),
            )

            assert res.success == trusted["success"], (
                f"{eid}: no_edit through cfe/eval gave success={res.success}, but the Phase 1 "
                f"replay path gave success={trusted['success']} on the same actions and state"
            )
            assert trusted["success"], f"{eid}: expert demo replay did not reach the goal"
    finally:
        env.close()
