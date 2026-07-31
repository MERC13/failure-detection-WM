"""Demo replay fidelity (SPEC.md §8) and BC data assembly.

SPEC.md §8 requires that "replay of unedited actions reproduces the original episode's success
flag exactly". Measured on 2026-07-30, that holds — but the recorded *state trajectory* does
not survive replay (drift > 1e-3 by the first step, peaking around 4.5-10.6). These tests pin
both facts so a later change cannot quietly start trusting recorded states.
"""

import numpy as np
import pytest

from cfe.policy.bc import (
    CONTACT_EPS,
    GRIPPER_DIM,
    Normalizer,
    PolicyConfig,
    build_arrays,
    classify_failure,
    find_target_object_key,
    object_positions,
    to_action,
)

# --- failure-mode breakdown (SPEC.md §4.3, amended 2026-07-31) --------------


def test_object_positions_ignores_the_robot_and_non_position_keys():
    """The breakdown has to work on every task, so it cannot key on one task's object names."""
    obs = {
        "alphabet_soup_1_pos": np.zeros(3),
        "basket_1_pos": np.ones(3),
        "robot0_eef_pos": np.zeros(3),  # the robot is not an object
        "alphabet_soup_1_quat": np.zeros(4),  # not a position
        "object-state": np.zeros(98),
    }
    assert sorted(object_positions(obs)) == ["alphabet_soup_1_pos", "basket_1_pos"]


def test_object_positions_excludes_end_effector_relative_keys():
    """Regression: `<obj>_to_robot0_eef_pos` is object-minus-eef, so it moves when the *arm*
    moves. Including it reported "object disturbed" for all 14 Phase 1 failures, in which the
    object provably never moved."""
    obs = {
        "alphabet_soup_1_pos": np.zeros(3),
        "alphabet_soup_1_to_robot0_eef_pos": np.ones(3),
        "salad_dressing_1_to_robot0_eef_pos": np.ones(3),
    }
    assert sorted(object_positions(obs)) == ["alphabet_soup_1_pos"]


def test_contact_threshold_clears_measured_reset_settle():
    """Objects settle at reset: 0.1062 m vertically but only 0.0010 m horizontally over 120
    idle steps. The threshold is horizontal and must sit well above that noise floor."""
    idle_horizontal_drift = 0.0010
    carried_to_basket = 0.5
    assert CONTACT_EPS > 5 * idle_horizontal_drift
    assert CONTACT_EPS < carried_to_basket / 10


def test_untouched_objects_classify_as_no_contact():
    """Nothing moved at all: the gripper never reached anything."""
    assert classify_failure(0.0, 0.0, timed_out=True) == "no_object_contact"
    assert classify_failure(CONTACT_EPS / 2, CONTACT_EPS / 2, timed_out=True) == "no_object_contact"


def test_nudging_a_distractor_is_not_touching_the_target():
    """Regression: 11 of the 14 Phase 1 failures moved *a distractor* by 1-10 cm while the
    target provably never moved. Reporting those as "an object moved" hid the fact that every
    failure is an approach failure."""
    assert (
        classify_failure(0.0, CONTACT_EPS * 10, timed_out=True)
        == "distractor_only_target_untouched"
    )


def test_moving_the_target_is_a_different_failure_mode():
    """A grasp-and-drop needs a different Phase 3 edit than a missed approach."""
    assert classify_failure(CONTACT_EPS * 10, 0.0, timed_out=True) == "target_moved_not_placed"
    assert (
        classify_failure(CONTACT_EPS * 10, 0.0, timed_out=False) == "target_moved_then_terminated"
    )


def test_target_object_comes_from_the_task_language():
    keys = [
        "alphabet_soup_1_pos",
        "basket_1_pos",
        "butter_1_pos",
        "milk_1_pos",
        "robot0_eef_pos",
        "alphabet_soup_1_to_robot0_eef_pos",
    ]
    lang = "pick up the alphabet soup and place it in the basket"
    assert find_target_object_key(keys, lang) == "alphabet_soup_1_pos"


def test_an_unidentifiable_target_raises_rather_than_guessing():
    """A silently wrong target turns the whole breakdown into noise."""
    keys = ["butter_1_pos", "milk_1_pos", "basket_1_pos"]
    with pytest.raises(ValueError, match="unique target object"):
        find_target_object_key(keys, "pick up the alphabet soup and place it in the basket")


# --- BC data assembly (fast: no sim, no model) ------------------------------


class _StubCache:
    """Minimal stand-in for EncodingCache with hand-built arrays."""

    def __init__(self, episodes):
        self._eps = episodes

    def read(self, eid, frame_idx=None):
        arr = self._eps[eid]["tokens"]
        return arr if frame_idx is None else arr[frame_idx]

    def read_meta(self, eid):
        return {k: v for k, v in self._eps[eid].items() if k != "tokens"}


def _episode(n_actions, n_tokens=4, dim=3, proprio_dim=9, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "tokens": rng.standard_normal((n_actions + 1, n_tokens, dim)).astype(np.float16),
        "actions": rng.standard_normal((n_actions, 7)).astype(np.float32),
        "proprio": rng.standard_normal((n_actions + 1, proprio_dim)).astype(np.float32),
    }


def _cfg(chunk=4, use_proprio=True):
    return PolicyConfig(
        chunk_size=chunk,
        hidden_sizes=(8,),
        dropout=0.0,
        include_cls_in_pool=False,
        use_proprio=use_proprio,
    )


def test_build_arrays_shapes_and_proprio_concat():
    cache = _StubCache({"demo_0": _episode(10)})
    d = build_arrays(cache, ["demo_0"], _cfg(chunk=4))
    assert d["x"].shape == (10, 3 + 9)  # pooled dim + proprio dim
    assert d["delta"].shape == (10, 4, 6)
    assert d["grip"].shape == (10, 4)
    assert set(np.unique(d["grip"])) <= {0.0, 1.0}


def test_build_arrays_without_proprio():
    cache = _StubCache({"demo_0": _episode(6)})
    assert build_arrays(cache, ["demo_0"], _cfg(use_proprio=False))["x"].shape == (6, 3)


def test_chunks_past_the_episode_end_repeat_the_final_action():
    ep = _episode(5)
    ep["actions"] = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    d = build_arrays(_StubCache({"demo_0": ep}), ["demo_0"], _cfg(chunk=3))
    # Last sample starts at t=4 with only one action left; the rest repeat action 4.
    last = d["delta"][-1]
    assert np.allclose(last[0], ep["actions"][4, :6])
    assert np.allclose(last[1], ep["actions"][4, :6])
    assert np.allclose(last[2], ep["actions"][4, :6])


def test_episodes_are_kept_separate():
    cache = _StubCache({"demo_0": _episode(4, seed=1), "demo_1": _episode(6, seed=2)})
    d = build_arrays(cache, ["demo_0", "demo_1"], _cfg())
    assert d["x"].shape[0] == 10
    assert (d["episode"] == 0).sum() == 4
    assert (d["episode"] == 1).sum() == 6


def test_gripper_target_is_binary_from_sign():
    ep = _episode(3)
    ep["actions"][:, GRIPPER_DIM] = [-1.0, 1.0, -1.0]
    d = build_arrays(_StubCache({"demo_0": ep}), ["demo_0"], _cfg(chunk=1))
    assert d["grip"][:, 0].tolist() == [0.0, 1.0, 0.0]


# --- action decoding --------------------------------------------------------


def test_to_action_binarises_the_gripper_and_clips_deltas():
    norm = Normalizer(x_mean=np.zeros(3), x_std=np.ones(3), d_mean=np.zeros(6), d_std=np.ones(6))
    out = np.zeros((2, 7), dtype=np.float32)
    out[0, :6] = 5.0  # far outside the env's [-1, 1] range
    out[0, GRIPPER_DIM] = 2.0  # positive logit -> closed
    out[1, GRIPPER_DIM] = -2.0  # negative logit -> open
    actions = to_action(out, norm)
    assert np.allclose(actions[0, :6], 1.0), "deltas must be clipped to [-1, 1]"
    assert actions[0, GRIPPER_DIM] == 1.0
    assert actions[1, GRIPPER_DIM] == -1.0
    assert set(np.unique(actions[:, GRIPPER_DIM])) <= {-1.0, 1.0}


def test_normalizer_round_trips_deltas():
    rng = np.random.default_rng(0)
    deltas = rng.standard_normal((20, 4, 6)).astype(np.float32) * [0.2, 0.5, 0.5, 0.02, 0.05, 0.04]
    norm = Normalizer.fit(rng.standard_normal((20, 12)).astype(np.float32), deltas)
    assert np.allclose(norm.denorm_delta(norm.norm_delta(deltas)), deltas, atol=1e-5)
    # Rotation dims are ~20x smaller than translation; normalisation must equalise them.
    assert norm.norm_delta(deltas).reshape(-1, 6).std(0).min() > 0.5


def test_normalizer_survives_a_constant_feature():
    x = np.ones((10, 3), dtype=np.float32)
    norm = Normalizer.fit(x, np.zeros((10, 2, 6), dtype=np.float32))
    assert np.all(np.isfinite(norm.norm_x(x)))


# --- the simulator (slow) ---------------------------------------------------


@pytest.mark.slow
def test_replay_of_unedited_actions_reproduces_the_success_flag(cfg, task_and_init_states, seed):
    """SPEC.md §8. The demos are expert *action* sequences; replay must still reach the goal."""
    import h5py

    from cfe.data.demos import demo_file_path, episode_ids, make_replay_env, replay_episode

    task, _ = task_and_init_states
    path = demo_file_path(cfg["dataset_dir"], cfg["suite"], task)
    ids = episode_ids(path)[:3]

    env = make_replay_env(cfg, task, seed)
    try:
        for eid in ids:
            with h5py.File(path, "r") as f:
                d = f["data"][eid]
                actions, state0 = d["actions"][()], d["states"][0]
            out = replay_episode(env, actions, state0)
            assert out["success"], f"{eid}: expert replay did not reach the goal"
            assert len(out["frames"]) == len(actions) + 1
            assert out["proprio"].shape == (len(actions) + 1, 9)
    finally:
        env.close()


@pytest.mark.slow
def test_recorded_states_are_not_reproduced_by_replay(cfg, task_and_init_states, seed):
    """Pins the finding that recorded `states[t]` must not be used as ground truth.

    If this ever starts passing bit-exactly, the physics changed and PROGRESS.md's guidance
    (and anything in Phase 2/3 built on it) needs revisiting — so it fails loudly either way.
    """
    import h5py

    from cfe.data.demos import demo_file_path, episode_ids, make_replay_env
    from scripts.verify_env import sim_state

    task, _ = task_and_init_states
    path = demo_file_path(cfg["dataset_dir"], cfg["suite"], task)
    eid = episode_ids(path)[0]
    with h5py.File(path, "r") as f:
        d = f["data"][eid]
        actions, states = d["actions"][()], d["states"][()]

    env = make_replay_env(cfg, task, seed)
    try:
        env.reset()
        env.set_init_state(states[0])
        # The reset itself IS bit-exact — that is invariant I1 and must not regress.
        assert np.array_equal(sim_state(env), states[0])
        drift = []
        for t in range(min(20, len(actions) - 1)):
            env.step(actions[t].copy())
            drift.append(float(np.abs(sim_state(env) - states[t + 1]).max()))
    finally:
        env.close()

    assert max(drift) > 1e-3, (
        "recorded states now reproduce under replay; PROGRESS.md's finding is stale and "
        "Phase 2/3 assumptions need revisiting"
    )
