"""Injected failure causes (SPEC.md §4.4).

Every injection carries a ground-truth `cause` and edit window that later become the answer key
for SPEC.md §6's cause-identification metrics. If an injection changes steps its recorded window
does not name, or moves the sequence length so the expert oracle misaligns, the resulting
numbers are confidently wrong in a way nothing downstream would flag. These check the
properties the rest of the pipeline assumes.
"""

import numpy as np
import pytest

from cfe.data.inject import (
    ACTION_DIM,
    CAUSES,
    GRIPPER_DIM,
    Injection,
    early_release,
    grasp_offset,
    gripper_phases,
    metres_per_unit_action,
    mistimed_close,
    realised_offset_from_proprio,
    wrong_target,
)

CLOSE, RELEASE, T = 40, 120, 150
SCALE = np.array([0.01, 0.013, 0.012])


def _demo(close: int = CLOSE, release: int = RELEASE, n: int = T) -> np.ndarray:
    """A synthetic demo with one clean close/release pair, like 44 of the 48 real ones."""
    rng = np.random.default_rng(0)
    a = np.zeros((n, ACTION_DIM))
    a[:, :6] = rng.uniform(-0.2, 0.2, size=(n, 6))
    a[:, GRIPPER_DIM] = -1.0
    a[close:release, GRIPPER_DIM] = 1.0
    return a


def _regrasp_demo() -> np.ndarray:
    """The 4-in-48 shape: a fumble and a regrasp before the real place-and-release."""
    a = _demo()
    a[60:70, GRIPPER_DIM] = -1.0
    return a


# --- phase detection --------------------------------------------------------


def test_phases_on_a_clean_demo():
    p = gripper_phases(_demo())
    assert (p.close_idx, p.release_idx) == (CLOSE, RELEASE)
    assert not p.regrasped


def test_regrasp_release_is_the_last_one_not_the_first():
    """Anchoring early_release at the fumble would corrupt the wrong phase of the episode."""
    p = gripper_phases(_regrasp_demo())
    assert p.close_idx == CLOSE
    assert p.release_idx == RELEASE
    assert p.regrasped


def test_a_demo_that_never_closes_is_refused_not_guessed():
    a = _demo()
    a[:, GRIPPER_DIM] = -1.0
    with pytest.raises(ValueError, match="at least one close"):
        gripper_phases(a)


def test_a_continuous_gripper_is_refused():
    a = _demo()
    a[10, GRIPPER_DIM] = 0.3
    with pytest.raises(ValueError, match="strictly -1 or"):
        gripper_phases(a)


# --- calibration ------------------------------------------------------------


def test_calibration_recovers_a_known_scale():
    """Construct travel from actions at a known ratio; the fit must return that ratio."""
    rng = np.random.default_rng(1)
    a = np.zeros((200, ACTION_DIM))
    a[:, :3] = rng.uniform(-0.5, 0.5, size=(200, 3))
    eef = np.zeros((201, 3))
    eef[1:] = np.cumsum(a[:, :3] * SCALE, axis=0)

    out = metres_per_unit_action(a, eef)
    assert out["m_per_unit"] == pytest.approx(SCALE, rel=1e-9)
    assert out["corr"] == pytest.approx(np.ones(3), abs=1e-9)


def test_calibration_rejects_a_length_mismatch():
    a = _demo()
    with pytest.raises(ValueError, match="expected T\\+1 for T"):
        metres_per_unit_action(a, np.zeros((len(a), 3)))


# --- the four causes --------------------------------------------------------


@pytest.mark.parametrize("cause", CAUSES)
def test_every_injection_preserves_length_and_the_gripper_alphabet(cause):
    """Length preservation is what keeps `expert_oracle` aligned to the same indices."""
    a = _demo()
    p = gripper_phases(a)
    inj = _make(cause, a, p)
    assert inj.actions.shape == a.shape
    assert set(np.unique(inj.actions[:, GRIPPER_DIM])) <= {-1.0, 1.0}
    assert inj.cause == cause


@pytest.mark.parametrize("cause", CAUSES)
def test_the_recorded_window_is_exactly_the_steps_that_changed(cause):
    a = _demo()
    p = gripper_phases(a)
    inj = _make(cause, a, p)
    changed = np.flatnonzero(np.any(inj.actions != a, axis=1))
    assert changed.size > 0
    assert (int(changed[0]), int(changed[-1]) + 1) == inj.window


def test_grasp_offset_lands_the_requested_displacement():
    """δ is in metres, so the summed action delta times the calibration must reproduce it."""
    a = _demo()
    p = gripper_phases(a)
    delta = np.array([0.03, -0.02])
    inj = grasp_offset(a, p, delta, window_steps=20, m_per_unit=SCALE)

    realised = (inj.actions[:, :2] - a[:, :2]).sum(axis=0) * SCALE[:2]
    assert realised == pytest.approx(delta, rel=1e-9)
    assert inj.params["realised_delta_m"] == pytest.approx(delta.tolist(), rel=1e-9)
    assert not inj.params["clipped"]


def test_grasp_offset_ends_at_the_grasp_and_is_not_compensated_afterwards():
    a = _demo()
    p = gripper_phases(a)
    inj = grasp_offset(a, p, np.array([0.03, 0.0]), window_steps=20, m_per_unit=SCALE)
    assert inj.window == (CLOSE - 20, CLOSE)
    assert np.array_equal(inj.actions[CLOSE:], a[CLOSE:])


def test_clipping_is_reported_rather_than_silently_shrinking_the_offset():
    a = _demo()
    a[:, 0] = 0.99  # already at the bound
    p = gripper_phases(a)
    inj = grasp_offset(a, p, np.array([0.5, 0.0]), window_steps=20, m_per_unit=SCALE)
    assert inj.params["clipped"] or inj.params["realised_delta_m"][0] < 0.5


def test_wrong_target_points_at_the_distractor():
    a = _demo()
    p = gripper_phases(a)
    inj = wrong_target(
        a,
        p,
        target_xy=np.array([0.0, 0.0]),
        distractor_xy=np.array([0.15, -0.10]),
        fraction=1.0,
        window_steps=20,
        m_per_unit=SCALE,
    )
    realised = (inj.actions[:, :2] - a[:, :2]).sum(axis=0) * SCALE[:2]
    assert realised == pytest.approx([0.15, -0.10], rel=1e-9)
    assert inj.params["separation_m"] == pytest.approx(np.hypot(0.15, 0.10))


def test_mistimed_close_early_shuts_the_gripper_sooner():
    a = _demo()
    p = gripper_phases(a)
    inj = mistimed_close(a, p, shift=-8)
    assert inj.window == (CLOSE - 8, CLOSE)
    assert np.all(inj.actions[CLOSE - 8 : CLOSE, GRIPPER_DIM] == 1.0)
    assert gripper_phases(inj.actions).close_idx == CLOSE - 8


def test_mistimed_close_late_delays_the_grasp():
    a = _demo()
    p = gripper_phases(a)
    inj = mistimed_close(a, p, shift=8)
    assert inj.window == (CLOSE, CLOSE + 8)
    assert np.all(inj.actions[CLOSE : CLOSE + 8, GRIPPER_DIM] == -1.0)
    assert gripper_phases(inj.actions).close_idx == CLOSE + 8


def test_a_zero_shift_is_refused_as_not_an_injection():
    a = _demo()
    with pytest.raises(ValueError, match="non-zero"):
        mistimed_close(a, gripper_phases(a), shift=0)


def test_early_release_opens_before_the_place_pose():
    a = _demo()
    p = gripper_phases(a)
    inj = early_release(a, p, steps=20)
    assert inj.window == (RELEASE - 20, RELEASE)
    assert np.all(inj.actions[RELEASE - 20 : RELEASE, GRIPPER_DIM] == -1.0)
    # The grasp itself is untouched — this is a release fault, not a failure to grasp.
    assert np.all(inj.actions[CLOSE : RELEASE - 20, GRIPPER_DIM] == 1.0)


def test_early_release_cannot_open_before_the_grasp():
    """An 'early release' that precedes the close is a different cause and would mislabel."""
    a = _demo()
    p = gripper_phases(a)
    inj = early_release(a, p, steps=10_000)
    assert inj.window[0] > p.close_idx


def test_injection_rejects_an_unknown_cause():
    with pytest.raises(ValueError, match="unknown cause"):
        Injection(actions=_demo(), window=(0, 1), cause="not_a_cause", params={})


def test_injected_actions_are_read_only():
    """A replayed result must keep corresponding to the actions it was filed under."""
    inj = grasp_offset(_demo(), gripper_phases(_demo()), np.array([0.02, 0.0]), 20, SCALE)
    with pytest.raises(ValueError):
        inj.actions[0, 0] = 5.0


# --- the independent check on the injection ---------------------------------


def test_realised_offset_reads_the_separation_from_proprio():
    """CLAUDE.md: the calibration's claim is checked against a measurement, not trusted."""
    base = np.zeros((50, 3))
    injected = base.copy()
    injected[:, 0] += 0.03
    out = realised_offset_from_proprio(base, injected, at_step=40)
    assert out["separation_norm_m"] == pytest.approx(0.03)
    assert out["at_step"] == 40


def _make(cause: str, a: np.ndarray, p) -> Injection:
    if cause == "grasp_offset":
        return grasp_offset(a, p, np.array([0.03, 0.0]), 20, SCALE)
    if cause == "wrong_target":
        return wrong_target(a, p, np.zeros(2), np.array([0.15, 0.0]), 1.0, 20, SCALE)
    if cause == "mistimed_close":
        return mistimed_close(a, p, shift=-8)
    if cause == "early_release":
        return early_release(a, p, steps=20)
    raise AssertionError(f"unhandled cause {cause!r}")


# --- episode-kind classification --------------------------------------------


def test_episode_kind_does_not_mistake_an_injection_for_a_demo():
    """The collision that nearly shipped: an injected id names the demo it corrupts, and a
    substring test for `_demo_` classified it as expert data. `train_dynamics.py` and
    `probe_representation.py` both select on this."""
    from cfe.data.demos import DEMO, INJECT, ROLLOUT, episode_kind

    assert episode_kind("libero_object_t0_demo_1") == DEMO
    assert episode_kind("libero_object_t0_rollout_12") == ROLLOUT
    assert episode_kind("libero_object_t0_inject_grasp_offset_src1_p2") == INJECT
    assert episode_kind("libero_object_t12_inject_early_release_src47_p0") == INJECT
    assert episode_kind("nonsense") == "unknown"


def test_no_injected_id_contains_the_demo_substring():
    """Belt and braces: even if a consumer still tests with `in`, it must not match."""
    from cfe.data.demos import episode_kind

    for cause in CAUSES:
        cid = f"libero_object_t0_inject_{cause}_src3_p1"
        assert "_demo_" not in cid
        assert episode_kind(cid) == "inject"
