"""The failure-set store (SPEC.md §4.4).

CLAUDE.md: "Failure episodes carry a `cause` tag that is `null` for naturally occurring failures
and a string for injected ones. Code that assumes non-null will crash on half the dataset."
These pin the null-cause path as much as the happy one.
"""

import numpy as np
import pytest

from cfe.data.failure_set import (
    INJECTED,
    NATURAL,
    FailureRecord,
    load_failure_set,
    read_record,
    summarise,
    write_record,
)


def _record(**over) -> FailureRecord:
    base = {
        "failure_id": "libero_object_t0_inject_grasp_offset_demo_0_a",
        "source": INJECTED,
        "suite": "libero_object",
        "task_id": 0,
        "task_name": "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "episode_id": "libero_object_t0_inject_grasp_offset_demo_0_a",
        "init_state": np.arange(110, dtype=np.float64),
        "actions": np.zeros((12, 7)),
        "success": False,
        "cause": "grasp_offset",
        "edit_window": (20, 40),
        "expert_episode_id": "libero_object_t0_demo_0",
        "params": {"requested_delta_m": [0.03, 0.0]},
    }
    return FailureRecord(**(base | over))


def _natural(**over) -> FailureRecord:
    natural = {
        "failure_id": "libero_object_t0_rollout_2",
        "source": NATURAL,
        "episode_id": "libero_object_t0_rollout_2",
        "cause": None,
        "edit_window": None,
        "expert_episode_id": None,
        "params": {"failure_mode": "no_object_contact"},
    }
    return _record(**(natural | over))


# --- the invariants the store enforces --------------------------------------


def test_the_failure_set_refuses_a_success():
    with pytest.raises(ValueError, match="failures only"):
        _record(success=True)


def test_an_injected_failure_must_carry_a_cause():
    with pytest.raises(ValueError, match="must carry a cause"):
        _record(cause=None)


def test_a_natural_failure_may_not_be_given_a_cause():
    """Guessing one would make SPEC.md §6's cause accuracy score a guess."""
    with pytest.raises(ValueError, match="no ground-truth cause"):
        _natural(cause="grasp_offset")


def test_an_injected_failure_must_carry_its_edit_window():
    with pytest.raises(ValueError, match="must carry its edit window"):
        _record(edit_window=None)


def test_is_injected_is_explicit_rather_than_a_truthiness_test():
    assert _record().is_injected
    assert not _natural().is_injected


# --- round trip -------------------------------------------------------------


def test_round_trip_preserves_every_field(tmp_path):
    r = _record()
    path = write_record(tmp_path, r)
    back = read_record(path)

    assert back.failure_id == r.failure_id
    assert back.cause == r.cause
    assert back.edit_window == (20, 40)
    assert back.expert_episode_id == r.expert_episode_id
    assert np.array_equal(back.init_state, r.init_state)
    assert np.array_equal(back.actions, r.actions)
    assert back.params == r.params


def test_a_natural_failure_round_trips_with_a_null_cause(tmp_path):
    """The half of the dataset CLAUDE.md warns about."""
    path = write_record(tmp_path, _natural())
    back = read_record(path)
    assert back.cause is None
    assert back.edit_window is None
    assert back.expert_episode_id is None


def test_writes_are_atomic_leaving_no_partial_file(tmp_path):
    write_record(tmp_path, _record())
    assert list(tmp_path.glob("*.tmp")) == []
    assert len(list(tmp_path.glob("*.json"))) == 1


# --- loading and filtering --------------------------------------------------


def test_loading_is_sorted_not_filesystem_ordered(tmp_path):
    """G3 takes the first N; filesystem enumeration order is not the same on two machines."""
    for fid in ("c_two", "a_zero", "b_one"):
        write_record(tmp_path, _record(failure_id=fid, episode_id=fid))
    assert [r.failure_id for r in load_failure_set(tmp_path)] == ["a_zero", "b_one", "c_two"]


def test_filters_select_source_task_and_cause(tmp_path):
    write_record(tmp_path, _record())
    write_record(tmp_path, _natural())
    write_record(tmp_path, _record(failure_id="other_task", episode_id="other", task_id=1))

    assert len(load_failure_set(tmp_path)) == 3
    assert len(load_failure_set(tmp_path, source=NATURAL)) == 1
    assert len(load_failure_set(tmp_path, task_id=1)) == 1
    assert len(load_failure_set(tmp_path, cause="grasp_offset")) == 2


def test_a_missing_directory_is_empty_not_an_error(tmp_path):
    assert load_failure_set(tmp_path / "nope") == []


def test_summary_counts_what_section_4_4_quotas_are_checked_against(tmp_path):
    records = [
        _record(),
        _record(failure_id="b", episode_id="b", cause="early_release"),
        _natural(),
        _record(failure_id="d", episode_id="d", task_id=2),
    ]
    s = summarise(records)
    assert s == {
        "n_total": 4,
        "n_injected": 3,
        "n_natural": 1,
        "by_cause": {"early_release": 1, "grasp_offset": 2, "natural (unlabelled)": 1},
        "by_task": {0: 3, 2: 1},
        "n_tasks": 2,
    }
