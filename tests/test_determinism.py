"""Invariant I5 (SPEC.md §1): same seed and config produce identical output."""

import numpy as np
import pytest


@pytest.mark.slow
def test_seeded_rollout_is_deterministic(cfg, task_and_init_states, seed):
    from scripts.verify_env import check_seed_determinism

    task, init_states = task_and_init_states
    result = check_seed_determinism(cfg, task, init_states, seed)
    assert result.passed, f"I5 determinism failed: {result.detail}"


def test_obs_comparison_rejects_a_single_changed_bit():
    """The I1 comparison must be exact. Guards against it being loosened to a tolerance."""
    from scripts.verify_env import compare_obs

    a = {"img": np.zeros((4, 4), dtype=np.uint8)}
    b = {"img": np.zeros((4, 4), dtype=np.uint8)}
    assert compare_obs(a, b) == {}

    b["img"][0, 0] = 1
    mismatch = compare_obs(a, b)
    assert mismatch, "a one-element difference must be reported"
    assert mismatch["n_differing_elements"] == 1


def test_obs_comparison_detects_missing_keys():
    from scripts.verify_env import compare_obs

    mismatch = compare_obs({"a": np.zeros(2)}, {})
    assert mismatch["reason"] == "observation keys differ"
    assert mismatch["only_in_first"] == ["a"]


def test_rollout_digest_is_order_and_value_sensitive():
    from scripts.verify_env import hash_arrays

    x, y = np.arange(4), np.arange(4)[::-1].copy()
    assert hash_arrays([x]) == hash_arrays([x])
    assert hash_arrays([x]) != hash_arrays([y])
    assert hash_arrays([x, y]) != hash_arrays([y, x])
