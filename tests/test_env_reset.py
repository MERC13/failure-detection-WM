"""Invariant I1 (SPEC.md §1): exact state reset.

SPEC.md §8 lists this first among the tests whose silent failure would poison results. The
comparison is bit-exact by design — if it ever needs a tolerance to pass, the correct response
is to stop and report, not to add one.
"""

import pytest


@pytest.mark.slow
def test_exact_reset_is_bit_identical(cfg, task_and_init_states, seed):
    from scripts.verify_env import check_exact_reset

    task, init_states = task_and_init_states
    result = check_exact_reset(cfg, task, init_states, seed)
    assert result.passed, f"I1 exact reset failed: {result.detail}"


@pytest.mark.slow
def test_reset_does_not_diverge_under_stepping(cfg, task_and_init_states, seed):
    from scripts.verify_env import check_reset_under_stepping

    task, init_states = task_and_init_states
    result = check_reset_under_stepping(cfg, task, init_states, seed)
    assert result.passed, f"I1 diverged once stepped: {result.detail}"
