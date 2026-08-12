"""Pure-function pieces of the ceiling test (Phase 3.5 diagnostic, no simulator, no torch here).

The window-scoring and IoU machinery these lean on (`window_iou`, the tie-break convention) are
already tested elsewhere (`tests/test_eval_metrics.py`, `tests/test_search_windows.py`); this
covers only what's new: window selection over a raw score curve, the half-open-window label
convention, the shared-length truncation check, and AUROC.
"""

import numpy as np
import pytest

from scripts.diagnose_fault_localization import (
    auroc,
    labels_from_window,
    position_features,
    shared_length,
    summarise_iou,
    top_m_windows,
    top_window,
)


def test_top_window_finds_the_true_peak():
    scores = np.array([0.0, 0.0, 5.0, 5.0, 0.0, 0.0])
    assert top_window(scores, w=2, stride=1) == (2, 4)


def test_top_window_none_when_episode_shorter_than_window():
    assert top_window(np.zeros(3), w=5, stride=1) is None


def test_top_window_ties_break_on_earlier_start():
    """Matches cfe/search/windows.py:_top's convention, so the two are comparable."""
    scores = np.zeros(6)
    assert top_window(scores, w=2, stride=1) == (0, 2)


def test_top_m_windows_ranks_best_first():
    scores = np.array([0.0, 9.0, 0.0, 5.0, 0.0, 1.0])
    got = top_m_windows(scores, w=1, stride=1, m=3)
    assert got == [(1, 2), (3, 4), (5, 6)]


def test_top_m_windows_matches_top_window_at_m_1():
    scores = np.array([0.0, 0.0, 5.0, 5.0, 0.0, 0.0])
    assert top_m_windows(scores, w=2, stride=1, m=1) == [top_window(scores, w=2, stride=1)]


def test_top_m_windows_empty_when_episode_shorter_than_window():
    assert top_m_windows(np.zeros(3), w=5, stride=1, m=3) == []


def test_top_m_windows_ties_break_on_earlier_start():
    """Matches cfe/search/windows.py:_top's convention, same as top_window."""
    scores = np.zeros(6)
    assert top_m_windows(scores, w=2, stride=1, m=3) == [(0, 2), (1, 3), (2, 4)]


def test_top_m_windows_caps_at_the_number_of_available_starts():
    scores = np.array([1.0, 2.0])
    assert len(top_m_windows(scores, w=1, stride=1, m=10)) == 2


def test_top_window_respects_stride():
    # The true peak is at start=1, but stride=2 only offers starts 0, 2, 4 — the diagnostic
    # must not silently fall back to an unstrided search and claim a window it never evaluated.
    scores = np.array([0.0, 9.0, 0.0, 1.0, 1.0])
    assert top_window(scores, w=1, stride=2)[0] in (0, 2, 4)


def test_labels_from_window_is_half_open():
    y = labels_from_window(t_total=10, edit_window=(3, 6))
    assert y.tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]


def test_labels_from_window_clips_to_episode_length():
    # A window whose end exceeds t_total should not raise or index out of bounds.
    y = labels_from_window(t_total=5, edit_window=(3, 100))
    assert y.tolist() == [0, 0, 0, 1, 1]


def test_labels_from_window_empty_window_is_all_clean():
    y = labels_from_window(t_total=5, edit_window=(4, 4))
    assert y.sum() == 0


def test_shared_length_equal_episodes_not_truncated():
    n, truncated = shared_length(50, 50)
    assert n == 50
    assert truncated is False


def test_shared_length_mismatched_episodes_truncated_to_the_shorter():
    n, truncated = shared_length(50, 47)
    assert n == 47
    assert truncated is True


def test_auroc_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(1.0)


def test_auroc_worst_case_separation():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(0.0)


def test_auroc_none_when_only_one_class_present():
    assert auroc(np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0])) is None


def test_position_features_shape_and_values():
    feats = position_features(t_total=5)
    assert feats.shape == (5, 2)
    assert feats[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert feats[:, 1].tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_position_features_single_step_does_not_divide_by_zero():
    feats = position_features(t_total=1)
    assert feats.shape == (1, 2)
    assert np.isfinite(feats).all()


def test_summarise_iou_groups_by_cause_and_pools():
    rows = [
        {"cause": "grasp_offset", "iou": 0.4},
        {"cause": "grasp_offset", "iou": 0.2},
        {"cause": "early_release", "iou": 0.0},
    ]
    out = summarise_iou(rows)
    assert out["grasp_offset"]["mean_iou"] == pytest.approx(0.3)
    assert out["grasp_offset"]["n"] == 2
    assert out["pooled"]["n"] == 3
    assert out["pooled"]["mean_iou"] == pytest.approx(0.2)
