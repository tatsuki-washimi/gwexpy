"""Input-contract regression tests for ``gwexpy.statistics.roc``.

Covers the 9 confirmed findings (issue #463 / Phase 1 supplement sweep
2026-06-23): degenerate inputs must raise ``ValueError`` instead of silently
returning a meaningless numeric default (AUC=0.5/0.0), and the sklearn-style
``{-1, +1}`` label convention must be handled rather than dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwexpy.statistics.roc import (
    calculate_roc,
    evaluate_detection_performance,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:`trapz` is deprecated:DeprecationWarning"
)


# --- valid baseline ---------------------------------------------------------


def test_valid_input_returns_finite_auc():
    fpr, tpr, auc = calculate_roc(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), n_points=16
    )
    assert np.all(np.isfinite(fpr))
    assert np.all(np.isfinite(tpr))
    assert auc == pytest.approx(1.0)


# --- S1a / S1b: empty class -------------------------------------------------


def test_empty_positive_class_raises():
    # (S1a) all-negative labels: previously returned AUC=0.5 silently.
    with pytest.raises(ValueError, match="positive class is empty"):
        calculate_roc(np.array([0, 0, 0]), np.array([0.1, 0.5, 0.9]))


def test_empty_negative_class_raises():
    # (S1b) all-positive labels: previously returned AUC=0.5 silently.
    with pytest.raises(ValueError, match="negative class is empty"):
        calculate_roc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9]))


# --- {-1, +1} sklearn label convention [NEW] --------------------------------


def test_sklearn_minus_one_plus_one_labels_handled():
    # Previously {-1, +1} collapsed n_neg=0 -> silent AUC=0.5.
    fpr, tpr, auc = calculate_roc(
        np.array([-1, 1, -1, 1]), np.array([0.1, 0.9, 0.2, 0.8])
    )
    assert auc == pytest.approx(1.0)


def test_one_two_labels_handled():
    # {1, 2} encoding: 2 is treated as negative (non pos_label).
    fpr, tpr, auc = calculate_roc(
        np.array([2, 1, 2, 1]), np.array([0.1, 0.9, 0.2, 0.8])
    )
    assert auc == pytest.approx(1.0)


def test_pos_label_argument_respected():
    fpr, tpr, auc = calculate_roc(
        np.array([5, 9, 5, 9]),
        np.array([0.1, 0.9, 0.2, 0.8]),
        pos_label=9,
    )
    assert auc == pytest.approx(1.0)


# --- S1c: n_points -----------------------------------------------------------


@pytest.mark.parametrize("n_points", [0, 1])
def test_n_points_below_two_raises(n_points):
    with pytest.raises(ValueError, match="n_points must be >= 2"):
        calculate_roc(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.9, 0.2, 0.8]),
            n_points=n_points,
        )


# --- S1d: all-equal scores ---------------------------------------------------


def test_all_equal_scores_raises():
    with pytest.raises(ValueError, match="identical"):
        calculate_roc(np.array([0, 1, 0, 1]), np.array([5.0, 5.0, 5.0, 5.0]))


# --- S1e: non-finite scores --------------------------------------------------


@pytest.mark.parametrize(
    "y_score",
    [
        np.array([0.1, np.nan, 0.5, 0.9]),
        np.array([0.1, np.inf, 0.5, 0.9]),
    ],
)
def test_non_finite_scores_raise(y_score):
    with pytest.raises(ValueError, match="non-finite"):
        calculate_roc(np.array([0, 0, 1, 1]), y_score)


# --- S1g: shape contract -----------------------------------------------------


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError, match="same shape"):
        calculate_roc(np.array([0, 1, 1]), np.array([0.1, 0.9]))


# --- empty input -------------------------------------------------------------


def test_empty_arrays_raise_clear_error():
    with pytest.raises(ValueError, match="at least one sample"):
        calculate_roc(np.array([]), np.array([]))


def test_evaluate_detection_performance_rejects_zero_trials():
    def _method(_ts):
        return 0.0

    def _generator(**_kwargs):
        return object()

    with pytest.raises(ValueError, match="n_trials"):
        evaluate_detection_performance(_method, _generator, n_trials=0)


# --- tied-FPR ordering [NEW] -------------------------------------------------


def test_tied_fpr_ordering_matches_sklearn():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=40)
    # ensure both classes present
    y_true[0], y_true[1] = 0, 1
    y_score = rng.random(size=40)

    _, _, auc = calculate_roc(y_true, y_score, n_points=200)
    ref = sklearn_metrics.roc_auc_score(y_true, y_score)
    # lexsort removes the tied-FPR bias; dense thresholding approaches sklearn.
    assert auc == pytest.approx(ref, abs=0.02)


# --- S1f: evaluate_detection_performance empty score map ---------------------


def test_evaluate_detection_empty_score_map_raises():
    from gwexpy.statistics.roc import _reduce_score

    class _EmptyMap:
        value = np.array([])

    with pytest.raises(ValueError, match="empty or all-NaN"):
        _reduce_score(_EmptyMap())


def test_evaluate_detection_all_nan_score_map_raises():
    from gwexpy.statistics.roc import _reduce_score

    class _NanMap:
        value = np.array([np.nan, np.nan, np.nan])

    with pytest.raises(ValueError, match="empty or all-NaN"):
        _reduce_score(_NanMap())
