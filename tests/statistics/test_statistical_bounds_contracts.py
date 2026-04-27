from __future__ import annotations

import numpy as np
import pytest

from gwexpy.statistics.roc import calculate_roc

pytestmark = pytest.mark.filterwarnings(
    "ignore:`trapz` is deprecated:DeprecationWarning"
)


def test_calculate_roc_returns_bounded_monotonic_rates_and_auc():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    fpr, tpr, auc = calculate_roc(y_true, y_score, n_points=8)

    assert fpr.shape == tpr.shape == (8,)
    assert np.all(np.isfinite(fpr))
    assert np.all(np.isfinite(tpr))
    assert np.all((0.0 <= fpr) & (fpr <= 1.0))
    assert np.all((0.0 <= tpr) & (tpr <= 1.0))
    assert np.all(np.diff(fpr) >= 0.0)
    assert 0.0 <= auc <= 1.0


@pytest.mark.parametrize(
    ("y_true", "y_score"),
    [
        (np.ones(4), np.linspace(0.0, 1.0, 4)),
        (np.zeros(4), np.linspace(0.0, 1.0, 4)),
    ],
)
def test_calculate_roc_degenerate_class_contract_returns_diagonal(y_true, y_score):
    fpr, tpr, auc = calculate_roc(y_true, y_score)

    np.testing.assert_array_equal(fpr, np.array([0, 1]))
    np.testing.assert_array_equal(tpr, np.array([0, 1]))
    assert auc == pytest.approx(0.5)
