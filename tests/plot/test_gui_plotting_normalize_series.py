"""Regression tests for GUI plotting ``normalize_series`` (#765-C1).

The helper lives under ``gwexpy.gui`` but has no Qt dependency; exercising it
from ``tests/plot/`` keeps the case on the standard PR-fast pytest path
(``tests/gui/`` is ignored in headless CI).
"""

from __future__ import annotations

import numpy as np

from gwexpy.gui.plotting.normalize import normalize_series


def test_ndarray_data_does_not_trigger_ambiguous_truth_value():
    x, y = normalize_series({"data": np.array([0.0, 2.0, 3.0]), "df": 0.5})
    np.testing.assert_array_equal(x, [0.0, 0.5, 1.0])
    np.testing.assert_array_equal(y, [0.0, 2.0, 3.0])


def test_single_zero_array_takes_precedence_over_value():
    x, y = normalize_series({"data": np.array([0.0]), "value": np.array([99.0])})
    np.testing.assert_array_equal(x, [0.0])
    np.testing.assert_array_equal(y, [0.0])


def test_none_data_uses_value_fallback():
    x, y = normalize_series(
        {"data": None, "value": np.array([4.0, 5.0]), "frequencies": [10, 20]}
    )
    np.testing.assert_array_equal(x, [10, 20])
    np.testing.assert_array_equal(y, [4.0, 5.0])


def test_missing_data_and_value_returns_none():
    assert normalize_series({"data": None, "value": None}) is None
    assert normalize_series({}) is None


def test_list_data_and_epoch_axis_contract_unchanged():
    x, y = normalize_series({"data": [1.0, 2.0], "dt": 0.25, "epoch": 10.0})
    np.testing.assert_array_equal(x, [10.0, 10.25])
    np.testing.assert_array_equal(y, [1.0, 2.0])
