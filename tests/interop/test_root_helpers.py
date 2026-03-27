"""Tests for ROOT interop helper functions (no ROOT required)."""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.interop.root_ import _extract_error_array, _get_label
from gwexpy.timeseries import TimeSeries


# ---------------------------------------------------------------------------
# _get_label
# ---------------------------------------------------------------------------


class TestGetLabel:
    def test_name_with_unit(self):
        obj = type("O", (), {"name": "signal"})()
        label = _get_label(obj, u.m)
        assert label == "signal [m]"

    def test_name_without_unit(self):
        obj = type("O", (), {"name": "signal"})()
        label = _get_label(obj, None)
        assert label == "signal"

    def test_no_name_uses_default(self):
        obj = type("O", (), {})()
        label = _get_label(obj, None, default_name="xaxis")
        assert label == "xaxis"

    def test_empty_name_uses_default(self):
        obj = type("O", (), {"name": ""})()
        label = _get_label(obj, None, default_name="yaxis")
        assert label == "yaxis"

    def test_none_name_uses_default(self):
        obj = type("O", (), {"name": None})()
        label = _get_label(obj, u.Hz, default_name="freq")
        assert label == "freq [Hz]"

    def test_dimensionless_unit_string(self):
        obj = type("O", (), {"name": "phase"})()
        label = _get_label(obj, u.rad)
        assert "phase" in label
        assert "rad" in label


# ---------------------------------------------------------------------------
# _extract_error_array
# ---------------------------------------------------------------------------


class _FakeSeries:
    """Minimal gwpy-like Series with .value and .xindex."""

    def __init__(self, data):
        self.value = np.asarray(data)
        self.xindex = self.value
        self.unit = u.m
        self.shape = self.value.shape

    def __len__(self):
        return len(self.value)


class TestExtractErrorArray:
    def test_plain_array(self):
        series = _FakeSeries(np.ones(5))
        error = np.ones(5) * 0.1
        result = _extract_error_array(series, error)
        np.testing.assert_array_almost_equal(result, 0.1)

    def test_plain_list(self):
        series = _FakeSeries(np.ones(5))
        result = _extract_error_array(series, [0.1] * 5)
        assert result.shape == (5,)

    def test_plain_array_shape_mismatch_raises(self):
        series = _FakeSeries(np.ones(5))
        with pytest.raises(ValueError, match="shape"):
            _extract_error_array(series, np.ones(3))

    def test_gwpy_series_error(self):
        """Error with .value and .xindex attributes (gwpy-like)."""
        series = _FakeSeries(np.ones(5))
        error = _FakeSeries(np.ones(5) * 0.2)
        result = _extract_error_array(series, error)
        np.testing.assert_array_almost_equal(result, 0.2)

    def test_gwpy_series_length_mismatch_raises(self):
        series = _FakeSeries(np.ones(5))
        error = _FakeSeries(np.ones(3))
        with pytest.raises(ValueError, match="length"):
            _extract_error_array(series, error)

    def test_astropy_quantity_same_unit(self):
        series = _FakeSeries(np.ones(5))
        error = u.Quantity(np.ones(5) * 0.1, u.m)
        result = _extract_error_array(series, error)
        np.testing.assert_array_almost_equal(result, 0.1)

    def test_astropy_quantity_convertible_unit(self):
        series = _FakeSeries(np.ones(5))
        error = u.Quantity(np.ones(5) * 100, u.cm)  # 100 cm = 1 m
        result = _extract_error_array(series, error)
        np.testing.assert_array_almost_equal(result, 1.0)

    def test_astropy_quantity_incompatible_unit_uses_raw_value(self):
        series = _FakeSeries(np.ones(5))
        error = u.Quantity(np.ones(5) * 0.5, u.s)  # incompatible → raw value
        result = _extract_error_array(series, error)
        np.testing.assert_array_almost_equal(result, 0.5)

    def test_astropy_quantity_shape_mismatch_raises(self):
        series = _FakeSeries(np.ones(5))
        error = u.Quantity(np.ones(3), u.m)
        with pytest.raises(ValueError, match="shape"):
            _extract_error_array(series, error)

    def test_timeseries_as_error(self):
        ts_series = TimeSeries(np.ones(10), t0=0, dt=0.01)
        ts_error = TimeSeries(np.ones(10) * 0.05, t0=0, dt=0.01)
        result = _extract_error_array(ts_series, ts_error)
        assert result.shape == (10,)
