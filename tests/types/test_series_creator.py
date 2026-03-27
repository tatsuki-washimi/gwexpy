"""Tests for gwexpy/types/series_creator.py."""
from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.types.series_creator import (
    _is_time_unit,
    _is_freq_unit,
    _is_angular_frequency,
    _to_quantity_1d,
    _to_hz,
    _to_angular_frequency,
    as_series,
)


# ---------------------------------------------------------------------------
# _is_time_unit
# ---------------------------------------------------------------------------

class TestIsTimeUnit:
    def test_seconds(self):
        assert _is_time_unit(u.s) is True

    def test_milliseconds(self):
        assert _is_time_unit(u.ms) is True

    def test_hz_is_not_time(self):
        assert _is_time_unit(u.Hz) is False

    def test_invalid_raises_false(self):
        # Lines 14-15 — ValueError/TypeError → False
        assert _is_time_unit("not_a_unit_xyz") is False

    def test_none_returns_false(self):
        assert _is_time_unit(None) is False


# ---------------------------------------------------------------------------
# _is_freq_unit
# ---------------------------------------------------------------------------

class TestIsFreqUnit:
    def test_hz(self):
        assert _is_freq_unit(u.Hz) is True

    def test_khz(self):
        assert _is_freq_unit(u.kHz) is True

    def test_seconds_is_not_freq(self):
        assert _is_freq_unit(u.s) is False

    def test_invalid_returns_false(self):
        # Lines 23-25 — ValueError/TypeError → False
        assert _is_freq_unit("invalid_unit_abc") is False


# ---------------------------------------------------------------------------
# _is_angular_frequency
# ---------------------------------------------------------------------------

class TestIsAngularFrequency:
    def test_rad_per_s(self):
        assert _is_angular_frequency(u.rad / u.s) is True

    def test_hz_not_angular(self):
        assert _is_angular_frequency(u.Hz) is False

    def test_invalid_returns_false(self):
        # Lines 31-32 — ValueError/TypeError → False
        assert _is_angular_frequency("bad") is False


# ---------------------------------------------------------------------------
# _to_quantity_1d
# ---------------------------------------------------------------------------

class TestToQuantity1d:
    def test_valid_1d(self):
        q = u.Quantity([1.0, 2.0], u.s)
        result = _to_quantity_1d(q)
        assert len(result) == 2

    def test_non_quantity_raises(self):
        # Line 38
        with pytest.raises(TypeError, match="expects a 1D astropy.units.Quantity"):
            _to_quantity_1d(np.array([1.0, 2.0]))

    def test_2d_raises(self):
        # Line 41
        q = u.Quantity(np.ones((2, 3)), u.s)
        with pytest.raises(ValueError, match="1D axis"):
            _to_quantity_1d(q)


# ---------------------------------------------------------------------------
# _to_hz
# ---------------------------------------------------------------------------

class TestToHz:
    def test_hz_passthrough(self):
        q = u.Quantity([10.0, 20.0], u.Hz)
        result = _to_hz(q)
        np.testing.assert_allclose(result.value, [10.0, 20.0])

    def test_khz_to_hz(self):
        q = u.Quantity([1.0], u.kHz)
        result = _to_hz(q)
        assert result[0].to(u.Hz).value == pytest.approx(1000.0)

    def test_angular_frequency_fallback(self):
        # Line 52-53 — angular freq that can't convert directly
        omega = u.Quantity([2 * np.pi], u.rad / u.s)
        result = _to_hz(omega)
        assert result[0].value == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------------------------
# _to_angular_frequency
# ---------------------------------------------------------------------------

class TestToAngularFrequency:
    def test_hz_to_rad_per_s(self):
        # Line 58
        hz = u.Quantity([1.0], u.Hz)
        result = _to_angular_frequency(hz)
        assert result[0].value == pytest.approx(2 * np.pi, rel=1e-5)


# ---------------------------------------------------------------------------
# as_series — time axis
# ---------------------------------------------------------------------------

class TestAsSeriesTime:
    def test_quantity_seconds(self):
        axis = u.Quantity(np.arange(10, dtype=float), u.s)
        ts = as_series(axis)
        assert len(ts) == 10

    def test_quantity_seconds_with_unit(self):
        # Line 120-128
        axis = u.Quantity(np.arange(5, dtype=float), u.s)
        ts = as_series(axis, unit=u.ms)
        assert len(ts) == 5

    def test_wrong_unit_for_time_axis_raises(self):
        # Line 122 — unit not time-like for time axis
        axis = u.Quantity(np.arange(5, dtype=float), u.s)
        with pytest.raises(ValueError, match="time-like"):
            as_series(axis, unit=u.Hz)

    def test_with_name(self):
        axis = u.Quantity(np.arange(5, dtype=float), u.s)
        ts = as_series(axis, name="myts")
        assert ts.name == "myts"


# ---------------------------------------------------------------------------
# as_series — frequency axis
# ---------------------------------------------------------------------------

class TestAsSeriesFreq:
    def test_quantity_hz(self):
        # Line 130-155
        axis = u.Quantity(np.linspace(0, 100, 50), u.Hz)
        fs = as_series(axis)
        assert len(fs) == 50

    def test_quantity_hz_no_unit(self):
        # Line 132 — unit=None → use axis unit
        axis = u.Quantity(np.linspace(0, 50, 25), u.Hz)
        fs = as_series(axis)
        assert len(fs) == 25

    def test_quantity_hz_with_unit(self):
        # Line 133-134
        axis = u.Quantity(np.linspace(0, 50, 25), u.Hz)
        fs = as_series(axis, unit=u.kHz)
        assert len(fs) == 25

    def test_freq_axis_with_non_freq_unit_raises(self):
        # Lines 135-138
        axis = u.Quantity(np.linspace(0, 50, 25), u.Hz)
        with pytest.raises(ValueError, match="frequency-like"):
            as_series(axis, unit=u.s)

    def test_angular_frequency_axis(self):
        # Lines 141-142 — angular frequency axis
        axis = u.Quantity(np.linspace(0, 2*np.pi*100, 50), u.rad/u.s)
        fs = as_series(axis)
        assert len(fs) == 50

    def test_angular_freq_output_unit(self):
        # Line 141-142 — value_unit is angular → _to_angular_frequency
        axis = u.Quantity(np.linspace(0, 2*np.pi*100, 50), u.rad/u.s)
        fs = as_series(axis, unit=u.rad/u.s)
        assert len(fs) == 50


# ---------------------------------------------------------------------------
# as_series — axis_unit path (Index with unit)
# ---------------------------------------------------------------------------

class TestAsSeriesAxisUnit:
    def test_axis_with_unit_attr(self):
        # Lines 87-94 — non-Quantity with .unit
        class FakeIndex:
            def __init__(self, vals, unit):
                self._vals = vals
                self.unit = unit
            def __iter__(self):
                return iter(self._vals)
            def __len__(self):
                return len(self._vals)
            def __getitem__(self, key):
                return self._vals[key]

        idx = FakeIndex(np.arange(5, dtype=float), "s")
        ts = as_series(idx)
        assert len(ts) == 5


# ---------------------------------------------------------------------------
# as_series — non-unit path (datetime-like)
# ---------------------------------------------------------------------------

class TestAsSeriesNoUnit:
    def test_unsupported_type_raises(self):
        # Lines 113-117 — can't convert to GPS
        with pytest.raises(TypeError, match="as_series expects"):
            as_series(42)  # single scalar with no unit

    def test_no_unit_axis_raises(self):
        # Line 157 — axis with no time or freq unit
        # e.g. dimensionless quantity
        axis = u.Quantity(np.arange(5, dtype=float), u.dimensionless_unscaled)
        with pytest.raises(ValueError, match="time-like"):
            as_series(axis)
