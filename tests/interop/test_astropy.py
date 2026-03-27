"""Tests for gwexpy/interop/astropy_.py."""
from __future__ import annotations

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries
from gwexpy.interop.astropy_ import from_astropy_timeseries, to_astropy_timeseries


def _make_ts(n=10, t0=1000000000.0, dt=1.0, unit="m"):
    return TimeSeries(np.arange(float(n)), t0=t0, dt=dt, unit=unit)


class TestToAstropyTimeSeries:
    def test_basic_conversion(self):
        from astropy.timeseries import TimeSeries as APTimeSeries
        ts = _make_ts()
        ap = to_astropy_timeseries(ts)
        assert isinstance(ap, APTimeSeries)

    def test_value_column_default(self):
        ts = _make_ts()
        ap = to_astropy_timeseries(ts)
        assert "value" in ap.colnames

    def test_custom_column_name(self):
        ts = _make_ts()
        ap = to_astropy_timeseries(ts, column="strain")
        assert "strain" in ap.colnames

    def test_correct_length(self):
        ts = _make_ts(n=5)
        ap = to_astropy_timeseries(ts)
        assert len(ap) == 5

    def test_data_values_match(self):
        ts = _make_ts(n=5)
        ap = to_astropy_timeseries(ts)
        np.testing.assert_array_equal(ap["value"], ts.value)

    def test_time_format_gps(self):
        from astropy.time import Time
        ts = _make_ts(n=3, t0=1000000000.0)
        ap = to_astropy_timeseries(ts, time_format="gps")
        assert isinstance(ap.time, Time)

    def test_time_format_other_falls_back_to_gps(self):
        """Non-gps time_format currently falls back to gps internally."""
        from astropy.time import Time
        ts = _make_ts(n=3)
        ap = to_astropy_timeseries(ts, time_format="unix")
        assert isinstance(ap.time, Time)


class TestFromAstropyTimeSeries:
    def _make_ap_ts(self, n=5, t0=1000000000.0, dt=1.0):
        from astropy.time import Time
        from astropy.timeseries import TimeSeries as APTimeSeries
        import astropy.units as u
        times = Time(t0 + np.arange(n) * dt, format="gps")
        data = {"value": np.arange(float(n)) * u.m}
        return APTimeSeries(data=data, time=times)

    def test_basic_roundtrip(self):
        ts = _make_ts(n=5)
        ap = to_astropy_timeseries(ts)
        ts2 = from_astropy_timeseries(TimeSeries, ap)
        np.testing.assert_array_almost_equal(ts2.value, ts.value)

    def test_t0_restored(self):
        ts = _make_ts(n=5, t0=1000000000.0, dt=1.0)
        ap = to_astropy_timeseries(ts)
        ts2 = from_astropy_timeseries(TimeSeries, ap)
        assert ts2.t0.value == pytest.approx(ts.t0.value, rel=1e-6)

    def test_dt_restored(self):
        ts = _make_ts(n=5, dt=0.5)
        ap = to_astropy_timeseries(ts)
        ts2 = from_astropy_timeseries(TimeSeries, ap)
        assert ts2.dt.value == pytest.approx(0.5, rel=1e-4)

    def test_unit_from_quantity_column(self):
        ap = self._make_ap_ts(n=5)
        ts2 = from_astropy_timeseries(TimeSeries, ap)
        assert str(ts2.unit) == "m"

    def test_unit_override(self):
        ts = _make_ts(n=5, unit="m")
        ap = to_astropy_timeseries(ts)
        ts2 = from_astropy_timeseries(TimeSeries, ap, unit="cm")
        assert str(ts2.unit) == "cm"

    def test_custom_column(self):
        ts = _make_ts(n=3)
        ap = to_astropy_timeseries(ts, column="strain")
        ts2 = from_astropy_timeseries(TimeSeries, ap, column="strain")
        np.testing.assert_array_almost_equal(ts2.value, ts.value)

    def test_single_point_dt_default(self):
        """Single-point series: dt defaults to 1.0."""
        from astropy.time import Time
        from astropy.timeseries import TimeSeries as APTimeSeries
        times = Time([1000000000.0], format="gps")
        ap = APTimeSeries(data={"value": [42.0]}, time=times)
        ts2 = from_astropy_timeseries(TimeSeries, ap)
        assert ts2.dt.value == pytest.approx(1.0)
