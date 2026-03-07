"""Tests for xarray interop adapter."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from gwexpy.interop.xarray_ import from_xarray, to_xarray
from gwexpy.timeseries import TimeSeries


def _make_ts(n=100):
    return TimeSeries(
        np.arange(n, dtype=np.float64),
        t0=1000000000.0, dt=0.01, unit="m", name="test",
    )


class TestToXarray:
    def test_returns_dataarray(self):
        ts = _make_ts()
        da = to_xarray(ts)
        assert isinstance(da, xr.DataArray)
        assert da.dims == ("time",)
        assert len(da) == 100

    def test_values_preserved(self):
        ts = _make_ts()
        da = to_xarray(ts)
        np.testing.assert_array_equal(da.values, ts.value)

    def test_attrs_contain_unit(self):
        ts = _make_ts()
        da = to_xarray(ts)
        assert da.attrs["unit"] == "m"

    def test_name_preserved(self):
        ts = _make_ts()
        da = to_xarray(ts)
        assert da.name == "test"

    def test_gps_time_coord(self):
        ts = _make_ts()
        da = to_xarray(ts, time_coord="gps")
        times = da.coords["time"].values
        assert np.isclose(times[0], ts.times.value[0])


class TestFromXarray:
    def test_roundtrip_gps(self):
        ts = _make_ts()
        da = to_xarray(ts, time_coord="gps")
        ts2 = from_xarray(TimeSeries, da, unit="m")
        np.testing.assert_array_equal(ts2.value, ts.value)
        assert np.isclose(ts2.t0.value, ts.t0.value, rtol=1e-6)

    def test_roundtrip_datetime(self):
        ts = _make_ts()
        da = to_xarray(ts, time_coord="datetime")
        ts2 = from_xarray(TimeSeries, da, unit="m")
        np.testing.assert_array_equal(ts2.value, ts.value)

    def test_unit_from_attrs(self):
        ts = _make_ts()
        da = to_xarray(ts, time_coord="gps")
        ts2 = from_xarray(TimeSeries, da)
        assert str(ts2.unit) == "m"
