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


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

from gwexpy.interop.xarray_ import (
    _detect_dim_role,
    _detect_dim_roles,
    _try_parse_unit,
)


class TestDetectDimRole:
    def test_heuristic_time_dim(self):
        da = xr.DataArray(np.zeros(5), dims=("time",))
        role = _detect_dim_role(da, "time")
        assert role == 0  # time role

    def test_heuristic_frequency_dim(self):
        da = xr.DataArray(np.zeros(5), dims=("frequency",))
        role = _detect_dim_role(da, "frequency")
        assert role is not None

    def test_unknown_dim_returns_none(self):
        da = xr.DataArray(np.zeros(5), dims=("unknown_dim_xyz",))
        role = _detect_dim_role(da, "unknown_dim_xyz")
        assert role is None

    def test_cf_axis_attribute(self):
        da = xr.DataArray(
            np.zeros(5),
            dims=("t",),
            coords={"t": xr.DataArray(np.arange(5), dims=("t",), attrs={"axis": "T"})},
        )
        role = _detect_dim_role(da, "t")
        assert role == 0  # T → time role


class TestDetectDimRoles:
    def test_basic(self):
        da = xr.DataArray(np.zeros((5, 3)), dims=("time", "x"))
        roles = _detect_dim_roles(da, ("time", "x"))
        assert 0 in roles  # time → role 0
        assert roles[0] == "time"

    def test_empty_returns_empty(self):
        da = xr.DataArray(np.zeros(5), dims=("abc_unknown_xyz",))
        roles = _detect_dim_roles(da, ("abc_unknown_xyz",))
        assert roles == {}


class TestTryParseUnit:
    def test_valid_unit(self):
        u = _try_parse_unit("m/s")
        assert u is not None

    def test_invalid_unit_returns_none_with_warning(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _try_parse_unit("not_a_real_unit_xyz_abc_123")
        assert result is None
