import numpy as np
import pytest
from astropy import units as u
from gwpy.time import LIGOTimeGPS

from gwexpy.timeseries import TimeSeries

try:
    import xarray as xr
except ImportError:
    xr = None


@pytest.mark.skipif(xr is None, reason="xarray not installed")
def test_xarray_interop():
    t0 = LIGOTimeGPS(1000000000, 0)
    dt = 0.5 * u.s
    data = np.random.randn(20)
    ts = TimeSeries(data, t0=t0, dt=dt, name="xr_test", unit="m")

    # 1. to_xarray
    da = ts.to_xarray()
    assert isinstance(da, xr.DataArray)
    assert da.name == "xr_test"
    assert da.attrs["unit"] == "m"
    assert da.coords["time"].shape == (20,)

    # 2. from_xarray
    ts_restored = TimeSeries.from_xarray(da)
    assert ts_restored.name == "xr_test"
    assert ts_restored.unit == u.m
    assert np.isclose(ts_restored.t0.value, float(t0), atol=1e-6)
    np.testing.assert_allclose(ts_restored.dt.value, dt.value)
    np.testing.assert_array_equal(ts_restored.value, data)
