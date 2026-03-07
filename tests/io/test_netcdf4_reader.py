"""Tests for NetCDF4 reader/writer roundtrip."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


class TestNetCDF4Roundtrip:
    def test_single_variable_roundtrip(self, tmp_path):
        path = tmp_path / "test.nc"
        data = np.arange(100, dtype=np.float64)
        ts = TimeSeries(data, t0=1000000000, dt=0.01, name="signal", unit="m")

        # Write
        tsd_out = TimeSeriesDict({"signal": ts})
        tsd_out.write(str(path), format="netcdf4")

        # Read
        tsd_in = TimeSeriesDict.read(str(path), format="netcdf4")
        assert "signal" in tsd_in
        np.testing.assert_allclose(tsd_in["signal"].value, data)
        assert np.isclose(tsd_in["signal"].dt.value, 0.01)

    def test_multi_variable(self, tmp_path):
        path = tmp_path / "multi.nc"
        tsd_out = TimeSeriesDict({
            "ch1": TimeSeries(np.ones(50), t0=0, dt=0.1, name="ch1"),
            "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.1, name="ch2"),
        })
        tsd_out.write(str(path), format="netcdf4")

        tsd_in = TimeSeriesDict.read(str(path), format="netcdf4")
        assert set(tsd_in.keys()) >= {"ch1", "ch2"}

    def test_unit_override(self, tmp_path):
        path = tmp_path / "unit.nc"
        ts = TimeSeries(np.ones(10), t0=0, dt=1.0, name="x", unit="m")
        TimeSeriesDict({"x": ts}).write(str(path), format="netcdf4")

        tsd = TimeSeriesDict.read(str(path), format="netcdf4", unit="V")
        assert str(tsd["x"].unit) == "V"

    def test_auto_time_coord_detection(self, tmp_path):
        """The reader should auto-detect time coordinates named 'time', 'Time', 't', etc."""
        for coord_name in ("time", "Time", "TIME", "t"):
            path = tmp_path / f"tc_{coord_name}.nc"
            times = np.arange(10, dtype=np.float64)
            ds = xr.Dataset(
                {"signal": xr.DataArray(np.ones(10), dims=[coord_name])},
                coords={coord_name: times},
            )
            ds.to_netcdf(str(path))

            tsd = TimeSeriesDict.read(str(path), format="netcdf4")
            assert "signal" in tsd
            assert len(tsd["signal"]) == 10

    def test_single_timeseries_read(self, tmp_path):
        from gwexpy.timeseries.io.netcdf4_ import read_timeseries_netcdf4

        path = tmp_path / "single.nc"
        ts = TimeSeries(np.arange(20, dtype=np.float64), t0=0, dt=0.5, name="x")
        TimeSeriesDict({"x": ts}).write(str(path), format="netcdf4")

        ts_in = read_timeseries_netcdf4(str(path))
        assert len(ts_in) == 20

    def test_empty_dataset_raises(self, tmp_path):
        path = tmp_path / "empty.nc"
        ds = xr.Dataset()
        ds.to_netcdf(str(path))
        with pytest.raises((ValueError, KeyError)):
            TimeSeriesDict.read(str(path), format="netcdf4")
