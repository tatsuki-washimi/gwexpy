"""Tests for Zarr reader/writer roundtrip."""

import os

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
if os.environ.get("GWEXPY_ALLOW_ZARR", "") != "1":
    pytest.skip("zarr tests require GWEXPY_ALLOW_ZARR=1", allow_module_level=True)

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


class TestZarrRoundtrip:
    def test_single_channel_roundtrip(self, tmp_path):
        path = tmp_path / "test.zarr"
        data = np.arange(100, dtype=np.float64)
        ts = TimeSeries(data, t0=1000000000, sample_rate=100, name="ch0")

        tsd_out = TimeSeriesDict({"ch0": ts})
        tsd_out.write(str(path), format="zarr")

        tsd_in = TimeSeriesDict.read(str(path), format="zarr")
        assert "ch0" in tsd_in
        np.testing.assert_allclose(tsd_in["ch0"].value, data)
        assert np.isclose(tsd_in["ch0"].sample_rate.value, 100)

    def test_multi_channel_roundtrip(self, tmp_path):
        path = tmp_path / "multi.zarr"
        tsd_out = TimeSeriesDict({
            "a": TimeSeries(np.ones(50), t0=0, sample_rate=10, name="a"),
            "b": TimeSeries(np.zeros(50), t0=0, sample_rate=10, name="b"),
        })
        tsd_out.write(str(path), format="zarr")

        tsd_in = TimeSeriesDict.read(str(path), format="zarr")
        assert set(tsd_in.keys()) >= {"a", "b"}
        np.testing.assert_array_equal(tsd_in["a"].value, np.ones(50))

    def test_attrs_preserved(self, tmp_path):
        path = tmp_path / "attrs.zarr"
        ts = TimeSeries(np.ones(10), t0=12345.0, sample_rate=256, name="x", unit="m")
        TimeSeriesDict({"x": ts}).write(str(path), format="zarr")

        tsd = TimeSeriesDict.read(str(path), format="zarr")
        assert np.isclose(tsd["x"].t0.value, 12345.0)
        assert np.isclose(tsd["x"].sample_rate.value, 256)

    def test_fallback_dt_to_sample_rate(self, tmp_path):
        """When only dt is stored (not sample_rate), sample_rate = 1/dt."""
        path = tmp_path / "dt_only.zarr"
        store = zarr.open_group(str(path), mode="w")
        creator = getattr(store, "create_array", None) or store.create_dataset
        arr = creator("sig", data=np.ones(10))
        arr.attrs["dt"] = 0.004  # 250 Hz
        arr.attrs["t0"] = 0.0

        tsd = TimeSeriesDict.read(str(path), format="zarr")
        assert np.isclose(tsd["sig"].sample_rate.value, 250.0)

    def test_default_sample_rate_when_no_attrs(self, tmp_path):
        """When no sample_rate or dt attrs exist, defaults to 1 Hz."""
        path = tmp_path / "bare.zarr"
        store = zarr.open_group(str(path), mode="w")
        creator = getattr(store, "create_array", None) or store.create_dataset
        creator("bare", data=np.ones(5))

        tsd = TimeSeriesDict.read(str(path), format="zarr")
        assert np.isclose(tsd["bare"].sample_rate.value, 1.0)

    def test_unit_override(self, tmp_path):
        path = tmp_path / "unit.zarr"
        ts = TimeSeries(np.ones(10), t0=0, sample_rate=1, name="x", unit="m")
        TimeSeriesDict({"x": ts}).write(str(path), format="zarr")

        tsd = TimeSeriesDict.read(str(path), format="zarr", unit="V")
        assert str(tsd["x"].unit) == "V"

    def test_single_timeseries_read(self, tmp_path):
        from gwexpy.timeseries.io.zarr_ import (
            read_timeseries_zarr,
            write_timeseriesdict_zarr,
        )

        path = tmp_path / "single.zarr"
        ts = TimeSeries(np.arange(20, dtype=np.float64), t0=0, sample_rate=10, name="x")
        write_timeseriesdict_zarr(TimeSeriesDict({"x": ts}), str(path))

        ts_in = read_timeseries_zarr(str(path))
        assert len(ts_in) == 20
