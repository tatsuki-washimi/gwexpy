"""Tests for Zarr reader/writer roundtrip."""

import os

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
if os.environ.get("GWEXPY_ALLOW_ZARR", "") != "1":
    pytest.skip("zarr tests require GWEXPY_ALLOW_ZARR=1", allow_module_level=True)

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


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

    def test_missing_timing_metadata_raises_clear_error(self, tmp_path):
        """Missing timing metadata should fail fast instead of silently assuming 1 Hz."""
        path = tmp_path / "bare.zarr"
        store = zarr.open_group(str(path), mode="w")
        creator = getattr(store, "create_array", None) or store.create_dataset
        creator("bare", data=np.ones(5))

        with pytest.raises(ValueError, match="sample_rate|dt|timing"):
            TimeSeriesDict.read(str(path), format="zarr")

    def test_missing_timing_metadata_recovers_with_sample_rate_override(self, tmp_path):
        """Explicit sample-rate override should recover bare Zarr arrays."""
        path = tmp_path / "bare_sample_rate_override.zarr"
        store = zarr.open_group(str(path), mode="w")
        creator = getattr(store, "create_array", None) or store.create_dataset
        creator("bare", data=np.arange(5, dtype=np.float64))

        tsd = TimeSeriesDict.read(
            str(path),
            format="zarr",
            sample_rate_override=32.0,
        )

        assert np.isclose(tsd["bare"].sample_rate.value, 32.0)
        assert np.isclose(tsd["bare"].dt.value, 1.0 / 32.0)

    def test_missing_timing_metadata_recovers_with_dt_override(self, tmp_path):
        """Explicit dt override should also recover bare Zarr arrays."""
        path = tmp_path / "bare_dt_override.zarr"
        store = zarr.open_group(str(path), mode="w")
        creator = getattr(store, "create_array", None) or store.create_dataset
        creator("bare", data=np.arange(5, dtype=np.float64))

        tsd = TimeSeriesDict.read(
            str(path),
            format="zarr",
            dt_override=0.125,
        )

        assert np.isclose(tsd["bare"].dt.value, 0.125)
        assert np.isclose(tsd["bare"].sample_rate.value, 8.0)

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

    def test_matrix_roundtrip(self, tmp_path):
        path = tmp_path / "matrix.zarr"
        matrix = TimeSeriesMatrix(
            np.arange(24, dtype=np.float64).reshape(2, 2, 6),
            t0=1000000000.0,
            sample_rate=16.0,
        )

        matrix.write(str(path), format="zarr")

        loaded = TimeSeriesMatrix.read(str(path), format="zarr")
        np.testing.assert_allclose(loaded.value, matrix.value)
        assert loaded.shape == matrix.shape
        assert np.isclose(float(loaded.sample_rate.value), 16.0)
