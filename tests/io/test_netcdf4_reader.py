"""Tests for NetCDF4 reader/writer roundtrip."""

from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


class TestNetCDF4Roundtrip:
    @pytest.mark.parametrize("fmt", ("nc", "netcdf4"))
    def test_canonical_and_legacy_aliases_roundtrip(self, tmp_path, fmt):
        path = tmp_path / f"alias_{fmt}.nc"
        data = np.arange(8, dtype=np.float64)
        ts = TimeSeries(data, t0=1234567890, dt=0.25, name="signal", unit="m")

        TimeSeriesDict({"signal": ts}).write(str(path), format=fmt)

        tsd = TimeSeriesDict.read(str(path), format=fmt)
        np.testing.assert_allclose(tsd["signal"].value, data)
        assert np.isclose(tsd["signal"].dt.value, 0.25)

    def test_single_variable_roundtrip(self, tmp_path):
        path = tmp_path / "test.nc"
        data = np.arange(100, dtype=np.float64)
        ts = TimeSeries(data, t0=1000000000, dt=0.01, name="signal", unit="m")

        # Write
        tsd_out = TimeSeriesDict({"signal": ts})
        tsd_out.write(str(path), format="nc")

        # Read
        tsd_in = TimeSeriesDict.read(str(path), format="nc")
        assert "signal" in tsd_in
        np.testing.assert_allclose(tsd_in["signal"].value, data)
        assert np.isclose(tsd_in["signal"].dt.value, 0.01)

    def test_multi_variable(self, tmp_path):
        path = tmp_path / "multi.nc"
        tsd_out = TimeSeriesDict({
            "ch1": TimeSeries(np.ones(50), t0=0, dt=0.1, name="ch1"),
            "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.1, name="ch2"),
        })
        tsd_out.write(str(path), format="nc")

        tsd_in = TimeSeriesDict.read(str(path), format="nc")
        assert set(tsd_in.keys()) >= {"ch1", "ch2"}

    @pytest.mark.parametrize("fmt", ("nc", "netcdf4"))
    def test_matrix_roundtrip(self, tmp_path, fmt):
        path = tmp_path / f"matrix_{fmt}.nc"
        matrix = TimeSeriesMatrix(
            np.arange(24, dtype=np.float64).reshape(2, 2, 6),
            t0=1234567890.0,
            dt=0.25,
        )

        matrix.write(str(path), format=fmt)

        loaded = TimeSeriesMatrix.read(str(path), format=fmt)
        np.testing.assert_allclose(loaded.value, matrix.value)
        assert loaded.shape == matrix.shape
        assert np.isclose(float(loaded.dt.value), 0.25)

    def test_unit_override(self, tmp_path):
        path = tmp_path / "unit.nc"
        ts = TimeSeries(np.ones(10), t0=0, dt=1.0, name="x", unit="m")
        TimeSeriesDict({"x": ts}).write(str(path), format="nc")

        tsd = TimeSeriesDict.read(str(path), format="nc", unit="V")
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

            tsd = TimeSeriesDict.read(str(path), format="nc")
            assert "signal" in tsd
            assert len(tsd["signal"]) == 10

    def test_single_timeseries_read(self, tmp_path):
        from gwexpy.timeseries.io.netcdf4_ import read_timeseries_netcdf4

        path = tmp_path / "single.nc"
        ts = TimeSeries(np.arange(20, dtype=np.float64), t0=0, dt=0.5, name="x")
        TimeSeriesDict({"x": ts}).write(str(path), format="nc")

        ts_in = read_timeseries_netcdf4(str(path))
        assert len(ts_in) == 20

    def test_empty_dataset_raises(self, tmp_path):
        path = tmp_path / "empty.nc"
        ds = xr.Dataset()
        ds.to_netcdf(str(path))
        with pytest.raises((ValueError, KeyError)):
            TimeSeriesDict.read(str(path), format="nc")

    def test_matrix_roundtrip_numpy_scalar_keys(self, tmp_path):
        """numpy.int64/float64 row/col keys must be serializable (not raise TypeError)."""
        from collections import OrderedDict

        from gwexpy.types.metadata import MetaData, MetaDataDict

        path = tmp_path / "matrix_np_keys.nc"
        data = np.arange(24, dtype=np.float64).reshape(2, 2, 6)
        matrix = TimeSeriesMatrix(data, t0=1234567890.0, dt=0.25)
        matrix.rows = MetaDataDict(
            OrderedDict({np.int64(0): MetaData(), np.int64(1): MetaData()}),
            expected_size=2,
            key_prefix="row",
        )
        matrix.cols = MetaDataDict(
            OrderedDict({np.int64(10): MetaData(), np.int64(20): MetaData()}),
            expected_size=2,
            key_prefix="col",
        )

        # Should not raise TypeError for numpy scalar keys
        matrix.write(str(path), format="nc")

        loaded = TimeSeriesMatrix.read(str(path), format="nc")
        row_keys = list(loaded.row_keys())
        col_keys = list(loaded.col_keys())
        assert row_keys == [0, 1]
        assert col_keys == [10, 20]
        np.testing.assert_allclose(loaded.value, data)

    def test_matrix_roundtrip_preserves_integer_keys(self, tmp_path):
        """Integer row/col keys must survive a write → read roundtrip."""
        from collections import OrderedDict

        from gwexpy.types.metadata import MetaData, MetaDataDict

        path = tmp_path / "matrix_int_keys.nc"
        data = np.arange(24, dtype=np.float64).reshape(2, 2, 6)
        matrix = TimeSeriesMatrix(data, t0=1234567890.0, dt=0.25)
        matrix.rows = MetaDataDict(
            OrderedDict({0: MetaData(), 1: MetaData()}),
            expected_size=2,
            key_prefix="row",
        )
        matrix.cols = MetaDataDict(
            OrderedDict({10: MetaData(), 20: MetaData()}),
            expected_size=2,
            key_prefix="col",
        )

        matrix.write(str(path), format="nc")

        loaded = TimeSeriesMatrix.read(str(path), format="nc")
        row_keys = list(loaded.row_keys())
        col_keys = list(loaded.col_keys())

        assert row_keys == [0, 1], f"Expected [0, 1], got {row_keys}"
        assert col_keys == [10, 20], f"Expected [10, 20], got {col_keys}"
        assert all(isinstance(k, int) for k in row_keys), "Row keys should be int"
        assert all(isinstance(k, int) for k in col_keys), "Col keys should be int"
        np.testing.assert_allclose(loaded.value, data)


class TestNetCDF4FixtureContract:
    """Verify that the generated test.nc fixture satisfies the reader contract.

    These tests exist to prevent fixture-contract drift: if generate_netcdf4()
    is ever modified to drop the time coordinate variable, these will fail fast.
    """

    @pytest.fixture(scope="class")
    def fixture_path(self):
        p = Path(__file__).resolve().parents[1] / "fixtures" / "data" / "test.nc"
        if not p.exists():
            pytest.skip("test.nc fixture not generated")
        return p

    def test_timeseriesdict_reads_fixture(self, fixture_path):
        tsd = TimeSeriesDict.read(str(fixture_path), format="nc")
        assert "ch1" in tsd
        assert len(tsd["ch1"]) == 100

    def test_timeseries_reads_fixture(self, fixture_path):
        ts = TimeSeries.read(str(fixture_path), format="nc")
        assert len(ts) == 100

    def test_timeseriesmatrix_reads_fixture(self, fixture_path):
        m = TimeSeriesMatrix.read(str(fixture_path), format="nc")
        assert len(m) > 0
