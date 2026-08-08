"""Tests for NetCDF4 reader/writer roundtrip."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
pytest.importorskip("netCDF4")

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

FIXTURE_NETCDF = "tests/fixtures/data/test.nc"
FIXTURE_V0112_LEGACY = Path("tests/fixtures/data/v0.1.12-legacy.nc")
FIXTURE_V0112_LEGACY_SHA256 = (
    "3ccc0cf0cb7baf444c9f90b91d98c32f24c9880fd0d2987af1a105c457911082"
)


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
        tsd_out = TimeSeriesDict(
            {
                "ch1": TimeSeries(np.ones(50), t0=0, dt=0.1, name="ch1"),
                "ch2": TimeSeries(np.zeros(50), t0=0, dt=0.1, name="ch2"),
            }
        )
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

    def test_v2_roundtrip_preserves_dtype_and_exact_timing(self, tmp_path):
        """The v2 schema stores the timing contract without datetime rounding."""
        path = tmp_path / "v2_exact.nc"
        t0 = 1234567890.0 + 1.0 / 3.0
        dt = 0.1
        values = np.array([2**53 + 1, 2**53 + 3, 2**53 + 5], dtype=np.int64)
        source = TimeSeries(values, t0=t0, dt=dt, name="integer")

        TimeSeriesDict({"integer": source}).write(str(path), format="nc")

        with xr.open_dataset(path, decode_times=False) as ds:
            assert ds.attrs["gwexpy_netcdf_schema_version"] == 2
            assert ds.attrs["gwexpy_t0_float_hex"] == float(t0).hex()
            assert ds.attrs["gwexpy_dt_numerator"] == str(dt.as_integer_ratio()[0])
            assert ds.attrs["gwexpy_dt_denominator"] == str(dt.as_integer_ratio()[1])
            assert ds.attrs["gwexpy_axis_encoding"] == "t(i)=t0+i*dt"
            assert ds["sample"].dtype == np.dtype("int64")
            assert ds["integer"].dtype == np.dtype("int64")

        loaded = TimeSeriesDict.read(str(path), format="nc")
        assert float(loaded["integer"].t0.value).hex() == float(t0).hex()
        assert float(loaded["integer"].dt.value).hex() == dt.hex()
        assert loaded["integer"].dtype == values.dtype
        np.testing.assert_array_equal(loaded["integer"].value, values)

    @pytest.mark.parametrize("reader", (TimeSeriesDict, TimeSeriesMatrix))
    def test_v2_reader_rejects_noncontiguous_sample_coordinate(self, tmp_path, reader):
        """Schema v2 accepts only the exact ``0..N-1`` int64 sample axis."""
        path = tmp_path / "noncontiguous-sample.nc"
        source = TimeSeries(np.arange(3), t0=1234567890.0, dt=0.1, name="x")
        TimeSeriesDict({"x": source}).write(path, format="nc")

        with xr.open_dataset(path, decode_times=False) as original:
            tampered = original.load().assign_coords(
                sample=np.array([0, 2, 3], dtype=np.int64)
            )
        tampered.to_netcdf(path, mode="w")

        with pytest.raises(ValueError, match=r"0\.\.N-1"):
            reader.read(path, format="nc")

    def test_v2_channel_selection_is_ordered_and_fail_closed(self, tmp_path):
        """Selection validates names before values are materialized."""
        path = tmp_path / "channels.nc"
        tsd = TimeSeriesDict(
            {
                "z": TimeSeries(np.arange(3), t0=0, dt=1),
                "a": TimeSeries(np.arange(3), t0=0, dt=1),
            }
        )
        tsd.write(path, format="nc")

        default = TimeSeriesDict.read(path, format="nc")
        assert list(default) == ["a", "z"]
        requested = TimeSeriesDict.read(path, format="nc", channels=["z", "a"])
        assert list(requested) == ["z", "a"]
        with pytest.raises(ValueError, match="duplicate"):
            TimeSeriesDict.read(path, format="nc", channels=["a", "a"])
        with pytest.raises(ValueError, match="not found"):
            TimeSeriesDict.read(path, format="nc", channels=["missing"])
        with pytest.raises(ValueError, match="exactly one"):
            TimeSeries.read(path, format="nc")

    @pytest.mark.parametrize("reader", (TimeSeries, TimeSeriesDict, TimeSeriesMatrix))
    def test_bounded_v2_read_matches_the_exact_positional_crop(self, tmp_path, reader):
        """Reader bounds use the same large-GPS crop contract as direct reads."""
        path = tmp_path / "bounded.nc"
        t0 = 1_234_567_890.1234567
        dt = 1.0 / 30.0
        source = TimeSeries(np.arange(768), t0=t0, dt=dt, name="x")
        if reader is TimeSeriesMatrix:
            TimeSeriesMatrix(source.value.reshape(1, 1, -1), t0=t0, dt=dt).write(
                path, format="nc"
            )
        else:
            TimeSeriesDict({"x": source}).write(path, format="nc")
        start = t0 + 100 * dt
        end = t0 + 600 * dt

        loaded = reader.read(path, format="nc", start=start, end=end)

        if reader is TimeSeries:
            values = loaded.value
        elif reader is TimeSeriesDict:
            values = loaded["x"].value
        else:
            values = loaded.value[0, 0]
        np.testing.assert_array_equal(values, source.value[100:600])

    def test_matrix_reader_preserves_a_v2_dict_axis_during_bounded_read(self, tmp_path):
        """A dict-shaped v2 file must not be realigned through float timestamps."""
        path = tmp_path / "bounded-dict-as-matrix.nc"
        t0 = 1_234_567_890.1234567
        dt = 1.0 / 30.0
        source = TimeSeries(np.arange(768), t0=t0, dt=dt, name="x")
        TimeSeriesDict({"x": source}).write(path, format="nc")

        loaded = TimeSeriesMatrix.read(
            path,
            format="nc",
            start=t0 + 100 * dt,
            end=t0 + 600 * dt,
        )
        expected = source[100:600]

        assert loaded.shape == (1, 1, 500)
        np.testing.assert_array_equal(loaded.value[0, 0], expected.value)
        assert float(loaded.t0.value).hex() == float(expected.t0.value).hex()
        assert float(loaded.dt.value).hex() == float(expected.dt.value).hex()

    def test_v2_writer_rejects_unsupported_dtype_before_creating_target(self, tmp_path):
        path = tmp_path / "unsupported.nc"
        invalid = TimeSeries(np.array([True, False]), t0=0, dt=1, name="flag")

        with pytest.raises(TypeError, match="unsupported dtype"):
            TimeSeriesDict({"flag": invalid}).write(path, format="nc")
        assert not path.exists()

    def test_v2_writer_rejects_heterogeneous_axes_before_creating_target(
        self, tmp_path
    ):
        path = tmp_path / "heterogeneous.nc"
        tsd = TimeSeriesDict(
            {
                "a": TimeSeries(np.arange(3), t0=0, dt=0.1),
                "b": TimeSeries(np.arange(3), t0=0, dt=0.2),
            }
        )

        with pytest.raises(ValueError, match="identical t0 and dt"):
            tsd.write(path, format="nc")
        assert not path.exists()

    def test_legacy_netcdf_warns_once(self, tmp_path):
        path = tmp_path / "legacy.nc"
        xr.Dataset(
            {"x": xr.DataArray(np.arange(3), dims=["time"])},
            coords={"time": np.arange(3, dtype=float)},
        ).to_netcdf(path)

        with pytest.warns(RuntimeWarning, match="unversioned legacy") as recorded:
            TimeSeriesDict.read(path, format="nc")
        assert len(recorded) == 1

    @pytest.mark.parametrize("reader", (TimeSeries, TimeSeriesDict, TimeSeriesMatrix))
    def test_v0112_legacy_fixture_warns_once_per_public_read(self, reader):
        """The immutable v0.1.12 fixture retains its one-warning contract."""
        assert sha256(FIXTURE_V0112_LEGACY.read_bytes()).hexdigest() == (
            FIXTURE_V0112_LEGACY_SHA256
        )

        with pytest.warns(RuntimeWarning, match="unversioned legacy") as recorded:
            loaded = reader.read(FIXTURE_V0112_LEGACY, format="nc")

        assert isinstance(loaded, reader)
        assert len(recorded) == 1

    @pytest.mark.parametrize("reader", (TimeSeries, TimeSeriesDict, TimeSeriesMatrix))
    def test_multi_source_legacy_netcdf_warns_once_per_top_level_read(
        self, tmp_path, reader
    ):
        """One public read of several legacy files must emit one warning."""
        first = tmp_path / "legacy-first.nc"
        second = tmp_path / "legacy-second.nc"
        for path, samples in ((first, np.arange(2.0)), (second, np.arange(2.0, 4.0))):
            xr.Dataset(
                {"x": xr.DataArray(samples, dims=["time"])},
                coords={"time": samples},
            ).to_netcdf(path)

        with pytest.warns(RuntimeWarning, match="unversioned legacy") as recorded:
            loaded = reader.read([str(first), str(second)], format="nc")

        assert isinstance(loaded, reader)
        legacy_warnings = [
            warning
            for warning in recorded
            if str(warning.message)
            == "Reading unversioned legacy NetCDF; timing precision is limited."
        ]
        assert len(legacy_warnings) == 1

    def test_bundled_fixture_has_time_coordinate(self):
        with xr.open_dataset(FIXTURE_NETCDF) as ds:
            assert "time" in ds.coords
            np.testing.assert_allclose(np.diff(ds["time"].values), 0.1, atol=1e-6)

    @pytest.mark.parametrize("cls", (TimeSeries, TimeSeriesDict, TimeSeriesMatrix))
    def test_bundled_fixture_satisfies_time_coordinate_contract(self, cls):
        loaded = cls.read(FIXTURE_NETCDF, format="nc")
        assert isinstance(loaded, cls)

        if cls is TimeSeries:
            assert len(loaded) == 100
            assert np.isclose(float(loaded.dt.value), 0.1)
        elif cls is TimeSeriesDict:
            assert "ch1" in loaded
            assert len(loaded["ch1"]) == 100
            assert np.isclose(float(loaded["ch1"].dt.value), 0.1)
        else:
            assert loaded.shape[-1] == 100
            assert np.isclose(float(loaded.dt.value), 0.1)

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

    def test_matrix_multi_file_preserves_row_col_keys(self, tmp_path):
        """Reading a list of NetCDF4 matrix files must preserve gwexpy_row/col_key (#443)."""
        from collections import OrderedDict

        from gwexpy.types.metadata import MetaData, MetaDataDict

        seg1 = TimeSeriesMatrix(
            np.arange(12, dtype=np.float64).reshape(2, 1, 6),
            t0=1000000000.0,
            dt=0.25,
        )
        seg2 = TimeSeriesMatrix(
            np.arange(12, 24, dtype=np.float64).reshape(2, 1, 6),
            t0=1000000000.0 + 6 * 0.25,
            dt=0.25,
        )
        for seg in (seg1, seg2):
            seg.rows = MetaDataDict(
                OrderedDict({"H1": MetaData(), "L1": MetaData()}),
                expected_size=2,
                key_prefix="row",
            )

        path1 = tmp_path / "seg1.nc"
        path2 = tmp_path / "seg2.nc"
        seg1.write(str(path1), format="nc")
        seg2.write(str(path2), format="nc")

        merged = TimeSeriesMatrix.read([str(path1), str(path2)], format="nc")

        assert list(merged.row_keys()) == ["H1", "L1"], (
            f"row keys lost after multi-file read: {list(merged.row_keys())}"
        )
        assert merged.shape[-1] == 12
