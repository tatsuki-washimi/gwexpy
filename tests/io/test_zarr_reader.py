"""Tests for Zarr reader/writer roundtrip."""

import json
import os
from collections import OrderedDict

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
if os.environ.get("GWEXPY_ALLOW_ZARR", "") != "1":
    pytest.skip("zarr tests require GWEXPY_ALLOW_ZARR=1", allow_module_level=True)

from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


class TestZarrRoundtrip:
    def _write_matrix_array(
        self,
        store,
        name,
        data,
        *,
        sample_rate=16.0,
        t0=1000000000.0,
        unit=None,
        row_key=None,
        col_key=None,
        row_index=None,
        col_index=None,
        include_row_key=True,
        include_col_key=True,
    ):
        creator = getattr(store, "create_array", None) or store.create_dataset
        arr = creator(name, data=np.asarray(data, dtype=np.float64))
        arr.attrs["sample_rate"] = float(sample_rate)
        arr.attrs["dt"] = 1.0 / float(sample_rate)
        arr.attrs["t0"] = float(t0)
        if unit is not None:
            arr.attrs["unit"] = str(unit)

        if include_row_key:
            arr.attrs["gwexpy_row_key"] = json.dumps(row_key)
            arr.attrs["gwexpy_key_format"] = "json"
            if row_index is not None:
                arr.attrs["gwexpy_row_index"] = int(row_index)

        if include_col_key:
            arr.attrs["gwexpy_col_key"] = json.dumps(col_key)
            arr.attrs["gwexpy_key_format"] = "json"
            if col_index is not None:
                arr.attrs["gwexpy_col_index"] = int(col_index)

        return arr

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

    def test_matrix_roundtrip_1col_integer_row_keys(self, tmp_path):
        from gwexpy.types.metadata import MetaData, MetaDataDict

        path = tmp_path / "matrix_int_rows_1col.zarr"
        data = np.arange(12, dtype=np.float64).reshape(2, 1, 6)
        matrix = TimeSeriesMatrix(data, t0=1000000000.0, sample_rate=16.0)
        matrix.rows = MetaDataDict(
            OrderedDict({10: MetaData(), 11: MetaData()}),
            expected_size=2,
            key_prefix="row",
        )

        matrix.write(str(path), format="zarr")

        loaded = TimeSeriesMatrix.read(str(path), format="zarr")
        assert list(loaded.row_keys()) == [10, 11]
        assert loaded.shape == (2, 1, 6)
        np.testing.assert_allclose(loaded.value, data)

        store = zarr.open_group(str(path), mode="r")
        sample_array = next(
            arr
            for arr in (store[key] for key in store.keys())
            if arr.attrs.get("gwexpy_row_index") == 0
        )
        assert sample_array.attrs.get("gwexpy_row_key") == json.dumps(10)
        assert sample_array.attrs.get("gwexpy_col_key") in (
            json.dumps("col0"),
            json.dumps(list(loaded.col_keys())[0]),
        )
        assert sample_array.attrs.get("gwexpy_key_format") == "json"
        assert sample_array.attrs.get("gwexpy_row_index") == 0
        assert sample_array.attrs.get("gwexpy_col_index") == 0

    def test_matrix_roundtrip_nested_tuple_keys(self, tmp_path):
        from gwexpy.types.metadata import MetaData, MetaDataDict

        row_key = (("H1", "X"), "raw")
        col_key = (("L1", "Y"), "cal")
        path = tmp_path / "matrix_nested_keys.zarr"
        data = np.arange(6, dtype=np.float64).reshape(1, 1, 6)
        matrix = TimeSeriesMatrix(data, t0=1000000000.0, sample_rate=16.0)
        matrix.rows = MetaDataDict(
            OrderedDict({row_key: MetaData()}),
            expected_size=1,
            key_prefix="row",
        )
        matrix.cols = MetaDataDict(
            OrderedDict({col_key: MetaData()}),
            expected_size=1,
            key_prefix="col",
        )

        matrix.write(str(path), format="zarr")

        loaded = TimeSeriesMatrix.read(str(path), format="zarr")
        assert list(loaded.row_keys()) == [row_key]
        assert list(loaded.col_keys()) == [col_key]
        np.testing.assert_allclose(loaded.value, data)

    def test_incomplete_matrix_schema_raises(self, tmp_path):
        path = tmp_path / "matrix_incomplete.zarr"
        store = zarr.open_group(str(path), mode="w")

        self._write_matrix_array(
            store,
            "missing_col",
            np.arange(6, dtype=np.float64),
            row_key=0,
            include_col_key=False,
        )

        with pytest.raises(ValueError, match="matrix|schema|malformed"):
            TimeSeriesMatrix.read(str(path), format="zarr")

    def test_duplicate_matrix_attributes_raises(self, tmp_path):
        path = tmp_path / "matrix_duplicate_attrs.zarr"
        store = zarr.open_group(str(path), mode="w")

        self._write_matrix_array(
            store,
            "cell00_a",
            np.arange(6, dtype=np.float64),
            row_key=0,
            col_key=0,
            row_index=0,
            col_index=0,
        )
        self._write_matrix_array(
            store,
            "cell00_b",
            np.arange(6, dtype=np.float64),
            row_key=0,
            col_key=0,
            row_index=0,
            col_index=0,
        )

        with pytest.raises(ValueError, match="duplicate|already|conflict"):
            TimeSeriesMatrix.read(str(path), format="zarr")

    def test_mismatched_sample_rate_raises(self, tmp_path):
        path = tmp_path / "matrix_mismatched_sample_rate.zarr"
        store = zarr.open_group(str(path), mode="w")

        self._write_matrix_array(
            store,
            "m00",
            np.arange(6, dtype=np.float64),
            row_key=0,
            col_key=0,
            row_index=0,
            col_index=0,
            sample_rate=16.0,
        )
        self._write_matrix_array(
            store,
            "m10",
            np.arange(6, dtype=np.float64) * 2,
            row_key=1,
            col_key=0,
            row_index=1,
            col_index=0,
            sample_rate=32.0,
        )

        with pytest.raises(ValueError, match="sample_rate values"):
            TimeSeriesMatrix.read(str(path), format="zarr")

    def test_mismatched_t0_raises(self, tmp_path):
        path = tmp_path / "matrix_mismatched_t0.zarr"
        store = zarr.open_group(str(path), mode="w")

        self._write_matrix_array(
            store,
            "m00",
            np.arange(6, dtype=np.float64),
            row_key=0,
            col_key=0,
            row_index=0,
            col_index=0,
            t0=0.0,
        )
        self._write_matrix_array(
            store,
            "m10",
            np.arange(6, dtype=np.float64),
            row_key=1,
            col_key=0,
            row_index=1,
            col_index=0,
            t0=1.0,
        )

        with pytest.raises(ValueError, match="matching t0 values"):
            TimeSeriesMatrix.read(str(path), format="zarr")

    def test_mismatched_unit_raises(self, tmp_path):
        path = tmp_path / "matrix_mismatched_unit.zarr"
        store = zarr.open_group(str(path), mode="w")

        self._write_matrix_array(
            store,
            "m00",
            np.arange(6, dtype=np.float64),
            row_key=0,
            col_key=0,
            row_index=0,
            col_index=0,
            unit="m",
        )
        self._write_matrix_array(
            store,
            "m10",
            np.arange(6, dtype=np.float64),
            row_key=1,
            col_key=0,
            row_index=1,
            col_index=0,
            unit="V",
        )

        with pytest.raises(ValueError, match="matching units"):
            TimeSeriesMatrix.read(str(path), format="zarr")

        with pytest.raises(ValueError, match="matching units"):
            TimeSeriesMatrix.read(str(path), format="zarr", unit="A")

    def test_unequal_sample_length_raises(self, tmp_path):
        path = tmp_path / "matrix_mismatched_length.zarr"
        store = zarr.open_group(str(path), mode="w")

        self._write_matrix_array(
            store,
            "m00",
            np.arange(6, dtype=np.float64),
            row_key=0,
            col_key=0,
            row_index=0,
            col_index=0,
        )
        self._write_matrix_array(
            store,
            "m10",
            np.arange(5, dtype=np.float64),
            row_key=1,
            col_key=0,
            row_index=1,
            col_index=0,
        )

        with pytest.raises(ValueError, match="matching sample counts"):
            TimeSeriesMatrix.read(str(path), format="zarr")

    def test_read_falls_back_to_dict_when_no_matrix_attrs(self, tmp_path):
        path = tmp_path / "matrix_dict_fallback.zarr"
        store = zarr.open_group(str(path), mode="w")
        creator = getattr(store, "create_array", None) or store.create_dataset
        for i in range(2):
            arr = creator(f"channel_{i}", data=np.ones(5, dtype=np.float64) * (i + 1))
            arr.attrs["sample_rate"] = 16.0
            arr.attrs["dt"] = 1.0 / 16.0
            arr.attrs["t0"] = 1000000000.0

        matrix = TimeSeriesMatrix.read(str(path), format="zarr")
        assert matrix.shape == (2, 1, 5)
        assert np.isclose(float(matrix.sample_rate.value), 16.0)
        row_values = sorted(float(row[0, 0]) for row in matrix.value)
        assert row_values == [1.0, 2.0]

    def test_matrix_roundtrip_preserves_integer_keys(self, tmp_path):
        from collections import OrderedDict

        from gwexpy.types.metadata import MetaData, MetaDataDict

        path = tmp_path / "matrix_int_keys.zarr"
        data = np.arange(24, dtype=np.float64).reshape(2, 2, 6)
        matrix = TimeSeriesMatrix(data, t0=1000000000.0, sample_rate=16.0)
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

        matrix.write(str(path), format="zarr")

        loaded = TimeSeriesMatrix.read(str(path), format="zarr")
        assert list(loaded.row_keys()) == [0, 1]
        assert list(loaded.col_keys()) == [10, 20]
        np.testing.assert_allclose(loaded.value, data)
