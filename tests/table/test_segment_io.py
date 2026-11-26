"""Tests for SegmentTable.read() promotion (#405), write() HDF5 round-trip (#355),
provenance/schema versioning (#408), and U-4 (#658) policy.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from gwpy.segments import Segment
from gwpy.types import Array as GwpyArray

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesMatrix,
)
from gwexpy.spectrogram import Spectrogram
from gwexpy.table import CellStatus, SegmentCell, SegmentTable, col
from gwexpy.table.segment.io import (
    MAX_SUPPORTED_SCHEMA_VERSION,
    SCHEMA_NAME,
    SCHEMA_VERSION,
    register_payload_serializer,
)
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


class TestReadPromotion:
    def test_read_csv_backward_compat(self, tmp_path):
        csv_path = tmp_path / "segs.csv"
        pd.DataFrame(
            {"start": [10.0, 20.0], "end": [14.0, 24.0], "snr": [5.5, 8.2]}
        ).to_csv(csv_path, index=False)

        st = SegmentTable.read(csv_path)
        assert len(st) == 2
        assert st.row(0)["span"] == Segment(10.0, 14.0)
        assert st["snr"].to_list() == pytest.approx([5.5, 8.2])

        # Also test write(..., format="csv") round-trip
        out_csv = tmp_path / "roundtrip.csv"
        st.write(out_csv)
        st2 = SegmentTable.read(out_csv)
        assert [st2.row(i)["span"] for i in range(2)] == [
            Segment(10.0, 14.0),
            Segment(20.0, 24.0),
        ]

    def test_read_unknown_format_raises(self, tmp_path):
        p = tmp_path / "data.unknown"
        p.write_text("hello", encoding="utf-8")
        with pytest.raises(ValueError, match="Could not determine format"):
            SegmentTable.read(p)
        with pytest.raises(ValueError, match="Unsupported format"):
            SegmentTable.read(p, format="parquet_nope")


class TestHDF5RoundTripAndLazyLoad:
    def test_hdf5_round_trip_all_types_and_lazy_restore(self, tmp_path):
        segs = [Segment(0, 4), Segment(4, 8)]
        st = SegmentTable.from_segments(
            segs,
            snr=[5.0, np.nan],
            label=["alpha", None],
            flag=[True, False],
            count=[10, 20],
            start=[1.0, 2.0],  # intentional collision with default span_cols!
        )
        st.add_column("freq", [10.0, 20.0], unit=u.Hz)

        ts0 = TimeSeries(np.arange(8.0), sample_rate=2.0, t0=0.0, unit="m", name="ch0")
        ts1 = TimeSeries(np.arange(8.0) * 2, sample_rate=2.0, t0=4.0, unit="m")
        st.add_series_column("ts", data=[ts0, ts1], kind="timeseries")

        # Irregular frequency axis
        fs0 = FrequencySeries(
            [1.0, 2.0, 3.0], frequencies=[1.0, 4.0, 10.0], unit="1/Hz", name="irreg"
        )
        st.add_series_column(
            "fs", data=[fs0, SegmentCell.missing()], kind="frequencyseries"
        )

        # Spectrogram
        sp0 = Spectrogram(np.ones((2, 3)), dt=1.0, f0=1.0, df=2.0, unit="1/Hz")
        st.add_series_column(
            "spec", data=[sp0, SegmentCell.masked()], kind="spectrogram"
        )

        # Dicts
        tsd = TimeSeriesDict({"H1": ts0, "L1": ts1})
        st.add_series_column(
            "tsd",
            data=[tsd, SegmentCell.failed(ValueError("bad channel"))],
            kind="timeseriesdict",
        )
        fsd = FrequencySeriesDict({"H1": fs0})
        st.add_series_column(
            "fsd",
            data=[fsd, SegmentCell.not_applicable()],
            kind="frequencyseriesdict",
        )

        # Matrix, Array/Quantity, Scalar
        mat = TimeSeriesMatrix(
            np.arange(12.0).reshape(2, 2, 3),
            dt=0.5,
            t0=0.0,
            rows=["r0", "r1"],
            cols=["c0", "c1"],
        )
        st.add_series_column("mat", data=[mat, SegmentCell()], kind="matrix")
        st.add_series_column(
            "arr",
            data=[u.Quantity([1.0, 2.0], u.m), GwpyArray([3.0, 4.0], unit="s")],
            kind="array",
        )
        st.add_series_column(
            "sc",
            data=[42, SegmentCell.valid(None)],
            kind="scalar",
        )

        # Record filter provenance
        filtered = st.filter(col("count") >= 10, name="keep_all")

        h5_path = tmp_path / "table.h5"
        filtered.write(h5_path)

        # Verify on-disk structure directly
        with h5py.File(h5_path, "r") as f:
            assert f.attrs["gwexpy_schema"] == SCHEMA_NAME
            assert int(f.attrs["gwexpy_schema_version"]) == SCHEMA_VERSION
            assert isinstance(f.attrs["gwexpy_schema_version"], (int, np.integer))
            assert "metadata" in f
            assert "schema" in f
            assert "status" in f
            assert "payload" in f
            assert "provenance" in f
            prov = json.loads(f["provenance"][()])
            assert prov["schema"] == SCHEMA_NAME
            assert prov["schema_version"] == SCHEMA_VERSION
            assert prov["allow_pickle"] is False
            assert prov["used_pickle"] is False
            assert prov["history"][-1]["name"] == "keep_all"

        # Read back lazily
        reloaded = SegmentTable.read(h5_path)
        assert len(reloaded) == 2
        assert reloaded["freq"].unit == u.Hz
        assert reloaded["start"].to_list() == [1.0, 2.0]
        assert np.isnan(reloaded.row(1)["snr"])
        assert reloaded.row(0)["label"] == "alpha"
        assert np.isnan(reloaded.row(1)["label"])

        # Lazy payload restoration check: VALID cells on disk start UNLOADED!
        assert not reloaded._payload["ts"][0].is_loaded()
        assert reloaded._payload["ts"][0].status == CellStatus.UNLOADED
        assert reloaded._payload["fs"][1].status == CellStatus.MISSING
        assert reloaded._payload["spec"][1].status == CellStatus.MASKED
        assert reloaded._payload["tsd"][1].status == CellStatus.FAILED
        assert reloaded._payload["fsd"][1].status == CellStatus.NOT_APPLICABLE
        assert reloaded._payload["mat"][1].status == CellStatus.UNLOADED
        assert reloaded._payload["mat"][1].loader is None

        # On-demand load after read() closed its initial handle
        ts_loaded = reloaded.row(0)["ts"]
        assert reloaded._payload["ts"][0].is_loaded()
        np.testing.assert_allclose(ts_loaded.value, ts0.value)
        assert ts_loaded.unit == u.m

        fs_loaded = reloaded.row(0)["fs"]
        np.testing.assert_allclose(fs_loaded.frequencies.value, [1.0, 4.0, 10.0])

        sp_loaded = reloaded.row(0)["spec"]
        assert sp_loaded.shape == (2, 3)

        tsd_loaded = reloaded.row(0)["tsd"]
        assert list(tsd_loaded.keys()) == ["H1", "L1"]

        fsd_loaded = reloaded.row(0)["fsd"]
        assert list(fsd_loaded.keys()) == ["H1"]

        mat_loaded = reloaded.row(0)["mat"]
        assert isinstance(mat_loaded, TimeSeriesMatrix)
        np.testing.assert_allclose(mat_loaded.value, mat.value)

        arr0 = reloaded.row(0)["arr"]
        arr1 = reloaded.row(1)["arr"]
        assert isinstance(arr0, u.Quantity) and arr0.unit == u.m
        assert isinstance(arr1, GwpyArray) and arr1.unit == u.s

        assert reloaded.row(0)["sc"] == 42
        assert reloaded.row(1)["sc"] is None
        assert reloaded.provenance["history"][-1]["name"] == "keep_all"

    def test_empty_table_round_trip(self, tmp_path):
        st = SegmentTable.from_segments([])
        st.add_series_column("ts", data=[], kind="timeseries")
        p = tmp_path / "empty.hdf5"
        st.write(p)
        r = SegmentTable.read(p)
        assert len(r) == 0
        assert r.columns == ["span", "ts"]

    def test_overwrite_false_and_lazy_write_source(self, tmp_path):
        calls = 0

        def _load():
            nonlocal calls
            calls += 1
            return TimeSeries(np.ones(4), sample_rate=4.0)

        st = SegmentTable.from_segments([Segment(0, 1)])
        st.add_series_column("ts", loader=[_load], kind="timeseries")
        p = tmp_path / "lazy.h5"
        st.write(p)
        assert calls == 1
        assert st._payload["ts"][0].status == CellStatus.UNLOADED

        with pytest.raises(FileExistsError):
            st.write(p, overwrite=False)


class TestPicklePolicyAndRegistry:
    def test_unregistered_object_requires_allow_pickle(self, tmp_path):
        st = SegmentTable.from_segments([Segment(0, 1)])
        st.add_series_column("obj", data=[{"k": [1, 2, 3]}], kind="object")
        p = tmp_path / "obj.h5"
        p.write_bytes(b"sentinel")

        with pytest.raises(ValueError, match="allow_pickle=True"):
            st.write(p)
        # Pre-validation did not truncate existing file
        assert p.read_bytes() == b"sentinel"

        st.write(p, allow_pickle=True)
        with pytest.raises(ValueError, match="allow_pickle=False"):
            SegmentTable.read(p)

        reloaded = SegmentTable.read(p, allow_pickle=True)
        assert reloaded.row(0)["obj"] == {"k": [1, 2, 3]}
        assert reloaded.provenance["used_pickle"] is True

        with pytest.raises(ValueError, match="allow_pickle=False"):
            SegmentTable.read(p, allow_pickle=False)

    def test_custom_payload_serializer(self, tmp_path):
        def _write_pair(grp, val):
            grp.create_dataset("pair", data=np.array(val, dtype=np.int64))
            grp.attrs["serializer"] = "test_pair"

        def _read_pair(grp):
            arr = grp["pair"][:]
            return (int(arr[0]), int(arr[1]))

        register_payload_serializer(
            "test_pair", _write_pair, _read_pair, overwrite=True
        )
        st = SegmentTable.from_segments([Segment(0, 1)])
        st._payload["p"] = [SegmentCell(value=(7, 9))]
        st._schema["p"] = "test_pair"
        p = tmp_path / "custom.h5"
        st.write(p)
        r = SegmentTable.read(p)
        assert r.row(0)["p"] == (7, 9)


class TestU4SchemaVersioningPolicy:
    @pytest.mark.parametrize("version", [0, -1, 1.5, "1", True])
    def test_rejects_invalid_schema_version(self, tmp_path, version):
        path = tmp_path / "invalid_version.h5"
        SegmentTable.from_segments([Segment(0, 1)]).write(path)
        with h5py.File(path, "r+") as handle:
            del handle.attrs["gwexpy_schema_version"]
            handle.attrs["gwexpy_schema_version"] = version
        with pytest.raises(ValueError, match="gwexpy_schema_version"):
            SegmentTable.read(path)

    def test_rejects_newer_schema_version_with_details(self, tmp_path):
        st = SegmentTable.from_segments([Segment(0, 1)], snr=[1.0])
        p = tmp_path / "future.h5"
        st.write(p)

        with h5py.File(p, "r+") as f:
            f.attrs["gwexpy_schema_version"] = 99

        with pytest.raises(ValueError, match="99") as excinfo:
            SegmentTable.read(p)
        assert str(MAX_SUPPORTED_SCHEMA_VERSION) in str(excinfo.value)

    def test_rejects_unknown_gwexpy_attr_and_bad_schema_name(self, tmp_path):
        st = SegmentTable.from_segments([Segment(0, 1)])
        p = tmp_path / "bad_attr.h5"
        st.write(p)

        with h5py.File(p, "r+") as f:
            f.attrs["gwexpy_unexpected"] = "oops"

        with pytest.raises(ValueError, match="gwexpy_unexpected"):
            SegmentTable.read(p)

        p2 = tmp_path / "bad_schema.h5"
        st.write(p2)
        with h5py.File(p2, "r+") as f:
            f.attrs["gwexpy_schema"] = "gwexpy.other"

        with pytest.raises(ValueError, match="gwexpy.other"):
            SegmentTable.read(p2)

    def test_legacy_without_version_warns_and_external_attrs_preserved(self, tmp_path):
        st = SegmentTable.from_segments([Segment(0, 1)], snr=[3.5])
        p = tmp_path / "legacy.h5"
        st.write(p)

        with h5py.File(p, "r+") as f:
            del f.attrs["gwexpy_schema_version"]
            del f.attrs["gwexpy_schema"]
            f.attrs["observatory"] = "KAGRA"
            f.attrs["run_number"] = 4

        with pytest.warns(UserWarning, match="gwexpy_schema_version"):
            loaded = SegmentTable.read(p)

        assert loaded.row(0)["snr"] == 3.5
        assert loaded._external_attrs["observatory"] == "KAGRA"
        assert int(loaded._external_attrs["run_number"]) == 4

        p2 = tmp_path / "resaved.h5"
        loaded.write(p2)
        with h5py.File(p2, "r") as f:
            assert f.attrs["gwexpy_schema"] == SCHEMA_NAME
            assert int(f.attrs["gwexpy_schema_version"]) == SCHEMA_VERSION
            assert f.attrs["observatory"] == "KAGRA"
            assert int(f.attrs["run_number"]) == 4
