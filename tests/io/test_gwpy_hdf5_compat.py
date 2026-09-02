from __future__ import annotations

import h5py
import numpy as np
import pytest
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
from gwpy.spectrogram import Spectrogram as GwpySpectrogram
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwpy.timeseries import TimeSeriesDict as GwpyTimeSeriesDict

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesList,
)
from gwexpy.io.hdf5_collection import (
    LAYOUT_DATASET,
    read_hdf5_keymap,
    read_hdf5_order,
    write_hdf5_manifest,
)
from gwexpy.spectrogram import Spectrogram, SpectrogramDict, SpectrogramList
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList


def test_gwpy_reads_timeseriesdict_hdf5(tmp_path):
    ts1 = TimeSeries(np.arange(4.0), sample_rate=2.0, t0=1.0, unit="m", name="A")
    ts2 = TimeSeries(np.arange(4.0) * 2, sample_rate=2.0, t0=1.0, unit="m", name="B")
    tsd = TimeSeriesDict({"H1:TEST": ts1, "L1:TEST": ts2})

    outfile = tmp_path / "tsd_gwpy.h5"
    tsd.write(outfile, format="hdf5", layout="dataset")

    gwpy_tsd = GwpyTimeSeriesDict.read(outfile, format="hdf5")
    for gwpy_ts, expected in zip(gwpy_tsd.values(), tsd.values()):
        np.testing.assert_allclose(gwpy_ts.value, expected.value)
        assert str(gwpy_ts.unit) == str(expected.unit)


def test_timeseriesdict_hdf5_append_preserves_existing_entries(tmp_path):
    old = TimeSeries(
        np.arange(4.0), sample_rate=2.0, t0=1.0, unit="m", name="old series"
    )
    new = TimeSeries(
        np.arange(4.0) + 10,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="new series",
    )

    outfile = tmp_path / "tsd_append.h5"
    TimeSeriesDict({"old": old}).write(
        outfile, format="hdf5", layout="dataset"
    )
    TimeSeriesDict({"new": new}).write(
        outfile, format="hdf5", layout="dataset", append=True
    )

    with h5py.File(outfile, "r") as h5f:
        assert set(h5f.keys()) == {"old", "new"}
        assert read_hdf5_keymap(h5f) == {"old": "old", "new": "new"}
        assert read_hdf5_order(h5f) == ["old", "new"]

    gwpy_tsd = GwpyTimeSeriesDict.read(outfile, format="hdf5")
    assert set(gwpy_tsd) == {"old", "new"}
    np.testing.assert_allclose(gwpy_tsd["old"].value, old.value)
    np.testing.assert_allclose(gwpy_tsd["new"].value, new.value)
    assert gwpy_tsd["old"].name == old.name
    assert gwpy_tsd["new"].name == new.name

    gwexpy_tsd = TimeSeriesDict.read(outfile, format="hdf5")
    assert list(gwexpy_tsd) == ["old", "new"]
    np.testing.assert_allclose(gwexpy_tsd["old"].value, old.value)
    np.testing.assert_allclose(gwexpy_tsd["new"].value, new.value)
    assert gwexpy_tsd["old"].name == old.name
    assert gwexpy_tsd["new"].name == new.name


def test_timeseriesdict_hdf5_append_takes_precedence_over_overwrite(tmp_path):
    old = TimeSeries(
        np.arange(3.0), sample_rate=2.0, t0=1.0, unit="m", name="old series"
    )
    new = TimeSeries(
        np.arange(3.0) + 10,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="new series",
    )

    outfile = tmp_path / "tsd_append_overwrite.h5"
    TimeSeriesDict({"old": old}).write(
        outfile, format="hdf5", layout="dataset"
    )
    TimeSeriesDict({"new": new}).write(
        outfile,
        format="hdf5",
        layout="dataset",
        append=True,
        overwrite=True,
    )

    with h5py.File(outfile, "r") as h5f:
        assert set(h5f) == {"old", "new"}
        assert read_hdf5_keymap(h5f) == {"old": "old", "new": "new"}
        assert read_hdf5_order(h5f) == ["old", "new"]

    result = TimeSeriesDict.read(outfile, format="hdf5")
    assert list(result) == ["old", "new"]
    np.testing.assert_allclose(result["old"].value, old.value)
    np.testing.assert_allclose(result["new"].value, new.value)
    assert result["old"].name == old.name
    assert result["new"].name == new.name


@pytest.mark.parametrize("mode", ["w", "w-", "x"])
def test_timeseriesdict_hdf5_append_rejects_create_modes_without_mutation(
    tmp_path, mode
):
    old = TimeSeries(
        np.arange(3.0), sample_rate=2.0, t0=1.0, unit="m", name="old series"
    )
    new = TimeSeries(
        np.arange(3.0) + 10,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="new series",
    )

    outfile = tmp_path / f"tsd_append_mode_{mode}.h5"
    TimeSeriesDict({"old": old}).write(
        outfile, format="hdf5", layout="dataset"
    )
    original_bytes = outfile.read_bytes()

    with pytest.raises(ValueError, match="append=True.*mode"):
        TimeSeriesDict({"new": new}).write(
            outfile,
            format="hdf5",
            layout="dataset",
            append=True,
            mode=mode,
        )

    assert outfile.read_bytes() == original_bytes
    with h5py.File(outfile, "r") as h5f:
        assert set(h5f) == {"old"}
        assert read_hdf5_keymap(h5f) == {"old": "old"}
        assert read_hdf5_order(h5f) == ["old"]
    result = TimeSeriesDict.read(outfile, format="hdf5")
    assert list(result) == ["old"]
    np.testing.assert_allclose(result["old"].value, old.value)
    assert result["old"].name == old.name


@pytest.mark.parametrize("mode", ["a", "r+"])
def test_timeseriesdict_hdf5_merge_mode_preflights_existing_logical_keys(
    tmp_path, mode
):
    old = TimeSeries(
        np.arange(3.0), sample_rate=2.0, t0=1.0, unit="m", name="old series"
    )
    fresh = TimeSeries(
        np.arange(3.0) + 10,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="fresh series",
    )
    replacement = TimeSeries(
        np.arange(3.0) + 20,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="replacement series",
    )

    outfile = tmp_path / f"tsd_merge_duplicate_{mode}.h5"
    TimeSeriesDict({"old": old}).write(
        outfile, format="hdf5", layout="dataset"
    )

    with pytest.raises(ValueError, match="logical key"):
        TimeSeriesDict({"fresh": fresh, "old": replacement}).write(
            outfile, format="hdf5", layout="dataset", mode=mode
        )

    with h5py.File(outfile, "r") as h5f:
        assert set(h5f) == {"old"}
        assert read_hdf5_keymap(h5f) == {"old": "old"}
        assert read_hdf5_order(h5f) == ["old"]
        np.testing.assert_allclose(h5f["old"][()], old.value)
    result = TimeSeriesDict.read(outfile, format="hdf5")
    assert list(result) == ["old"]
    np.testing.assert_allclose(result["old"].value, old.value)
    assert result["old"].name == old.name


@pytest.mark.parametrize("mode", ["a", "r+"])
def test_timeseriesdict_hdf5_merge_mode_preserves_distinct_logical_keys(
    tmp_path, mode
):
    old = TimeSeries(
        np.arange(3.0), sample_rate=2.0, t0=1.0, unit="m", name="old series"
    )
    new = TimeSeries(
        np.arange(3.0) + 10,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="new series",
    )

    outfile = tmp_path / f"tsd_merge_distinct_{mode}.h5"
    TimeSeriesDict({"old": old}).write(
        outfile, format="hdf5", layout="dataset"
    )
    TimeSeriesDict({"new": new}).write(
        outfile, format="hdf5", layout="dataset", mode=mode
    )

    with h5py.File(outfile, "r") as h5f:
        assert set(h5f) == {"old", "new"}
        assert read_hdf5_keymap(h5f) == {"old": "old", "new": "new"}
        assert read_hdf5_order(h5f) == ["old", "new"]
    result = TimeSeriesDict.read(outfile, format="hdf5")
    assert list(result) == ["old", "new"]
    np.testing.assert_allclose(result["old"].value, old.value)
    np.testing.assert_allclose(result["new"].value, new.value)
    assert result["old"].name == old.name
    assert result["new"].name == new.name


@pytest.mark.parametrize("duplicate_key", ["mapped", "fallback"])
def test_timeseriesdict_hdf5_append_preflights_existing_logical_keys(
    tmp_path, duplicate_key
):
    mapped = TimeSeries(
        np.arange(3.0), sample_rate=2.0, t0=1.0, unit="m", name="mapped series"
    )
    fallback = TimeSeries(
        np.arange(3.0) + 10,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="fallback series",
    )
    fresh = TimeSeries(
        np.arange(3.0) + 20,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="fresh series",
    )
    replacement = TimeSeries(
        np.arange(3.0) + 30,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="replacement series",
    )

    outfile = tmp_path / f"tsd_duplicate_{duplicate_key}.h5"
    TimeSeriesDict(
        {"mapped_physical": mapped, "fallback": fallback}
    ).write(outfile, format="hdf5", layout="dataset")
    with h5py.File(outfile, "r+") as h5f:
        write_hdf5_manifest(
            h5f,
            kind="TimeSeriesDict",
            layout=LAYOUT_DATASET,
            keymap={"mapped_physical": "mapped"},
            order=["mapped_physical", "fallback"],
        )

    with pytest.raises(ValueError, match="logical key"):
        TimeSeriesDict(
            {"fresh": fresh, duplicate_key: replacement}
        ).write(outfile, format="hdf5", layout="dataset", append=True)

    with h5py.File(outfile, "r") as h5f:
        assert set(h5f) == {"mapped_physical", "fallback"}
        assert read_hdf5_keymap(h5f) == {"mapped_physical": "mapped"}
        assert read_hdf5_order(h5f) == ["mapped_physical", "fallback"]
        np.testing.assert_allclose(h5f["mapped_physical"][()], mapped.value)
        np.testing.assert_allclose(h5f["fallback"][()], fallback.value)


def test_timeseriesdict_hdf5_append_reconciles_partial_stale_manifest(tmp_path):
    old_a = TimeSeries(
        np.arange(3.0), sample_rate=2.0, t0=1.0, unit="m", name="old a"
    )
    old_b = TimeSeries(
        np.arange(3.0) + 10,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="old b",
    )
    new = TimeSeries(
        np.arange(3.0) + 20,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="new series",
    )
    private = "__gwexpy_t0_rollback_hidden"

    outfile = tmp_path / "tsd_partial_manifest.h5"
    with h5py.File(outfile, "w") as h5f:
        old_a.write(h5f, format="hdf5", path="old_a")
        old_b.write(h5f, format="hdf5", path="old_b")
        h5f.create_group("wrong_kind")
        h5f.create_dataset(private, data=[99.0])
        write_hdf5_manifest(
            h5f,
            kind="TimeSeriesDict",
            layout=LAYOUT_DATASET,
            keymap={
                "old_b": "logical_b",
                "missing": "ghost",
                "wrong_kind": "wrong_kind",
                private: "private",
            },
            order=["missing", "old_b", "old_b", "wrong_kind", private],
        )

    TimeSeriesDict({"new": new}).write(
        outfile, format="hdf5", layout="dataset", append=True
    )

    with h5py.File(outfile, "r") as h5f:
        assert set(h5f) == {"old_a", "old_b", "wrong_kind", private, "new"}
        assert read_hdf5_order(h5f) == ["old_b", "old_a", "new"]
        assert read_hdf5_keymap(h5f) == {
            "old_a": "old_a",
            "old_b": "logical_b",
            "new": "new",
        }
        assert isinstance(h5f["wrong_kind"], h5py.Group)
        np.testing.assert_allclose(h5f[private][()], [99.0])

    result = TimeSeriesDict.read(outfile, format="hdf5")
    assert list(result) == ["logical_b", "old_a", "new"]
    np.testing.assert_allclose(result["logical_b"].value, old_b.value)
    np.testing.assert_allclose(result["old_a"].value, old_a.value)
    np.testing.assert_allclose(result["new"].value, new.value)
    assert result["logical_b"].name == old_b.name
    assert result["old_a"].name == old_a.name
    assert result["new"].name == new.name


def test_timeseriesdict_hdf5_append_filters_unrelated_and_linked_root_objects(
    tmp_path,
):
    listed = TimeSeries(
        np.arange(3.0), sample_rate=2.0, t0=1.0, unit="m", name="listed series"
    )
    omitted = GwpyTimeSeries(
        np.arange(3.0) + 10,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="omitted native series",
    )
    new = TimeSeries(
        np.arange(3.0) + 20,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="new series",
    )

    outfile = tmp_path / "tsd_filter_merge_candidates.h5"
    with h5py.File(outfile, "w") as h5f:
        listed.write(h5f, format="hdf5", path="listed")
        omitted.write(h5f, format="hdf5", path="omitted")
        h5f.create_dataset("unrelated", data=[101.0, 102.0, 103.0])
        non_time = h5f.create_dataset(
            "non_time_axis", data=[201.0, 202.0, 203.0]
        )
        non_time.attrs["x0"] = 0.0
        non_time.attrs["dx"] = 1.0
        non_time.attrs["xunit"] = "Hz"
        h5f["soft_alias"] = h5py.SoftLink("/omitted")
        write_hdf5_manifest(
            h5f,
            kind="TimeSeriesDict",
            layout=LAYOUT_DATASET,
            keymap={"listed": "logical_listed"},
            order=["listed"],
        )

    TimeSeriesDict({"new": new}).write(
        outfile, format="hdf5", layout="dataset", append=True
    )

    with h5py.File(outfile, "r") as h5f:
        assert set(h5f) == {
            "listed",
            "omitted",
            "unrelated",
            "non_time_axis",
            "soft_alias",
            "new",
        }
        assert isinstance(h5f.get("soft_alias", getlink=True), h5py.SoftLink)
        np.testing.assert_allclose(h5f["unrelated"][()], [101.0, 102.0, 103.0])
        np.testing.assert_allclose(
            h5f["non_time_axis"][()], [201.0, 202.0, 203.0]
        )
        assert read_hdf5_keymap(h5f) == {
            "listed": "logical_listed",
            "omitted": "omitted",
            "new": "new",
        }
        assert read_hdf5_order(h5f) == ["listed", "omitted", "new"]

    result = TimeSeriesDict.read(outfile, format="hdf5")
    assert list(result) == ["logical_listed", "omitted", "new"]
    np.testing.assert_allclose(result["logical_listed"].value, listed.value)
    np.testing.assert_allclose(result["omitted"].value, omitted.value)
    np.testing.assert_allclose(result["new"].value, new.value)
    assert result["logical_listed"].name == listed.name
    assert result["omitted"].name == omitted.name
    assert result["new"].name == new.name


def test_timeseriesdict_hdf5_append_accepts_legacy_time_axis_candidates(tmp_path):
    complex_values = np.array([1 + 2j, 3 + 4j])
    new = TimeSeries(
        np.arange(3.0),
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="new series",
    )

    outfile = tmp_path / "tsd_legacy_candidates.h5"
    with h5py.File(outfile, "w") as h5f:
        complex_ds = h5f.create_dataset("complex", data=complex_values)
        complex_ds.attrs["xunit"] = "ms"
        empty_ds = h5f.create_dataset("empty", data=np.array([], dtype=float))
        empty_ds.attrs["xunit"] = "s"
        non_scalar = h5f.create_dataset("non_scalar_xunit", data=[99.0])
        non_scalar.attrs["xunit"] = np.array(["s"], dtype=h5py.string_dtype())

    TimeSeriesDict({"new": new}).write(
        outfile, format="hdf5", layout="dataset", append=True
    )

    with h5py.File(outfile, "r") as h5f:
        assert set(h5f) == {"complex", "empty", "non_scalar_xunit", "new"}
        assert read_hdf5_keymap(h5f) == {
            "complex": "complex",
            "empty": "empty",
            "new": "new",
        }
        assert read_hdf5_order(h5f) == ["complex", "empty", "new"]
        np.testing.assert_allclose(h5f["non_scalar_xunit"][()], [99.0])

    result = TimeSeriesDict.read(outfile, format="hdf5")
    assert list(result) == ["complex", "empty", "new"]
    np.testing.assert_allclose(result["complex"].value, complex_values)
    assert result["empty"].size == 0
    np.testing.assert_allclose(result["new"].value, new.value)
    assert result["new"].name == new.name


def test_timeseriesdict_hdf5_append_preserves_manifest_explicit_legacy_entries(
    tmp_path,
):
    ordered_values = np.array([1.0, 2.0, 3.0])
    mapped_values = np.array([11.0, 12.0, 13.0])
    new = TimeSeries(
        np.arange(3.0) + 20,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="new series",
    )

    outfile = tmp_path / "tsd_explicit_legacy_candidates.h5"
    with h5py.File(outfile, "w") as h5f:
        ordered = h5f.create_dataset("legacy_ordered", data=ordered_values)
        ordered.attrs["x0"] = 12.5
        ordered.attrs["dx"] = 0.25
        ordered.attrs["unit"] = "m"
        ordered.attrs["name"] = "ordered legacy series"
        mapped = h5f.create_dataset("legacy_mapped", data=mapped_values)
        mapped.attrs["x0"] = 22.5
        mapped.attrs["dx"] = 0.5
        mapped.attrs["unit"] = "m"
        mapped.attrs["name"] = "mapped legacy series"
        h5f.create_dataset("unrelated", data=[101.0, 102.0, 103.0])
        invalid = h5f.create_dataset("invalid_frequency", data=[201.0, 202.0])
        invalid.attrs["xunit"] = "Hz"
        invalid_string = h5f.create_dataset(
            "invalid_string", data=np.array([b"a", b"b"], dtype="S1")
        )
        invalid_string.attrs["xunit"] = "s"
        write_hdf5_manifest(
            h5f,
            kind="TimeSeriesDict",
            layout=LAYOUT_DATASET,
            keymap={
                "legacy_mapped": "logical_mapped",
                "invalid_frequency": "invalid_frequency",
                "invalid_string": "invalid_string",
            },
            order=["legacy_ordered", "invalid_frequency", "invalid_string"],
        )

    TimeSeriesDict({"new": new}).write(
        outfile, format="hdf5", layout="dataset", append=True
    )

    with h5py.File(outfile, "r") as h5f:
        assert set(h5f) == {
            "legacy_ordered",
            "legacy_mapped",
            "unrelated",
            "invalid_frequency",
            "invalid_string",
            "new",
        }
        assert "xunit" not in h5f["legacy_ordered"].attrs
        assert "xunit" not in h5f["legacy_mapped"].attrs
        np.testing.assert_allclose(h5f["unrelated"][()], [101.0, 102.0, 103.0])
        np.testing.assert_allclose(h5f["invalid_frequency"][()], [201.0, 202.0])
        np.testing.assert_array_equal(h5f["invalid_string"][()], [b"a", b"b"])
        assert read_hdf5_keymap(h5f) == {
            "legacy_ordered": "legacy_ordered",
            "legacy_mapped": "logical_mapped",
            "new": "new",
        }
        assert read_hdf5_order(h5f) == [
            "legacy_ordered",
            "legacy_mapped",
            "new",
        ]

    result = TimeSeriesDict.read(outfile, format="hdf5")
    assert list(result) == ["legacy_ordered", "logical_mapped", "new"]
    np.testing.assert_allclose(result["legacy_ordered"].value, ordered_values)
    np.testing.assert_allclose(result["logical_mapped"].value, mapped_values)
    np.testing.assert_allclose(result["new"].value, new.value)
    assert result["legacy_ordered"].name == "ordered legacy series"
    assert result["logical_mapped"].name == "mapped legacy series"
    assert str(result["legacy_ordered"].unit) == "m"
    assert str(result["logical_mapped"].unit) == "m"
    assert result["legacy_ordered"].t0.value == 12.5
    assert result["logical_mapped"].t0.value == 22.5
    assert result["legacy_ordered"].dt.value == 0.25
    assert result["logical_mapped"].dt.value == 0.5
    assert result["new"].name == new.name


def test_timeseriesdict_hdf5_append_rejects_ambiguous_existing_logical_keys(
    tmp_path,
):
    old_a = TimeSeries(np.arange(3.0), sample_rate=2.0, t0=1.0, unit="m")
    old_b = TimeSeries(np.arange(3.0) + 10, sample_rate=2.0, t0=1.0, unit="m")
    new = TimeSeries(np.arange(3.0) + 20, sample_rate=2.0, t0=1.0, unit="m")

    outfile = tmp_path / "tsd_ambiguous_manifest.h5"
    TimeSeriesDict({"old_a": old_a, "old_b": old_b}).write(
        outfile, format="hdf5", layout="dataset"
    )
    with h5py.File(outfile, "r+") as h5f:
        write_hdf5_manifest(
            h5f,
            kind="TimeSeriesDict",
            layout=LAYOUT_DATASET,
            keymap={"old_a": "duplicate", "old_b": "duplicate"},
            order=["old_a", "old_b"],
        )

    with pytest.raises(ValueError, match="ambiguous.*logical key"):
        TimeSeriesDict({"new": new}).write(
            outfile, format="hdf5", layout="dataset", append=True
        )

    with h5py.File(outfile, "r") as h5f:
        assert set(h5f) == {"old_a", "old_b"}
        assert read_hdf5_keymap(h5f) == {
            "old_a": "duplicate",
            "old_b": "duplicate",
        }
        assert read_hdf5_order(h5f) == ["old_a", "old_b"]
        np.testing.assert_allclose(h5f["old_a"][()], old_a.value)
        np.testing.assert_allclose(h5f["old_b"][()], old_b.value)


def test_timeseriesdict_hdf5_append_reconciles_native_file_without_manifest(
    tmp_path,
):
    native = GwpyTimeSeries(
        np.arange(3.0), sample_rate=2.0, t0=1.0, unit="m", name="native series"
    )
    new = TimeSeries(
        np.arange(3.0) + 10,
        sample_rate=2.0,
        t0=1.0,
        unit="m",
        name="new series",
    )

    outfile = tmp_path / "tsd_native_append.h5"
    native.write(outfile, format="hdf5", path="native")
    with h5py.File(outfile, "r") as h5f:
        assert read_hdf5_keymap(h5f) == {}
        assert read_hdf5_order(h5f) == []

    TimeSeriesDict({"new": new}).write(
        outfile, format="hdf5", layout="dataset", append=True
    )

    with h5py.File(outfile, "r") as h5f:
        assert set(h5f) == {"native", "new"}
        assert read_hdf5_keymap(h5f) == {"native": "native", "new": "new"}
        assert read_hdf5_order(h5f) == ["native", "new"]

    gwpy_result = GwpyTimeSeriesDict.read(outfile, format="hdf5")
    assert set(gwpy_result) == {"native", "new"}
    np.testing.assert_allclose(gwpy_result["native"].value, native.value)
    np.testing.assert_allclose(gwpy_result["new"].value, new.value)
    assert gwpy_result["native"].name == native.name
    assert gwpy_result["new"].name == new.name

    gwexpy_result = TimeSeriesDict.read(outfile, format="hdf5")
    assert list(gwexpy_result) == ["native", "new"]
    np.testing.assert_allclose(gwexpy_result["native"].value, native.value)
    np.testing.assert_allclose(gwexpy_result["new"].value, new.value)
    assert gwexpy_result["native"].name == native.name
    assert gwexpy_result["new"].name == new.name


def test_gwpy_reads_timeserieslist_hdf5(tmp_path):
    ts1 = TimeSeries(np.arange(3.0), sample_rate=1.0, t0=0, unit="m")
    ts2 = TimeSeries(np.arange(3.0) * 2, sample_rate=1.0, t0=0, unit="m")
    tsl = TimeSeriesList(ts1, ts2)

    outfile = tmp_path / "tsl_gwpy.h5"
    tsl.write(outfile, format="hdf5", layout="dataset")

    with h5py.File(outfile, "r") as h5f:
        order = read_hdf5_order(h5f) or list(h5f.keys())
        for idx, ds_name in enumerate(order):
            gwpy_ts = GwpyTimeSeries.read(h5f, format="hdf5", path=ds_name)
            expected = tsl[idx]
            np.testing.assert_allclose(gwpy_ts.value, expected.value)
            assert str(gwpy_ts.unit) == str(expected.unit)


def test_gwpy_reads_frequencyseriesdict_hdf5(tmp_path):
    fs = FrequencySeries(np.arange(3.0), frequencies=np.arange(3.0), unit="1")
    fsd = FrequencySeriesDict({"H1:ASD": fs})

    outfile = tmp_path / "fsd_gwpy.h5"
    fsd.write(outfile, format="hdf5", layout="dataset")

    with h5py.File(outfile, "r") as h5f:
        keymap = read_hdf5_keymap(h5f)
        order = read_hdf5_order(h5f) or list(h5f.keys())
        for ds_name in order:
            gwpy_fs = GwpyFrequencySeries.read(h5f, format="hdf5", path=ds_name)
            orig_key = keymap.get(ds_name, ds_name)
            expected = fsd[orig_key]
            np.testing.assert_allclose(gwpy_fs.value, expected.value)
            assert str(gwpy_fs.unit) == str(expected.unit)


def test_gwpy_reads_frequencyserieslist_hdf5(tmp_path):
    fsl = FrequencySeriesList(
        FrequencySeries(np.arange(3.0), frequencies=np.arange(3.0), unit="1"),
        FrequencySeries(np.arange(3.0) * 2, frequencies=np.arange(3.0), unit="1"),
    )

    outfile = tmp_path / "fsl_gwpy.h5"
    fsl.write(outfile, format="hdf5", layout="dataset")

    with h5py.File(outfile, "r") as h5f:
        order = read_hdf5_order(h5f) or list(h5f.keys())
        for idx, ds_name in enumerate(order):
            gwpy_fs = GwpyFrequencySeries.read(h5f, format="hdf5", path=ds_name)
            expected = fsl[idx]
            np.testing.assert_allclose(gwpy_fs.value, expected.value)
            assert str(gwpy_fs.unit) == str(expected.unit)


def test_gwpy_reads_spectrogramdict_hdf5(tmp_path):
    sg = Spectrogram(
        np.arange(6.0).reshape(2, 3),
        times=np.arange(2.0),
        frequencies=np.arange(3.0),
        unit="m",
    )
    sgd = SpectrogramDict({"H1:SPEC": sg})

    outfile = tmp_path / "sgd_gwpy.h5"
    sgd.write(outfile, format="hdf5", layout="dataset")

    with h5py.File(outfile, "r") as h5f:
        keymap = read_hdf5_keymap(h5f)
        order = read_hdf5_order(h5f) or list(h5f.keys())
        for ds_name in order:
            gwpy_sg = GwpySpectrogram.read(h5f, format="hdf5", path=ds_name)
            orig_key = keymap.get(ds_name, ds_name)
            expected = sgd[orig_key]
            np.testing.assert_allclose(gwpy_sg.value, expected.value)
            assert str(gwpy_sg.unit) == str(expected.unit)


def test_gwpy_reads_spectrogramlist_hdf5(tmp_path):
    sgl = SpectrogramList(
        [
            Spectrogram(
                np.arange(6.0).reshape(2, 3),
                times=np.arange(2.0),
                frequencies=np.arange(3.0),
                unit="m",
            )
        ]
    )

    outfile = tmp_path / "sgl_gwpy.h5"
    sgl.write(outfile, format="hdf5", layout="dataset")

    with h5py.File(outfile, "r") as h5f:
        order = read_hdf5_order(h5f) or list(h5f.keys())
        for idx, ds_name in enumerate(order):
            gwpy_sg = GwpySpectrogram.read(h5f, format="hdf5", path=ds_name)
            expected = sgl[idx]
            np.testing.assert_allclose(gwpy_sg.value, expected.value)
            assert str(gwpy_sg.unit) == str(expected.unit)
