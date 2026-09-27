"""Fail closed when an HDF5 collection manifest lists an unreadable entry."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from gwexpy.frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesList,
)
from gwexpy.histogram import Histogram, HistogramDict, HistogramList
from gwexpy.spectrogram import Spectrogram, SpectrogramDict, SpectrogramList
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList

_MANIFEST_ATTRS = (
    "gwexpy_keymap",
    "gwexpy_order",
    "gwexpy_kind",
    "gwexpy_layout",
    "gwexpy_layout_version",
)

_COLLECTION_CASES = (
    pytest.param("TS", "Dict", TimeSeriesDict, id="timeseries-dict"),
    pytest.param("TS", "List", TimeSeriesList, id="timeseries-list"),
    pytest.param("FS", "Dict", FrequencySeriesDict, id="frequencyseries-dict"),
    pytest.param("FS", "List", FrequencySeriesList, id="frequencyseries-list"),
    pytest.param("SPEC", "Dict", SpectrogramDict, id="spectrogram-dict"),
    pytest.param("SPEC", "List", SpectrogramList, id="spectrogram-list"),
    pytest.param("HIST", "Dict", HistogramDict, id="histogram-dict"),
    pytest.param("HIST", "List", HistogramList, id="histogram-list"),
)

_DISCOVERY_CASES = tuple(
    case
    for case in _COLLECTION_CASES
    if case.values[0] != "TS" or case.values[1] != "Dict"
)

_GROUP_MISSING_CASES = tuple(
    case for case in _COLLECTION_CASES if case.values[0] in {"TS", "FS", "SPEC"}
)

_GROUP_PAYLOAD_CASES = _COLLECTION_CASES


def _entry(family: str, name: str):
    offset = "ABC".index(name) * 10.0
    values = np.array([1.0, 2.0, 3.0]) + offset
    if family == "TS":
        return TimeSeries(values, sample_rate=2.0, t0=100.0, unit="m", name=name)
    if family == "FS":
        return FrequencySeries(
            values, frequencies=np.array([4.0, 6.0, 8.0]), unit="m", name=name
        )
    if family == "SPEC":
        return Spectrogram(
            np.stack([values, values + 1.0]),
            times=np.array([100.0, 102.0]),
            frequencies=np.array([4.0, 6.0, 8.0]),
            unit="m",
            name=name,
        )
    return Histogram(
        values=values,
        edges=np.array([0.0, 1.0, 2.0, 3.0]),
        unit="m",
        xunit="s",
        name=name,
    )


def _collection(family: str, shape: str, collection_cls):
    entries = [_entry(family, name) for name in "ABC"]
    if shape == "Dict":
        return collection_cls(dict(zip("ABC", entries)))
    if family == "TS":
        return collection_cls(*entries)
    return collection_cls(entries)


def _public_read(collection_cls, family: str, path):
    if family == "SPEC":
        return collection_cls().read(path, format="hdf5")
    return collection_cls.read(path, format="hdf5")


def _corrupt_middle_entry(path, shape: str, family: str) -> None:
    physical_name = "B" if shape == "Dict" else "1"
    attr_name = "unit" if family == "HIST" else "x0"
    invalid_value = (
        "audit-invalid-unit-@???" if family == "HIST" else "audit-invalid-axis"
    )
    with h5py.File(path, "r+") as h5f:
        entry_group = h5f[physical_name]
        target = (
            entry_group["data"]
            if isinstance(entry_group, h5py.Group) and "data" in entry_group
            else entry_group
        )
        target.attrs[attr_name] = invalid_value


@pytest.mark.parametrize(("family", "shape", "collection_cls"), _COLLECTION_CASES)
@pytest.mark.parametrize("layout", ("dataset", "group"))
def test_manifest_backed_collection_read_rejects_unreadable_entry(
    tmp_path, family, shape, collection_cls, layout
):
    path = tmp_path / f"{family}-{shape}-{layout}.h5"
    _collection(family, shape, collection_cls).write(path, format="hdf5", layout=layout)
    _corrupt_middle_entry(path, shape, family)

    with pytest.raises(
        (KeyError, TypeError, ValueError, OSError), match="audit-invalid"
    ):
        _public_read(collection_cls, family, path)


@pytest.mark.parametrize("layout", ("dataset", "group"))
def test_timeseriesdict_auto_read_rejects_unreadable_manifest_entry(tmp_path, layout):
    path = tmp_path / f"TS-Dict-auto-{layout}.h5"
    _collection("TS", "Dict", TimeSeriesDict).write(path, format="hdf5", layout=layout)
    _corrupt_middle_entry(path, "Dict", "TS")

    with pytest.raises(
        (KeyError, TypeError, ValueError, OSError), match="audit-invalid"
    ):
        TimeSeriesDict.read(path)


@pytest.mark.parametrize(("family", "shape", "collection_cls"), _GROUP_MISSING_CASES)
def test_manifest_backed_group_collection_read_rejects_missing_listed_entry(
    tmp_path, family, shape, collection_cls
):
    path = tmp_path / f"{family}-{shape}-group-missing-B.h5"
    _collection(family, shape, collection_cls).write(
        path, format="hdf5", layout="group"
    )
    with h5py.File(path, "r+") as h5f:
        h5f.attrs["gwexpy_order"] = json.dumps(
            ["C", "B", "A"] if shape == "Dict" else ["2", "1", "0"]
        )
        if shape == "Dict":
            h5f.attrs["gwexpy_keymap"] = json.dumps(
                {letter: f"logical-{letter}" for letter in "ABC"}
            )
        del h5f["B" if shape == "Dict" else "1"]

    with pytest.raises(KeyError):
        _public_read(collection_cls, family, path)


@pytest.mark.parametrize(("family", "shape", "collection_cls"), _GROUP_PAYLOAD_CASES)
def test_manifest_backed_group_read_does_not_fallback_to_other_dataset(
    tmp_path, family, shape, collection_cls
):
    path = tmp_path / f"{family}-{shape}-group-wrong-payload.h5"
    _collection(family, shape, collection_cls).write(
        path, format="hdf5", layout="group"
    )
    physical_name = "B" if shape == "Dict" else "1"
    with h5py.File(path, "r+") as h5f:
        group = h5f[physical_name]
        if family == "HIST":
            del group["data"]
            group.create_dataset("values", data=np.full(3, 999.0))
            group.create_dataset("edges", data=np.arange(4, dtype=np.float64))
            group.attrs["unit"] = "m"
            group.attrs["xunit"] = "s"
        else:
            payload = group["data"]
            payload[...] = np.full(payload.shape, 999.0)
            group.move("data", "unrelated")

    with pytest.raises((KeyError, TypeError, ValueError, OSError)):
        _public_read(collection_cls, family, path)


@pytest.mark.parametrize(("family", "shape", "collection_cls"), _DISCOVERY_CASES)
@pytest.mark.parametrize("layout", ("dataset", "group"))
def test_manifest_free_discovery_keeps_skipping_unreadable_entry(
    tmp_path, family, shape, collection_cls, layout
):
    path = tmp_path / f"{family}-{shape}-{layout}-discovery.h5"
    _collection(family, shape, collection_cls).write(path, format="hdf5", layout=layout)
    with h5py.File(path, "r+") as h5f:
        for attr in _MANIFEST_ATTRS:
            del h5f.attrs[attr]
    _corrupt_middle_entry(path, shape, family)

    result = _public_read(collection_cls, family, path)

    assert len(result) == 2
    if shape == "Dict":
        assert list(result) == ["A", "C"]
    else:
        assert [entry.name for entry in result] == ["A", "C"]
