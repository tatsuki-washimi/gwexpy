"""Round-trip metadata conformance for non-TimeSeries object families (D2).

The conformance harness historically validated only the TimeSeries family, so
metadata-loss bugs in FrequencySeries / Spectrogram round-trips were invisible.
These exercise the new family validators against the always-available HDF5
backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.spectrogram import Spectrogram
from tests.io_conformance.validators import (
    assert_eventtable_close,
    assert_frequencyseries_close,
    assert_histogram_close,
    assert_segmentlist_close,
    assert_spectrogram_close,
)


def test_frequencyseries_hdf5_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    values = np.arange(16.0)
    expected = FrequencySeries(values, df=0.5, f0=10.0, name="H1:FS", unit="m")
    path = tmp_path / "fs.h5"

    expected.write(path, format="hdf5")
    actual = FrequencySeries.read(path, format="hdf5")

    assert_frequencyseries_close(
        actual, values, df=0.5, f0=10.0, name="H1:FS", unit="m"
    )


def test_spectrogram_hdf5_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    values = np.arange(24.0).reshape(6, 4)
    expected = Spectrogram(
        values, dt=0.25, df=2.0, t0=1_000_000_000.0, f0=0.0, name="H1:SG", unit="m"
    )
    path = tmp_path / "sg.h5"

    expected.write(path, format="hdf5")
    actual = Spectrogram.read(path, format="hdf5")

    assert_spectrogram_close(
        actual,
        values,
        dt=0.25,
        df=2.0,
        t0=1_000_000_000.0,
        f0=0.0,
        name="H1:SG",
        unit="m",
    )


def test_histogram_hdf5_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    from gwexpy.histogram import Histogram

    values = np.array([3.0, 5.0, 2.0, 1.0])
    edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    expected = Histogram(values, edges, name="H1:HIST", unit="m")
    path = tmp_path / "hist.h5"

    expected.write(path, format="hdf5")
    actual = Histogram.read(path, format="hdf5")

    assert_histogram_close(
        actual, values, expected_edges=edges, name="H1:HIST", unit="m"
    )


def test_segmentlist_hdf5_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    from gwexpy.segments import Segment, SegmentList

    expected = SegmentList([Segment(0, 1), Segment(2, 3), Segment(10, 12)])
    path = tmp_path / "segs.h5"

    expected.write(path, format="hdf5", path="segments")
    actual = SegmentList.read(path, format="hdf5", path="segments")

    assert_segmentlist_close(actual, expected)


def test_eventtable_hdf5_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    from gwexpy.table import EventTable

    expected = EventTable(data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], names=["a", "b"])
    path = tmp_path / "events.h5"

    expected.write(path, format="hdf5", path="events")
    actual = EventTable.read(path, format="hdf5", path="events")

    assert_eventtable_close(actual, expected)
