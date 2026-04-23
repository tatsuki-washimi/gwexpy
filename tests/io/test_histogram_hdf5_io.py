from __future__ import annotations

import numpy as np

from gwexpy.histogram import Histogram, HistogramDict, HistogramList


def _make_histogram(name: str, scale: float = 1.0) -> Histogram:
    return Histogram(
        values=np.array([1.0, 2.0]) * scale,
        edges=np.array([0.0, 1.0, 2.0]),
        unit="V",
        xunit="Hz",
        sumw2=np.array([0.1, 0.2]) * scale,
        name=name,
    )


def test_histogram_hdf5_roundtrip(tmp_path):
    hist = _make_histogram("hist")
    path = tmp_path / "hist.h5"

    hist.write(path, format="hdf5")
    hist2 = Histogram.read(path, format="hdf5")

    assert hist2.name == hist.name
    assert hist2.unit == hist.unit
    assert hist2.xunit == hist.xunit
    np.testing.assert_allclose(hist2.value, hist.value)
    np.testing.assert_allclose(hist2.sumw2.value, hist.sumw2.value)


def test_histogramdict_hdf5_roundtrip(tmp_path):
    hd = HistogramDict({
        "H1:A": _make_histogram("A"),
        "L1:B": _make_histogram("B", scale=2.0),
    })
    path = tmp_path / "hd.h5"

    hd.write(path, format="hdf5")
    hd2 = HistogramDict.read(path, format="hdf5")

    assert list(hd2.keys()) == list(hd.keys())
    np.testing.assert_allclose(hd2["L1:B"].value, hd["L1:B"].value)


def test_histogramlist_hdf5_roundtrip(tmp_path):
    hl = HistogramList([
        _make_histogram("A"),
        _make_histogram("B", scale=3.0),
    ])
    path = tmp_path / "hl.h5"

    hl.write(path, format="hdf5")
    hl2 = HistogramList.read(path, format="hdf5")

    assert len(hl2) == len(hl)
    np.testing.assert_allclose(hl2[0].value, hl[0].value)
