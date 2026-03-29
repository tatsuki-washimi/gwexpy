import numpy as np
import pytest
from astropy import units as u

from gwexpy.histogram import Histogram, HistogramDict, HistogramList


def test_histogram_dict_basics():
    h1 = Histogram([10, 20], [0, 1, 2], unit="count")
    h2 = Histogram([5, 15], [0, 1, 2], unit="count")

    hd = HistogramDict()
    hd["a"] = h1
    hd["b"] = h2

    assert len(hd) == 2
    assert "a" in hd

    # test dict mapping (rebin)
    hd_rebinned = hd.rebin([0.5, 1.5])
    assert hd_rebinned["a"].values.value[0] == 15
    assert hd_rebinned["b"].values.value[0] == 10
    assert type(hd_rebinned) is HistogramDict

def test_histogram_list_basics():
    h1 = Histogram([10, 20], [0, 1, 2], unit="count")
    h2 = Histogram([5, 15], [0, 1, 2], unit="count")

    hl = HistogramList([h1, h2])

    assert len(hl) == 2
    assert type(hl) is HistogramList

    # test list mapping (rebin)
    hl_rebinned = hl.rebin([0.5, 1.5])
    assert hl_rebinned[0].values.value[0] == 15
    assert hl_rebinned[1].values.value[0] == 10
    assert type(hl_rebinned) is HistogramList

def test_histogram_dict_integral():
    h1 = Histogram([10, 20], [0, 1, 2], unit="count")
    h2 = Histogram([30, 40], [0, 1, 2], unit="count")
    hd = HistogramDict({"a": h1, "b": h2})

    res = hd.integral(0.5, 1.5)

    # Dictionary plain method mapping
    assert type(res) is dict
    assert "a" in res
    assert "b" in res

    # integral returns (value, error)
    val_a, err_a = res["a"]
    val_b, err_b = res["b"]

    assert val_a.value == 15
    assert val_b.value == 35

def test_histogram_collections_hdf5(tmp_path):
    h1 = Histogram([10, 20], [0, 1, 2], unit="count", xunit="Hz", name="a")
    h2 = Histogram([30, 40], [0, 1, 2], unit="count", xunit="Hz", name="b")

    hd = HistogramDict({"a": h1, "b": h2})

    fpath = tmp_path / "coll.h5"
    hd.write(fpath, format="hdf5")

    hd2 = HistogramDict.read(fpath, format="hdf5")

    assert "a" in hd2
    assert "b" in hd2
    assert hd2["a"].values.value[0] == 10
    assert hd2["b"].values.value[0] == 30

    hl = HistogramList([h1, h2])
    fpath2 = tmp_path / "list.h5"
    hl.write(fpath2, format="hdf5")

    hl2 = HistogramList.read(fpath2, format="hdf5")
    assert len(hl2) == 2
    assert hl2[0].values.value[0] == 10

