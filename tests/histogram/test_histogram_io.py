import h5py
import numpy as np
import pytest

from gwexpy.histogram import Histogram


def test_hdf5_roundtrip(tmp_path):
    hist = Histogram(
        values=[10, 20],
        edges=[0, 1, 2],
        unit="V",
        xunit="Hz",
        sumw2=[1, 2],
        name="test_h",
    )
    fpath = tmp_path / "test.h5"

    hist.write(fpath, format="hdf5")

    h2 = Histogram.read(fpath, format="hdf5")

    assert h2.name == "test_h"
    assert h2.unit == hist.unit
    assert h2.xunit == hist.xunit
    assert np.allclose(h2.values.value, hist.values.value)
    assert np.allclose(h2.sumw2.value, hist.sumw2.value)


def test_hdf5_roundtrip_preserves_flow_bin_metadata(tmp_path):
    hist = Histogram(
        values=[10, 20],
        edges=[0, 1, 2],
        unit="V",
        xunit="Hz",
        sumw2=[1, 2],
        underflow=3,
        overflow=4,
        underflow_sumw2=5,
        overflow_sumw2=6,
        name="flow_h",
    )
    fpath = tmp_path / "flow.h5"

    hist.write(fpath, format="hdf5")
    h2 = Histogram.read(fpath, format="hdf5")

    assert h2.underflow.unit == hist.underflow.unit
    assert h2.overflow.unit == hist.overflow.unit
    assert np.allclose(h2.underflow.value, hist.underflow.value)
    assert np.allclose(h2.overflow.value, hist.overflow.value)

    assert h2.underflow_sumw2 is not None
    assert h2.overflow_sumw2 is not None
    assert hist.underflow_sumw2 is not None
    assert hist.overflow_sumw2 is not None
    assert h2.underflow_sumw2.unit == hist.underflow_sumw2.unit
    assert h2.overflow_sumw2.unit == hist.overflow_sumw2.unit
    assert np.allclose(h2.underflow_sumw2.value, hist.underflow_sumw2.value)
    assert np.allclose(h2.overflow_sumw2.value, hist.overflow_sumw2.value)


def test_hdf5_read_legacy_file_without_flow_bin_attrs(tmp_path):
    fpath = tmp_path / "legacy.h5"
    with h5py.File(fpath, "w") as h5f:
        group = h5f.create_group("data")
        group.create_dataset("values", data=[10, 20])
        group.create_dataset("edges", data=[0, 1, 2])
        group.create_dataset("sumw2", data=[1, 2])
        group.attrs["unit"] = "V"
        group.attrs["xunit"] = "Hz"
        group.attrs["name"] = "legacy_h"

    hist = Histogram.read(fpath, format="hdf5")

    assert hist.name == "legacy_h"
    assert hist.underflow.value == 0
    assert hist.overflow.value == 0
    assert hist.underflow.unit == hist.unit
    assert hist.overflow.unit == hist.unit
    assert hist.underflow_sumw2 is not None
    assert hist.overflow_sumw2 is not None
    assert hist.underflow_sumw2.value == 0
    assert hist.overflow_sumw2.value == 0
    assert hist.underflow_sumw2.unit == hist.unit**2
    assert hist.overflow_sumw2.unit == hist.unit**2
