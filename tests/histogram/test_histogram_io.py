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
