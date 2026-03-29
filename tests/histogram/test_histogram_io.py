import os

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
