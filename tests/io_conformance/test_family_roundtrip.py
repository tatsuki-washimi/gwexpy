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
    assert_frequencyseries_close,
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
