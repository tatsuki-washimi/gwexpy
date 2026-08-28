"""Published Spectrogram HDF5 value, axis, and metadata contract."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

pytestmark = pytest.mark.skipif(
    os.environ.get("GWEXPY_POST_RELEASE_QUALIFICATION") != "1",
    reason="post-release qualification is opt-in",
)


def test_spectrogram_hdf5_roundtrip_preserves_public_contract(tmp_path: Path) -> None:
    from gwexpy.spectrogram import Spectrogram

    original = Spectrogram(
        np.arange(24, dtype=np.float64).reshape(4, 6),
        unit="V**2/Hz",
        t0=1_234_567_890,
        dt=0.25 * u.s,
        f0=2 * u.Hz,
        df=0.5 * u.Hz,
        name="qualification spectrum",
        channel="X1:QUALIFICATION",
    )
    destination = tmp_path / "spectrogram.hdf5"

    original.write(destination, format="hdf5", path="spectrogram")
    restored = Spectrogram.read(destination, format="hdf5", path="spectrogram")

    assert type(restored) is Spectrogram
    np.testing.assert_array_equal(restored.value, original.value)
    assert restored.shape == original.shape
    assert restored.unit == original.unit
    assert restored.t0 == original.t0
    assert restored.dt == original.dt
    assert restored.f0 == original.f0
    assert restored.df == original.df
    assert restored.name == original.name
    assert str(restored.channel) == str(original.channel)
