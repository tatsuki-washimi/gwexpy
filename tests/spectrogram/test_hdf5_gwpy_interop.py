from __future__ import annotations

import numpy as np
from gwpy.spectrogram import Spectrogram as GwpySpectrogram

from gwexpy.io.hdf5_collection import safe_hdf5_key
from gwexpy.spectrogram import Spectrogram, SpectrogramDict, SpectrogramList


def test_spectrogramdict_hdf5_gwpy_read(tmp_path):
    data = np.arange(12.0).reshape(3, 4)
    times = np.arange(3.0)
    freqs = np.arange(4.0)
    sg = Spectrogram(data, times=times, frequencies=freqs, unit="m")
    sgd = SpectrogramDict({"H1:SPEC": sg})

    outfile = tmp_path / "sgd.h5"
    sgd.write(outfile, format="hdf5")

    path = safe_hdf5_key("H1:SPEC")
    gw = GwpySpectrogram.read(outfile, format="hdf5", path=path)
    np.testing.assert_allclose(gw.value, sg.value)
    np.testing.assert_allclose(gw.times.value, sg.times.value)
    np.testing.assert_allclose(gw.frequencies.value, sg.frequencies.value)


def test_spectrogramlist_hdf5_gwpy_read(tmp_path):
    data = np.arange(6.0).reshape(2, 3)
    times = np.arange(2.0)
    freqs = np.arange(3.0)
    sg1 = Spectrogram(data, times=times, frequencies=freqs, unit="m")
    sg2 = Spectrogram(data * 2, times=times, frequencies=freqs, unit="m")
    sgl = SpectrogramList([sg1, sg2])

    outfile = tmp_path / "sgl.h5"
    sgl.write(outfile, format="hdf5")

    gw = GwpySpectrogram.read(outfile, format="hdf5", path="0")
    np.testing.assert_allclose(gw.value, sg1.value)
    np.testing.assert_allclose(gw.times.value, sg1.times.value)
    np.testing.assert_allclose(gw.frequencies.value, sg1.frequencies.value)
