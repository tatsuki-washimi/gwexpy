from __future__ import annotations

import numpy as np

from gwexpy.spectrogram import Spectrogram, SpectrogramDict, SpectrogramList


def test_spectrogramdict_group_layout_roundtrip(tmp_path):
    data = np.arange(6.0).reshape(2, 3)
    times = np.arange(2.0)
    freqs = np.arange(3.0)
    sg = Spectrogram(data, times=times, frequencies=freqs, unit="m")
    sgd = SpectrogramDict({"H1:SPEC": sg})

    outfile = tmp_path / "sgd_group.h5"
    sgd.write(outfile, format="hdf5", layout="group")

    sgd2 = SpectrogramDict().read(outfile, format="hdf5")
    assert list(sgd2.keys()) == list(sgd.keys())
    np.testing.assert_allclose(sgd2["H1:SPEC"].value, sg.value)


def test_spectrogramlist_group_layout_roundtrip(tmp_path):
    data = np.arange(6.0).reshape(2, 3)
    times = np.arange(2.0)
    freqs = np.arange(3.0)
    sg1 = Spectrogram(data, times=times, frequencies=freqs, unit="m")
    sg2 = Spectrogram(data * 2, times=times, frequencies=freqs, unit="m")
    sgl = SpectrogramList([sg1, sg2])

    outfile = tmp_path / "sgl_group.h5"
    sgl.write(outfile, format="hdf5", layout="group")

    sgl2 = SpectrogramList().read(outfile, format="hdf5")
    assert len(sgl2) == len(sgl)
    np.testing.assert_allclose(sgl2[0].value, sg1.value)
