from __future__ import annotations

import numpy as np

from gwexpy.spectrogram import Spectrogram, SpectrogramDict, SpectrogramList


def test_spectrogramdict_hdf5_file_roundtrip(tmp_path):
    data = np.arange(12.0).reshape(3, 4)
    times = np.arange(3.0)
    freqs = np.arange(4.0)
    s1 = Spectrogram(data, times=times, frequencies=freqs, unit="m")
    s2 = Spectrogram(data * 2, times=times, frequencies=freqs, unit="m")
    sgd = SpectrogramDict({"H1:SPEC": s1, "L1:SPEC": s2})

    outfile = tmp_path / "sgd.h5"
    sgd.write(outfile, format="hdf5")

    sgd2 = SpectrogramDict().read(outfile, format="hdf5")
    assert list(sgd2.keys()) == list(sgd.keys())
    for k in sgd:
        np.testing.assert_allclose(sgd2[k].value, sgd[k].value)
        np.testing.assert_allclose(sgd2[k].times.value, sgd[k].times.value)
        np.testing.assert_allclose(sgd2[k].frequencies.value, sgd[k].frequencies.value)
        assert str(sgd2[k].unit) == str(sgd[k].unit)


def test_spectrogramlist_hdf5_file_roundtrip(tmp_path):
    data = np.arange(6.0).reshape(2, 3)
    times = np.arange(2.0)
    freqs = np.arange(3.0)
    s1 = Spectrogram(data, times=times, frequencies=freqs, unit="m")
    s2 = Spectrogram(data * 3, times=times, frequencies=freqs, unit="m")
    sgl = SpectrogramList([s1, s2])

    outfile = tmp_path / "sgl.h5"
    sgl.write(outfile, format="hdf5")

    sgl2 = SpectrogramList().read(outfile, format="hdf5")
    assert len(sgl2) == len(sgl)
    for i in range(len(sgl)):
        np.testing.assert_allclose(sgl2[i].value, sgl[i].value)
        np.testing.assert_allclose(sgl2[i].times.value, sgl[i].times.value)
        np.testing.assert_allclose(sgl2[i].frequencies.value, sgl[i].frequencies.value)
        assert str(sgl2[i].unit) == str(sgl[i].unit)
