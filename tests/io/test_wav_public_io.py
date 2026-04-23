"""Public contract tests for WAV direct I/O."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io.registry.base import IORegistryError
from scipy.io import wavfile

from gwexpy.gui.loaders.loaders import load_products
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


def _write_stereo_wav(path):
    wavfile.write(
        str(path),
        8000,
        np.array([[0, 1000], [1000, 0], [-1000, 1000], [0, -1000]], dtype=np.int16),
    )


def test_wav_public_read_entrypoints_and_single_write(tmp_path):
    path = tmp_path / "stereo.wav"
    _write_stereo_wav(path)

    tsd = TimeSeriesDict.read(path, format="wav")
    assert sorted(tsd.keys()) == ["channel_0", "channel_1"]

    ts = TimeSeries.read(path, format="wav")
    assert ts.name == "channel_0"
    assert len(ts) == 4

    mono = TimeSeries(np.array([0, 1000, -1000, 500], dtype=np.int16), sample_rate=8000)
    out = tmp_path / "single.wav"
    mono.write(out, format="wav")
    assert isinstance(TimeSeries.read(out, format="wav"), TimeSeries)


def test_wav_write_boundary_excludes_dict_and_keeps_matrix_read(tmp_path):
    path = tmp_path / "stereo.wav"
    _write_stereo_wav(path)

    with pytest.raises(IORegistryError):
        TimeSeriesDict.read(path, format="wav").write(tmp_path / "dict.wav", format="wav")

    matrix = TimeSeriesMatrix.read(path, format="wav")
    assert matrix.shape == (2, 1, 4)


def test_wav_load_products_matches_timeseriesdict_interpretation(tmp_path):
    path = tmp_path / "stereo.wav"
    _write_stereo_wav(path)

    products = load_products(str(path))
    assert "TS" in products
    assert sorted(products["TS"].keys()) == ["channel_0", "channel_1"]

