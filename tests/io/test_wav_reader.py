"""Tests for WAV format reader/writer roundtrip."""

import numpy as np
import pytest
from scipy.io import wavfile

from gwexpy.timeseries.io.wav import read_timeseries_wav, read_timeseriesdict_wav


class TestWavReader:
    def test_mono_roundtrip(self, tmp_path):
        rate = 44100
        duration = 0.1
        t = np.linspace(0, duration, int(rate * duration), endpoint=False)
        data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
        path = tmp_path / "mono.wav"
        wavfile.write(str(path), rate, data)

        tsd = read_timeseriesdict_wav(str(path))
        assert len(tsd) == 1
        assert "channel_0" in tsd
        ts = tsd["channel_0"]
        assert np.isclose(ts.sample_rate.value, rate)
        np.testing.assert_array_equal(ts.value, data)

    def test_stereo_roundtrip(self, tmp_path):
        rate = 22050
        n = 1000
        left = np.arange(n, dtype=np.int16)
        right = np.arange(n, dtype=np.int16) * -1
        data = np.column_stack([left, right])
        path = tmp_path / "stereo.wav"
        wavfile.write(str(path), rate, data)

        tsd = read_timeseriesdict_wav(str(path))
        assert len(tsd) == 2
        assert "channel_0" in tsd
        assert "channel_1" in tsd
        np.testing.assert_array_equal(tsd["channel_0"].value, left)
        np.testing.assert_array_equal(tsd["channel_1"].value, right)

    def test_single_timeseries(self, tmp_path):
        rate = 8000
        data = np.ones(100, dtype=np.int16) * 1000
        path = tmp_path / "single.wav"
        wavfile.write(str(path), rate, data)

        ts = read_timeseries_wav(str(path))
        assert np.isclose(ts.sample_rate.value, rate)
        np.testing.assert_array_equal(ts.value, data)

    def test_t0_defaults_to_zero(self, tmp_path):
        path = tmp_path / "t0.wav"
        wavfile.write(str(path), 1000, np.zeros(10, dtype=np.int16))

        ts = read_timeseries_wav(str(path))
        assert float(ts.t0.value) == 0.0

    def test_float32_wav(self, tmp_path):
        rate = 16000
        data = np.random.default_rng(42).random(500).astype(np.float32)
        path = tmp_path / "float32.wav"
        wavfile.write(str(path), rate, data)

        ts = read_timeseries_wav(str(path))
        np.testing.assert_allclose(ts.value, data, atol=1e-6)

    def test_empty_wav_raises(self, tmp_path):
        path = tmp_path / "empty.wav"
        path.write_bytes(b"")
        with pytest.raises(Exception):
            read_timeseries_wav(str(path))
