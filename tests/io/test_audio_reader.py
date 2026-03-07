"""Tests for audio format reader/writer (MP3, FLAC, OGG, M4A via pydub)."""

import numpy as np
import pytest

pydub = pytest.importorskip("pydub")
from pydub import AudioSegment

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def _make_audio_segment(n_samples=1000, rate=44100, channels=1):
    """Create a simple AudioSegment for testing."""
    data = (np.sin(np.linspace(0, 2 * np.pi, n_samples)) * 32767).astype(np.int16)
    if channels > 1:
        data = np.column_stack([data] * channels).flatten()
    raw = data.tobytes()
    return AudioSegment(raw, frame_rate=rate, sample_width=2, channels=channels)


class TestAudioReader:
    def test_wav_format_roundtrip(self, tmp_path):
        """Test read/write via audio.py using WAV (no ffmpeg needed)."""
        seg = _make_audio_segment(rate=22050)
        path = tmp_path / "test.wav"
        seg.export(str(path), format="wav")

        from gwexpy.timeseries.io.audio import read_timeseriesdict_audio

        tsd = read_timeseriesdict_audio(str(path), format_hint="wav")
        assert "channel_0" in tsd
        ts = tsd["channel_0"]
        assert np.isclose(ts.sample_rate.value, 22050)
        # Normalised to [-1, 1]
        assert np.all(np.abs(ts.value) <= 1.0 + 1e-6)

    def test_stereo_channels(self, tmp_path):
        seg = _make_audio_segment(channels=2)
        path = tmp_path / "stereo.wav"
        seg.export(str(path), format="wav")

        from gwexpy.timeseries.io.audio import read_timeseriesdict_audio

        tsd = read_timeseriesdict_audio(str(path), format_hint="wav")
        assert len(tsd) == 2
        assert "channel_0" in tsd
        assert "channel_1" in tsd

    def test_unit_override(self, tmp_path):
        seg = _make_audio_segment()
        path = tmp_path / "unit.wav"
        seg.export(str(path), format="wav")

        from gwexpy.timeseries.io.audio import read_timeseriesdict_audio

        tsd = read_timeseriesdict_audio(str(path), format_hint="wav", unit="Pa")
        assert str(tsd["channel_0"].unit) == "Pa"

    def test_t0_defaults_to_zero(self, tmp_path):
        seg = _make_audio_segment()
        path = tmp_path / "t0.wav"
        seg.export(str(path), format="wav")

        from gwexpy.timeseries.io.audio import read_timeseries_audio

        ts = read_timeseries_audio(str(path), format_hint="wav")
        assert float(ts.t0.value) == 0.0

    def test_write_read_roundtrip(self, tmp_path):
        """Write a TimeSeries via audio writer, read back, check approximate match."""
        from gwexpy.timeseries.io.audio import (
            read_timeseries_audio,
            write_timeseries_audio,
        )

        data = np.sin(np.linspace(0, 4 * np.pi, 500))
        ts = TimeSeries(data, t0=0, dt=1.0 / 44100, name="sig")
        path = tmp_path / "roundtrip.wav"
        write_timeseries_audio(ts, str(path), format_hint="wav")

        ts_back = read_timeseries_audio(str(path), format_hint="wav")
        # 16-bit quantization limits precision
        assert len(ts_back) == len(ts)
        # Correlation should be very high
        corr = np.corrcoef(ts.value, ts_back.value)[0, 1]
        assert corr > 0.99
