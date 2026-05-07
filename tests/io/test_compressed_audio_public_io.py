"""Public contract tests for compressed audio direct I/O."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
from gwpy.io.registry import default_registry as io_registry

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def _make_tsd():
    return TimeSeriesDict(
        {
            "L": TimeSeries(
                np.array([0.0, 0.25, -0.25, 0.5], dtype=float), sample_rate=8, name="L"
            ),
            "R": TimeSeries(
                np.array([0.5, -0.5, 0.25, -0.25], dtype=float), sample_rate=8, name="R"
            ),
        }
    )


def _has_audio_codec_backend() -> bool:
    return shutil.which("ffmpeg") is not None or shutil.which("avconv") is not None


@pytest.mark.parametrize("fmt", ["flac", "ogg", "mp3", "m4a"])
def test_compressed_audio_missing_pydub_raises_clean_importerror(
    monkeypatch, tmp_path, fmt
):
    from gwexpy.timeseries.io import audio as audio_io

    def _boom():
        raise ImportError(
            "pydub is required for reading audio files (MP3, FLAC, OGG, M4A). "
            "Install with `pip install pydub`."
        )

    monkeypatch.setattr(audio_io, "_import_pydub", _boom)
    path = tmp_path / f"sample.{fmt}"
    path.write_bytes(b"")

    with pytest.raises(ImportError, match="pydub is required"):
        TimeSeries.read(path, format=fmt)
    with pytest.raises(ImportError, match="pydub is required"):
        _make_tsd().write(path, format=fmt)


def test_compressed_audio_missing_codec_on_read_raises_clean_importerror(
    monkeypatch, tmp_path
):
    from gwexpy.timeseries.io import audio as audio_io

    class MissingCodecAudioSegment:
        @classmethod
        def from_file(cls, *args, **kwargs):
            raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(audio_io, "_import_pydub", lambda: MissingCodecAudioSegment)
    path = tmp_path / "sample.mp3"
    path.write_bytes(b"not an audio payload")

    with pytest.raises(ImportError, match="ffmpeg or libav"):
        TimeSeriesDict.read(path, format="mp3")


def test_compressed_audio_missing_codec_on_write_raises_clean_importerror(
    monkeypatch, tmp_path
):
    from gwexpy.timeseries.io import audio as audio_io

    class MissingCodecAudioSegment:
        def __init__(self, *args, **kwargs):
            pass

        def export(self, *args, **kwargs):
            raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(audio_io, "_import_pydub", lambda: MissingCodecAudioSegment)
    path = tmp_path / "sample.mp3"

    with pytest.raises(ImportError, match="ffmpeg or libav"):
        _make_tsd().write(path, format="mp3")


@pytest.mark.parametrize("fmt", ["flac", "ogg", "mp3", "m4a"])
def test_compressed_audio_public_extensions_auto_identify(tmp_path, fmt):
    path = tmp_path / f"sample.{fmt}"
    path.write_bytes(b"not an audio payload")

    assert fmt in io_registry.identify_format(
        "read", TimeSeriesDict, str(path), None, (), {}
    )
    assert fmt in io_registry.identify_format(
        "read", TimeSeries, str(path), None, (), {}
    )


@pytest.mark.parametrize("fmt", ["flac", "ogg", "mp3", "m4a"])
def test_compressed_audio_public_roundtrip_when_dependency_available(tmp_path, fmt):
    pytest.importorskip("pydub")
    if not _has_audio_codec_backend():
        pytest.skip("ffmpeg/libav is required for compressed audio round-trip")

    tsd = _make_tsd()
    path = tmp_path / f"sample.{fmt}"

    tsd.write(path, format=fmt)
    back = TimeSeriesDict.read(path, format=fmt)

    assert len(back) == 2
    assert sorted(str(k) for k in back.keys()) == ["channel_0", "channel_1"]
