"""C2 WAV construction gate for single-channel public reads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from gwexpy.timeseries.io import wav as wav_io


@pytest.fixture
def stereo_wav(tmp_path: Path) -> Path:
    path = tmp_path / "stereo.wav"
    wavfile.write(path, 64, np.arange(128, dtype=np.int16).reshape(64, 2))
    return path


@pytest.mark.parametrize("channels", [None, ["channel_1"]])
def test_single_wav_constructs_only_requested_channel(
    stereo_wav: Path, channels: list[str] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructed: list[str] = []
    original = wav_io.TimeSeries

    def count_construction(*args: object, **kwargs: object) -> object:
        constructed.append(str(kwargs["name"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(wav_io, "TimeSeries", count_construction)
    kwargs = {} if channels is None else {"channels": channels}
    result = wav_io.read_timeseries_wav(stereo_wav, **kwargs)
    expected = "channel_0" if channels is None else "channel_1"
    assert result.name == expected
    assert constructed == [expected]


def test_dict_wav_keeps_requested_channel_order(
    stereo_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructed: list[str] = []
    original = wav_io.TimeSeries

    def count_construction(*args: object, **kwargs: object) -> object:
        constructed.append(str(kwargs["name"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(wav_io, "TimeSeries", count_construction)
    result = wav_io.read_timeseriesdict_wav(stereo_wav, channels=["channel_1"])
    assert list(result) == ["channel_1"]
    assert constructed == ["channel_1"]


def test_unit_override_retains_full_construction_contract(
    stereo_wav: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructed: list[str] = []
    original = wav_io.TimeSeries

    def count_construction(*args: object, **kwargs: object) -> object:
        constructed.append(str(kwargs["name"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(wav_io, "TimeSeries", count_construction)
    result = wav_io.read_timeseriesdict_wav(
        stereo_wav, channels=["channel_1"], unit="V"
    )
    assert list(result) == ["channel_1"]
    assert constructed == ["channel_0", "channel_1"]
