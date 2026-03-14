"""
Audio format reader/writer for gwexpy (MP3, FLAC, OGG, M4A via pydub).

pydub requires ffmpeg or libav for most formats.
FLAC decoding may work without ffmpeg if the audioop module is available.
"""

from __future__ import annotations

from datetime import datetime
from collections.abc import Iterable

import numpy as np
from astropy import units as u
from gwpy.io.registry import default_registry as io_registry

from gwexpy.io.utils import (
    apply_unit,
    datetime_to_gps,
    ensure_dependency,
    filter_by_channels,
    set_provenance,
)

from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
from ._registration import register_timeseries_format


def _import_pydub():
    try:
        pydub = ensure_dependency("pydub")
        return pydub.AudioSegment
    except ImportError as exc:
        raise ImportError(
            "pydub is required for reading audio files (MP3, FLAC, OGG, M4A). "
            "Install with `pip install pydub`. "
            "MP3/M4A encoding also requires ffmpeg (`apt install ffmpeg` or equivalent)."
        ) from exc


def read_timeseriesdict_audio(
    source,
    *,
    format_hint: str | None = None,
    channels: Iterable[str] | None = None,
    unit: str | u.Unit | None = None,
    epoch: float | datetime | None = None,
    **kwargs,
) -> TimeSeriesDict:
    """Read an audio file into a TimeSeriesDict via pydub.

    Parameters
    ----------
    source : str or path-like
        Path to the audio file.
    format_hint : str, optional
        Audio format hint passed to pydub (e.g. ``"mp3"``, ``"flac"``).
        If *None*, pydub infers from the file extension.
    channels : iterable of str, optional
        Channel names to keep (e.g. ``["channel_0"]``).
    unit : str, optional
        Physical unit override.
    epoch : float or datetime, optional
        Override the start time (GPS seconds or datetime).
        If not provided, defaults to 0.0 (audio files do not carry absolute timestamps).
    """
    AudioSegment = _import_pydub()

    seg_kwargs = {}
    if format_hint is not None:
        seg_kwargs["format"] = format_hint

    seg = AudioSegment.from_file(str(source), **seg_kwargs)

    n_channels = seg.channels
    sample_rate = seg.frame_rate
    sample_width = seg.sample_width  # bytes per sample

    raw = np.array(seg.get_array_of_samples())

    # pydub interleaves samples: [L0, R0, L1, R1, ...]
    if n_channels > 1:
        raw = raw.reshape(-1, n_channels)
    else:
        raw = raw.reshape(-1, 1)

    # Normalise to float64 in [-1, 1]
    max_val = float(2 ** (8 * sample_width - 1))
    data = raw.astype(np.float64) / max_val

    tsd = TimeSeriesDict()
    dt = 1.0 / sample_rate

    # epoch 処理
    if epoch is not None:
        if isinstance(epoch, (int, float)):
            t0 = float(epoch)
        elif isinstance(epoch, datetime):
            t0 = datetime_to_gps(epoch)
        else:
            raise TypeError(f"epoch must be float or datetime, got {type(epoch)}")
    else:
        t0 = 0.0  # audio files carry no absolute timestamp

    for i in range(n_channels):
        name = f"channel_{i}"
        ts = TimeSeries(
            data[:, i],
            t0=t0,
            dt=dt,
            name=name,
            channel=name,
        )
        ts = apply_unit(ts, unit) if unit else ts
        tsd[name] = ts

    if channels:
        tsd = TimeSeriesDict(filter_by_channels(tsd, channels))

    set_provenance(
        tsd,
        {
            "format": format_hint or "audio",
            "sample_rate": sample_rate,
            "sample_width_bytes": sample_width,
            "original_channels": n_channels,
            "epoch_source": "user" if epoch else "default",
            "unit_source": "override" if unit else "normalised_float",
        },
    )
    return tsd


def read_timeseries_audio(source, **kwargs) -> TimeSeries:
    tsd = read_timeseriesdict_audio(source, **kwargs)
    if not tsd:
        raise ValueError("No data found in audio file")
    return tsd[next(iter(tsd.keys()))]


def read_timeseriesmatrix_audio(source, **kwargs) -> TimeSeriesMatrix:
    tsd = read_timeseriesdict_audio(source, **kwargs)
    return tsd.to_matrix()


# -- Writers -------------------------------------------------------------------


def write_timeseriesdict_audio(tsd, target, *, format_hint=None, **kwargs):
    """Write a TimeSeriesDict to an audio file via pydub.

    The data is rescaled from float to 16-bit PCM before export.

    Parameters
    ----------
    tsd : TimeSeriesDict
        Data to write.
    target : str or path-like
        Output file path.
    format_hint : str, optional
        Audio format (e.g. ``"mp3"``, ``"flac"``).  Inferred from
        extension if *None*.
    """
    AudioSegment = _import_pydub()

    keys = list(tsd.keys())
    if not keys:
        raise ValueError("Cannot write empty TimeSeriesDict to audio")

    first = tsd[keys[0]]
    sample_rate = int(round(first.sample_rate.value))
    n_samples = first.shape[0]
    n_channels = len(keys)

    # Interleave channels into int16
    interleaved = np.empty(n_samples * n_channels, dtype=np.int16)
    for idx, key in enumerate(keys):
        arr = np.asarray(tsd[key].value, dtype=np.float64)
        peak = np.max(np.abs(arr)) or 1.0
        scaled = (arr / peak * 32767).astype(np.int16)
        interleaved[idx::n_channels] = scaled

    seg = AudioSegment(
        interleaved.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,  # 16-bit
        channels=n_channels,
    )

    export_fmt = format_hint
    if export_fmt is None:
        ext = str(target).rsplit(".", 1)[-1].lower()
        export_fmt = ext

    seg.export(str(target), format=export_fmt, **kwargs)


def write_timeseries_audio(ts, target, **kwargs):
    write_timeseriesdict_audio(
        TimeSeriesDict({ts.name or "channel_0": ts}), target, **kwargs
    )


# -- Registration --------------------------------------------------------------

_AUDIO_EXTENSIONS = {
    "mp3": ".mp3",
    "flac": ".flac",
    "ogg": ".ogg",
    "m4a": ".m4a",
}


def _register_audio_format(name, extension):
    def reader_dict(src, **kw):
        """
        Read audio data into a TimeSeriesDict.
        """
        return read_timeseriesdict_audio(src, format_hint=name, **kw)

    def reader_single(src, **kw):
        """
        Read audio data into a TimeSeries.
        """
        return read_timeseries_audio(src, format_hint=name, **kw)

    def reader_matrix(src, **kw):
        """
        Read audio data into a TimeSeriesMatrix.
        """
        return read_timeseriesmatrix_audio(src, format_hint=name, **kw)

    def writer_dict(tsd, tgt, **kw):
        """
        Write TimeSeriesDict to an audio file.
        """
        return write_timeseriesdict_audio(tsd, tgt, format_hint=name, **kw)

    def writer_single(ts, tgt, **kw):
        """
        Write TimeSeries to an audio file.
        """
        return write_timeseries_audio(ts, tgt, format_hint=name, **kw)

    register_timeseries_format(
        name,
        reader_dict=reader_dict,
        reader_single=reader_single,
        reader_matrix=reader_matrix,
        writer_dict=writer_dict,
        writer_single=writer_single,
        extension=extension,
    )


for _fmt, _ext in _AUDIO_EXTENSIONS.items():
    _register_audio_format(_fmt, _ext)
