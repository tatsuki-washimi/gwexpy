"""
Audio format reader/writer for gwexpy (MP3, FLAC, OGG, M4A via pydub).

pydub requires ffmpeg or libav for most formats.
FLAC decoding may work without ffmpeg if the audioop module is available.
"""

from __future__ import annotations

import numpy as np
from gwpy.io.registry import default_registry as io_registry

from gwexpy.io.utils import apply_unit, filter_by_channels, set_provenance

from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


def _import_pydub():
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise ImportError(
            "pydub is required for reading audio files (MP3, FLAC, OGG, M4A). "
            "Install with `pip install pydub`. "
            "MP3/M4A encoding also requires ffmpeg (`apt install ffmpeg` or equivalent)."
        ) from exc
    return AudioSegment


def read_timeseriesdict_audio(
    source,
    *,
    format_hint=None,
    channels=None,
    unit=None,
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
    write_timeseriesdict_audio(TimeSeriesDict({ts.name or "channel_0": ts}), target, **kwargs)


# -- Registration --------------------------------------------------------------

_AUDIO_FORMATS = {
    "mp3": ".mp3",
    "flac": ".flac",
    "ogg": ".ogg",
    "m4a": ".m4a",
}

for _fmt, _ext in _AUDIO_FORMATS.items():
    # Readers
    io_registry.register_reader(
        _fmt,
        TimeSeriesDict,
        lambda src, _f=_fmt, **kw: read_timeseriesdict_audio(src, format_hint=_f, **kw),
        force=True,
    )
    io_registry.register_reader(
        _fmt,
        TimeSeries,
        lambda src, _f=_fmt, **kw: read_timeseries_audio(src, format_hint=_f, **kw),
        force=True,
    )
    io_registry.register_reader(
        _fmt,
        TimeSeriesMatrix,
        lambda src, _f=_fmt, **kw: read_timeseriesmatrix_audio(src, format_hint=_f, **kw),
        force=True,
    )

    # Writers
    io_registry.register_writer(
        _fmt,
        TimeSeriesDict,
        lambda tsd, tgt, _f=_fmt, **kw: write_timeseriesdict_audio(tsd, tgt, format_hint=_f, **kw),
        force=True,
    )
    io_registry.register_writer(
        _fmt,
        TimeSeries,
        lambda ts, tgt, _f=_fmt, **kw: write_timeseries_audio(ts, tgt, format_hint=_f, **kw),
        force=True,
    )

    # Identifiers
    io_registry.register_identifier(
        _fmt,
        TimeSeriesDict,
        lambda *args, _e=_ext, **kwargs: str(args[1]).lower().endswith(_e),
    )
    io_registry.register_identifier(
        _fmt,
        TimeSeries,
        lambda *args, _e=_ext, **kwargs: str(args[1]).lower().endswith(_e),
    )
