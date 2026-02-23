"""
WAV format reader for gwexpy.
Wrapper around scipy.io.wavfile to support TimeSeriesDict and metadata.
"""

from __future__ import annotations

import numpy as np
from gwpy.io.registry import default_registry as io_registry
from scipy.io import wavfile

from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix


def read_timeseriesdict_wav(source, **kwargs):
    """
    Read a WAV file into a TimeSeriesDict.

    Channels are named 'channel_0', 'channel_1', etc.
    """
    rate, data = wavfile.read(source, **kwargs)

    # Check dimensions
    if data.ndim == 1:
        # Mono
        n_channels = 1
        data = data[:, np.newaxis]
    else:
        # Multi-channel
        n_channels = data.shape[1]

    tsd = TimeSeriesDict()
    dt = 1.0 / rate

    # WAV does not store absolute time, so default to 0 (GPS)
    t0 = 0.0

    for i in range(n_channels):
        name = f"channel_{i}"

        ts = TimeSeries(
            data[:, i],
            t0=t0,
            dt=dt,
            name=name,
            channel=name,
        )
        tsd[name] = ts

    return tsd


def read_timeseries_wav(source, **kwargs):
    """
    Read a WAV file into a TimeSeries.
    If multiple channels are present, returns the first one.
    """
    tsd = read_timeseriesdict_wav(source, **kwargs)
    if not tsd:
        raise ValueError("No data found in WAV file")
    return tsd[next(iter(tsd.keys()))]


# -- Registration

for fmt in ["wav"]:
    io_registry.register_reader(
        fmt, TimeSeriesDict, read_timeseriesdict_wav, force=True
    )
    # Override gwpy's default wav reader to ensure consistent behavior
    io_registry.register_reader(fmt, TimeSeries, read_timeseries_wav, force=True)
    io_registry.register_reader(
        fmt,
        TimeSeriesMatrix,
        lambda *a, **k: read_timeseriesdict_wav(*a, **k).to_matrix(),
        force=True,
    )

    io_registry.register_identifier(
        fmt,
        TimeSeriesDict,
        lambda *args, **kwargs: str(args[1]).lower().endswith(f".{fmt}"),
    )
    io_registry.register_identifier(
        fmt,
        TimeSeries,
        lambda *args, **kwargs: str(args[1]).lower().endswith(f".{fmt}"),
    )
