"""
WAV format reader for gwexpy.
Wrapper around scipy.io.wavfile to support TimeSeriesDict and metadata.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import numpy as np
from astropy import units as u
from scipy.io import wavfile

from gwexpy.io.utils import (
    apply_unit,
    datetime_to_gps,
    extract_audio_metadata,
    filter_by_channels,
    set_provenance,
)

from .. import TimeSeries, TimeSeriesDict
from ._registration import register_timeseries_format


def read_timeseriesdict_wav(
    source,
    *,
    channels: Iterable[str] | None = None,
    unit: str | u.Unit | None = None,
    epoch: float | datetime | None = None,
    extract_metadata: bool = False,
    **kwargs,
):
    """
    Read a WAV file into a TimeSeriesDict.

    Parameters
    ----------
    source : str or Path
        Path to the WAV file.
    channels : iterable of str, optional
        Channel names to keep (e.g., ["channel_0"]).
    unit : str or Unit, optional
        Physical unit override.
    epoch : float or datetime, optional
        Override the start time (GPS seconds or datetime).
        If not provided, defaults to 0.0 (WAV files do not carry absolute timestamps).
    extract_metadata : bool, optional
        If True, attempt to extract audio metadata (title, artist, album, etc.)
        using tinytag. Metadata is stored in the provenance. Requires the
        optional ``tinytag`` package (``pip install gwexpy[audio]``).
        Default: False (for backward compatibility).
    **kwargs
        Additional arguments passed to scipy.io.wavfile.read.

    Returns
    -------
    TimeSeriesDict
        Dictionary of TimeSeries objects, one per channel.

    Notes
    -----
    Channels are named 'channel_0', 'channel_1', etc.
    """
    # Filter kwargs for wavfile.read
    scipy_kwargs = {
        k: v for k, v in kwargs.items() if k not in ("start", "end", "format")
    }
    rate, data = wavfile.read(source, **scipy_kwargs)

    # Check dimensions
    if data.ndim == 1:
        # Mono
        n_channels = 1
        data = data[:, np.newaxis]
    else:
        # Multi-channel
        n_channels = data.shape[1]

    # epoch 処理
    if epoch is not None:
        if isinstance(epoch, (int, float)):
            t0 = float(epoch)
        elif isinstance(epoch, datetime):
            t0 = datetime_to_gps(epoch)
        else:
            raise TypeError(f"epoch must be float or datetime, got {type(epoch)}")
    else:
        t0 = 0.0  # 既存の動作を維持

    tsd = TimeSeriesDict()
    dt = 1.0 / rate

    for i in range(n_channels):
        name = f"channel_{i}"

        ts = TimeSeries(
            data[:, i],
            t0=t0,
            dt=dt,
            name=name,
            channel=name,
        )
        # unit 適用
        if unit is not None:
            ts = apply_unit(ts, unit)
        tsd[name] = ts

    # channels フィルタリング
    if channels is not None:
        tsd = TimeSeriesDict(filter_by_channels(tsd, channels))

    # Build provenance metadata
    provenance = {
        "format": "wav",
        "epoch_source": "user" if epoch else "default",
        "unit_source": "override" if unit else "wav",
    }

    # Extract audio metadata if requested
    if extract_metadata:
        metadata = extract_audio_metadata(source)
        if metadata:
            provenance.update(metadata)

    set_provenance(tsd, provenance)

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
# Override gwpy's default wav reader to ensure consistent behavior

register_timeseries_format(
    "wav",
    reader_dict=read_timeseriesdict_wav,
    reader_single=read_timeseries_wav,
    extension="wav",
)
