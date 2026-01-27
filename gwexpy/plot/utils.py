from __future__ import annotations

import numpy as np
from gwpy.plot.utils import *  # noqa: F403


def adaptive_decimate(ts, target_points=10000):
    """
    Decimate a TimeSeries object using a min-max algorithm to preserve peaks.

    Parameters
    ----------
    ts : TimeSeries
        The input time series.
    target_points : int, optional
        Approximate number of points in the output. Default is 10000.

    Returns
    -------
    TimeSeries
        The decimated time series.
    """
    if len(ts) <= target_points:
        return ts

    # Calculate bin size. We take 2 points (min and max) per bin.
    n_bins = target_points // 2
    if n_bins < 1:
        n_bins = 1

    bin_size = len(ts) // n_bins
    if bin_size <= 1:
        return ts

    # Reshape and find min/max
    # Truncate if not divisible
    n_samples = (len(ts) // bin_size) * bin_size
    trimmed_data = ts.value[:n_samples]
    reshaped = trimmed_data.reshape(-1, bin_size)

    mins = reshaped.min(axis=1)
    maxs = reshaped.max(axis=1)

    # We want to preserve the temporal order of min/max within each bin?
    # Actually, just alternating min/max is usually enough for envelope plotting.
    # But better to check which comes first?
    # For simplicity and speed (which is the goal), alternating is often fine
    # but let's try to be a bit smarter if it's not too slow.

    # Interleave min and max
    decimated_data = np.empty(mins.size * 2, dtype=ts.dtype)
    decimated_data[0::2] = mins
    decimated_data[1::2] = maxs

    # New times axis
    ts.dt.value * bin_size / 2.0
    new_times = np.linspace(
        ts.times[0].value, ts.times[n_samples - 1].value, decimated_data.size
    )

    # Create new TimeSeries
    from gwexpy.timeseries import TimeSeries

    new_ts = TimeSeries(
        decimated_data,
        times=new_times,
        unit=ts.unit,
        name=ts.name,
        channel=getattr(ts, "channel", None),
    )
    return new_ts
