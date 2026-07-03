"""gwexpy.statistics.dq_flag - DataQualityFlag generation from statistical results."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gwpy.segments import DataQualityFlag, Segment, SegmentList

if TYPE_CHECKING:
    from ..spectrogram import Spectrogram


def to_segments(
    p_value_map: Spectrogram,
    alpha: float = 0.05,
    min_duration: float = 0.0,
    fmin: float | None = None,
    fmax: float | None = None,
) -> DataQualityFlag:
    """Generate DataQualityFlag (segments) from a p-value map.

    A segment is created when p < alpha.

    Parameters
    ----------
    p_value_map : Spectrogram
        Input spectrogram of p-values.
    alpha : float, default=0.05
        Significance level.
    min_duration : float, default=0.0
        Minimum duration of segments to keep in seconds.
    fmin, fmax : float, optional
        Frequency range to consider. If None, all frequencies are checked.
        If ANY frequency in the range has p < alpha, the time is marked as bad.

    Returns
    -------
    DataQualityFlag

    """
    # 1. Filter by frequency if requested
    if fmin is not None or fmax is not None:
        p_value_map = p_value_map.crop(fmin=fmin, fmax=fmax)

    # 2. Identify "bad" time steps (where p < alpha in ANY frequency bin)
    # p_value_map.value is (n_times, n_freqs)
    is_bad = np.any(p_value_map.value < alpha, axis=1)

    # 3. Form segments
    times = p_value_map.times.value

    name = f"{p_value_map.name}_veto" if p_value_map.name else "non_gaussian_veto"

    # Guard the empty-spectrogram case: with zero time steps, times[0]/times[-1]
    # below would raise IndexError (issue #459). Return an empty flag instead.
    if len(times) == 0:
        return DataQualityFlag(name=name, active=SegmentList(), known=SegmentList())

    # Prefer the spectrogram's own time resolution; recomputing from array
    # differences is impossible for a single time step and the old hardcoded
    # dt=1.0 fallback silently mis-sized the segment (issue #459).
    try:
        dt = float(p_value_map.dt.value)
    except (AttributeError, TypeError, ValueError):
        dt = float(times[1] - times[0]) if len(times) > 1 else float(np.nan)

    segments = SegmentList()
    active_seg_start = None

    for i, bad in enumerate(is_bad):
        if bad:
            if active_seg_start is None:
                active_seg_start = times[i] - dt/2.0
        else:
            if active_seg_start is not None:
                seg_end = times[i] - dt/2.0
                if seg_end - active_seg_start >= min_duration:
                    segments.append(Segment(active_seg_start, seg_end))
                active_seg_start = None

    # Close last segment if active
    if active_seg_start is not None:
        seg_end = times[-1] + dt/2.0
        if seg_end - active_seg_start >= min_duration:
            segments.append(Segment(active_seg_start, seg_end))

    flag = DataQualityFlag(
        name=name,
        active=segments,
        known=SegmentList([Segment(times[0]-dt/2.0, times[-1]+dt/2.0)])
    )

    return flag
