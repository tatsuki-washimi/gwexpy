"""Shared helpers for handling list/tuple sources in TimeSeries readers.

Most gwexpy readers operate on a single file. When a user passes a list
(or tuple) of paths the behaviour should be one of:

- merge the per-file results into a single container (when multi-file
  semantics are well defined, e.g. more channels and/or contiguous time
  segments), or
- raise a clear ``ValueError`` early (when merging is not meaningful,
  e.g. a single WAV file is a self-contained audio recording).

The merge semantics mirror the ObsPy-based seismic reader
(:mod:`gwexpy.timeseries.io.seismic`): channels that appear in more than
one file are concatenated along the time axis (sorted by start time,
gaps filled with ``pad``), while channels unique to one file are simply
added to the result.
"""

from __future__ import annotations

from math import floor

import numpy as np

__all__ = [
    "expand_multi_source",
    "read_multi_dict",
    "reject_multi_source",
]


def expand_multi_source(source):
    """Return ``source`` as a list if it is a list or tuple, else `None`.

    Parameters
    ----------
    source : object
        The source argument passed to a reader.

    Returns
    -------
    list or None
        A list of the individual sources if ``source`` is a list or
        tuple, otherwise `None` (single-source input).

    """
    if isinstance(source, (list, tuple)):
        return list(source)
    return None


def reject_multi_source(source, format_name):
    """Raise a clear ``ValueError`` if ``source`` is a list or tuple.

    Use this in readers for formats where reading multiple files into a
    single container has no meaningful semantics (e.g. WAV/audio files),
    so users get an actionable error instead of an opaque backend
    ``TypeError``.

    Parameters
    ----------
    source : object
        The source argument passed to a reader.
    format_name : str
        Format identifier used in the error message.

    Raises
    ------
    ValueError
        If ``source`` is a list or tuple.

    """
    if isinstance(source, (list, tuple)):
        raise ValueError(
            f"format '{format_name}' does not support reading multiple "
            f"files; got {len(source)} paths"
        )


def _regular_placement_plan(series, gap, pad):
    """Plan GWpy-compatible placements for simple, regular TimeSeries.

    Return ``None`` for cases that need the original append semantics.  In
    particular, an explicit/cached xindex needs GWpy's index update rules.
    """
    from .. import TimeSeries

    first = series[0]
    if (
        (gap == "pad" and not np.isscalar(pad))
        or type(first) is not TimeSeries
        or any(
            type(ts) is not TimeSeries
            or getattr(ts, "_xindex", None) is not None
            or not ts.size
            or ts.xunit != first.xunit
            for ts in series
        )
    ):
        return None

    try:
        origin = first.xspan[0]
        step = first.dx.value
        if not np.isfinite(step) or step <= 0:
            return None
        length = len(first)
        placements = [(first, 0, 0)]
        for ts in series[1:]:
            # An append keeps the first series' unit, dtype and cadence.
            first.is_compatible(ts)
            other_span = ts.xspan
            end = origin + length * step
            contiguous = abs(float(end - other_span[0])) < 2**-18
            if not contiguous:
                # The anti-contiguous check in GWpy also leads to this branch;
                # only a positive pad or gap='ignore' can complete normally.
                if gap == "pad":
                    padding = floor((other_span[0] - end) / step + 0.5)
                    if padding < 1:
                        return None
                elif gap == "ignore":
                    padding = 0
                else:
                    return None
            else:
                padding = 0
            placements.append((ts, length + padding, padding))
            length += padding + len(ts)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        return None
    return length, placements


def _place_segment(destination, start, source):
    """Copy one input segment into its final slot, converting units as GWpy does."""
    stop = start + len(source)
    if source.unit == destination.unit:
        destination.value[start:stop] = source.value
    else:
        destination[start:stop] = source


def _merge_series(series, gap, pad):
    """Merge regular segments with one placement each; defer other cases to GWpy."""
    if len(series) == 1:
        return series[0]

    plan = _regular_placement_plan(series, gap, pad)
    if plan is None:
        merged = series[0]
        for ts in series[1:]:
            merged = merged.append(ts, inplace=False, gap=gap, pad=pad)
        return merged

    total, placements = plan
    first = series[0]
    merged = np.empty(total, dtype=first.dtype).view(type(first))
    merged.__array_finalize__(first)
    for ts, start, padding in placements:
        if padding:
            # Keep GWpy's cast (including integer/NaN warnings) exactly.
            merged.value[start - padding : start] = (np.ones(padding) * pad).astype(
                first.dtype
            )
        _place_segment(merged, start, ts)
    return merged


def read_multi_dict(reader_func, sources, format_name, *, pad=None, gap=None, **kwargs):
    """Read several single-file sources and merge them into one dict.

    Parameters
    ----------
    reader_func : callable
        Single-file ``TimeSeriesDict`` reader, called as
        ``reader_func(source, **kwargs)`` for each entry of ``sources``.
    sources : list
        Individual sources (paths or file-like objects).
    format_name : str
        Format identifier used in error messages.
    pad : float, optional
        Fill value for gaps between time segments of the same channel
        (default: NaN, matching the seismic reader).
    gap : str, optional
        Gap handling mode forwarded to :meth:`TimeSeries.append`
        (default: ``"pad"``).
    **kwargs
        Additional keyword arguments forwarded to ``reader_func``.

    Returns
    -------
    TimeSeriesDict
        Union of the channels found in all files; duplicate channels are
        concatenated along the time axis (sorted by start time).

    """
    from .. import TimeSeriesDict

    if not sources:
        raise ValueError(f"no {format_name} files provided")
    if pad is None:
        pad = np.nan
    if gap is None:
        gap = "pad"

    segments: dict = {}
    order: list = []
    first_tsd = None
    for src in sources:
        tsd = reader_func(src, **kwargs)
        if first_tsd is None:
            first_tsd = tsd
        for key, ts in tsd.items():
            if key not in segments:
                segments[key] = []
                order.append(key)
            segments[key].append(ts)

    out = TimeSeriesDict()
    for key in order:
        # No later channel needs these parts; release the input list as soon as
        # this channel has its final output allocation.
        series = sorted(segments.pop(key), key=lambda ts: float(ts.t0.value))
        try:
            merged = _merge_series(series, gap, pad)
        except ValueError as exc:
            raise ValueError(
                f"failed to merge channel '{key}' across {format_name} files: {exc}"
            ) from exc
        out[key] = merged

    # Propagate provenance from the first file (if any) so merged reads
    # carry the same metadata shape as single-file reads.
    provenance = {}
    attrs = getattr(first_tsd, "attrs", None)
    if isinstance(attrs, dict):
        provenance.update(attrs)
    else:
        provenance.update(getattr(first_tsd, "_gwexpy_io", None) or {})
    if provenance:
        from gwexpy.io.utils import set_provenance

        provenance["n_sources"] = len(sources)
        set_provenance(out, provenance)

    return out
