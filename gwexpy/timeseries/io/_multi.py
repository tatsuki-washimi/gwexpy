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
        series = sorted(segments[key], key=lambda ts: float(ts.t0.value))
        merged = series[0]
        for ts in series[1:]:
            try:
                merged = merged.append(ts, inplace=False, gap=gap, pad=pad)
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
