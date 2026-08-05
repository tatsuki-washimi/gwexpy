"""Shared handling of the ``start``/``end`` read selectors (issue #611).

GWpy's registry lets any reader accept ``start`` and ``end`` through
``**kwargs``, and several GWexpy readers accepted them, dropped them, and
returned the whole file.  That is worse than not accepting them at all: the
caller asked for a window, got the full span back, and nothing in the result
says so.  Downstream code that trusts the request — a PSD over "the quiet
minute", a coincidence search over a specific segment — then computes a
correct-looking number from the wrong samples.

This module leaves a reader exactly two honest options and removes the third:

``apply_time_selection``
    Take the selectors out of ``kwargs``, let the reader load whatever the
    backend gives it, and crop the assembled object.  Every reader that reads
    whole files uses this, because for them cropping costs two lines and
    refusing would remove function GWpy itself provides.

``reject_time_selection``
    Refuse the request with a message that names the format and says what to do
    instead.  Reserved for paths where the windowed result cannot be verified.
    Deliberately not a silent no-op and not a warning: a warning is routinely
    filtered, and the wrong array still reaches the caller.

Backend-level push-down (asking NetCDF4, Zarr or ObsPy to read only the
requested slice) is a performance change, not a correctness one, and is out of
scope for v0.1.13.  Reading whole and cropping is slower but returns the right
samples.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np

from ..interop.errors import IoNotImplementedError

__all__ = [
    "TIME_SELECTION_KEYS",
    "apply_time_selection",
    "pop_time_selection",
    "reject_time_selection",
]

#: Keyword names GWpy's registry injects to request a sub-span of a source.
TIME_SELECTION_KEYS = ("start", "end")


def pop_time_selection(kwargs: dict[str, Any]) -> tuple[Any, Any]:
    """Remove ``start``/``end`` from *kwargs* and return them.

    Mutates *kwargs* in place, because the caller's next move is always to
    forward the remainder to a backend that would reject the selectors.

    Returns
    -------
    (start, end) : tuple
        Either element is `None` when the corresponding keyword was absent or
        explicitly passed as `None`.

    """
    return kwargs.pop("start", None), kwargs.pop("end", None)


def _normalize_bound(bound: Any) -> float | None:
    """Return *bound* as a plain GPS float, or `None`.

    GWpy's registry runs ``to_gps`` on ``start``/``end`` before any reader
    sees them, so readers reached through it only ever see numbers.  GWexpy's
    ``TimeSeriesDict.read``/``TimeSeriesList.read`` are plain classmethods
    whose fast-path branches bypass that layer, so the Quantity / date-string
    / `~datetime.datetime` inputs GWpy documents as valid reached ``float()``
    raw and crashed there.  This applies the same normalisation at the one
    choke point every branch shares.

    Numeric input is passed through ``float()`` untouched rather than through
    ``to_gps``: ``to_gps`` quantises to integer nanoseconds, and the values on
    the registry path already went through it, so re-applying it here would
    only move precision loss onto the bypass paths that currently do better.
    """
    if bound is None:
        return None
    if isinstance(bound, (int, float, np.integer, np.floating)):
        return float(bound)

    from gwexpy.time import to_gps

    gps = to_gps(bound)
    if isinstance(gps, (np.ndarray, list)) and np.ndim(gps) > 0:
        gps = gps[0]
    return float(gps)


def _empty_intersection(series: Any, *, after: bool) -> Any:
    """Return an empty slice of *series* at the span edge nearest the window.

    Built by direct slicing, not by ``crop(edge, edge)``: the series' own span
    edge is frequently not representable in the integer nanoseconds that
    GWexpy's ``TimeSeries.crop`` wrapper round-trips every bound through, and
    an edge that rounds *below* ``x0`` reaches gwpy's unguarded
    ``floor((end - x0) / dx)`` as a negative index that wraps to the end of
    the array.  Measured: ``t0 = 1/3`` with a window entirely before the data
    returned nine samples instead of zero — the exact silent-wrong-samples
    defect #611 exists to remove, reintroduced by the fix's own edge case.

    `~gwexpy.timeseries.TimeSeriesMatrix` cannot be sliced along time from
    here (axis order is class-internal), but its ``crop`` is
    ``searchsorted``-based — no ``to_gps`` round-trip, no negative index — so
    the zero-width window is safe for it.
    """
    if getattr(series, "ndim", 1) == 1:
        return series[len(series) :] if after else series[:0]
    lo, hi = (float(getattr(bound, "value", bound)) for bound in series.span)
    edge = hi if after else lo
    return series.crop(edge, edge)


def _crop_to_span(series: Any, start: float | None, end: float | None) -> Any:
    """Crop one series to ``[start, end)``, treating its span as the limit.

    Only the bounds the caller actually gave are forwarded to
    :meth:`~gwpy.types.series.Series.crop`; a missing or out-of-span bound
    becomes `None`, never a value synthesised from the series' own span.  The
    distinction is load-bearing: GWexpy's crop wrapper round-trips every
    non-`None` bound through ``to_gps`` (integer nanoseconds), so passing the
    span's exclusive upper edge — often not nanosecond-representable — rounds
    it *down* and silently drops the final sample.  Measured end-to-end: a
    ``dt = 1/30`` file read with ``start`` only returned 47 samples where
    ``full_read.crop(start)`` returns 48, for hdf5, zarr, csv and wav alike.
    With `None` forwarded instead, ``crop`` does not touch that side at all,
    and the in-range bounds the user did give reach ``crop`` bit-identical to
    the oracle's, so ``result == full_read.crop(start, end)`` holds exactly.

    Out-of-span bounds are clamped to `None` silently, without gwpy's
    "crop given start smaller than current start" warning: a bounded read
    legitimately asks for a superset of what one file holds (that is how
    multi-file and mixed-epoch reads work), and gwpy's own
    ``read_hdf5_timeseries`` clamps with ``max``/``min`` just as silently.

    Windows that miss the data entirely never reach ``crop`` — see
    :func:`_empty_intersection` for why.

    ``start``/``end`` here are already plain floats (`None` allowed); type
    normalisation happens once in :func:`apply_time_selection`.
    """
    # ``span`` is a Segment of plain floats for TimeSeries but a Quantity in
    # seconds for TimeSeriesMatrix, and float() on the latter raises TypeError
    # ("only dimensionless scalar quantities...") rather than dropping the unit.
    lo, hi = (float(getattr(bound, "value", bound)) for bound in series.span)

    if end is not None and end <= lo:
        return _empty_intersection(series, after=False)
    if start is not None and start >= hi:
        return _empty_intersection(series, after=True)
    if start is not None and start <= lo:
        start = None
    if end is not None and end >= hi:
        end = None
    if start is None and end is None:
        return series
    return series.crop(start, end)


def apply_time_selection(data: Any, start: Any, end: Any) -> Any:
    """Crop *data* to ``[start, end)``, or return it unchanged.

    Parameters
    ----------
    data : TimeSeries, TimeSeriesDict, TimeSeriesMatrix, or any object with
        ``span`` and ``crop``.  Mappings are cropped entry by entry, so channels
        with different epochs each get their own clamp.
    start, end : float, `~astropy.units.Quantity`, date `str`, or None
        GPS bounds, in any form :func:`gwexpy.time.to_gps` accepts.  Both
        `None` means "no selection was requested", and *data* is returned
        as-is rather than round-tripped through :meth:`crop`.

    Returns
    -------
    The cropped object, of the same type as *data*.

    Notes
    -----
    Cropping is non-destructive: mappings are rebuilt rather than mutated, so
    passing a plain :class:`gwpy.timeseries.TimeSeriesDict` — whose own
    ``crop`` is in-place — does not modify the caller's object.

    ``_gwexpy_io`` provenance is carried across explicitly, as a copy so the
    bounded result and the source never alias one mutable dict.  GWexpy's
    ``TimeSeriesDict.crop`` drops it (measured on the base tree, so this is
    pre-existing rather than introduced here), which would otherwise make a
    bounded read lose metadata that an unbounded read of the same source keeps.

    """
    start = _normalize_bound(start)
    end = _normalize_bound(end)
    if start is None and end is None:
        return data

    if isinstance(data, MutableMapping):
        out: Any = type(data)()
        for key, series in data.items():
            out[key] = _crop_to_span(series, start, end)
        provenance = getattr(data, "_gwexpy_io", None)
        if provenance is not None:
            out._gwexpy_io = {**provenance}
        return out

    return _crop_to_span(data, start, end)


def reject_time_selection(
    format_name: str,
    kwargs: Mapping[str, Any],
    *,
    hint: str | None = None,
) -> None:
    """Raise if a ``start``/``end`` request reached a reader that cannot honour it.

    Call this *before* doing any work, so the failure is cheap and unambiguous.
    Passing the selectors explicitly as `None` is not a request and does not
    raise — GWpy's own call sites do that routinely.

    Parameters
    ----------
    format_name : str
        Format identifier as the user typed it (e.g. ``"ats.mth5"``), used in
        the message so the reader can be found from the traceback alone.
    kwargs : mapping
        The reader's keyword arguments, inspected but not modified.
    hint : str, optional
        Extra guidance appended after the standard remedy.

    Raises
    ------
    gwexpy.interop.errors.IoNotImplementedError
        A `NotImplementedError` subclass, when either selector is not `None`.

    """
    requested = [key for key in TIME_SELECTION_KEYS if kwargs.get(key) is not None]
    if not requested:
        return

    names = " and ".join(f"'{key}'" for key in requested)
    lines = [
        f"The '{format_name}' reader cannot restrict a read to {names}: it has "
        "no windowed read path, and silently returning the full file would give "
        "you numerically wrong data for the span you asked for (issue #611).",
        "Read the source in full and crop the result instead, e.g. "
        "`TimeSeries.read(source, format=...).crop(start, end)`.",
    ]
    if hint:
        lines.append(hint)
    raise IoNotImplementedError(" ".join(lines))
