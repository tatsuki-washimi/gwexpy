"""`TimeSeries` read post-processing that preserves explicit HDF5 axes.

GWpy pads or validates a freshly read series against the requested
``[start, end)`` interval using ``sample_rate``.  An irregularly sampled
series has no sample rate, so GWpy's step raises `AttributeError` even when
nothing needs padding. The readers defined here also skip affine merging for
an explicit HDF5 axis that does not match its stored ``x0``/``dx`` values,
then report an uncovered interval explicitly. Reads that combine several
sources and every other format are handled by GWpy unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import h5py
import numpy as np
from gwpy.time import to_gps
from gwpy.timeseries.connect import TimeSeriesDictRead, TimeSeriesRead

from gwexpy.io.hdf5_series import _explicit_axis_span, needs_explicit_axis

__all__ = [
    "IrregularAwareTimeSeriesDictRead",
    "IrregularAwareTimeSeriesRead",
    "mark_irregular_hdf5_read",
    "native_hdf5_dict_read",
]

# Set by the native HDF5 reader when an explicit xindex must be preserved, and
# removed again by ``merge``. Other formats keep GWpy's behavior unchanged.
_IRREGULAR_HDF5_READ_ATTR = "_gwex_irregular_hdf5_read"


def mark_irregular_hdf5_read(series: Any) -> Any:
    """Flag a series whose explicit HDF5 axis must bypass affine merging."""
    if needs_explicit_axis(series):
        setattr(series, _IRREGULAR_HDF5_READ_ATTR, True)
    return series


def _take_irregular_hdf5_mark(series: Any) -> bool:
    """Remove the native-HDF5 flag from ``series`` and report whether it was set."""
    state = getattr(series, "__dict__", None)
    return bool(state is not None and state.pop(_IRREGULAR_HDF5_READ_ATTR, False))


def _hdf5_source_has_irregular_axis(source: Any) -> bool:
    """Return `True` if any dataset in an HDF5 ``source`` stores ``xindex``.

    An unreadable source reports `False`, leaving its error to GWpy's reader.
    A file-like source is left at its original position.
    """

    def stored(container: h5py.HLObject) -> bool:
        def visit(_name: str, node: h5py.HLObject) -> bool | None:
            found = isinstance(node, h5py.Dataset) and "xindex" in node.attrs
            return True if found else None

        if isinstance(container, h5py.Dataset):
            return "xindex" in container.attrs
        return bool(container.visititems(visit))

    if isinstance(source, h5py.HLObject):
        return stored(source)
    position = None
    if not isinstance(source, (str, os.PathLike)):
        try:
            position = source.tell()
        except (AttributeError, OSError, TypeError, ValueError):
            return False
    try:
        with h5py.File(source, "r") as h5file:
            return stored(h5file)
    except (OSError, TypeError, ValueError):
        return False
    finally:
        if position is not None:
            source.seek(position)


def native_hdf5_dict_read(source: Any, parent: Any) -> Any:
    """Return the reader for a plain (GWpy-layout) HDF5 dict ``source``.

    This is ``parent.read`` itself unless the source stores an irregular
    axis, which GWpy's own post-processing cannot return.
    """
    if _hdf5_source_has_irregular_axis(source):
        return IrregularAwareTimeSeriesDictRead(None, parent)
    return parent.read


def _require_covered(series: Any, gap: Any, start: Any, end: Any) -> None:
    """Raise if an irregular ``series`` does not cover ``[start, end)``.

    Mirrors GWpy's ``gap="raise"`` check.  ``gap="pad"`` raises as well,
    because padding has no meaning without a sample rate.  ``gap="ignore"``
    (or any other value) performs no check, as in GWpy.
    """
    if gap not in ("pad", "raise"):
        return
    span = _explicit_axis_span(series) if needs_explicit_axis(series) else series.span
    uncovered = (start is not None and float(start) < float(span[0])) or (
        end is not None and float(end) > float(span[1])
    )
    if not uncovered:
        return
    requested = type(span)(
        span[0] if start is None else start,
        span[1] if end is None else end,
    )
    detail = (
        "" if gap == "raise" else "; an irregularly sampled series cannot be padded"
    )
    msg = (
        f"{type(series).__name__} with span {span} does not cover "
        f"requested interval {requested}{detail}"
    )
    raise ValueError(msg)


def _crop_explicit_axis(series: Any, start: Any, end: Any) -> Any:
    """Crop an explicit-axis series without consulting its affine ``dx``."""
    axis = series.xindex.value
    if start is not None:
        start = (
            start.to_value(series.xunit)
            if hasattr(start, "to_value")
            else to_gps(start)
        )
    if end is not None:
        end = end.to_value(series.xunit) if hasattr(end, "to_value") else to_gps(end)
    idx0 = None if start is None else int(np.searchsorted(axis, float(start), "left"))
    idx1 = None if end is None else int(np.searchsorted(axis, float(end), "left"))
    return series[idx0:idx1]


class IrregularAwareTimeSeriesRead(TimeSeriesRead):
    """Read data into a `TimeSeries`, including irregularly sampled data."""

    def merge(  # type: ignore[override]
        self,
        items: Sequence[Any],
        pad: float | None = None,
        gap: Any = None,
        start: Any = None,
        end: Any = None,
    ) -> Any:
        """Combine the series read from each source into one `TimeSeries`.

        A single series with an explicit HDF5 axis that cannot be represented
        by x0/dx is returned as read, after checking its requested interval.
        Everything else is merged by GWpy.
        """
        from_hdf5 = [_take_irregular_hdf5_mark(item) for item in items]
        if len(items) != 1 or not from_hdf5[0] or not needs_explicit_axis(items[0]):
            return super().merge(items, pad=pad, gap=gap, start=start, end=end)

        series = items[0]
        _require_covered(series, gap, start, end)
        series = _crop_explicit_axis(series, start, end)
        if not isinstance(series, self._cls):
            series = self._cls(series)
        return series


class IrregularAwareTimeSeriesDictRead(TimeSeriesDictRead):
    """Read data into a `TimeSeriesDict`, including irregularly sampled data."""

    def __call__(
        self,
        source: Any,
        *args: Any,
        start: Any = None,
        end: Any = None,
        pad: float | None = None,
        gap: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Defer explicit-axis window selection until after unbounded reads."""
        from gwexpy.io.time_selection import apply_time_selection

        requested_start = to_gps(start) if start is not None else None
        requested_end = to_gps(end) if end is not None else None
        result = super().__call__(
            source,
            *args,
            start=None,
            end=None,
            pad=pad,
            gap=gap,
            **kwargs,
        )
        effective_gap = gap if gap is not None else ("raise" if pad is None else "pad")
        regular = self._cls()
        explicit = self._cls()
        for key, value in result.items():
            if needs_explicit_axis(value):
                _require_covered(value, effective_gap, requested_start, requested_end)
                explicit[key] = _crop_explicit_axis(
                    value, requested_start, requested_end
                )
            else:
                regular[key] = value
        if regular:
            if effective_gap in ("pad", "raise"):
                merged_regular = super().merge(
                    [regular],
                    pad=pad,
                    gap=effective_gap,
                    start=requested_start,
                    end=requested_end,
                )
            elif requested_start is not None or requested_end is not None:
                merged_regular = type(regular)()
                for key, value in regular.items():
                    merged_regular[key] = apply_time_selection(
                        value, requested_start, requested_end
                    )
            else:
                merged_regular = regular
        else:
            merged_regular = {}
        out = self._cls()
        for key in result:
            out[key] = explicit[key] if key in explicit else merged_regular[key]
        provenance = getattr(result, "_gwexpy_io", None)
        if provenance is not None:
            out._gwexpy_io = {**provenance}
        return out

    def merge(  # type: ignore[override]
        self,
        items: Sequence[Any],
        pad: float | None = None,
        gap: Any = None,
        start: Any = None,
        end: Any = None,
    ) -> Any:
        """Combine the dicts read from each source into one `TimeSeriesDict`.

        For a single source, irregularly sampled entries are returned as
        read, after checking that they cover the requested interval; the
        regular entries of the same dict go through GWpy's merge.
        """
        if len(items) != 1 or not any(
            needs_explicit_axis(value) for value in items[0].values()
        ):
            return super().merge(items, pad=pad, gap=gap, start=start, end=end)

        source = items[0]
        regular = type(source)()
        explicit = type(source)()
        for key, value in source.items():
            if needs_explicit_axis(value):
                _require_covered(value, gap, start, end)
                explicit[key] = _crop_explicit_axis(value, start, end)
            else:
                regular[key] = value
        merged_regular = (
            super().merge([regular], pad=pad, gap=gap, start=start, end=end)
            if regular
            else {}
        )
        out = self._cls()
        for key, value in source.items():
            if key in explicit:
                out[key] = explicit[key]
            else:
                out[key] = merged_regular[key]
        return out
