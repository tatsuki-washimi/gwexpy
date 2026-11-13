"""`TimeSeries` read post-processing that tolerates irregular time axes.

GWpy pads or validates a freshly read series against the requested
``[start, end)`` interval using ``sample_rate``.  An irregularly sampled
series has no sample rate, so GWpy's step raises `AttributeError` even when
nothing needs padding.  The readers defined here skip that step for an
irregular series read from a single native HDF5 source and report an
uncovered interval explicitly.  Regular series, reads that combine several
sources, and every other format are handled by GWpy unchanged.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import h5py
from gwpy.timeseries.connect import TimeSeriesDictRead, TimeSeriesRead

from gwexpy.io.hdf5_series import has_irregular_axis

__all__ = [
    "IrregularAwareTimeSeriesDictRead",
    "IrregularAwareTimeSeriesRead",
    "mark_irregular_hdf5_read",
    "native_hdf5_dict_read",
]

# Set by the native HDF5 reader on an irregular series it returns, and removed
# again by ``merge``.  Other formats keep GWpy's behaviour for irregular data.
_IRREGULAR_HDF5_READ_ATTR = "_gwex_irregular_hdf5_read"


def mark_irregular_hdf5_read(series: Any) -> Any:
    """Flag an irregular ``series`` as freshly read from native HDF5."""
    if has_irregular_axis(series):
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
    span = series.span
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

        A single irregularly sampled series read from native HDF5 is
        returned as read, after checking that it covers the requested
        interval.  Everything else is merged by GWpy.
        """
        from_hdf5 = [_take_irregular_hdf5_mark(item) for item in items]
        if len(items) != 1 or not from_hdf5[0] or not has_irregular_axis(items[0]):
            return super().merge(items, pad=pad, gap=gap, start=start, end=end)

        series = items[0]
        _require_covered(series, gap, start, end)
        if not isinstance(series, self._cls):
            series = self._cls(series)
        return series


class IrregularAwareTimeSeriesDictRead(TimeSeriesDictRead):
    """Read data into a `TimeSeriesDict`, including irregularly sampled data."""

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
            has_irregular_axis(value) for value in items[0].values()
        ):
            return super().merge(items, pad=pad, gap=gap, start=start, end=end)

        source = items[0]
        regular = type(source)()
        for key, value in source.items():
            if has_irregular_axis(value):
                _require_covered(value, gap, start, end)
            else:
                regular[key] = value
        merged_regular = (
            super().merge([regular], pad=pad, gap=gap, start=start, end=end)
            if regular
            else {}
        )
        out = self._cls()
        for key, value in source.items():
            out[key] = merged_regular[key] if key in merged_regular else value
        return out
