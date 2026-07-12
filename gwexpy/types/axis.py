from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.units import Quantity

__all__ = ["AxisDescriptor", "coerce_1d_quantity"]

_REGULAR_RTOL = 1e-9
_REGULAR_ATOL_ULPS = 8


def coerce_1d_quantity(index, unit=None) -> Quantity:
    """Ensure input is a 1D Quantity."""
    if not isinstance(index, Quantity):
        index = Quantity(index, unit=unit)
    elif unit is not None:
        index = index.to(unit)

    if index.ndim != 1:
        if index.ndim == 0:
            index = index.reshape(1)
        else:
            raise ValueError(f"Axis index must be 1D, got {index.ndim}D")
    return index


@dataclass
class AxisDescriptor:
    """Describe a named one-dimensional axis and its coordinate values."""

    name: str
    index: Quantity  # 1D

    def __post_init__(self):
        self.index = coerce_1d_quantity(self.index)

    @property
    def unit(self):
        """Return the axis unit."""
        return self.index.unit

    def __eq__(self, other):
        if not isinstance(other, AxisDescriptor):
            return NotImplemented
        try:
            return (
                self.name == other.name
                and self.index.shape == other.index.shape
                and np.all(self.index == other.index)
            )
        except (AttributeError, TypeError, ValueError):
            return False

    @property
    def size(self):
        """Return the number of axis samples."""
        return self.index.size

    @property
    def regular(self) -> bool:
        """Return whether the axis has regular linear spacing.

        Integer coordinates are compared exactly. Other numeric coordinates
        have adjacent intervals compared with a relative tolerance of ``1e-9``
        and an absolute tolerance of eight ULPs at the largest coordinate
        magnitude. This avoids both small-scale axes being accepted by a fixed
        absolute tolerance and large-offset axes being rejected because of
        subtraction round-off.

        Logarithmic (equal-ratio) axes are not regular under this linear-spacing
        contract.
        """
        if self.size <= 1:
            return True
        values = np.asarray(self.index.to_value(self.unit))
        if not np.all(np.isfinite(values)):
            return False
        if np.issubdtype(values.dtype, np.integer):
            # Python integers preserve exact intervals and cannot overflow.
            diffs = np.diff(values.astype(object))
            return all(diff == diffs[0] for diff in diffs[1:])
        diffs = np.diff(values)
        coordinate_scale = np.max(np.abs(values))
        atol = _REGULAR_ATOL_ULPS * abs(np.spacing(coordinate_scale))
        return bool(
            np.allclose(
                diffs,
                diffs[0],
                rtol=_REGULAR_RTOL,
                atol=atol,
                equal_nan=False,
            )
        )

    @property
    def delta(self) -> Quantity | None:
        """Return the constant axis spacing when the axis is regular."""
        if self.size > 1 and self.regular:
            return self.index[1] - self.index[0]
        return None

    def to_value(self, q):
        """Convert Quantity to axis unit value, or return float if dimensionless/compatible."""
        if isinstance(q, Quantity):
            return q.to_value(self.unit)
        return float(q)

    def iloc_nearest(self, value):
        """Return the integer index nearest to value."""
        val = self.to_value(value)
        # Assume monotonic for speed, or general search? User said "nearest/searchsorted"
        # but also "axis coordinates are not assumed to be regular".
        # If not regular, we can still use abs diffargmin.
        idx = np.abs(self.index.value - val).argmin()
        return idx

    def iloc_slice(self, s: slice):
        """Convert a coordinate slice (start, stop, step) to an integer slice.

        Coordinate start and stop bounds require an ascending axis because they
        are resolved with :func:`numpy.searchsorted`. Descending and unordered
        axes remain valid for :meth:`iloc_nearest`.
        """
        start_idx = None
        stop_idx = None
        step_idx = None

        # Handling start
        if s.start is not None:
            val = self.to_value(s.start)
            # searchsorted works if sorted
            # If we assume sorted (ascending):
            start_idx = np.searchsorted(self.index.value, val, side="left")

        # Handling stop
        if s.stop is not None:
            val = self.to_value(s.stop)
            stop_idx = np.searchsorted(
                self.index.value, val, side="left"
            )  # 'right' might be more 'python slice' style?
            # Typically python slice includes start, excludes stop.
            # searchsorted('left') gives index i such that a[i-1] < v <= a[i].
            # If we want to exclude the value itself if it equals, left is good?
            # Actually, for `sel`, users expect range [start, stop).

        # Handling step (coordinate step -> integer step?)
        # This is tricky for non-regular. If regular, we can compute it.
        if s.step is not None:
            # If coordinate step is given, we need to map to integer step
            if self.regular and self.delta is not None and self.delta.value != 0:
                step_val = self.to_value(s.step)
                step_idx = int(round(step_val / self.delta.value))
            else:
                raise ValueError("Cannot use coordinate step slice on irregular axis")

        return slice(start_idx, stop_idx, step_idx)
