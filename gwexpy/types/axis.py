from dataclasses import dataclass

import numpy as np
from astropy.units import Quantity

__all__ = ["AxisDescriptor", "coerce_1d_quantity"]


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
    name: str
    index: Quantity  # 1D

    def __post_init__(self):
        # validation
        self.index = coerce_1d_quantity(self.index)

    @property
    def unit(self):
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
        return self.index.size

    @property
    def regular(self) -> bool:
        if hasattr(self.index, "regular"):
            return self.index.regular
        if self.size <= 1:
            return True
        diffs = np.diff(self.index.value)
        return bool(np.isclose(np.diff(diffs), 0).all())

    @property
    def delta(self) -> Quantity | None:
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
        """Convert a coordinate slice (start, stop, step) to an integer slice."""
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
