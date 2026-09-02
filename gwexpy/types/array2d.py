from __future__ import annotations

from gwpy.types.array2d import Array2D as GwpyArray2D

from ._stats import StatisticalMethodsMixin
from .axis import AxisDescriptor
from .axis_api import AxisApiMixin

__all__ = ["Array2D"]


class Array2D(AxisApiMixin, StatisticalMethodsMixin, GwpyArray2D):
    """2D array with a unified axis API."""

    _metadata_slots = GwpyArray2D._metadata_slots + ("_axis0_name", "_axis1_name")

    def __new__(cls, data, axis_names=None, **kwargs):
        """Create a 2D array with explicit axis naming."""
        obj = super().__new__(cls, data, **kwargs)
        if axis_names is None:
            name0, name1 = "axis0", "axis1"
        else:
            if len(axis_names) != 2:
                raise ValueError("axis_names must have length 2 for Array2D")
            name0, name1 = axis_names
        obj._axis0_name = name0
        obj._axis1_name = name1
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return

        current_a0 = getattr(self, "_axis0_name", None)
        current_a1 = getattr(self, "_axis1_name", None)
        parent_a0 = getattr(obj, "_axis0_name", None)
        parent_a1 = getattr(obj, "_axis1_name", None)

        if parent_a0 is not None:
            self._axis0_name = parent_a0
        elif current_a0 is None:
            self._axis0_name = "axis0"

        if parent_a1 is not None:
            self._axis1_name = parent_a1
        elif current_a1 is None:
            self._axis1_name = "axis1"

    @property
    def axes(self):
        """Return axis descriptors for both dimensions."""
        return (
            AxisDescriptor(self._axis0_name, self.xindex),
            AxisDescriptor(self._axis1_name, self.yindex),
        )

    def _set_axis_name(self, index, name):
        if index == 0:
            self._axis0_name = name
        elif index == 1:
            self._axis1_name = name
        else:
            raise IndexError(index)

    def _isel_tuple(self, item_tuple):
        return self.__getitem__(item_tuple)

    def swapaxes(self, axis1, axis2):
        """Swap axes, preserving GWpy metadata on numeric calls."""
        if isinstance(axis1, str) or isinstance(axis2, str):
            return AxisApiMixin.swapaxes(self, axis1, axis2)
        return GwpyArray2D.swapaxes(self, axis1, axis2)

    def transpose(self, *axes):
        """Transpose through GWpy unless a named-axis extension is used."""
        normalized = (
            axes[0] if len(axes) == 1 and isinstance(axes[0], (tuple, list)) else axes
        )
        if any(isinstance(axis, str) for axis in normalized):
            return AxisApiMixin.transpose(self, *axes)
        return GwpyArray2D.transpose(self, *axes)

    @property
    def T(self):  # noqa: N802
        """Return the GWpy two-dimensional transpose with swapped axes."""
        transposed = GwpyArray2D.T.fget(self)
        transposed._axis0_name = self._axis1_name
        transposed._axis1_name = self._axis0_name
        return transposed

    def _swapaxes_int(self, a, b):
        if {a, b} != {0, 1}:
            raise ValueError(f"Invalid axis indices: {a}, {b}")

        # Access old indices before swap
        try:
            old_x = self.xindex
        except AttributeError:
            old_x = None
        try:
            old_y = self.yindex
        except AttributeError:
            old_y = None

        new_data = GwpyArray2D.swapaxes(self, a, b)

        # Create fresh instance to ensure metadata correctness
        # Passing xindex/yindex explicitly swaps them
        return self.__class__(
            new_data.value,
            unit=new_data.unit,
            axis_names=(self._axis1_name, self._axis0_name),
            xindex=old_y,
            yindex=old_x,
        )

    def _transpose_int(self, axes: tuple[int, ...]):
        if axes == (0, 1):
            return self.copy()
        elif axes == (1, 0):
            return self._swapaxes_int(0, 1)
        else:
            raise ValueError(f"Invalid transpose axes for 2D: {axes}")

    def imshow(self, **kwargs):
        """Plot this array with `matplotlib.axes.Axes.imshow`.

        Inherited from GWpy.
        """
        return super().imshow(**kwargs)

    def pcolormesh(self, **kwargs):
        """Plot this array with `matplotlib.axes.Axes.pcolormesh`.

        Inherited from GWpy.
        """
        return super().pcolormesh(**kwargs)
