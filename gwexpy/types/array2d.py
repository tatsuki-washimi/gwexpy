from __future__ import annotations

from typing import Any

from gwpy.types.array import Array as GwpyArray
from gwpy.types.array2d import Array2D as GwpyArray2D

from gwexpy.time._reference import (
    TIME_REFERENCE_ABSOLUTE,
    TIME_REFERENCE_RELATIVE,
    TimeReferenceMixin,
    check_time_setter_value,
    inherit_time_reference,
    resolve_time_inputs,
    set_time_reference,
)

from ._stats import StatisticalMethodsMixin
from .axis import AxisDescriptor
from .axis_api import AxisApiMixin

__all__ = ["Array2D"]


class Array2D(
    AxisApiMixin,
    StatisticalMethodsMixin,
    TimeReferenceMixin,
    GwpyArray2D,
):
    """2D array with a unified axis API."""

    _metadata_slots = GwpyArray2D._metadata_slots + ("_axis0_name", "_axis1_name")
    _time_reference_axis_attrs = ("_epoch",)

    def __new__(
        cls,
        data,
        unit=None,
        x0=None,
        dx=None,
        xindex=None,
        xunit=None,
        y0=None,
        dy=None,
        yindex=None,
        yunit=None,
        *,
        axis_names=None,
        **kwargs,
    ):
        """Create a 2D array with explicit axis naming."""
        time_reference, cleared = resolve_time_inputs(
            {"epoch": kwargs.get("epoch")},
            explicit_axis=False,
            inherit=isinstance(data, GwpyArray),
        )
        if "epoch" in cleared:
            kwargs = dict(kwargs)
            kwargs["epoch"] = None
        obj = super().__new__(
            cls,
            data,
            unit=unit,
            x0=x0,
            dx=dx,
            xindex=xindex,
            xunit=xunit,
            y0=y0,
            dy=dy,
            yindex=yindex,
            yunit=yunit,
            **kwargs,
        )
        if time_reference is not None:
            set_time_reference(obj, time_reference)
        elif isinstance(data, TimeReferenceMixin):
            inherit_time_reference(obj, data)
        elif isinstance(data, GwpyArray):
            set_time_reference(
                obj,
                TIME_REFERENCE_ABSOLUTE
                if getattr(data, "epoch", None) is not None
                else TIME_REFERENCE_RELATIVE,
            )
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

        for key, val in getattr(obj, "__dict__", {}).items():
            if key.startswith("_gwex_") and key not in self.__dict__:
                self.__dict__[key] = val

    @property
    def epoch(self) -> Any:
        """GPS epoch associated with this array, or ``None`` if omitted."""
        return GwpyArray2D.epoch.__get__(self, type(self))

    @epoch.setter
    def epoch(self, value: Any) -> None:
        check_time_setter_value(value, name="epoch")
        GwpyArray2D.epoch.__set__(self, value)

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
