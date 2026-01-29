from __future__ import annotations

from gwpy.types.array2d import Array2D as GwpyArray2D

from gwexpy.fitting.mixin import FittingMixin

from .array2d import Array2D

__all__ = ["Plane2D"]


class Plane2D(FittingMixin, Array2D):
    """
    2D Array wrapper where the two axes are semantically significant as Axis 1 and Axis 2.
    """

    # Do NOT add _axis1_name/_axis2_name to slots, as they duplicate Array2D slots or cause confusion.
    # We rely on Array2D's _axis0_name/_axis1_name for storage.

    def __new__(cls, data, axis1_name="axis1", axis2_name="axis2", **kwargs):
        if "axis_names" in kwargs:
            axis1_name, axis2_name = kwargs.pop("axis_names")
        obj = super().__new__(cls, data, axis_names=(axis1_name, axis2_name), **kwargs)
        return obj

    @property
    def axis1(self):
        """First axis descriptor (dimension 0)."""
        return self.axes[0]

    @property
    def axis2(self):
        """Second axis descriptor (dimension 1)."""
        return self.axes[1]

    def _swapaxes_int(self, a, b):
        if {a, b} != {0, 1}:
            raise ValueError(f"Invalid axis indices: {a}, {b}")
        new_data = GwpyArray2D.swapaxes(self, a, b)

        # Explicit construction for Plane2D logic
        # New y (axis 0) = old x (axis 1)
        # New x (axis 1) = old y (axis 0)
        obj = Plane2D(
            new_data.value,
            unit=new_data.unit,
            axis1_name=self.axis2.name,
            axis2_name=self.axis1.name,
            yindex=self.xindex,
            xindex=self.yindex,
        )
        # Safety force set
        obj.xindex = self.yindex
        obj.yindex = self.xindex
        return obj
