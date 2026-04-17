from __future__ import annotations

from gwpy.types.array2d import Array2D as GwpyArray2D

from gwexpy.fitting.mixin import FittingMixin

from .array2d import Array2D

__all__ = ["Plane2D"]


class Plane2D(FittingMixin, Array2D):
    """Two-dimensional array with explicit semantic names for each axis.

    `Plane2D` is used by field and transform APIs when a derived slice should
    keep human-readable axis meaning instead of anonymous array dimensions.

    Examples
    --------
    >>> import numpy as np
    >>> from gwexpy.types import Plane2D
    >>> plane = Plane2D(np.ones((2, 3)), axis1_name="time", axis2_name="frequency")
    >>> plane.axis1.name, plane.axis2.name
    ('time', 'frequency')
    """

    # Do NOT add _axis1_name/_axis2_name to slots, as they duplicate Array2D slots or cause confusion.
    # We rely on Array2D's _axis0_name/_axis1_name for storage.

    def __new__(
        cls,
        data,
        axis1_name: str = "axis1",
        axis2_name: str = "axis2",
        **kwargs,
    ):
        """Create a plane with explicit names for both axes.

        Parameters
        ----------
        data : array-like
            Two-dimensional values to wrap.
        axis1_name : str, optional
            Semantic name for dimension 0.
        axis2_name : str, optional
            Semantic name for dimension 1.
        **kwargs
            Forwarded to :class:`gwexpy.types.Array2D`.
        """
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
