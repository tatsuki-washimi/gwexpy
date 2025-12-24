
from gwpy.types.array import Array as GwpyArray
from astropy.units import dimensionless_unscaled
import numpy as np
from .axis import AxisDescriptor
from .axis_api import AxisApiMixin

__all__ = ["Array"]

class Array(AxisApiMixin, GwpyArray):
    """
    N-dimensional array with axis unified API.
    """
    _metadata_slots = getattr(GwpyArray, "_metadata_slots", ()) + ("_axis_names",)

    def __new__(cls, data, axis_names=None, **kwargs):
        obj = super().__new__(cls, data, **kwargs)
        if axis_names is None:
             axis_names = [f"axis{i}" for i in range(obj.ndim)]
        else:
             axis_names = list(axis_names)
        obj._axis_names = axis_names
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return

        parent_names = getattr(obj, "_axis_names", None)
        if parent_names is None or len(parent_names) != self.ndim:
            self._axis_names = [f"axis{i}" for i in range(self.ndim)]
        else:
            self._axis_names = list(parent_names)

    @property
    def axes(self):
        if len(self._axis_names) != self.ndim:
             self._axis_names = [f"axis{i}" for i in range(self.ndim)]
        return tuple(
            AxisDescriptor(name, np.arange(self.shape[i]) * dimensionless_unscaled)
            for i, name in enumerate(self._axis_names)
        )

    def _set_axis_name(self, index, name):
        self._axis_names[index] = name

    def _isel_tuple(self, item_tuple):
        return self.__getitem__(item_tuple)

    def _swapaxes_int(self, a, b):
        new_obj = GwpyArray.swapaxes(self, a, b)
        if hasattr(new_obj, "_axis_names") and len(new_obj._axis_names) == new_obj.ndim:
             new_obj._axis_names[a], new_obj._axis_names[b] = new_obj._axis_names[b], new_obj._axis_names[a]
        return new_obj
