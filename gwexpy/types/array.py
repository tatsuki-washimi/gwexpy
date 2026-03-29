from __future__ import annotations

from typing import Any

import numpy as np
from astropy.units import dimensionless_unscaled
from gwpy.types.array import Array as GwpyArray

from ._stats import StatisticalMethodsMixin
from .axis import AxisDescriptor
from .axis_api import AxisApiMixin

__all__ = ["Array"]


class Array(AxisApiMixin, StatisticalMethodsMixin, GwpyArray):
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

        # Propagate custom _gwex_ attributes
        for key, val in getattr(obj, "__dict__", {}).items():
            if key.startswith("_gwex_") and key not in self.__dict__:
                self.__dict__[key] = val

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
            new_obj._axis_names[a], new_obj._axis_names[b] = (
                new_obj._axis_names[b],
                new_obj._axis_names[a],
            )
        return new_obj

    def rms(self, axis=None, keepdims=False, ignore_nan=True):
        func = np.nanmean if ignore_nan else np.mean
        val = np.sqrt(func(np.square(self.value), axis=axis, keepdims=keepdims))
        return val * self.unit

    def _propagate_gwex_attrs(self, other: Any) -> None:
        """Propagate _gwex_ prefixed attributes to another object."""
        for key, val in self.__dict__.items():
            if key.startswith("_gwex_") and key not in other.__dict__:
                other.__dict__[key] = val
