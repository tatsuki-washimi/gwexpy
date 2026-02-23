from __future__ import annotations

import numpy as np
from astropy.units import Quantity, dimensionless_unscaled
from gwpy.types.array import Array as GwpyArray

from .array import Array
from .axis import AxisDescriptor, coerce_1d_quantity
from .plane2d import Plane2D

__all__ = ["Array3D"]


class Array3D(Array):
    """
    3D Array with explicit axis management.
    """

    _metadata_slots = Array._metadata_slots + (
        "_axis0_name",
        "_axis1_name",
        "_axis2_name",
        "_axis0_index",
        "_axis1_index",
        "_axis2_index",
    )

    def __new__(
        cls,
        data,
        unit=None,
        axis0=None,
        axis1=None,
        axis2=None,
        axis_names=None,
        **kwargs,
    ):
        obj = super().__new__(cls, data, unit=unit, **kwargs)
        if obj.ndim != 3:
            raise ValueError(f"Array3D must be 3-dimensional, got {obj.ndim}D")

        if axis_names is None:
            obj._axis0_name = "axis0"
            obj._axis1_name = "axis1"
            obj._axis2_name = "axis2"
        else:
            if len(axis_names) != 3:
                raise ValueError("axis_names must be length 3")
            obj._axis0_name, obj._axis1_name, obj._axis2_name = [
                str(x) for x in axis_names
            ]

        if axis0 is None:
            obj._axis0_index = np.arange(obj.shape[0]) * dimensionless_unscaled
        else:
            obj._axis0_index = coerce_1d_quantity(axis0)

        if axis1 is None:
            obj._axis1_index = np.arange(obj.shape[1]) * dimensionless_unscaled
        else:
            obj._axis1_index = coerce_1d_quantity(axis1)

        if axis2 is None:
            obj._axis2_index = np.arange(obj.shape[2]) * dimensionless_unscaled
        else:
            obj._axis2_index = coerce_1d_quantity(axis2)

        if getattr(obj, "_axis_names", None) is not None:
            obj._axis_names = [obj._axis0_name, obj._axis1_name, obj._axis2_name]

        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return

        def copy_attr(name, default):
            if getattr(self, name, None) is None:
                val = getattr(obj, name, None)
                if val is not None:
                    setattr(self, name, val)
                else:
                    setattr(self, name, default)

        copy_attr("_axis0_name", "axis0")
        copy_attr("_axis1_name", "axis1")
        copy_attr("_axis2_name", "axis2")

        # Safe initialization of indices respecting ndim
        # If ndim < 1, assume scalar or broken

        if self.ndim >= 1 and getattr(self, "_axis0_index", None) is None:
            # Try copy if shape matches
            if getattr(obj, "ndim", 0) >= 1 and obj.shape[0] == self.shape[0]:
                self._axis0_index = getattr(obj, "_axis0_index", None)
            if getattr(self, "_axis0_index", None) is None:
                self._axis0_index = np.arange(self.shape[0]) * dimensionless_unscaled

        if self.ndim >= 2 and getattr(self, "_axis1_index", None) is None:
            if getattr(obj, "ndim", 0) >= 2 and obj.shape[1] == self.shape[1]:
                self._axis1_index = getattr(obj, "_axis1_index", None)
            if getattr(self, "_axis1_index", None) is None:
                self._axis1_index = np.arange(self.shape[1]) * dimensionless_unscaled

        if self.ndim >= 3 and getattr(self, "_axis2_index", None) is None:
            if getattr(obj, "ndim", 0) >= 3 and obj.shape[2] == self.shape[2]:
                self._axis2_index = getattr(obj, "_axis2_index", None)
            if getattr(self, "_axis2_index", None) is None:
                self._axis2_index = np.arange(self.shape[2]) * dimensionless_unscaled

        if getattr(self, "_axis_names", None) is not None and self.ndim == 3:
            self._axis_names = [self._axis0_name, self._axis1_name, self._axis2_name]

    @property
    def axes(self):
        return (
            AxisDescriptor(self._axis0_name, self._axis0_index),
            AxisDescriptor(self._axis1_name, self._axis1_index),
            AxisDescriptor(self._axis2_name, self._axis2_index),
        )

    def _set_axis_name(self, index, name):
        if index == 0:
            self._axis0_name = name
        elif index == 1:
            self._axis1_name = name
        elif index == 2:
            self._axis2_name = name
        else:
            raise IndexError(index)
        if (
            getattr(self, "_axis_names", None) is not None
            and len(self._axis_names) == 3
        ):
            self._axis_names = [self._axis0_name, self._axis1_name, self._axis2_name]

    def __getitem__(self, item):
        return self._getitem_with_axis_metadata(item)

    def _isel_tuple(self, item_tuple):
        return self._getitem_with_axis_metadata(item_tuple)

    def _getitem_with_axis_metadata(self, item):
        raw = GwpyArray.__getitem__(self, item)
        items_list = self._normalize_item(item)
        if items_list is None:
            return self._to_plain(raw)

        current_axes = [
            (self._axis0_name, self._axis0_index),
            (self._axis1_name, self._axis1_index),
            (self._axis2_name, self._axis2_index),
        ]
        kept_axes = []

        for i, sl in enumerate(items_list):
            name, idx_arr = current_axes[i]
            if idx_arr is None:
                return self._to_plain(raw)
            if isinstance(sl, slice):
                kept_axes.append((name, idx_arr[sl]))
            elif self._is_int_index(sl):
                continue
            else:
                return self._to_plain(raw)

        expected_ndim = len(kept_axes)
        if getattr(raw, "ndim", None) != expected_ndim:
            return self._to_plain(raw)

        if expected_ndim == 3:
            value, unit = self._value_unit(raw)
            meta = self._metadata_kwargs(raw)
            return Array3D(
                value,
                unit=unit,
                axis_names=[n for n, _ in kept_axes],
                axis0=kept_axes[0][1],
                axis1=kept_axes[1][1],
                axis2=kept_axes[2][1],
                copy=False,
                **meta,
            )
        if expected_ndim == 2:
            value, unit = self._value_unit(raw)
            meta = self._metadata_kwargs(raw)
            return Plane2D(
                value,
                unit=unit,
                axis1_name=kept_axes[0][0],
                axis2_name=kept_axes[1][0],
                xindex=kept_axes[0][1], # Axis 0
                yindex=kept_axes[1][1], # Axis 1
                copy=False,
                **meta,
            )

        return self._to_plain(raw)

    @staticmethod
    def _is_int_index(value):
        return isinstance(value, (int, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        )

    @staticmethod
    def _normalize_item(item):
        if not isinstance(item, tuple):
            item = (item,)
        if any(val is None for val in item):
            return None
        if Ellipsis in item:
            if item.count(Ellipsis) > 1:
                return None
            ellipsis_idx = item.index(Ellipsis)
            num_specified = len(item) - 1
            fill = 3 - num_specified
            if fill < 0:
                return None
            item = (
                item[:ellipsis_idx] + (slice(None),) * fill + item[ellipsis_idx + 1 :]
            )
        if len(item) > 3:
            return None
        if len(item) < 3:
            item = item + (slice(None),) * (3 - len(item))
        if any(val is None for val in item):
            return None
        return list(item)

    @staticmethod
    def _value_unit(raw):
        if isinstance(raw, Quantity):
            return raw.value, raw.unit
        return raw, None

    def _metadata_kwargs(self, raw):
        meta = {}
        for attr in getattr(GwpyArray, "_metadata_slots", ()):
            if hasattr(raw, attr):
                val = getattr(raw, attr)
            elif hasattr(self, attr):
                val = getattr(self, attr)
            else:
                continue
            if val is not None:
                meta[attr] = val
        return meta

    @staticmethod
    def _to_plain(raw):
        if isinstance(raw, Quantity):
            return raw.view(Quantity)
        return raw

    def _apply_axis_metadata(self, obj, order):
        names = [self._axis0_name, self._axis1_name, self._axis2_name]
        indices = [self._axis0_index, self._axis1_index, self._axis2_index]
        obj._axis0_name, obj._axis1_name, obj._axis2_name = [names[i] for i in order]
        obj._axis0_index, obj._axis1_index, obj._axis2_index = [
            indices[i] for i in order
        ]
        if getattr(obj, "_axis_names", None) is not None:
            obj._axis_names = [names[i] for i in order]

    def _swapaxes_int(self, a, b):
        new_data = GwpyArray.swapaxes(self, a, b)
        order = [0, 1, 2]
        order[a], order[b] = order[b], order[a]
        self._apply_axis_metadata(new_data, order)
        return new_data

    def _transpose_int(self, axes: tuple[int, ...]):
        if len(axes) != 3 or set(axes) != {0, 1, 2}:
            raise ValueError(f"Invalid transpose axes for 3D: {axes}")
        new_data = GwpyArray.transpose(self, axes)
        self._apply_axis_metadata(new_data, list(axes))
        return new_data

    def plane(self, drop_axis, drop_index, *, axis1=None, axis2=None):
        drop_axis_idx = self._get_axis_index(drop_axis)
        remaining = [i for i in range(3) if i != drop_axis_idx]
        target_indices = list(remaining)

        if axis1 is not None or axis2 is not None:
            idx1 = self._get_axis_index(axis1) if axis1 is not None else None
            idx2 = self._get_axis_index(axis2) if axis2 is not None else None

            if idx1 is not None and idx1 not in remaining:
                raise ValueError("axis1 must refer to a remaining axis")
            if idx2 is not None and idx2 not in remaining:
                raise ValueError("axis2 must refer to a remaining axis")
            if idx1 is not None and idx2 is not None and idx1 == idx2:
                raise ValueError("axis1 and axis2 cannot be the same axis")

            pool = [i for i in remaining if i not in (idx1, idx2)]
            if idx1 is None:
                idx1 = pool.pop(0)
            if idx2 is None:
                idx2 = pool.pop(0)

            target_indices = [idx1, idx2]

        sl = [slice(None)] * 3
        sl[drop_axis_idx] = drop_index
        sliced = self._getitem_with_axis_metadata(tuple(sl))
        if not isinstance(sliced, Plane2D):
            return sliced

        if target_indices != remaining:
            sliced = sliced.T

        return sliced
