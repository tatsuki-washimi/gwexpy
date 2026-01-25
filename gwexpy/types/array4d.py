"""4D Array with explicit axis management."""

import numpy as np
from astropy.units import Quantity, dimensionless_unscaled
from gwpy.types.array import Array as GwpyArray

from .array import Array
from .axis import AxisDescriptor, coerce_1d_quantity

__all__ = ["Array4D"]


class Array4D(Array):
    """4D Array with explicit axis management.

    This class extends :class:`Array` to provide explicit management of
    4 axes, each with a name and index (Quantity array).

    Parameters
    ----------
    data : array-like
        4-dimensional input data.
    unit : `~astropy.units.Unit`, optional
        Physical unit of the data.
    axis0 : `~astropy.units.Quantity` or array-like, optional
        Index values for axis 0 (1D).
    axis1 : `~astropy.units.Quantity` or array-like, optional
        Index values for axis 1 (1D).
    axis2 : `~astropy.units.Quantity` or array-like, optional
        Index values for axis 2 (1D).
    axis3 : `~astropy.units.Quantity` or array-like, optional
        Index values for axis 3 (1D).
    axis_names : iterable of str, optional
        Names for each axis (length 4). Defaults to
        ``["axis0", "axis1", "axis2", "axis3"]``.
    **kwargs
        Additional keyword arguments passed to :class:`Array`.

    Raises
    ------
    ValueError
        If the input data is not 4-dimensional.
    """

    _metadata_slots = Array._metadata_slots + (
        "_axis0_name",
        "_axis1_name",
        "_axis2_name",
        "_axis3_name",
        "_axis0_index",
        "_axis1_index",
        "_axis2_index",
        "_axis3_index",
    )

    def __new__(
        cls,
        data,
        unit=None,
        axis0=None,
        axis1=None,
        axis2=None,
        axis3=None,
        axis_names=None,
        **kwargs,
    ):
        obj = super().__new__(cls, data, unit=unit, **kwargs)
        if obj.ndim != 4:
            raise ValueError(f"Array4D must be 4-dimensional, got {obj.ndim}D")

        # Axis names
        if axis_names is None:
            obj._axis0_name = "axis0"
            obj._axis1_name = "axis1"
            obj._axis2_name = "axis2"
            obj._axis3_name = "axis3"
        else:
            if len(axis_names) != 4:
                raise ValueError("axis_names must be length 4")
            obj._axis0_name, obj._axis1_name, obj._axis2_name, obj._axis3_name = [
                str(x) for x in axis_names
            ]

        # Axis indices
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

        if axis3 is None:
            obj._axis3_index = np.arange(obj.shape[3]) * dimensionless_unscaled
        else:
            obj._axis3_index = coerce_1d_quantity(axis3)

        # Update generic axis_names if present
        if getattr(obj, "_axis_names", None) is not None:
            obj._axis_names = [
                obj._axis0_name,
                obj._axis1_name,
                obj._axis2_name,
                obj._axis3_name,
            ]

        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None:
            return

        def copy_attr(name, default):
            val = getattr(obj, name, None)
            if val is not None:
                setattr(self, name, val)
            elif getattr(self, name, None) is None:
                setattr(self, name, default)

        copy_attr("_axis0_name", "axis0")
        copy_attr("_axis1_name", "axis1")
        copy_attr("_axis2_name", "axis2")
        copy_attr("_axis3_name", "axis3")

        if self.ndim >= 1:  # Always check axis 0 if dimension exists
            val = getattr(obj, "_axis0_index", None)
            current = getattr(self, "_axis0_index", None)

            if (
                val is not None
                and getattr(obj, "ndim", 0) >= 1
                and obj.shape[0] == self.shape[0]
            ):
                # Force copy if self is default or None?
                if current is None:
                    self._axis0_index = val
                # Also overwrite if current is dimensionless (default) and val has unit
                elif (
                    hasattr(current, "unit") and current.unit == dimensionless_unscaled
                ):
                    if hasattr(val, "unit") and val.unit != dimensionless_unscaled:
                        self._axis0_index = val

            # Default fallback
            if getattr(self, "_axis0_index", None) is None:
                self._axis0_index = np.arange(self.shape[0]) * dimensionless_unscaled

        if self.ndim >= 2:
            val = getattr(obj, "_axis1_index", None)
            current = getattr(self, "_axis1_index", None)

            if (
                val is not None
                and getattr(obj, "ndim", 0) >= 2
                and obj.shape[1] == self.shape[1]
            ):
                if current is None:
                    self._axis1_index = val
                elif (
                    hasattr(current, "unit") and current.unit == dimensionless_unscaled
                ):
                    if hasattr(val, "unit") and val.unit != dimensionless_unscaled:
                        self._axis1_index = val

            if getattr(self, "_axis1_index", None) is None:
                self._axis1_index = np.arange(self.shape[1]) * dimensionless_unscaled

        if self.ndim >= 3:
            val = getattr(obj, "_axis2_index", None)
            current = getattr(self, "_axis2_index", None)

            if (
                val is not None
                and getattr(obj, "ndim", 0) >= 3
                and obj.shape[2] == self.shape[2]
            ):
                if current is None:
                    self._axis2_index = val
                elif (
                    hasattr(current, "unit") and current.unit == dimensionless_unscaled
                ):
                    if hasattr(val, "unit") and val.unit != dimensionless_unscaled:
                        self._axis2_index = val

            if getattr(self, "_axis2_index", None) is None:
                self._axis2_index = np.arange(self.shape[2]) * dimensionless_unscaled

        if self.ndim >= 4:
            val = getattr(obj, "_axis3_index", None)
            current = getattr(self, "_axis3_index", None)

            if (
                val is not None
                and getattr(obj, "ndim", 0) >= 4
                and obj.shape[3] == self.shape[3]
            ):
                if current is None:
                    self._axis3_index = val
                elif (
                    hasattr(current, "unit") and current.unit == dimensionless_unscaled
                ):
                    if hasattr(val, "unit") and val.unit != dimensionless_unscaled:
                        self._axis3_index = val

            if getattr(self, "_axis3_index", None) is None:
                self._axis3_index = np.arange(self.shape[3]) * dimensionless_unscaled

        if getattr(self, "_axis_names", None) is not None and self.ndim == 4:
            self._axis_names = [
                self._axis0_name,
                self._axis1_name,
                self._axis2_name,
                self._axis3_name,
            ]

    @property
    def axes(self):
        """Tuple of AxisDescriptor objects for each dimension."""
        return (
            AxisDescriptor(self._axis0_name, self._axis0_index),
            AxisDescriptor(self._axis1_name, self._axis1_index),
            AxisDescriptor(self._axis2_name, self._axis2_index),
            AxisDescriptor(self._axis3_name, self._axis3_index),
        )

    def _set_axis_name(self, index, name):
        """Set the name of a specific axis by index."""
        if index == 0:
            self._axis0_name = name
        elif index == 1:
            self._axis1_name = name
        elif index == 2:
            self._axis2_name = name
        elif index == 3:
            self._axis3_name = name
        else:
            raise IndexError(index)
        if (
            getattr(self, "_axis_names", None) is not None
            and len(self._axis_names) == 4
        ):
            self._axis_names = [
                self._axis0_name,
                self._axis1_name,
                self._axis2_name,
                self._axis3_name,
            ]

    def __getitem__(self, item):
        """Get item with axis metadata preservation."""
        return self._getitem_with_axis_metadata(item)

    def _isel_tuple(self, item_tuple):
        """Internal method for isel to use the same getitem logic."""
        return self._getitem_with_axis_metadata(item_tuple)

    def _getitem_with_axis_metadata(self, item):
        """Perform __getitem__ and reconstruct axis metadata."""
        raw = GwpyArray.__getitem__(self, item)
        items_list = self._normalize_item(item)
        if items_list is None:
            return self._to_plain(raw)

        current_axes = [
            (self._axis0_name, self._axis0_index),
            (self._axis1_name, self._axis1_index),
            (self._axis2_name, self._axis2_index),
            (self._axis3_name, self._axis3_index),
        ]
        kept_axes = []

        for i, sl in enumerate(items_list):
            name, idx_arr = current_axes[i]
            if isinstance(sl, slice):
                kept_axes.append((name, idx_arr[sl]))
            elif self._is_int_index(sl):
                # int index: axis is dropped
                continue
            else:
                return self._to_plain(raw)

        expected_ndim = len(kept_axes)
        if getattr(raw, "ndim", None) != expected_ndim:
            return self._to_plain(raw)

        if expected_ndim == 4:
            value, unit = self._value_unit(raw)
            meta = self._metadata_kwargs(raw)
            return Array4D(
                value,
                unit=unit,
                axis_names=[n for n, _ in kept_axes],
                axis0=kept_axes[0][1],
                axis1=kept_axes[1][1],
                axis2=kept_axes[2][1],
                axis3=kept_axes[3][1],
                copy=False,
                **meta,
            )

        # For lower dimensions, return plain array/Quantity
        return self._to_plain(raw)

    @staticmethod
    def _is_int_index(value):
        """Check if value is an integer index (not boolean)."""
        return isinstance(value, (int, np.integer)) and not isinstance(
            value, (bool, np.bool_)
        )

    @staticmethod
    def _normalize_item(item):
        """Normalize indexing item to a list of length 4.

        Returns None if the item cannot be normalized (e.g., contains None,
        multiple Ellipsis, or >4 elements).
        """
        if not isinstance(item, tuple):
            item = (item,)
        if any(val is None for val in item):
            return None
        if Ellipsis in item:
            if item.count(Ellipsis) > 1:
                return None
            ellipsis_idx = item.index(Ellipsis)
            num_specified = len(item) - 1
            fill = 4 - num_specified
            if fill < 0:
                return None
            item = (
                item[:ellipsis_idx] + (slice(None),) * fill + item[ellipsis_idx + 1 :]
            )
        if len(item) > 4:
            return None
        if len(item) < 4:
            item = item + (slice(None),) * (4 - len(item))
        if any(val is None for val in item):
            return None
        return list(item)

    @staticmethod
    def _value_unit(raw):
        """Extract value and unit from raw array."""
        if isinstance(raw, Quantity):
            return raw.value, raw.unit
        return raw, None

    def _metadata_kwargs(self, raw):
        """Extract metadata kwargs from raw array for reconstruction."""
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
        """Convert raw array to a plain Quantity if applicable."""
        if isinstance(raw, Quantity):
            return raw.view(Quantity)
        return raw

    def _apply_axis_metadata(self, obj, order):
        """Apply axis metadata to obj in the given order."""
        names = [
            self._axis0_name,
            self._axis1_name,
            self._axis2_name,
            self._axis3_name,
        ]
        indices = [
            self._axis0_index,
            self._axis1_index,
            self._axis2_index,
            self._axis3_index,
        ]
        (
            obj._axis0_name,
            obj._axis1_name,
            obj._axis2_name,
            obj._axis3_name,
        ) = [names[i] for i in order]
        (
            obj._axis0_index,
            obj._axis1_index,
            obj._axis2_index,
            obj._axis3_index,
        ) = [indices[i] for i in order]
        if getattr(obj, "_axis_names", None) is not None:
            obj._axis_names = [names[i] for i in order]

    def _swapaxes_int(self, a, b):
        """Swap two axes and update axis metadata."""
        new_data = GwpyArray.swapaxes(self, a, b)
        order = [0, 1, 2, 3]
        order[a], order[b] = order[b], order[a]
        self._apply_axis_metadata(new_data, order)
        return new_data

    def _transpose_int(self, axes: tuple[int, ...]):
        """Transpose axes and update axis metadata."""
        if len(axes) != 4 or set(axes) != {0, 1, 2, 3}:
            raise ValueError(f"Invalid transpose axes for 4D: {axes}")
        new_data = GwpyArray.transpose(self, axes)
        self._apply_axis_metadata(new_data, list(axes))
        return new_data
