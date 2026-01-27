from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

from .axis import AxisDescriptor

if TYPE_CHECKING:
    from .mixin._protocols import AxisApiHost

__all__ = ["AxisApiMixin"]


class AxisApiMixin(ABC):
    @property
    @abstractmethod
    def axes(self) -> tuple[AxisDescriptor, ...]:
        """Tuple of AxisDescriptor objects for each dimension.

        Returns
        -------
        tuple of AxisDescriptor
            Each descriptor contains the axis name and index values.
        """
        pass

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Names of all axes as a tuple of strings.

        Returns
        -------
        tuple of str
            The name of each axis in order.
        """
        return tuple(ax.name for ax in self.axes)

    def axis(self, key: int | str) -> AxisDescriptor:
        """Get an axis descriptor by index or name.

        Parameters
        ----------
        key : int or str
            Axis index (0-based) or name.

        Returns
        -------
        AxisDescriptor
            The requested axis descriptor.

        Raises
        ------
        KeyError
            If axis name not found.
        TypeError
            If key is not int or str.
        """
        if isinstance(key, int):
            return self.axes[key]
        elif isinstance(key, str):
            for ax in self.axes:
                if ax.name == key:
                    return ax
            raise KeyError(f"Axis '{key}' not found, available: {self.axis_names}")
        else:
            raise TypeError(f"Axis key must be int or str, got {type(key)}")

    def _get_axis_index(self, key: int | str) -> int:
        if isinstance(key, int):
            if key < 0:
                key += len(self.axes)
            if not 0 <= key < len(self.axes):
                raise IndexError(f"Axis index {key} out of range")
            return key
        elif isinstance(key, str):
            names = self.axis_names
            if key in names:
                return names.index(key)
            raise KeyError(f"Axis '{key}' not found in {names}")
        else:
            raise TypeError(f"Axis key must be int or str, got {type(key)}")

    def rename_axes(
        self: AxisApiHost, mapping: dict[str, str], *, inplace: bool = False
    ) -> Any:
        """Rename axes using a mapping of old names to new names.

        Parameters
        ----------
        mapping : dict
            Mapping from old axis names to new names.
        inplace : bool, optional
            If True, modify in place. Otherwise return a copy.

        Returns
        -------
        self or copy
        """
        if not inplace:
            new_obj = self.copy()
            new_obj.rename_axes(mapping, inplace=True)
            return new_obj

        old_names = self.axis_names
        new_names_list = list(old_names)

        for old, new in mapping.items():
            if old not in old_names:
                raise ValueError(f"Axis '{old}' not found in {old_names}")
            idx = old_names.index(old)
            new_names_list[idx] = new

        if len(set(new_names_list)) != len(new_names_list):
            raise ValueError(f"Duplicate axis names resulted: {new_names_list}")

        for i, new_name in enumerate(new_names_list):
            if new_name != old_names[i]:
                self._set_axis_name(i, new_name)

        return self

    def _set_axis_name(self, index: int, name: str):
        pass

    def isel(self, indexers=None, **kwargs):
        """Select by integer indices along specified axes.

        Parameters
        ----------
        indexers : dict, optional
            Mapping of axis name/index to integer index or slice.
        **kwargs
            Additional indexers as keyword arguments.

        Returns
        -------
        subset
            Sliced array.
        """
        if indexers is None:
            indexers = {}
        indexers = {**indexers, **kwargs}

        num_axes = len(self.axes)
        slices = [slice(None)] * num_axes

        for key, sel in indexers.items():
            axis_idx = self._get_axis_index(key)
            slices[axis_idx] = sel

        return self._isel_tuple(tuple(slices))

    def sel(self, indexers=None, *, method="nearest", **kwargs):
        """Select by coordinate values along specified axes.

        Parameters
        ----------
        indexers : dict, optional
            Mapping of axis name to coordinate value or slice.
        method : str, optional
            Selection method: 'nearest' (default).
        **kwargs
            Additional indexers as keyword arguments.

        Returns
        -------
        subset
            Sliced array at nearest coordinate values.
        """
        if indexers is None:
            indexers = {}
        indexers = {**indexers, **kwargs}

        isel_indexers = {}

        for key, val in indexers.items():
            ax = self.axis(key)
            ax_idx = self._get_axis_index(key)

            if isinstance(val, slice):
                isel_indexers[ax_idx] = ax.iloc_slice(val)
            else:
                isel_indexers[ax_idx] = ax.iloc_nearest(val)

        return self.isel(isel_indexers)

    def swapaxes(self: AxisApiHost, axis1: int | str, axis2: int | str) -> Any:
        idx1 = self._get_axis_index(axis1)
        idx2 = self._get_axis_index(axis2)
        if idx1 == idx2:
            return self.copy()
        return self._swapaxes_int(idx1, idx2)

    def transpose(self, *axes):
        """Permute the dimensions of an array."""
        # Normalize axes
        ndim = len(self.axes)
        if not axes:
            axes = tuple(range(ndim))[::-1]
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        axes = tuple(axes)

        if len(axes) != ndim:
            raise ValueError("axes don't match array")

        # Convert to int indices
        perm_int = [self._get_axis_index(ax) for ax in axes]

        return self._transpose_int(tuple(perm_int))

    @property
    def T(self):
        return self.transpose()

    @abstractmethod
    def _isel_tuple(self, item_tuple):
        pass

    @abstractmethod
    def _swapaxes_int(self, a: int, b: int):
        pass

    def _transpose_int(self, axes: tuple[int, ...]):
        """Default transpose implementation using mixin hooks if subclass handles simple swaps?
        Or delegate to super().transpose then fix metadata.
        """
        # Since we are mixin, we expect 'super()' to be the numpy array usually.
        # But calling super().transpose() might call generic object implementation?
        # We rely on subclass MRO to trigger numpy implementation.
        # But wait, if AxisApiMixin is first, super() is the next class (GwpyArray).

        base = cast(Any, super())
        new_obj = base.transpose(*axes)
        # new_obj is the transposed array (view).
        # We need to reorder metadata.

        # Original axes list
        old_axes_info = list(zip(range(len(self.axes)), self.axis_names))

        # Apply permutation to info
        # axes is the new order of dimension lookups.
        # i.e. new_obj axis i comes from self axis axes[i].

        if hasattr(new_obj, "_set_axis_name"):
            for i, origin_idx in enumerate(axes):
                origin_name = old_axes_info[origin_idx][1]
                new_obj._set_axis_name(i, origin_name)

        return new_obj
