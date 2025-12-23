
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Union

from .axis import AxisDescriptor

__all__ = ["AxisApiMixin"]

class AxisApiMixin(ABC):
    
    @property
    @abstractmethod
    def axes(self) -> Tuple[AxisDescriptor, ...]:
        pass

    @property
    def axis_names(self) -> Tuple[str, ...]:
        return tuple(ax.name for ax in self.axes)

    def axis(self, key: Union[int, str]) -> AxisDescriptor:
        if isinstance(key, int):
            return self.axes[key]
        elif isinstance(key, str):
            for ax in self.axes:
                if ax.name == key:
                    return ax
            raise KeyError(f"Axis '{key}' not found, available: {self.axis_names}")
        else:
            raise TypeError(f"Axis key must be int or str, got {type(key)}")

    def _get_axis_index(self, key: Union[int, str]) -> int:
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

    def rename_axes(self, mapping: Dict[str, str], *, inplace=False):
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

    def swapaxes(self, axis1, axis2):
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
            axes = axes[0]
        
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
    
    def _transpose_int(self, axes: Tuple[int, ...]):
        """Default transpose implementation using mixin hooks if subclass handles simple swaps? 
           Or delegate to super().transpose then fix metadata.
        """
        # Since we are mixin, we expect 'super()' to be the numpy array usually.
        # But calling super().transpose() might call generic object implementation?
        # We rely on subclass MRO to trigger numpy implementation.
        # But wait, if AxisApiMixin is first, super() is the next class (GwpyArray).
        
        new_obj = super().transpose(*axes)
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
