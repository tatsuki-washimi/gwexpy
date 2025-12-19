
import numpy as np
from astropy.units import Quantity, dimensionless_unscaled

from .array import Array
from .axis import AxisDescriptor, coerce_1d_quantity
from .plane2d import Plane2D

__all__ = ["Array3D"]

class Array3D(Array):
    """
    3D Array with explicit axis management.
    """
    _metadata_slots = Array._metadata_slots + (
        "_axis0_name", "_axis1_name", "_axis2_name",
        "_axis0_index", "_axis1_index", "_axis2_index",
    )

    def __new__(cls, data, unit=None, axis0=None, axis1=None, axis2=None,
                axis_names=None, **kwargs):
        obj = super().__new__(cls, data, unit=unit, **kwargs)
        if obj.ndim != 3:
             raise ValueError(f"Array3D must be 3-dimensional, got {obj.ndim}D")

        if axis_names is None:
            obj._axis0_name = "axis0"
            obj._axis1_name = "axis1"
            obj._axis2_name = "axis2"
        else:
            if len(axis_names) != 3: raise ValueError("axis_names must be length 3")
            obj._axis0_name, obj._axis1_name, obj._axis2_name = [str(x) for x in axis_names]

        if axis0 is None: obj._axis0_index = np.arange(obj.shape[0]) * dimensionless_unscaled
        else: obj._axis0_index = coerce_1d_quantity(axis0)

        if axis1 is None: obj._axis1_index = np.arange(obj.shape[1]) * dimensionless_unscaled
        else: obj._axis1_index = coerce_1d_quantity(axis1)

        if axis2 is None: obj._axis2_index = np.arange(obj.shape[2]) * dimensionless_unscaled
        else: obj._axis2_index = coerce_1d_quantity(axis2)

        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None: return
        
        def copy_attr(name, default):
            if getattr(self, name, None) is None:
                val = getattr(obj, name, None)
                if val is not None: setattr(self, name, val)
                else: setattr(self, name, default)

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

    @property
    def axes(self):
        return (
            AxisDescriptor(self._axis0_name, self._axis0_index),
            AxisDescriptor(self._axis1_name, self._axis1_index),
            AxisDescriptor(self._axis2_name, self._axis2_index),
        )

    def _set_axis_name(self, index, name):
        if index == 0: self._axis0_name = name
        elif index == 1: self._axis1_name = name
        elif index == 2: self._axis2_name = name
        else: raise IndexError(index)

    def _isel_tuple(self, item_tuple):
        new_data = self.__getitem__(item_tuple)
        if new_data.ndim == 0:
            return new_data 
        
        # Reconstruct axes tracking logic (same as before)
        current_axes = [
            (self._axis0_name, self._axis0_index),
            (self._axis1_name, self._axis1_index),
            (self._axis2_name, self._axis2_index),
        ]
        kept_axes = []
        
        items_list = list(item_tuple)
        if len(items_list) < 3:
             items_list += [slice(None)] * (3 - len(items_list))
             
        def slice_axis(idx_array, sl):
            if isinstance(sl, slice): return idx_array[sl]
            try:
                 int(sl); return None
            except: 
                 return idx_array[sl] 
        
        for i, sl in enumerate(items_list):
            if i >= 3: break
            name, idx_arr = current_axes[i]
            new_idx = slice_axis(idx_arr, sl)
            if new_idx is not None:
                kept_axes.append((name, new_idx))
        
        if new_data.ndim == 3:
            if len(kept_axes) != 3: return new_data 
            return Array3D(
                new_data, 
                axis_names=[n for n, _ in kept_axes],
                axis0=kept_axes[0][1], axis1=kept_axes[1][1], axis2=kept_axes[2][1]
            )
        elif new_data.ndim == 2:
            if len(kept_axes) != 2: return new_data
            p_obj = Plane2D(
                new_data,
                axis1_name=kept_axes[0][0], axis2_name=kept_axes[1][0],
                yindex=kept_axes[0][1], xindex=kept_axes[1][1]  
            )
            # Force indices to ensure correct axis mapping
            p_obj.yindex = kept_axes[0][1]
            p_obj.xindex = kept_axes[1][1]
            return p_obj
        else:
            if new_data.ndim == 0: return new_data
            # For 1D, return simple Quantity/Array?
            # Or perhaps just return new_data as is (which is already an Array subclass instance)
            if new_data.ndim == 1:
                # If we don't have Array1D, just return it.
                return new_data
            raise NotImplementedError("Unexpected dim return from Array3D slicing")

    def _swapaxes_int(self, a, b):
        new_data = GwpyArray.swapaxes(self, a, b) # numpy swap
        # Manually swap metadata on new object
        # Note: new_data is a view/copy of Array3D.
        # It has metadata copied via __array_finalize__.
        
        # We need to swap names and indices to match data swap
        
        # Helper to swap attrs
        def swap_attr(obj, attr_list, i, j):
            v1 = getattr(obj, attr_list[i])
            v2 = getattr(obj, attr_list[j])
            setattr(obj, attr_list[i], v2)
            setattr(obj, attr_list[j], v1)
            
        names = ["_axis0_name", "_axis1_name", "_axis2_name"]
        indices = ["_axis0_index", "_axis1_index", "_axis2_index"]
        
        swap_attr(new_data, names, a, b)
        swap_attr(new_data, indices, a, b)
        
        return new_data

    def plane(self, drop_axis, drop_index, *, axis1=None, axis2=None):
        drop_axis_idx = self._get_axis_index(drop_axis)
        all_indices = {0, 1, 2}
        remaining = sorted(list(all_indices - {drop_axis_idx}))
        target_indices = list(remaining)
        
        if axis1 is not None or axis2 is not None:
            pool = list(remaining)
            idx1 = -1
            idx2 = -1
            
            if axis1 is not None:
                idx1 = self._get_axis_index(axis1)
                if idx1 in pool: pool.remove(idx1)
            if axis2 is not None:
                idx2 = self._get_axis_index(axis2)
                if idx2 in pool: pool.remove(idx2)
            
            if axis1 is None: idx1 = pool.pop(0)
            if axis2 is None: idx2 = pool.pop(0)
                
            if idx1 == idx2: raise ValueError("axis1 and axis2 cannot be the same axis")
            if idx1 == drop_axis_idx or idx2 == drop_axis_idx: raise ValueError("cannot use dropped axis as output axis")
                
            target_indices = [idx1, idx2]

        sl = [slice(None)] * 3
        sl[drop_axis_idx] = drop_index
        sliced = self._isel_tuple(tuple(sl))
        
        # If we got Plane2D, check order
        if isinstance(sliced, Plane2D):
             # _isel_tuple returns in remaining order 
             current_order = remaining
             if target_indices != current_order:
                 sliced = sliced.T
                 
        return sliced
