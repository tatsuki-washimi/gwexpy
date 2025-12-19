
import numpy as np
from astropy.units import Quantity, dimensionless_unscaled

from .array3d import Array3D
from .plane2d import Plane2D
from .axis import AxisDescriptor, coerce_1d_quantity

__all__ = ["TimePlaneTransform"]

class TimePlaneTransform:
    """
    A container for 3D data with a preferred (time, axis1, axis2) structure,
    commonly produced by time-frequency or similar transforms.
    
    This class wraps an Array3D to enforce semantic structure:
    - Axis 0 is "time"
    - Axis 1 and 2 are symmetric spatial/frequency dimensions.
    """

    def __init__(self, data3d, *, kind="custom", meta=None):
        """
        Initialize a TimePlaneTransform.

        Parameters
        ----------
        data3d : Array3D or tuple
            The underlying 3D data.
            Preferred: an `Array3D` instance.
            Supported tuple format: (value, time_axis, axis1, axis2, unit/metadata).
        kind : str, optional
            A string describing the type of transform (e.g., "stlt", "bispectrum"). Default is "custom".
        meta : dict, optional
            Additional metadata dictionary. Default is None (stored as empty dict).
        """
        self._kind = str(kind)
        self._meta = meta if meta is not None else {}

        if isinstance(data3d, Array3D):
            self._data = data3d
        elif isinstance(data3d, tuple):
            # (value, time_axis, axis1, axis2, unit/metadata)
            # unpack loosely
            if len(data3d) == 5:
                val, t_ax, ax1, ax2, u_or_m = data3d
                unit = u_or_m if not isinstance(u_or_m, dict) else None
                extra_meta = u_or_m if isinstance(u_or_m, dict) else {}
            elif len(data3d) == 4:
                val, t_ax, ax1, ax2 = data3d
                unit = None
                extra_meta = {}
            else:
                 raise ValueError("Tuple data3d must be length 4 or 5: (value, time, ax1, ax2, [unit/meta])")
            
            # Construct names from provided axes objects if possible, else defaults.
            # But the tuple typically provides coordinates (Quantity/array), not AxisDescriptor.
            # We assign default names "time", "axis1", "axis2" unless user provided descriptors (unlikely/simple case).
            self._data = Array3D(
                val,
                unit=unit,
                axis_names=["time", "axis1", "axis2"],
                axis0=t_ax,
                axis1=ax1,
                axis2=ax2,
                **extra_meta
            )
        else:
            raise TypeError("data3d must be an Array3D or a tuple.")
        
        # Validation
        if self._data.ndim != 3:
            raise ValueError(f"TimePlaneTransform requires 3D data, got {self._data.ndim}D")
        
    @property
    def kind(self):
        return self._kind
    
    @property
    def meta(self):
        return self._meta
    
    @property
    def value(self):
        return self._data.value
    
    @property
    def shape(self):
        return self._data.shape
    
    @property
    def ndim(self):
        return self._data.ndim
    
    @property
    def unit(self):
        return self._data.unit

    @property
    def axes(self):
        """
        Return the 3 AxisDescriptors: (time, axis1, axis2).
        """
        return self._data.axes
    
    @property
    def times(self):
        """Coordinate array of the time axis (axis 0)."""
        return self.axes[0].index
    
    @property
    def axis1(self):
        """AxisDescriptor for the first symmetric axis (axis 1)."""
        return self.axes[1]

    @property
    def axis2(self):
        """AxisDescriptor for the second symmetric axis (axis 2)."""
        return self.axes[2]

    def _axis_id(self, key):
        """
        Resolve axis key (int or str) to integer index (0, 1, 2).
        """
        if isinstance(key, int):
            if key < 0: key += 3
            if not (0 <= key <= 2):
                raise ValueError(f"Axis index {key} out of range (0-2)")
            return key
        
        if isinstance(key, str):
            # Check names
            for i, ax in enumerate(self.axes):
                if ax.name == key:
                    return i
            # Special case: "time" always maps to 0 if not explicitly named something else?
            # User said: "axis0 name should be time by convention, but do not enforce... match against axes[i].name"
            # But "TimePlaneTransform is user-facing", so 'time' alias for axis 0 is reasonable if not conflicting?
            # User instruction: "if str: match against axes[i].name; else raise KeyError"
            raise KeyError(f"No axis named '{key}' found. Axes are: {[ax.name for ax in self.axes]}")
        
        raise TypeError(f"Invalid axis key type: {type(key)}")

    def plane(self, drop_axis, drop_index, *, axis1=None, axis2=None):
        """
        Extract a 2D plane by slicing at a specific index along one axis.

        Parameters
        ----------
        drop_axis : int or str
            The axis to slice along (remove).
        drop_index : int
            Integer index to select along `drop_axis`.
        axis1 : str or int, optional
            The generic axis 1 for the resulting Plane2D.
        axis2 : str or int, optional
            The generic axis 2 for the resulting Plane2D.
        
        Returns
        -------
        Plane2D
        """
        # Resolve drop_axis to int to pass to Array3D (or let Array3D handle it if it handles names)
        # Array3D.plane is internal-ish? No, public. It calls _get_axis_index.
        # However, we want to support standard "time" referencing even if underlying axis is named "t".
        # But for now let's stick to strict name matching or int, via self._axis_id if we want full control.
        # But Array3D.plane() signature is `plane(drop_axis, drop_index, *, axis1=None, axis2=None)`
        # If we pass strings to Array3D.plane, it uses its own valid names.
        # If we use `_axis_id` here, we get an int, which is safe to pass to Array3D.plane.
        
        idx = self._axis_id(drop_axis)    
        
        # If axis1/axis2 args are provided, they might be strings referring to names in THIS object.
        # Array3D.plane expects names or ints.
        
        # We delegate.
        # Note: Array3D.plane expects drop_index to be int/slice? Spec says "drop_index: int (isel-style)".
        return self._data.plane(idx, drop_index, axis1=axis1, axis2=axis2)

    def at_time(self, t, *, method="nearest"):
        """
        Extract a Plane2D at the specific time `t`.
        
        Parameters
        ----------
        t : Quantity or float
            Time value. If float, assumed to be in the unit of the time axis.
        method : str, optional
            "nearest" (default). Future versions may support interpolation.
            
        Returns
        -------
        Plane2D
        """
        # 1. Resolve time index
        time_ax = self.axes[0]
        
        if method == "nearest":
            idx = time_ax.iloc_nearest(t)
        else:
            raise NotImplementedError(f"Method '{method}' not implemented yet.")
        
        # 2. Slice
        # drop_axis=0 (time)
        return self.plane(0, idx)

    def to_array3d(self):
        """Return the underlying Array3D object (advanced usage)."""
        return self._data
