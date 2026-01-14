import numpy as np
from astropy.units import dimensionless_unscaled

from .array3d import Array3D

__all__ = ["TimePlaneTransform", "LaplaceGram"]


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
                raise ValueError(
                    "Tuple data3d must be length 4 or 5: (value, time, ax1, ax2, [unit/meta])"
                )

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
                **extra_meta,
            )
        else:
            raise TypeError("data3d must be an Array3D or a tuple.")

        # Validation
        if self._data.ndim != 3:
            raise ValueError(
                f"TimePlaneTransform requires 3D data, got {self._data.ndim}D"
            )

    @property
    def kind(self):
        """String describing the type of transform (e.g., 'stlt', 'bispectrum')."""
        return self._kind

    @property
    def meta(self):
        """Additional metadata dictionary."""
        return self._meta

    @property
    def value(self):
        """The underlying data values as numpy array."""
        return self._data.value

    @property
    def shape(self):
        """Shape of the 3D data array (time, axis1, axis2)."""
        return self._data.shape

    @property
    def ndim(self):
        """Number of dimensions (always 3)."""
        return self._data.ndim

    @property
    def unit(self):
        """Physical unit of the data values."""
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
            if key < 0:
                key += 3
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
            raise KeyError(
                f"No axis named '{key}' found. Axes are: {[ax.name for ax in self.axes]}"
            )

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
        # Access index quantity directly locally to perform interpolation logic
        times = time_ax.index

        # Ensure t is comparable (value check)
        if hasattr(t, "unit") and hasattr(times, "unit") and times.unit is not None:
            t_val = t.to(times.unit).value
            times_val = times.value
        elif hasattr(t, "value"):  # t is Quantity, times might be dimensionless
            t_val = t.value
            times_val = times.value if hasattr(times, "value") else times
        else:
            t_val = t
            times_val = times.value if hasattr(times, "value") else times

        if method == "nearest":
            # idx = time_ax.iloc_nearest(t)
            # Use direct numpy for consistency with linear logic if desired, but reuse existing if reliable.
            # Assuming axis class works, but let's be robust:
            idx = np.abs(times_val - t_val).argmin()
            return self.plane(0, idx)

        elif method == "linear":
            # Find insertion point
            i = np.searchsorted(times_val, t_val, side="right")

            # Clamp to boundaries
            if i <= 0:
                return self.plane(0, 0)
            if i >= len(times_val):
                return self.plane(0, len(times_val) - 1)

            idx_prev = i - 1
            idx_next = i

            t_prev = times_val[idx_prev]
            t_next = times_val[idx_next]

            # Check for zero duration interval (duplicates)
            denom = t_next - t_prev
            if denom == 0:
                return self.plane(0, idx_prev)

            alpha = (t_val - t_prev) / denom

            p_prev = self.plane(0, idx_prev)
            p_next = self.plane(0, idx_next)

            # Interpolate values
            # Plane2D should expose .value? Checked file_view earlier, doesn't explicitly show Plane2D def but usage implies it.
            # TimePlaneTransform.value uses self._data.value.
            # We assume p_prev.value is accessible (standard gwexpy object).

            val_interp = (1.0 - alpha) * p_prev.value + alpha * p_next.value

            # Construct new Plane2D
            # Plane2D(data, axis1_name=..., axis2_name=..., yindex=..., xindex=...)
            # Plane2D axis1 -> Array2D axis0 (yindex)
            # Plane2D axis2 -> Array2D axis1 (xindex)

            return p_prev.__class__(
                val_interp,
                unit=p_prev.unit,
                axis1_name=p_prev.axis1.name,
                axis2_name=p_prev.axis2.name,
                yindex=p_prev.axis1.index,
                xindex=p_prev.axis2.index,
            )

        else:
            raise NotImplementedError(
                f"Method '{method}' not implemented. Supported: 'nearest', 'linear'."
            )

        # 2. Slice (nearest case handled above)
        # return self.plane(0, idx)

    def at_sigma(self, sigma):
        """
        Extract a 2D plane (Spectrogram-like) at a specific sigma index (if axis1 is sigma)
        or value.

        This assumes axis 1 is sigma.
        """
        # If sigma is an index (int)
        if hasattr(self, "axis1") and self.axis1.name == "sigma":
            # We can try to match value if needed, but strict index for now?
            # User requested `.at_sigma(sigma)` which implies value lookup potentially.
            # Similar to at_time logic.
            # Let's assume simplest implementation: nearest lookup or index.

            sigma_ax = self.axis1.index
            if isinstance(sigma, int) and (
                not hasattr(sigma_ax, "shape") or sigma < len(sigma_ax)
            ):
                # It might be an index
                # But if sigma is 0.0 (float), it's a value.
                pass

            # Implementation of value lookup
            s_val = sigma
            if hasattr(sigma, "value"):
                s_val = sigma.value

            ax_val = sigma_ax
            if hasattr(sigma_ax, "value"):
                ax_val = sigma_ax.value

            if isinstance(s_val, (int, np.integer)) and (
                s_val >= 0 and s_val < len(ax_val)
            ):
                # Ambiguous if values are integers. Assume value first?
                # But typically sigma is float.
                pass

            # Nearest neighbor
            idx = np.abs(ax_val - s_val).argmin()
            return self.plane(1, idx)
        else:
            raise ValueError(
                "This transform does not appear to have 'sigma' as axis 1."
            )

    def to_array3d(self):
        """Return the underlying Array3D object (advanced usage)."""
        return self._data


class LaplaceGram(TimePlaneTransform):
    """
    3D container for Short-Time Laplace Transform data.
    Structure: (time, sigma, frequency).
    """

    def __init__(self, data3d, **kwargs):
        # Enforce axis names if Array3D is created here?
        # User passes constructed Array3D usually.
        # Just ensure base init works.
        super().__init__(data3d, **kwargs)
        if self.axis1.name != "sigma":
            # Try to rename if generic
            if self.axis1.name == "axis1":
                self._data._set_axis_name(1, "sigma")
        if self.axis2.name != "frequency":
            if self.axis2.name == "axis2":
                self._data._set_axis_name(2, "frequency")

    @property
    def sigmas(self):
        """The sigma (Re[s]) axis index."""
        return self.axis1.index

    @property
    def frequencies(self):
        """The frequency axis index."""
        return self.axis2.index

    def normalize_per_sigma(self, eps=1e-30):
        """
        Normalize the magnitude of the transform along the frequency axis (last axis).
        Each (time, sigma) slice is normalized such that the sum over frequency bins is 1.
        This helps in identifying poles by equalizing the energy contribution across different damping values.

        Parameters
        ----------
        eps : float, optional
            Small value to avoid division by zero. Default is 1e-30.

        Returns
        -------
        LaplaceGram
            A new LaplaceGram instance containing the normalized magnitude.
        """
        # Calculate magnitude
        mag = np.abs(self.value)

        # Sum over frequency axis (axis 2 / -1)
        denom = np.sum(mag, axis=-1, keepdims=True)
        denom = np.maximum(denom, eps)

        # Normalize
        norm_data = mag / denom

        # Create new Array3D for the result
        # Normalized data is effectively a probability distribution or shape profile, so dimensionless
        new_data = Array3D(
            norm_data,
            unit=dimensionless_unscaled,
            axis_names=[self.axes[0].name, self.axis1.name, self.axis2.name],
            axis0=self.times,
            axis1=self.axis1.index,
            axis2=self.axis2.index,
        )

        # Return wrapped in LaplaceGram
        return LaplaceGram(new_data, kind=self.kind, meta=self.meta.copy())
