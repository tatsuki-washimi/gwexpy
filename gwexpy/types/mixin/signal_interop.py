from typing import Any, Optional
import numpy as np



class SignalAnalysisMixin:
    """Mixin providing signal analysis methods like smoothing and peak finding."""

    def smooth(self, width: Any, method: str = "amplitude", ignore_nan: bool = True) -> Any:
        """
        Smooth the series.

        Parameters
        ----------
        width : `int`
            Number of samples for the smoothing winow.
        method : `str`, optional
            Smoothing target: 'amplitude', 'power', 'complex', 'db'.
        ignore_nan : `bool`, optional
            If True, ignore NaNs.

        Returns
        -------
        Series
            Smoothed series.
        """
        from astropy import units as u

        # Determine value logic based on method
        # This logic is mostly common between Time/Frequency series if we abstract 'value' and 'unit'

        if method == 'complex':
            from scipy.ndimage import uniform_filter1d

            def _smooth(x):
                if ignore_nan:
                    import pandas as pd
                    return pd.Series(x).rolling(window=width, center=True, min_periods=1).mean().values
                else:
                    return uniform_filter1d(x, size=width)

            re = _smooth(self.value.real)
            im = _smooth(self.value.imag)
            val = re + 1j * im
            unit = self.unit

        elif method == 'amplitude':
            mag = np.abs(self.value)
            if ignore_nan:
                import pandas as pd
                val = pd.Series(mag).rolling(window=width, center=True, min_periods=1).mean().values
            else:
                from scipy.ndimage import uniform_filter1d
                val = uniform_filter1d(mag, size=width)
            unit = self.unit
        elif method == 'power':
            pwr = np.abs(self.value)**2
            if ignore_nan:
                import pandas as pd
                val = pd.Series(pwr).rolling(window=width, center=True, min_periods=1).mean().values
            else:
                from scipy.ndimage import uniform_filter1d
                val = uniform_filter1d(pwr, size=width)
            unit = self.unit**2
        elif method == 'db':
            # To dB
            mag = np.abs(self.value)
            with np.errstate(divide='ignore'):
                 db = 20 * np.log10(mag)
            if ignore_nan:
                import pandas as pd
                val = pd.Series(db).rolling(window=width, center=True, min_periods=1).mean().values
            else:
                from scipy.ndimage import uniform_filter1d
                val = uniform_filter1d(db, size=width)
            unit = u.Unit('dB')

        else:
            raise ValueError(f"Unknown smoothing method: {method}")

        # Construct new object.
        # We assume self.__class__ constructor takes (value, unit=...) plus other meta
        # Since TimeSeries/FrequencySeries constructors differ slightly (t0, dt vs f0, df),
        # we need to pass metadata blindly or clone self.

        # Safest way to clone without knowing specific args is usually slicing or creating and setting props
        # But we want to return a new object of the same type.

        # Try to infer constructor args or copy metadata
        self.copy()
        # Update data and unit
        # Note: gwpy series copy() is shallow-ish but creates new object.
        # Setting .value directly might be tricky if it's a property wrapper, but typically ok for bare arrays.
        # But changing unit usually requires creating new Quantity-like object or setting property.

        # A cleaner way for gwpy objects:
        return self.__class__(
            val,
            unit=unit,
            name=self.name,
            channel=self.channel,
            # Pass through extensive metadata if possible, but minimal set is safe
            **self._get_meta_for_constructor()
        )

    def _get_meta_for_constructor(self):
        """Helper to extract metadata for reconstruction. Override in subclasses."""
        # Default fallback
        meta = {}
        if hasattr(self, 'epoch'): meta['epoch'] = self.epoch
        # TimeSeries specific
        if hasattr(self, 'sample_rate'): meta['sample_rate'] = self.sample_rate
        # FrequencySeries specific
        if hasattr(self, 'frequencies') and len(self.frequencies) > 0:
             meta['frequencies'] = self.frequencies
        return meta

    def find_peaks(self, threshold: Optional[float] = None, method: str = "amplitude", **kwargs: Any) -> Any:
        """
        Find peaks in the series.

        Wraps `scipy.signal.find_peaks`.
        """
        import scipy.signal  # noqa: F401 - availability check

        # Prepare target array
        if method == 'amplitude':
            target = np.abs(self.value)
        elif method == 'power':
            target = np.abs(self.value)**2
        elif method == 'db':
            target = 20 * np.log10(np.abs(self.value))
        else:
            raise ValueError(f"Unknown method {method}")

        if threshold is not None:
            if hasattr(threshold, 'unit'):  # astropy.units.Quantity
                current_unit = self.unit
                if method == 'power': current_unit = current_unit**2
                elif method == 'db': current_unit = None # db is unitless-ish

                if current_unit and hasattr(threshold, 'to'):
                    threshold = threshold.to(current_unit).value
                elif hasattr(threshold, 'value'):
                     threshold = threshold.value

            kwargs['height'] = threshold

        # Unit support for distance and width
        # This requires knowing the 'dx' (dt or df)
        dx = getattr(self, "dt", None) or getattr(self, "df", None)

        if dx is not None:
            dx_val = dx.value if hasattr(dx, "value") else dx
            dx_unit = getattr(dx, "unit", None)

            if dx_unit is None and hasattr(self, "xunit"):
                 dx_unit = getattr(self, "xunit", None)

            if dx_unit is None or (hasattr(dx_unit, 'physical_type') and dx_unit.physical_type == 'dimensionless'):
                # Fallback based on class type if unit is missing or dimensionless
                # meticulous heuristic checks
                cls_name = self.__class__.__name__
                if "Time" in cls_name:
                    from astropy import units as u
                    dx_unit = u.s
                elif "Frequency" in cls_name:
                    from astropy import units as u
                    dx_unit = u.Hz

            # Distance (dx -> samples)
            dist = kwargs.get('distance', None)
            if dist is not None and hasattr(dist, "to") and dx_unit:
                # Convert dist to dx_unit then divide by dx_val
                kwargs['distance'] = int(dist.to(dx_unit).value / dx_val)

            # Width (dx -> samples)
            wid = kwargs.get('width', None)
            if wid is not None:
                 def _convert_width(w):
                     if hasattr(w, "to") and dx_unit:
                         return w.to(dx_unit).value / dx_val
                     return w

                 if np.iterable(wid):
                      kwargs['width'] = tuple(_convert_width(w) for w in wid) if isinstance(wid, tuple) else [_convert_width(w) for w in wid]
                 else:
                      kwargs['width'] = _convert_width(wid)

        peaks_indices, props = scipy.signal.find_peaks(target, **kwargs)

        if len(peaks_indices) == 0:
             # Return empty container of same type
             return self[[]], props

        # Use slicing to return subset (preserves type and metadata)
        out = self[peaks_indices]
        if self.name:
            out.name = f"{self.name}_peaks"
        return out, props


class InteropMixin:
    """Mixin for interoperability methods (Torch, TensorFlow, etc)."""

    def to_torch(
        self,
        device: Optional[str] = None,
        dtype: Any = None,
        requires_grad: bool = False,
        copy: bool = False,
    ) -> Any:
        """Convert to torch.Tensor."""
        from gwexpy.interop.torch_ import to_torch
        return to_torch(self, device=device, dtype=dtype, requires_grad=requires_grad, copy=copy)

    def to_tensorflow(self, dtype: Any = None) -> Any:
        """Convert to tensorflow.Tensor."""
        from gwexpy.interop.tensorflow_ import to_tf
        return to_tf(self, dtype=dtype)

    def to_jax(self) -> Any:
        """Convert to JAX Array."""
        from gwexpy.interop.jax_ import to_jax
        return to_jax(self)

    def to_cupy(self, dtype=None) -> Any:
        """Convert to CuPy Array."""
        from gwexpy.interop.cupy_ import to_cupy
        return to_cupy(self, dtype=dtype)

    def to_dask(self, chunks='auto') -> Any:
        """Convert to Dask Array."""
        from gwexpy.interop.dask_ import to_dask
        return to_dask(self, chunks=chunks)

    def to_zarr(self, store, path=None, **kwargs) -> Any:
        """Save to Zarr storage."""
        from gwexpy.interop.zarr_ import to_zarr
        return to_zarr(self, store, path=path, **kwargs)
