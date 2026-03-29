from __future__ import annotations

from typing import Any, cast

import numpy as np


class SignalAnalysisMixin:
    """Mixin providing signal analysis methods like smoothing and peak finding."""

    value: Any
    unit: Any
    xunit: Any
    dt: Any
    df: Any

    def smooth(
        self, width: Any, method: str = "amplitude", ignore_nan: bool = True
    ) -> Any:
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
        from typing import cast

        h = cast(Any, self)

        # Determine value logic based on method
        # This logic is mostly common between Time/Frequency series if we abstract 'value' and 'unit'

        if method == "complex":
            from scipy.ndimage import uniform_filter1d

            def _smooth(x):
                if ignore_nan:
                    import pandas as pd

                    return (
                        pd.Series(x)
                        .rolling(window=width, center=True, min_periods=1)
                        .mean()
                        .values
                    )
                else:
                    return uniform_filter1d(x, size=width)

            re = _smooth(h.value.real)
            im = _smooth(h.value.imag)
            val = re + 1j * im
            unit = h.unit

        elif method == "amplitude":
            mag = np.abs(h.value)
            if ignore_nan:
                import pandas as pd

                val = (
                    pd.Series(mag)
                    .rolling(window=width, center=True, min_periods=1)
                    .mean()
                    .values
                )
            else:
                from scipy.ndimage import uniform_filter1d

                val = uniform_filter1d(mag, size=width)
            unit = h.unit
        elif method == "power":
            pwr = np.abs(h.value) ** 2
            if ignore_nan:
                import pandas as pd

                val = (
                    pd.Series(pwr)
                    .rolling(window=width, center=True, min_periods=1)
                    .mean()
                    .values
                )
            else:
                from scipy.ndimage import uniform_filter1d

                val = uniform_filter1d(pwr, size=width)
            unit = h.unit**2
        elif method == "db":
            # To dB
            mag = np.abs(h.value)
            with np.errstate(divide="ignore"):
                db = 20 * np.log10(mag)
            if ignore_nan:
                import pandas as pd

                val = (
                    pd.Series(db)
                    .rolling(window=width, center=True, min_periods=1)
                    .mean()
                    .values
                )
            else:
                from scipy.ndimage import uniform_filter1d

                val = uniform_filter1d(db, size=width)
            unit = u.Unit("dB")

        else:
            raise ValueError(f"Unknown smoothing method: {method}")

        # Construct new object.
        # We assume self.__class__ constructor takes (value, unit=...) plus other meta
        # Since TimeSeries/FrequencySeries constructors differ slightly (t0, dt vs f0, df),
        # we need to pass metadata blindly or clone self.

        # Safest way to clone without knowing specific args is usually slicing or creating and setting props
        # But we want to return a new object of the same type.

        # Try to infer constructor args or copy metadata
        h.copy()
        # Update data and unit
        # Note: gwpy series copy() is shallow-ish but creates new object.
        # Setting .value directly might be tricky if it's a property wrapper, but typically ok for bare arrays.
        # But changing unit usually requires creating new Quantity-like object or setting property.

        # A cleaner way for gwpy objects:
        matrix_cls = cast(Any, h.__class__)
        return matrix_cls(
            val,
            unit=unit,
            name=h.name,
            channel=h.channel,
            # Pass through extensive metadata if possible, but minimal set is safe
            **h._get_meta_for_constructor(),
        )

    def _get_meta_for_constructor(self):
        """Helper to extract metadata for reconstruction. Override in subclasses."""
        # Default fallback
        meta = {}
        if hasattr(self, "epoch"):
            meta["epoch"] = self.epoch
        # TimeSeries specific
        if hasattr(self, "sample_rate"):
            meta["sample_rate"] = self.sample_rate
        # FrequencySeries specific
        if hasattr(self, "frequencies") and len(self.frequencies) > 0:
            meta["frequencies"] = self.frequencies
        return meta

    def find_peaks(
        self,
        height: Any | None = None,
        threshold: Any | None = None,
        distance: Any | None = None,
        prominence: Any | None = None,
        width: Any | None = None,
        method: str = "amplitude",
        **kwargs: Any,
    ) -> Any:
        """
        Find peaks in the series.

        Wraps `scipy.signal.find_peaks` with support for unit quantities.
        """
        import scipy.signal

        h = cast(Any, self)

        # Prepare target array based on method
        if method == "amplitude":
            target = np.abs(h.value)
            current_unit = h.unit
        elif method == "power":
            target = np.abs(h.value) ** 2
            current_unit = h.unit**2 if h.unit else None
        elif method == "db":
            target = 20 * np.log10(np.abs(h.value))
            current_unit = None  # dB is unitless-ish
        else:
            raise ValueError(f"Unknown method {method}")

        def _to_val(x: Any, unit: Any = None) -> Any:
            if hasattr(x, "to") and unit is not None:
                return x.to(unit).value
            return getattr(x, "value", x)

        # Handle unit quantities for data-related parameters
        h = _to_val(height, current_unit)
        t = _to_val(threshold, current_unit)
        p = _to_val(prominence, current_unit)

        # Unit support for distance and width
        # This requires knowing the 'dx' (dt or df)
        dx = getattr(h, "dt", None)
        if dx is None:
            dx = getattr(h, "df", None)

        if dx is not None:
            dx_val = dx.value if hasattr(dx, "value") else dx
            dx_unit = getattr(dx, "unit", None)

            if dx_unit is None and hasattr(h, "xunit"):
                dx_unit = getattr(h, "xunit", None)

            if dx_unit is None or (
                hasattr(dx_unit, "physical_type")
                and dx_unit.physical_type == "dimensionless"
            ):
                # Fallback based on class type if unit is missing or dimensionless
                # meticulous heuristic checks
                cls_name = h.__class__.__name__
                if "Time" in cls_name:
                    from astropy import units as u

                    dx_unit = u.s
                elif "Frequency" in cls_name:
                    from astropy import units as u

                    dx_unit = u.Hz

            # Distance (time/frequency -> samples)
            dist = _to_val(distance)
            if distance is not None and hasattr(distance, "to") and dx_unit:
                dist = int(distance.to(dx_unit).value / dx_val)

            # Width (time/frequency -> samples)
            wid = width
            if width is not None:
                if np.iterable(width) and not isinstance(width, (str, bytes)):
                    wid = [
                        (
                            w.to(dx_unit).value / dx_val
                            if hasattr(w, "to")
                            else _to_val(w)
                        )
                        for w in width
                    ]
                    if isinstance(width, tuple):
                        wid = tuple(wid)
                elif hasattr(width, "to") and dx_unit:
                    wid = width.to(dx_unit).value / dx_val
                else:
                    wid = _to_val(width)
        else:
            dist = _to_val(distance)
            wid = _to_val(width)

        # Call scipy
        peaks_indices, props = scipy.signal.find_peaks(
            target,
            height=h,
            threshold=t,
            distance=dist,
            prominence=p,
            width=wid,
            **kwargs,
        )

        if len(peaks_indices) == 0:
            # Return empty container of same type
            series = cast(Any, self)
            return series[[]], props

        # Use slicing to return subset (preserves type and metadata)
        out = h[peaks_indices]
        if h.name:
            out.name = f"{h.name}_peaks"
        return out, props


class InteropMixin:
    """Mixin for interoperability methods (Torch, TensorFlow, etc)."""

    def to_torch(
        self,
        device: str | None = None,
        dtype: Any = None,
        requires_grad: bool = False,
        copy: bool = False,
    ) -> Any:
        """Convert to torch.Tensor."""
        from gwexpy.interop.torch_ import to_torch

        return to_torch(
            self, device=device, dtype=dtype, requires_grad=requires_grad, copy=copy
        )

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

    def to_dask(self, chunks="auto") -> Any:
        """Convert to Dask Array."""
        from gwexpy.interop.dask_ import to_dask

        return to_dask(self, chunks=chunks)

    def to_zarr(self, store, path=None, **kwargs) -> Any:
        """Save to Zarr storage."""
        from gwexpy.interop.zarr_ import to_zarr

        return to_zarr(self, store, path=path, **kwargs)
