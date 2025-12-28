"""
gwexpy.timeseries.timeseries - Extended TimeSeries class

This module provides the main TimeSeries class for gwexpy, which extends
gwpy's TimeSeries with additional functionality.

The implementation is modularized across several files:
- _core.py: Core class definition and basic operations
- _spectral.py: Spectral transforms (FFT, CWT, etc.)
- _timeseries_legacy.py: Remaining methods (signal, resampling, analysis, interop)

This module integrates all Mixins into a single TimeSeries class.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from typing import Optional, Any

# Import Mixins
from ._spectral import TimeSeriesSpectralMixin
from ._signal import TimeSeriesSignalMixin
from ._resampling import TimeSeriesResamplingMixin
from ._analysis import TimeSeriesAnalysisMixin
from ._interop import TimeSeriesInteropMixin
from gwexpy.fitting.mixin import FittingMixin

# Import legacy for remaining methods
from ._timeseries_legacy import TimeSeries as _LegacyTimeSeries


class TimeSeries(
    TimeSeriesInteropMixin,    # Interoperability (highest priority)
    TimeSeriesAnalysisMixin,   # Analysis
    TimeSeriesResamplingMixin, # Resampling
    TimeSeriesSignalMixin,     # Signal processing
    TimeSeriesSpectralMixin,   # Spectral transforms
    FittingMixin,              # Fitting functionality
    _LegacyTimeSeries,         # All legacy methods including BaseTimeSeries
):




    """
    Extended TimeSeries with all gwexpy functionality.

    This class combines functionality from multiple modules:
    - Core operations: is_regular, _check_regular, tail, crop, append, find_peaks
    - Spectral transforms: fft, psd, cwt, laplace, etc.
    - Signal processing: analytic_signal, mix_down, xcorr, etc.
    - Analysis: impute, standardize, rolling_*, etc.
    - Interoperability: to_pandas, to_torch, to_xarray, etc.

    Inherits from gwpy.timeseries.TimeSeries for full compatibility.
    """

    def __new__(cls, data, *args, **kwargs):
        from gwexpy.timeseries.utils import _coerce_t0_gps

        should_coerce = True
        xunit = kwargs.get("xunit", None)
        if xunit is not None:
            try:
                should_coerce = u.Unit(xunit).is_equivalent(u.s)
            except (ValueError, TypeError):
                should_coerce = False
        else:
            dt = kwargs.get("dt", None)
            if isinstance(dt, u.Quantity):
                phys = getattr(dt.unit, "physical_type", None)
                if dt.unit != u.dimensionless_unscaled and phys != "time":
                    should_coerce = False

        if should_coerce:
            if "t0" in kwargs and kwargs["t0"] is not None:
                kwargs["t0"] = _coerce_t0_gps(kwargs["t0"])
            if "epoch" in kwargs and kwargs["epoch"] is not None:
                kwargs["epoch"] = _coerce_t0_gps(kwargs["epoch"])
        return super().__new__(cls, data, *args, **kwargs)

    # ===============================
    # Override methods from _core.py
    # (These take precedence over _LegacyTimeSeries versions)
    # ===============================

    @property
    def is_regular(self) -> bool:
        """Return True if this TimeSeries has a regular sample rate."""
        try:
            idx = getattr(self, "xindex", None)
            if idx is None:
                return True
            if hasattr(idx, "regular"):
                 return idx.regular

            times_val = np.asarray(idx)
            if len(times_val) < 2:
                return True
            diffs = np.diff(times_val)
            return np.allclose(diffs, diffs[0], atol=1e-12, rtol=1e-10)
        except (AttributeError, ValueError, TypeError):
            return False

    def _check_regular(self, method_name: Optional[str] = None):
        """Helper to ensure the series is regular before applying certain transforms."""
        if not self.is_regular:
            method = method_name or "This method"
            raise ValueError(
                f"{method} requires a regular sample rate (constant dt). "
                "Consider using .asfreq() or .interpolate() to regularized the series first."
            )

    def tail(self, n: int = 5) -> "TimeSeries":
        """Return the last `n` samples of this series."""
        if n is None:
            return self
        n = int(n)
        if n <= 0:
            return self[:0]
        return self[-n:]

    def crop(self, start: Any = None, end: Any = None, copy: bool = False) -> "TimeSeries":
        """
        Crop this series to the given GPS start and end times.
        Accepts any time format supported by gwexpy.time.to_gps.
        """
        from gwexpy.time import to_gps
        if start is not None:
             start = to_gps(start)
             if isinstance(start, (np.ndarray, list)) and np.ndim(start) > 0:
                 start = start[0]
             start = float(start)
        if end is not None:
             end = to_gps(end)
             if isinstance(end, (np.ndarray, list)) and np.ndim(end) > 0:
                 end = end[0]
             end = float(end)

        from gwpy.timeseries import TimeSeries as BaseTimeSeries
        return BaseTimeSeries.crop(self, start=start, end=end, copy=copy)

    def append(
        self,
        other: Any,
        inplace: bool = True,
        pad: Any = None,
        gap: Any = None,
        resize: bool = True,
    ) -> "TimeSeries":
        """
        Append another TimeSeries (GWpy-compatible), returning gwexpy TimeSeries.
        """
        from gwpy.timeseries import TimeSeries as BaseTimeSeries
        res = BaseTimeSeries.append(self, other, inplace=inplace, pad=pad, gap=gap, resize=resize)
        if inplace:
            return self
        if isinstance(res, self.__class__):
            return res
        return self.__class__(
            res.value,
            times=res.times,
            unit=res.unit,
            name=res.name,
            channel=getattr(res, "channel", None),
        )

    def find_peaks(
        self,
        height: Any = None,
        threshold: Any = None,
        distance: Any = None,
        prominence: Any = None,
        width: Any = None,
        wlen: Optional[int] = None,
        rel_height: float = 0.5,
        plateau_size: Any = None,
    ) -> tuple["TimeSeries", dict[str, Any]]:
        """
        Find peaks in the TimeSeries.

        Wraps `scipy.signal.find_peaks`.

        Returns
        -------
        peaks : `TimeSeries`
            A new TimeSeries containing only the peak values, indexed by their times.
        props : `dict`
            Dictionary of peak properties returned by `scipy.signal.find_peaks`.
        """
        from scipy.signal import find_peaks

        val = self.value

        def _to_val(x, unit=None):
             if hasattr(x, "value"):
                  if unit and hasattr(x, "to"):
                      return x.to(unit).value
                  return x.value
             return x

        h = _to_val(height, self.unit)
        t = _to_val(threshold, self.unit)
        p = _to_val(prominence, self.unit)

        dist = distance
        wid = width

        if self.dt is not None:
             fs = self.sample_rate.to("Hz").value
             if hasattr(dist, "to"):
                  dist = int(dist.to("s").value * fs)

             if np.iterable(wid):
                  new_wid = []
                  for w in wid:
                       if hasattr(w, "to"):
                            new_wid.append(w.to("s").value * fs)
                       else:
                            new_wid.append(w)
                  wid = tuple(new_wid) if isinstance(wid, tuple) else new_wid
             elif hasattr(wid, "to"):
                  wid = wid.to("s").value * fs

        peaks_indices, props = find_peaks(
             val,
             height=h,
             threshold=t,
             distance=dist,
             prominence=p,
             width=wid,
             wlen=wlen,
             rel_height=rel_height,
             plateau_size=plateau_size
        )

        if len(peaks_indices) == 0:
             return self.__class__([], times=[], unit=self.unit, name=self.name, channel=self.channel), props

        peak_times = self.times[peaks_indices]
        peak_vals = val[peaks_indices]

        out = self.__class__(
             peak_vals,
             times=peak_times,
             unit=self.unit,
             name=f"{self.name}_peaks" if self.name else "peaks",
             channel=self.channel
        )
        return out, props


__all__ = ["TimeSeries"]
