"""
Core TimeSeries class definition and basic operations.

This module contains the base TimeSeries class with essential functionality:
- Basic operations (tail, crop, append)
- Regularity checking (is_regular, _check_regular)
- Peak finding
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Any

from gwpy.timeseries import TimeSeries as BaseTimeSeries


class TimeSeriesCore(BaseTimeSeries):
    """
    Core Ti meSeries class with basic operations.
    
    This is the base class that other mixins will extend.
    Inherits from gwpy.timeseries.TimeSeries for compatibility.
    """

    # ===============================
    # Properties
    # ===============================

    @property
    def is_regular(self) -> bool:
        """Return True if this TimeSeries has a regular sample rate."""
        # Use underlying index safely to avoid triggering GWpy AttributeErrors on irregular series
        try:
            # Try to get the index without triggering property logic that checks .dt
            idx = getattr(self, "xindex", None)
            if idx is None:
                return True
            if hasattr(idx, "regular"):
                 return idx.regular
            
            # Manual check
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

    # ===============================
    # Basic Operations
    # ===============================

    def tail(self, n: int = 5) -> "TimeSeriesCore":
        """Return the last `n` samples of this series."""
        if n is None:
            return self
        n = int(n)
        if n <= 0:
            return self[:0]
        return self[-n:]

    def crop(self, start: Any = None, end: Any = None, copy: bool = False) -> "TimeSeriesCore":
        """
        Crop this series to the given GPS start and end times.
        Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).
        """
        from gwexpy.time import to_gps
        # Convert inputs to GPS if provided
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
            
        return super().crop(start=start, end=end, copy=copy)

    def append(
        self,
        other: Any,
        inplace: bool = True,
        pad: Any = None,
        gap: Any = None,
        resize: bool = True,
    ) -> "TimeSeriesCore":
        """
        Append another TimeSeries (GWpy-compatible), returning gwexpy TimeSeries.
        """
        res = super().append(other, inplace=inplace, pad=pad, gap=gap, resize=resize)
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
    ) -> tuple["TimeSeriesCore", dict[str, Any]]:
        """
        Find peaks in the TimeSeries.
        
        Wraps `scipy.signal.find_peaks`.
        
        Returns
        -------
        peaks : TimeSeries
             A TimeSeries containing only the peak values at their corresponding times.
        properties : dict
             Properties returned by scipy.signal.find_peaks.
        """
        from scipy.signal import find_peaks
        
        # Handle unit quantities
        val = self.value
        
        def _to_val(x, unit=None):
             if hasattr(x, "value"):
                  if unit and hasattr(x, "to"):
                      return x.to(unit).value
                  return x.value
             return x
             
        # Height/Threshold: relative to data units
        h = _to_val(height, self.unit)
        t = _to_val(threshold, self.unit)
        p = _to_val(prominence, self.unit)  # Prominence same unit as data
        
        # Distance/Width: time or samples
        # Scipy uses samples.
        dist = distance
        wid = width
        
        if self.dt is not None:
             fs = self.sample_rate.to("Hz").value
             # If distance is time quantity
             if hasattr(dist, "to"):
                  dist = int(dist.to("s").value * fs)
             
             # If width is quantity (or tuple of quantities)
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
                  
        # Call scipy
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
             # Return empty
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
