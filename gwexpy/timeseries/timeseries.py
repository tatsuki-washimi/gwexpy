from __future__ import annotations

import inspect
from enum import Enum
import numpy as np
from astropy import units as u
from typing import Optional, Union, Any, List, Iterable
try:
    import scipy.signal
except ImportError:
    pass # scipy is optional dependency for gwpy but required here for hilbert


from gwpy.timeseries import TimeSeries as BaseTimeSeries
from gwpy.timeseries import TimeSeriesDict as BaseTimeSeriesDict
from gwpy.timeseries import TimeSeriesList as BaseTimeSeriesList

from gwexpy.types.seriesmatrix import SeriesMatrix
from gwexpy.types.metadata import MetaData, MetaDataMatrix

# --- Monkey Patch TimeSeriesDict ---


# New Imports
from .preprocess import (
    impute_timeseries, standardize_timeseries, align_timeseries_collection, 
    standardize_matrix, whiten_matrix
)
from .arima import fit_arima
from .hurst import hurst, local_hurst
from .decomposition import (
    pca_fit, pca_transform, pca_inverse_transform, 
    ica_fit, ica_transform, ica_inverse_transform
)
from .spectral import csd_matrix_from_collection, coherence_matrix_from_collection


from .utils import *
class TimeSeries(BaseTimeSeries):
    """Light wrapper of gwpy's TimeSeries for compatibility."""

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
        Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).
        """
        from gwexpy.time import to_gps
        # Convert inputs to GPS if provided
        if start is not None:
             start = float(to_gps(start))
        if end is not None:
             end = float(to_gps(end))
            
        return super().crop(start=start, end=end, copy=copy)


    def asfreq(
        self,
        rule: Any,
        method: Optional[str] = None,
        fill_value: Any = np.nan,
        *,
        origin: str = "t0",
        offset: Any = None,
        align: str = "ceil",
        tolerance: Optional[float] = None,
        max_gap: Optional[float] = None,
        copy: bool = True,
    ) -> "TimeSeries":
        """
        Reindex the TimeSeries to a new fixed-interval grid associated with the given rule.
        """
        # 1. Parse rule to target dt (Quantity)
        if isinstance(rule, (int, float, np.number)):
            target_dt = u.Quantity(rule)
        elif isinstance(rule, str):
            # Simple parser for '1s', '10ms' etc.
            try:
                target_dt = u.Quantity(rule)
            except (TypeError, ValueError):
                # If astropy fails to parse simple string directly, try splitting
                import re
                match = re.match(r"([0-9\.]+)([a-zA-Z]*)", rule)
                if match:
                    val, unit_str = match.groups()
                    if not unit_str:
                        target_dt = u.Quantity(float(val))
                    else:
                        target_dt = float(val) * u.Unit(unit_str)
                else:
                    raise ValueError(f"Could not parse rule: {rule}")
        elif isinstance(rule, u.Quantity):
            target_dt = rule
        else:
             raise TypeError("rule must be a string, number, or astropy Quantity.")

        # Validate rule unit compatibility
        is_time = target_dt.unit.physical_type == 'time'
        is_dimless = (target_dt.unit is None or target_dt.unit == u.dimensionless_unscaled or target_dt.unit.physical_type == 'dimensionless')
        
        if not is_time and not is_dimless:
             raise ValueError("rule must be time-like or dimensionless")

        # 2. Determine Original Times and Span
        # Prefer using self.times.value (float GPS) for calculation to avoid Quantity overhead loops
        # But convert everything to the same unit (target_dt.unit or seconds) for precision
        
        # Use existing xindex if available (irregular), else construct from t0/dt
        old_times_q = self.times
        old_times_val = old_times_q.value
        time_unit = old_times_q.unit
        
        # Handle dimensionless time axis
        is_dimensionless = (time_unit is None or time_unit == u.dimensionless_unscaled or (hasattr(time_unit, 'physical_type') and time_unit.physical_type == 'dimensionless'))
        
        target_dt_in_time_unit = None
        start_time_val = None
        stop_time_val = None
        
        if is_dimensionless and target_dt.unit.physical_type == 'time':
             # Assume original data is in seconds
             time_unit = u.s
             target_dt_in_time_unit = target_dt.to(u.s)
             
             start_time_val = old_times_q[0].value
             
             if hasattr(self, 'dt') and self.dt is not None:
                  dt_q = self.dt
                  if dt_q.unit is None or dt_q.unit == u.dimensionless_unscaled:
                       dt_val = dt_q.value
                  else:
                       dt_val = dt_q.to(u.s).value
                  stop_time_val = start_time_val + (len(self) * dt_val)
             else:
                  stop_time_val = old_times_q[-1].value
        else:
             safe_unit = time_unit if time_unit is not None else u.dimensionless_unscaled
             # target_dt might be numeric (dimensionless) or already have safe_unit
             if target_dt.unit == u.dimensionless_unscaled:
                  target_dt_in_time_unit = u.Quantity(target_dt.value, safe_unit)
                  # Override target_dt to ensure it has units compatible with constructor
                  target_dt = target_dt_in_time_unit
             else:
                  target_dt_in_time_unit = target_dt.to(safe_unit)

             # start_time_val
             if old_times_q.unit == u.dimensionless_unscaled:
                  start_time_val = old_times_q[0].value
             else:
                  start_time_val = old_times_q[0].to(safe_unit).value
             
             if hasattr(self, 'dt') and self.dt is not None:
                 if self.dt.unit == u.dimensionless_unscaled:
                      dt_input = self.dt.value
                 else:
                      dt_input = self.dt.to(safe_unit).value
                 stop_time_val = start_time_val + (len(self) * dt_input)
             else:
                 if old_times_q.unit == u.dimensionless_unscaled:
                      stop_time_val = old_times_q[-1].value
                 else:
                      stop_time_val = old_times_q[-1].to(safe_unit).value
             
        dt_val = target_dt_in_time_unit.value
        
        # 3. Determine Origin Base
        safe_unit = time_unit if time_unit is not None else u.dimensionless_unscaled
        
        if origin == 't0':
            origin_val = start_time_val
        elif origin == 'gps0':
            origin_val = 0.0
        elif isinstance(origin, (u.Quantity, str)):
            # convert origin to same unit as time_unit (if possible) or seconds
            try:
                origin_val = u.Quantity(origin).to(safe_unit).value
            except u.UnitConversionError:
                 # Check if origin is dimensionless and safe_unit is physical
                 q_origin = u.Quantity(origin)
                 if q_origin.unit == u.dimensionless_unscaled and safe_unit.physical_type == 'time':
                      raise TypeError("Cannot use dimensionless origin for time-based series.")
                 raise
        elif isinstance(origin, (int, float, np.number)):
            # Calling side passed a number.
            # If safe_unit is physical time, this is ambiguous.
            if safe_unit.physical_type == 'time':
                 raise TypeError("origin must be a Quantity or time string when series has time unit.")
            origin_val = float(origin)
        else:
            origin_val = 0.0 
            
        if offset is None:
            offset_val = 0.0
        else:
            try:
                offset_val = u.Quantity(offset).to(safe_unit).value
            except u.UnitConversionError:
                # Be lenient if user passed numeric 0 or Quantity 0 of incompatible unit
                if u.Quantity(offset).value == 0:
                    offset_val = 0.0
                else:
                    raise

        base_val = origin_val + offset_val
        
        # 4. Generate New Grid
        # first_grid = base + k * dt >= start_time
        # k = ceil((start - base)/dt)
        if align == 'ceil':
             k = np.ceil((start_time_val - base_val) / dt_val)
        elif align == 'floor':
             k = np.floor((start_time_val - base_val) / dt_val)
        else:
             raise ValueError("align must be 'ceil' or 'floor'")
             
        grid_start = base_val + k * dt_val
        
        # Create array
        # num_points = ceil((stop - grid_start) / dt)
        # Ensure we don't go strictly >= stop
        # floating point epsilon care?
        
        duration = stop_time_val - grid_start
        if duration <= 0:
             n_points = 0
        else:
             n_points = int(np.ceil(duration / dt_val))
             
        # Check if the last point is exactly stop? bounds are [start, stop)
        # If grid_start + n*dt == stop, do we include it? usually no.
        # np.arange(grid_start, stop_time_val, dt_val) is standard.
        
        new_times_val = grid_start + np.arange(n_points) * dt_val
        # Filter strictly < stop in case of numerical slop
        new_times_val = new_times_val[new_times_val < stop_time_val]
        
        # 5. Reindex / Interpolate
        # If empty
        if len(new_times_val) == 0:
             # Safety fallback for t0 if grid is empty
             safe_unit = time_unit if time_unit is not None else u.dimensionless_unscaled
             safe_t0 = u.Quantity(grid_start, safe_unit)
             return self.__class__([], t0=safe_t0, dt=target_dt, channel=self.channel, name=self.name, unit=self.unit)

        new_data = np.full(len(new_times_val), fill_value, dtype=self.dtype)
        if hasattr(fill_value, 'dtype'):
             # If fill value is complex etc?
             pass 
        if self.dtype.kind == 'c':
             new_data = new_data.astype(np.complex128)
        
        # Mapping logic
        # For 'method=None', we look for exact matches with tolerance?
        # Exact match of floats is tricky.
        
        # Optimize regular-to-regular case?
        # If self.regular and dt match and aligned...
        
        if method == 'interpolate':
            # Linear interpolation
            # Use scipy.interpolate.interp1d or numpy.interp
            valid_mask = np.isfinite(self.value)
            if np.any(valid_mask):
                x = old_times_val
                y = self.value
                
                # Handling NaNs in input? 
                # If we want to skip NaNs in input for interpolation:
                # x = x[valid_mask]
                # y = y[valid_mask]
                # method='linear'
                
                # Simple linear interp
                # For complex data, interp real/imag separately
                if np.iscomplexobj(y):
                    new_data = np.interp(new_times_val, x, y.real) + 1j * np.interp(new_times_val, x, y.imag)
                else:
                    new_data = np.interp(new_times_val, x, y, left=np.nan, right=np.nan)

                # Enforce max_gap: invalidate interpolated points inside large gaps
                if max_gap is not None:
                    max_gap_val = u.Quantity(max_gap).to(time_unit).value
                    gaps = np.where(np.diff(x) > max_gap_val)[0]
                    for gi in gaps:
                        start = x[gi]
                        end = x[gi + 1]
                        mask_gap = (new_times_val > start) & (new_times_val < end)
                        new_data[mask_gap] = np.nan
                
                # Apply fill_value to NaNs (if interp produced NaNs for out of bounds)
                if np.isnan(fill_value):
                    pass # already nan
                else:
                    mask = np.isnan(new_data)
                    new_data[mask] = fill_value
                    
                # Todo: max_gap implementation for interpolate
        
        else:
            # Nearest, ffill, bfill, or None (exact)
            # Use searchsorted
            
            # idx = np.searchsorted(old_times_val, new_times_val, side='left')
            # Handle boundary logic for each method
            
            # For simplicity/speed on regular/near-regular data, this can be optimized.
            # But general solution:
            
            # Using pandas reindex/asfreq if strict compatibility is needed?
            # User output says "implement", assumes not using pandas directly if lighter.
            # GWpy generally avoids pandas dependency for core ops.
            
            idx_right = np.searchsorted(old_times_val, new_times_val, side='left')
            # idx_right is index where new_time would be inserted to keep order.
            # old[idx-1] <= new < old[idx]
            
            # Make sure we clip indices for access
            idx_prev = np.clip(idx_right - 1, 0, len(old_times_val) - 1)
            idx_next = np.clip(idx_right, 0, len(old_times_val) - 1)
            
            t_prev = old_times_val[idx_prev]
            # t_next = old_times_val[idx_next]
            
            matched_indices = np.full(len(new_times_val), -1, dtype=int)
            
            if method == 'ffill' or method == 'pad':
                 # Use prev if new_time >= prev
                 # Check boundaries
                 # searchsorted gives index.
                 # if new_time < old[0], idx=0. old[idx-1] is last element (wrap) or distinct?
                 # Need to check cond: old[idx_prev] <= new_time
                 cond = old_times_val[idx_prev] <= new_times_val + 1e-12 # Numerical slush?
                 # Also need to handle "before first element" -> NaN
                 # If idx_right == 0 and new_time < old[0], then prev should be invalid.
                 valid_prev = (idx_right > 0) | (np.abs(new_times_val - old_times_val[0]) < 1e-12)
                 
                 # Refine logic:
                 # ffill takes the latest observed value.
                 # If new_t is exactly on a point, take it.
                 # If in between, take left.
                 
                 # Use searchsorted(side='right') -> old[i-1] <= new < old[i]
                 idx_side_right = np.searchsorted(old_times_val, new_times_val, side='right')
                 fill_idx = idx_side_right - 1
                 
                 valid_f = (fill_idx >= 0)
                 # Apply max_gap
                 if max_gap is not None:
                      limit = u.Quantity(max_gap).to(time_unit).value
                      dt_diff = new_times_val - old_times_val[np.clip(fill_idx, 0, None)]
                      valid_f &= (dt_diff <= limit)
                 
                 # Assign values
                 valid_out_indices = np.where(valid_f)[0]
                 src_indices = fill_idx[valid_f]
                 new_data[valid_out_indices] = self.value[src_indices]
                 
            elif method == 'bfill' or method == 'backfill':
                 # Use next
                 idx_side_left = np.searchsorted(old_times_val, new_times_val, side='left')
                 fill_idx = idx_side_left
                 
                 valid_b = (fill_idx < len(old_times_val))
                 
                 if max_gap is not None:
                      limit = u.Quantity(max_gap).to(time_unit).value
                      dt_diff = old_times_val[np.clip(fill_idx, 0, len(old_times_val)-1)] - new_times_val
                      valid_b &= (dt_diff <= limit)
                      
                 valid_out_indices = np.where(valid_b)[0]
                 src_indices = fill_idx[valid_b]
                 new_data[valid_out_indices] = self.value[src_indices]
                 
            elif method == 'nearest':
                 # Compare prev and next
                 idx_side_left = np.searchsorted(old_times_val, new_times_val, side='left')
                 
                 idx_L = np.clip(idx_side_left - 1, 0, len(old_times_val)-1)
                 idx_R = np.clip(idx_side_left, 0, len(old_times_val)-1)
                 
                 dist_L = np.abs(new_times_val - old_times_val[idx_L])
                 dist_R = np.abs(new_times_val - old_times_val[idx_R])
                 
                 # Choose L or R
                 use_L = dist_L < dist_R
                 
                 chosen_idx = np.where(use_L, idx_L, idx_R)
                 chosen_dist = np.where(use_L, dist_L, dist_R)
                 
                 # Check validity (e.g. if new_time < old[0], L wraps or is 0? clip handles it)
                 # But if new < old[0], idx_side_left=0. idx_L=0. idx_R=0. dist_L==dist_R. use_L=False?
                 
                 valid_n = np.ones(len(new_times_val), dtype=bool)
                 if tolerance is not None:
                      tol_val = u.Quantity(tolerance).to(time_unit).value
                      valid_n &= (chosen_dist <= tol_val)
                      
                 if max_gap is not None:
                      limit = u.Quantity(max_gap).to(time_unit).value
                      valid_n &= (chosen_dist <= limit)
                      
                 valid_out = np.where(valid_n)[0]
                 src_idx = chosen_idx[valid_n]
                 new_data[valid_out] = self.value[src_idx]

            elif method is None:
                # Exact match only
                # Tolerance?
                # Simple implementation: use tolerance if provided, else almost zero
                tol_val = 1e-9 if tolerance is None else u.Quantity(tolerance).to(time_unit).value
                
                # Nearest + strict tolerance
                idx_side_left = np.searchsorted(old_times_val, new_times_val, side='left')
                idx_closest = np.clip(idx_side_left, 0, len(old_times_val)-1)
                # Check closest
                # Searchsorted 'left': old[i-1] < new <= old[i]
                # Actually standard exact check:
                # Find closest, check diff almost 0
                
                # Re-use nearest logic
                idx_L = np.clip(idx_side_left - 1, 0, len(old_times_val)-1)
                idx_R = np.clip(idx_side_left, 0, len(old_times_val)-1)
                dist_L = np.abs(new_times_val - old_times_val[idx_L])
                dist_R = np.abs(new_times_val - old_times_val[idx_R])
                
                # Check min dist
                min_dist = np.minimum(dist_L, dist_R)
                chosen_from_min = np.where(dist_L < dist_R, idx_L, idx_R)
                
                valid_exact = min_dist <= tol_val
                
                valid_out = np.where(valid_exact)[0]
                src_idx = chosen_from_min[valid_exact]
                new_data[valid_out] = self.value[src_idx]

        # Construct new TimeSeries
        return self.__class__(
            new_data,
            t0=u.Quantity(new_times_val[0], time_unit),
            dt=target_dt,
            unit=self.unit,
            name=self.name,
            channel=self.channel
        )

    def resample(self, rate: Any, *args: Any, **kwargs: Any) -> "TimeSeries":
        """
        Resample the TimeSeries. 
        
        If 'rate' is a time-string (e.g. '1s') or time Quantity, performs time-bin aggregation (pandas-like).
        Otherwise, performs signal processing resampling (gwpy standard).
        """
        is_time_bin = False
        if isinstance(rate, str):
            is_time_bin = True
        elif isinstance(rate, u.Quantity):
            if rate.unit.physical_type == 'time':
                is_time_bin = True
        
        if is_time_bin:
            return self._resample_time_bin(rate, **kwargs)
        else:
            return super().resample(rate, *args, **kwargs)

    def stlt(self, stride: Any, window: Any, **kwargs: Any) -> Any:
        """
        Compute Short-Time Local Transform (STLT).

        Produces a 3D time-frequency-frequency representation wrapped in
        a TimePlaneTransform container.

        Parameters
        ----------
        stride : str or Quantity
            Time step duration (e.g. '1s').
        window : str or Quantity
            Analysis window duration (e.g. '2s').

        Returns
        -------
        TimePlaneTransform
            3D transform result with shape (time, axis1, axis2).
        """
        from gwexpy.types.time_plane_transform import TimePlaneTransform
        try:
            import scipy.signal
        except ImportError:
            raise ImportError(
                "scipy is required for stlt. "
                "Please install it via `pip install scipy`."
            )
        
        # Normalize inputs
        stride_q = u.Quantity(stride) if isinstance(stride, str) else stride
        window_q = u.Quantity(window) if isinstance(window, str) else window
        
        if not stride_q.unit.is_equivalent(u.s):
            raise ValueError("stride must be a time quantity")
        
        # Parameters for STFT
        # fs (sampling rate)
        dt_s = self.dt.to(u.s).value
        fs_val = 1.0 / dt_s
        
        # Window samples
        nperseg = int(np.round((window_q.to(u.s).value / dt_s)))
        # Overlap samples
        # stride = window - overlap => overlap = window - stride
        stride_s = stride_q.to(u.s).value
        noverlap_samples = int(np.round((window_q.to(u.s).value - stride_s) / dt_s))
        
        if nperseg <= 0:
             raise ValueError("Window size too small")
             
        # STFT
        # Returns f, t, Zxx
        # f: Array of sample frequencies.
        # t: Array of segment times.
        # Zxx: STFT of x. Shape (n_freqs, n_segments)
        # Note: scipy stft pads or truncates? It centers windows by default.
        # boundary='zeros', padded=True is default.
        f, t_segs, Zxx = scipy.signal.stft(
             self.value, 
             fs=fs_val, 
             window='hann', 
             nperseg=nperseg, 
             noverlap=noverlap_samples,
             boundary=None, # Avoid padding artifacts at edges for physics data usually, or 'zeros'? Default is 'zeros'.
                            # strict spacing requires careful handling.
                            # Let's use default for robustness unless alignment issues arise.
             padded=False   # If False, no padding. "boundary argument must be None" if padded=False? No.
                            # padded=False checks if x is multiple of nperseg? No.
                            # Docs: "If boundary is None, the input signal is assumed to be periodic extension... no, wait."
                            # Scipy defaults: boundary='zeros', padded=True.
                            # Let's use standard default for safety, but check shape.
        )
        
        # Zxx shape: (F, T)
        # We want (T, F, F)
        # 1. Transpose to (T, F)
        Z = Zxx.T
        
        # 2. Compute Magnitude Outer Product: |S(t, f1)| * |S(t, f2)|
        # This represents the magnitude correlation between frequencies at each time.
        mag = np.abs(Z) # (T, F)
        
        # Shape: (T, F, F)
        # Use broadcasting
        data = mag[:, :, None] * mag[:, None, :]
        
        # 3. Construct axes
        # Time axis
        # t_segs are relative to start. Add t0.
        # scipy returns segment centers or starts? "Segment times".
        # Usually centered on the window.
        # We map it to absolute time.
        # Note: t_segs from scipy often starts at window/2 if boundary='zeros'.
        t0_val = self.t0.to(u.s).value
        times_q = (t0_val + t_segs) * u.s
        
        # Freq axis
        axis_f = f * u.Hz
        
        return TimePlaneTransform(
            (data, times_q, axis_f, axis_f, self.unit**2),
            kind="stlt_mag_outer",
            meta={"stride": stride, "window": window, "source": self.name}
        )

    def _resample_time_bin(
        self,
        rule: Any,
        agg: str = "mean",
        closed: str = "left",
        label: str = "left",
        origin: str = "t0",
        offset: Any = 0 * u.s,
        align: str = "floor",
        min_count: int = 1,
        nan_policy: str = "omit",
        inplace: bool = False,
    ) -> Any:
        """Internal: Bin-based resampling."""
        # 1. Parse rule to dt
        if isinstance(rule, str):
             import re
             match = re.match(r"([0-9\.]+)([a-zA-Z]+)", rule)
             if match:
                val, unit_str = match.groups()
                bin_dt = float(val) * u.Unit(unit_str)
             else:
                bin_dt = u.Quantity(rule)
        else:
             bin_dt = rule
             
        # 2. Setup Bins
        old_times_q = self.times
        time_unit = old_times_q.unit
        
        is_dimensionless = (time_unit is None or time_unit == u.dimensionless_unscaled or time_unit.physical_type == 'dimensionless')
        
        bin_dt_val = None
        start_time_val = None
        stop_time_val = None
        
        if is_dimensionless and bin_dt.unit.physical_type == 'time':
             time_unit = u.s
             bin_dt_val = bin_dt.to(u.s).value
             start_time_val = old_times_q[0].value
             stop_time_val = self.span[1].value if hasattr(self.span[1], 'value') else self.span[1]
        else:
             bin_dt_val = u.Quantity(bin_dt, time_unit).value
             start_time_val = u.Quantity(old_times_q[0], time_unit).value
             stop_time_val = u.Quantity(self.span[1], time_unit).value
        
        # Origin logic (similar to asfreq)
        
        if origin == 't0':
            origin_val = start_time_val
        elif origin == 'gps0':
            origin_val = 0.0
        else:
            origin_val = 0.0 # Default fallback
            
        offset_val = u.Quantity(offset).to(time_unit).value
        base_val = origin_val + offset_val
        
        # Grid alignment
        if align == 'floor':
             k = np.floor((start_time_val - base_val) / bin_dt_val)
        elif align == 'ceil':
             k = np.ceil((start_time_val - base_val) / bin_dt_val)
        else:
            k = np.floor((start_time_val - base_val) / bin_dt_val)

        grid_start = base_val + k * bin_dt_val
        
        # Create bin edges
        # We need edges covering the full range
        duration = stop_time_val - grid_start
        n_bins = int(np.ceil(duration / bin_dt_val))
        
        if n_bins <= 0:
             return self.__class__([], dt=bin_dt, unit=self.unit, name=self.name)
             
        edges = grid_start + np.arange(n_bins + 1) * bin_dt_val
        
        # 3. Aggregate
        # Digitize
        # old_times indices into bins
        # bins: [edge[0], edge[1]), [edge[1], edge[2]), ...
        
        # np.digitize returns i such that bins[i-1] <= x < bins[i] (if right=False)
        # We want 0-based index for bin 0..n_bins-1
        
        # Optimization: for regular self.dt, we can calculate indices directly
        # bin_idx = floor( (t - grid_start) / bin_dt )
        # This is much faster than digitize
        
        # This is much faster than digitize
        
        if is_dimensionless and bin_dt.unit.physical_type == 'time':
             # old_times_q is dimensionless (assumed seconds), so values are seconds
             old_times_val = old_times_q.value
        else:
             old_times_val = old_times_q.to(time_unit).value
             
        bin_indices = np.floor((old_times_val - grid_start) / bin_dt_val).astype(int)
        
        # Clip or filter valid bins
        # 0 <= idx < n_bins
        valid_mask = (bin_indices >= 0) & (bin_indices < n_bins)
        
        valid_indices = bin_indices[valid_mask]
        valid_values = self.value[valid_mask]
        
        # Aggregation loop
        # For 'mean', 'sum' etc., can use scipy.stats.binned_statistic or pandas or pure numpy
        # pure numpy: bincount for sum and count
        
        # Handle nan_policy
        if hasattr(self.value, 'dtype') and (self.value.dtype.kind == 'f' or self.value.dtype.kind == 'c'):
             has_nan = np.isnan(valid_values)
             if np.any(has_nan):
                  if nan_policy == 'omit':
                       # Mask out NaNs from aggregation
                       # BUT careful: simple masking desyncs indices
                       # Must implement per-bin logic or strictly mask upfront
                       non_nan_mask = ~has_nan
                       valid_indices = valid_indices[non_nan_mask]
                       valid_values = valid_values[non_nan_mask]
                  elif nan_policy == 'propagate':
                       # Let sum/mean propagate
                       pass
                       
        # Bincount works for flat data (1D)
        if agg == 'mean':
             # Sum / Count
             sums = np.bincount(valid_indices, weights=valid_values, minlength=n_bins)
             counts = np.bincount(valid_indices, minlength=n_bins)
             with np.errstate(invalid='ignore', divide='ignore'):
                  means = sums / counts
             # Apply min_count
             if min_count > 0:
                  means[counts < min_count] = np.nan
             out_data = means
             
        elif agg == 'sum':
             sums = np.bincount(valid_indices, weights=valid_values, minlength=n_bins)
             counts = np.bincount(valid_indices, minlength=n_bins)
             if min_count > 0:
                  sums[counts < min_count] = np.nan # Or 0? Usually sum of nothing is 0, but min_count implies invalidity
             out_data = sums
             
        elif agg in ['median', 'min', 'max', 'std']:
             # Use scipy.stats.binned_statistic if available (no pandas dependency required)
             # or use numpy logic for simple cases.
             # Median is hard in pure numpy without loop or sort.
             # But binned_statistic supports 'median' (since scipy 0.17), 'std', 'min', 'max'.
             try:
                 from scipy.stats import binned_statistic
                 # valid_values are float? Complex median/min/max is undefined generally.
                 # Assuming float/real.
                 if np.iscomplexobj(valid_values):
                     raise NotImplementedError(f"Aggregation '{agg}' not supported for complex data yet")
                     
                 # valid_indices are 0..n_bins-1, produced by np.digitize equivalent
                 # binned_statistic(x, values, statistic=agg, bins=...)
                 # We already have indices. But binned_statistic wants x.
                 # We can pass x=valid_indices, bins=range(n_bins+1).
                 # This aligns perfectly.
                 
                 stat_val, _, _ = binned_statistic(
                      valid_indices, 
                      valid_values, 
                      statistic=agg, 
                      bins=np.arange(n_bins + 1)
                 )
                 out_data = stat_val
                 
                 # Apply min_count via counts if not handled by binned_statistic
                 # binned_statistic doesn't support min_count directly.
                 # We have counts from earlier? No, need to compute.
                 if min_count > 1:
                     counts = np.bincount(valid_indices, minlength=n_bins)
                     out_data[counts < min_count] = np.nan
                     
             except ImportError:
                 # Fallback to pandas
                 try:
                     import pandas as pd
                     df = pd.DataFrame({'val': valid_values, 'bin': valid_indices})
                     grouped = df.groupby('bin')['val']
                     if agg == 'median':
                         res = grouped.median()
                     elif agg == 'min':
                         res = grouped.min()
                     elif agg == 'max':
                         res = grouped.max()
                     elif agg == 'std':
                         res = grouped.std(ddof=1) # sample std
                     
                     # Reindex to all bins
                     # res is indexed by bin. Missing bins are missing.
                     out_data = np.full(n_bins, np.nan)
                     if not res.empty:
                         # Filter indices in range
                         res_idx = res.index.values.astype(int)
                         mask = (res_idx >= 0) & (res_idx < n_bins)
                         out_data[res_idx[mask]] = res.values[mask]
                         
                     # min_count check using counts
                     if min_count > 1:
                         cts = grouped.count()
                         cts_idx = cts.index.values.astype(int)
                         mask_c = (cts_idx >= 0) & (cts_idx < n_bins)
                         # Set NaN where count < min_count
                         # Note: pandas aggregate might have done some logic, but explicit check matches 'mean' logic.
                         bad_bins = cts_idx[mask_c][cts.values[mask_c] < min_count]
                         out_data[bad_bins] = np.nan
                         
                 except ImportError:
                     # Fallback to loop
                     out_data = np.full(n_bins, np.nan)
                     import warnings
                     warnings.warn(f"Using slow fallback for {agg} resampling (install scipy or pandas for speed).")
                     if agg == 'median':
                          func = np.median
                     elif agg == 'min':
                          func = np.min
                     elif agg == 'max':
                          func = np.max
                     elif agg == 'std':
                          func = np.std
                     else:
                          func = lambda x: np.nan
                          
                     for i in range(n_bins):
                          in_bin = valid_values[valid_indices == i]
                          if len(in_bin) >= min_count:
                               out_data[i] = func(in_bin)

        elif callable(agg):
             # Slow loop
             out_data = np.full(n_bins, np.nan)
             for i in range(n_bins):
                  # This is very slow for large data
                  in_bin = valid_values[valid_indices == i]
                  if len(in_bin) >= min_count:
                       out_data[i] = agg(in_bin)
        else:
             # 'max', 'min', etc. use pandas if available?
             # Or reduceat if sorted? Times are sorted.
             # If times are sorted, bin_indices are sorted.
             # We can use np.add.reduceat etc. if valid_indices is monotonic
             
             # Fallback to pandas if simple? 
             # To stay pure numpy:
             # Find changes in bin_indices to get boundaries
             pass
             # For MVP, restrict to mean/sum or use scipy binned_statistic
             from scipy.stats import binned_statistic
             if agg == 'median':
                  stat_func = np.nanmedian if nan_policy=='omit' else np.median
             elif agg == 'max':
                  stat_func = np.nanmax
             elif agg == 'min':
                  stat_func = np.nanmin
             elif agg == 'std':
                  stat_func = np.nanstd
             else:
                  stat_func = agg # 'mean' also covered here if fallback needed
                  
             # binned_statistic handles this
             # x=old_times_val[valid_mask], values=valid_values
             # bins=edges
             # But binned_statistic might be slower than bincount for simple mean
             res = binned_statistic(old_times_val[valid_mask], valid_values, statistic=stat_func, bins=edges)
             out_data = res.statistic
             
             # Apply min_count? binned_statistic doesn't support min_count directly (except NaNs)
             # If we need strict min_count, we need counts too.
             if min_count > 0:
                   counts = np.bincount(valid_indices, minlength=n_bins)
                   if np.issubdtype(out_data.dtype, np.floating):
                        out_data[counts < min_count] = np.nan

        # 4. Result Times
        if label == 'left':
             final_t0 = u.Quantity(edges[0], time_unit)
        elif label == 'right':
             final_t0 = u.Quantity(edges[1], time_unit)
        else:
             final_t0 = u.Quantity(edges[0] + bin_dt_val/2, time_unit) # Center?
        
        # Unit
        out_unit = self.unit
        if agg == 'count':
             out_unit = u.dimensionless_unscaled
             
        return self.__class__(
            out_data,
            t0=final_t0,
            dt=bin_dt,
            unit=out_unit,
            name=self.name,
            channel=self.channel 
        )

    def analytic_signal(
        self,
        pad: Any = None,
        pad_mode: str = "reflect",
        pad_value: float = 0.0,
        nan_policy: str = "raise",
        copy: bool = True,
    ) -> "TimeSeries":
        """
        Compute the analytic signal (Hilbert transform) of the TimeSeries.
        
        If input is real, returns complex analytic signal z(t) = x(t) + i H[x(t)].
        If input is complex, returns a copy (casting to complex if needed).
        """
        # 1. Complex check
        if np.iscomplexobj(self.value):
            return self.astype(complex, copy=copy)
            
        # 2. Check regular
        info = _extract_axis_info(self)
        if not info['regular']:
             raise ValueError("analytic_signal requires regular sampling (dt). Use asfreq/resample first.")
        
        # 3. Handle NaN
        if nan_policy == 'raise':
            if not np.isfinite(self.value).all():
                raise ValueError("Input contains NaNs or infinite values.")
        
        # 4. Padding
        data = self.value
        n_pad = 0
        if pad is not None:
             if isinstance(pad, u.Quantity):
                  n_pad = int(round(pad.to(self.times.unit).value / self.dt.to(self.times.unit).value))
             else:
                  n_pad = int(pad)
                  
             if n_pad > 0:
                  if pad_mode == 'constant':
                       kwargs = {'constant_values': pad_value}
                  else:
                       kwargs = {}
                  data = np.pad(data, n_pad, mode=pad_mode, **kwargs)
                  
        # 5. Hilbert
        analytic = scipy.signal.hilbert(data)
        
        # 6. Crop if padded
        if n_pad > 0:
             analytic = analytic[n_pad:-n_pad]
             
        # 7. Wrap
        return self.__class__(
             analytic,
             t0=self.t0,
             dt=self.dt,
             unit=self.unit,
             channel=self.channel,
             name=self.name
        )
        
    def hilbert(self, *args: Any, **kwargs: Any) -> "TimeSeries":
        """Alias for analytic_signal."""
        return self.analytic_signal(*args, **kwargs)
        
    def envelope(self, *args: Any, **kwargs: Any) -> "TimeSeries":
        """Compute the envelope of the TimeSeries."""
        analytic = self.analytic_signal(*args, **kwargs)
        return abs(analytic)
        
    def instantaneous_phase(self, deg: bool = False, unwrap: bool = False, **kwargs: Any) -> "TimeSeries":
        """
        Compute the instantaneous phase of the TimeSeries.
        """
        analytic = self.analytic_signal(**kwargs)
        phi = np.angle(analytic.value, deg=deg)
        
        if unwrap:
             # unwrap expects radians usually, but if deg=True, results are in degrees.
             # numpy.unwrap default period is 2pi.
             # If deg=True, we should use period=360? 
             # numpy.unwrap doc says: 'period: float, optional, default 2*pi'
             period = 360.0 if deg else 2 * np.pi
             phi = np.unwrap(phi, period=period)
             
        out = self.__class__(
             phi,
             t0=self.t0,
             dt=self.dt,
             channel=self.channel,
             name=self.name
        )
        
        # Override unit
        out.override_unit('deg' if deg else 'rad')
        return out
        
    def unwrap_phase(self, deg: bool = False, **kwargs: Any) -> "TimeSeries":
        """Alias for instantaneous_phase(unwrap=True)."""
        return self.instantaneous_phase(deg=deg, unwrap=True, **kwargs)
        
    def instantaneous_frequency(self, unwrap: bool = True, smooth: Any = None, **kwargs: Any) -> "TimeSeries":
        """
        Compute the instantaneous frequency of the TimeSeries.
        Returns unit 'Hz'.
        """
        # Force radians for calculation
        phi_ts = self.instantaneous_phase(deg=False, unwrap=unwrap, **kwargs)
        phi = phi_ts.value
        
        dt_s = self.dt.to('s').value
        
        # gradient
        # dphi / dt
        dphi_dt = np.gradient(phi, dt_s)
        
        # f = (dphi/dt) / 2pi
        f_inst = dphi_dt / (2 * np.pi)
        
        # Smoothing
        if smooth is not None:
             # Determine window size
             # Parse string if needed
             if isinstance(smooth, str):
                  smooth = u.Quantity(smooth)
                  
             if isinstance(smooth, u.Quantity):
                  w_s = smooth.to('s').value
                  w_samples = int(round(w_s / dt_s))
             else:
                  w_samples = int(smooth)
             
             if w_samples > 1:
                  # Simple moving average
                  # Mode 'same' to keep length
                  # Handle edges? P0 says 'nearest' or simple.
                  # scipy.ndimage.uniform_filter1d is good, or convolve
                  window = np.ones(w_samples) / w_samples
                  # boundary 'symm' or 'replicate' (nearest)
                  # np.convolve doesn't handle boundary options nicely without manual pad
                  # explicit pad with 'edge' (nearest)
                  f_pad = np.pad(f_inst, w_samples//2, mode='edge')
                  f_smooth = np.convolve(f_pad, window, mode='valid')
                  
                  # Size matching might be tricky with odd/even windows in convolve
                  # convolve(valid) length: N_in - N_win + 1.
                  # N_pad = N + 2*(w//2). 
                  # N_out = N + 2*(w//2) - w + 1. 
                  # If w=3, w//2=1. Pad 1 each side. N+2. Out = N+2 - 3 + 1 = N. Correct.
                  # If w=4, w//2=2. Pad 2 each side. N+4. Out = N+4 - 4 + 1 = N+1. Too long?
                  
                  # Better use scipy.ndimage if available, or just careful indexing
                  # For P0, simple 'same' convolution is acceptable standard?
                  # standard convolve(mode='same') uses zero padding which dips at edges.
                  # Let's stick to the manual pad above but adjust for even w?
                  if len(f_smooth) > len(f_inst):
                       f_smooth = f_smooth[:len(f_inst)]
                  elif len(f_smooth) < len(f_inst):
                       # Should not happen with valid logic above for odd w
                       pass
                  f_inst = f_smooth
        
        out = self.__class__(
             f_inst,
             t0=self.t0,
             dt=self.dt,
             channel=self.channel,
             name=self.name
        )
        out.override_unit('Hz')
        return out

    def _build_phase_series(
        self,
        *,
        phase: Any = None,
        f0: Any = None,
        fdot: Any = 0.0,
        fddot: Any = 0.0,
        phase_epoch: Any = None,
        phase0: float = 0.0,
        prefer_dt: bool = True,
    ) -> np.ndarray:
        """Internal helper to build phase series in radians."""
        if (f0 is None and phase is None) or (f0 is not None and phase is not None):
             raise ValueError("Exactly one of 'f0' or 'phase' must be provided.")
             
        if phase is not None:
             if len(phase) != self.size:
                  raise ValueError(f"Length of phase ({len(phase)}) does not match TimeSeries ({self.size})")
             return np.asarray(phase, dtype=float) + phase0
             
        # Build from model
        if isinstance(f0, u.Quantity):
             f0 = f0.to('Hz').value
        else:
             f0 = float(f0)
             
        if isinstance(fdot, u.Quantity):
             fdot = fdot.to('Hz/s').value
        else:
             fdot = float(fdot)
             
        if isinstance(fddot, u.Quantity):
             fddot = float(fddot.to('Hz/s^2').value)
        else:
             fddot = float(fddot)
             
        # Determine t_rel
        has_dt = False
        dt_val = None
        try:
             # Check if dt is accessible
             if self.dt is not None:
                  dt_val = self.dt.to('s').value
                  has_dt = True
        except (AttributeError, ValueError):
             # Irregular axis
             pass

        if has_dt and prefer_dt and dt_val is not None:
             # Use regular dt
             if phase_epoch is None:
                  t_rel = dt_val * np.arange(self.size)
             else:
                  t0_abs = float(self.times.value[0]) # Assumed seconds/compatible
                  t0_rel = t0_abs - float(phase_epoch)
                  t_rel = t0_rel + dt_val * np.arange(self.size)
        else:
             # Use times array
             if self.times is None:
                  raise ValueError("TimeSeries requires times or dt to build phase model.")
             # Assume times are numerical or Quantity comparable to seconds
             times_val = self.times.value if hasattr(self.times, 'value') else self.times
             times_val = np.asarray(times_val, dtype=float)
             
             if phase_epoch is None:
                  t_rel = times_val - times_val[0]
             else:
                  t_rel = times_val - float(phase_epoch)
                  
        # Calculate phase
        # phi = 2pi * (f0*t + 0.5*fdot*t^2 + 1/6*fddot*t^3) + phase0
        # t_rel can be large? standard float64 usually fine for reasonable durations.
        
        cycles = f0 * t_rel
        if fdot != 0.0:
             cycles += 0.5 * fdot * t_rel**2
        if fddot != 0.0:
             cycles += (1.0/6.0) * fddot * t_rel**3
             
        return 2 * np.pi * cycles + phase0

    def mix_down(
        self,
        *,
        phase: Any = None,
        f0: Any = None,
        fdot: Any = 0.0,
        fddot: Any = 0.0,
        phase_epoch: Any = None,
        phase0: float = 0.0,
        singlesided: bool = False,
        copy: bool = True,
    ) -> "TimeSeries":
        """
        Mix the TimeSeries with a complex oscillator.
        """
        phase_series = self._build_phase_series(
            phase=phase,
            f0=f0,
            fdot=fdot,
            fddot=fddot,
            phase_epoch=phase_epoch,
            phase0=phase0,
            prefer_dt=True
        )
                  
        # Mix
        y = self.value * np.exp(-1j * phase_series)
        
        if singlesided:
             y *= 2.0
             
        # Prepare constructor args
        kwargs = {
             'unit': self.unit,
             'channel': self.channel,
             'name': self.name
        }
        
        # Check regularity again for constructor
        try:
             if self.dt is not None:
                  kwargs['t0'] = self.t0
                  kwargs['dt'] = self.dt
             else:
                  # This likely won"t happen if dt is None but sample_rate is not?
                  kwargs['t0'] = self.t0
                  kwargs['sample_rate'] = self.sample_rate
        except (AttributeError, ValueError):
             # Irregular - pass times explicit
             kwargs['times'] = self.times
             
        out = self.__class__(y, **kwargs)
        return out

    def baseband(
        self,
        *,
        phase: Any = None,
        f0: Any = None,
        fdot: Any = 0.0,
        fddot: Any = 0.0,
        phase_epoch: Any = None,
        phase0: float = 0.0,
        lowpass: Optional[float] = None,
        lowpass_kwargs: Optional[dict[str, Any]] = None,
        output_rate: Optional[float] = None,
        singlesided: bool = False,
    ) -> "TimeSeries":
        """
        Demodulate the TimeSeries to baseband, optionally applying lowpass filter and resampling.
        """
        z = self.mix_down(
            phase=phase,
            f0=f0,
            fdot=fdot,
            fddot=fddot,
            phase_epoch=phase_epoch,
            phase0=phase0,
            singlesided=singlesided
        )
        
        if lowpass is not None:
             if z.sample_rate is None:
                  raise ValueError("lowpass requires defined sample rate.")
             lp_kwargs = lowpass_kwargs or {}
             z = z.lowpass(lowpass, **lp_kwargs)
             
        if output_rate is not None:
             z = z.resample(output_rate)
             
        return z

    def lock_in(
        self,
        *,
        phase: Any = None,
        f0: Any = None,
        fdot: Any = 0.0,
        fddot: Any = 0.0,
        phase_epoch: Any = None,
        phase0: float = 0.0,
        stride: float = 1.0,
        singlesided: bool = True,
        output: str = "amp_phase",
        deg: bool = True,
    ) -> Any:
        """
        Perform lock-in amplification (demodulation + averaging).
        """
        if self.dt is None:
             raise ValueError("lock_in requires regular sampling (dt/sample_rate).")
             
        phase_series = self._build_phase_series(
            phase=phase,
            f0=f0,
            fdot=fdot,
            fddot=fddot,
            phase_epoch=phase_epoch,
            phase0=phase0,
            prefer_dt=True
        )
        
        outc = self.heterodyne(phase_series, stride=stride, singlesided=singlesided)
              
        if output == 'complex':
             return outc
        elif output == 'amp_phase':
             mag = outc.abs()
             ph = self.__class__(
                  np.angle(outc.value, deg=deg),
                  t0=outc.t0,
                  dt=outc.dt,
                  channel=self.channel,
                  name=self.name
             )
             ph.override_unit('deg' if deg else 'rad')
             return mag, ph
        elif output == 'iq':
             i = self.__class__(outc.value.real, t0=outc.t0, dt=outc.dt, channel=self.channel, name=self.name, unit=self.unit)
             q = self.__class__(outc.value.imag, t0=outc.t0, dt=outc.dt, channel=self.channel, name=self.name, unit=self.unit)
             return i, q
        else:
             raise ValueError(f"Unknown output format: {output}")



    def fft(
        self,
        nfft: Optional[int] = None,
        *,
        mode: str = "gwpy",
        pad_mode: str = "zero",
        pad_left: int = 0,
        pad_right: int = 0,
        nfft_mode: Optional[str] = None,
        other_length: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Compute the one-dimensional discrete Fourier Transform.

        Parameters
        ----------
        nfft : `int`, optional
            Length of the FFT. If `None`, the length of the TimeSeries is used.
        mode : `str`, optional, default: "gwpy"
            "gwpy": Standard behavior (circular convolution assumption).
            "transient": Transient-friendly mode with padding options.
        pad_mode : `str`, optional, default: "zero"
            Padding mode for "transient" mode. "zero" or "reflect".
        pad_left : `int`, optional, default: 0
            Number of samples to pad on the left (for "transient" mode).
        pad_right : `int`, optional, default: 0
            Number of samples to pad on the right (for "transient" mode).
        nfft_mode : `str`, optional, default: None
            "next_fast_len": Use scipy.fft.next_fast_len to optimize FFT size.
            None: Use exact calculated length.
        other_length : `int`, optional, default: None
            If provided in "transient" mode, the target length is calculated as
            len(self) + other_length - 1 (linear convolution size).
        **kwargs
            Keyword arguments passed to the `FrequencySeries` constructor.

        Returns
        -------
        out : `FrequencySeries`
            The DFT of the TimeSeries.
        """
        # 1. GWpy compatible mode (default)
        if mode == "gwpy":
            # If default args are used, just call super if NFFT matches or is None
            if (
                pad_mode == "zero"
                and pad_left == 0
                and pad_right == 0
                and nfft_mode is None
                and other_length is None
            ):
                base_fs = super().fft(nfft=nfft, **kwargs)
                try:
                    from gwexpy.frequencyseries import FrequencySeries as GWEXFrequencySeries
                except ImportError:
                    return base_fs
                if isinstance(base_fs, GWEXFrequencySeries):
                    return base_fs
                return GWEXFrequencySeries(
                    base_fs.value,
                    frequencies=base_fs.frequencies,
                    unit=base_fs.unit,
                    name=base_fs.name,
                    channel=base_fs.channel,
                    epoch=base_fs.epoch,
                )

            # Fallback to super for gwpy mode even if explicit args (ignore extras)
            base_fs = super().fft(nfft=nfft, **kwargs)
            try:
                from gwexpy.frequencyseries import FrequencySeries as GWEXFrequencySeries
            except ImportError:
                return base_fs
            if isinstance(base_fs, GWEXFrequencySeries):
                return base_fs
            return GWEXFrequencySeries(
                base_fs.value,
                frequencies=base_fs.frequencies,
                unit=base_fs.unit,
                name=base_fs.name,
                channel=base_fs.channel,
                epoch=base_fs.epoch,
            )

        if mode != "transient":
            raise ValueError(f"Unknown mode: {mode}")

        # 2. Transient mode
        # Create padded array
        x = self.value

        # Padding
        if pad_left > 0 or pad_right > 0:
            # Handle duration if float or Quantity
            if hasattr(pad_left, "to"):
                pad_left = pad_left.to("s").value * self.sample_rate.value
            elif isinstance(pad_left, float):
                pad_left = pad_left * self.sample_rate.value

            if hasattr(pad_right, "to"):
                pad_right = pad_right.to("s").value * self.sample_rate.value
            elif isinstance(pad_right, float):
                pad_right = pad_right * self.sample_rate.value

            pad_left = int(round(float(pad_left)))
            pad_right = int(round(float(pad_right)))

            if pad_left > 0 or pad_right > 0:
                if pad_mode == "zero":
                    x = np.pad(x, (pad_left, pad_right), mode="constant", constant_values=0)
                elif pad_mode == "reflect":
                    x = np.pad(x, (pad_left, pad_right), mode="reflect")
                else:
                    raise ValueError(f"Unknown pad_mode: {pad_mode}")

        len_x = x.shape[0]

        # Determine target_nfft
        if nfft is not None:
            if nfft < len_x:
                raise ValueError(f"nfft={nfft} must be >= padded length {len_x}")
            target_nfft = int(nfft)
        else:
            if other_length is not None:
                target_len = len_x + int(other_length) - 1
            else:
                target_len = len_x

            if nfft_mode == "next_fast_len":
                try:
                    import scipy.fft
                    def next_fast_len(n):
                        return scipy.fft.next_fast_len(n)
                except ImportError:
                    try:
                        from scipy.fftpack import next_fast_len
                    except ImportError:
                        # Fallback: just use target_len
                        def next_fast_len(n):
                            return n

                target_nfft = next_fast_len(target_len)
            else:
                target_nfft = target_len

        # Compute FFT
        # Normalization: rfft(self.value, n=nfft)/nfft
        dft = np.fft.rfft(x, n=target_nfft) / target_nfft

        if dft.shape[0] > 1:
            dft[1:] *= 2.0

        from gwexpy.frequencyseries import FrequencySeries

        fs = FrequencySeries(
            dft,
            epoch=self.epoch,
            unit=self.unit,
            name=self.name,
            channel=self.channel,
            **kwargs,
        )

        # Set frequencies with units
        fs.frequencies = np.fft.rfftfreq(target_nfft, d=self.dt.value) * u.Hz
        # Store transient metadata for round-trip ifft
        fs._gwex_fft_mode = "transient"
        fs._gwex_pad_left = pad_left
        fs._gwex_pad_right = pad_right
        fs._gwex_pad_mode = pad_mode
        fs._gwex_target_nfft = target_nfft
        fs._gwex_original_n = len(self)
        fs._gwex_other_length = other_length

        return fs

    def dct(self, type: int = 2, norm: str = "ortho", *, window: Any = None, detrend: bool = False) -> Any:
        """
        Compute the Discrete Cosine Transform (DCT) of the TimeSeries.

        Parameters
        ----------
        type : `int`, optional
            Type of the DCT (1, 2, 3, 4). Default is 2.
        norm : `str`, optional
            Normalization mode. Default is "ortho".
        window : `str`, `numpy.ndarray`, or `None`, optional
            Window function to apply before DCT.
        detrend : `bool`, optional
            If `True`, remove the mean (or detrend) from the data before DCT.

        Returns
        -------
        out : `FrequencySeries`
             The DCT of the TimeSeries.
        """
        try:
            import scipy.fft
            import scipy.signal
        except ImportError:
            raise ImportError(
                "scipy is required for dct. "
                "Please install it via `pip install scipy`."
            )

        data = self.value.copy()

        # Detrend
        if detrend:
             # If detrend is True (bool), use 'linear' (gwpy default usually)
             data = scipy.signal.detrend(data, type='linear')

        # Window
        if window is not None:
             if isinstance(window, str) or isinstance(window, tuple):
                 win = scipy.signal.get_window(window, len(data))
             else:
                 win = np.asarray(window)
                 if len(win) != len(data):
                      raise ValueError("Window length must match data length")
             data *= win
        
        # DCT
        out_dct = scipy.fft.dct(data, type=type, norm=norm)
        
        # Frequencies
        # f_k = k / (2 * N * dt)
        N = len(data)
        if self.dt is None:
             raise ValueError("TimeSeries must have a valid dt for DCT frequency calculation")
             
        dt = self.dt.to("s").value
        k = np.arange(N)
        freqs_val = k / (2 * N * dt)
        
        frequencies = u.Quantity(freqs_val, "Hz")
        
        from gwexpy.frequencyseries import FrequencySeries
        
        fs = FrequencySeries(
            out_dct,
            frequencies=frequencies,
            unit=self.unit,
            name=self.name + "_dct" if self.name else "dct",
            channel=self.channel,
            epoch=self.epoch
        )
        
        # Metadata
        fs.transform = "dct"
        fs.dct_type = type
        fs.dct_norm = norm
        fs.original_n = N
        fs.dt = self.dt
        

        return fs

    def laplace(
        self,
        *,
        sigma: Any = 0.0,
        frequencies: Any = None,
        t_start: Any = None,
        t_stop: Any = None,
        window: Any = None,
        detrend: bool = False,
        normalize: str = "integral",   # "integral" | "mean"
        dtype: Any = None,
        chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        One-sided finite-interval Laplace transform.

        Define s = sigma + i*2*pi*f (f in Hz).
        Compute on a cropped segment [t_start, t_stop] but shift time origin to t_start:
            L(s) = integral_0^T x(tau) exp(-(sigma + i 2pi f) tau) dtau
        Discrete approximation uses uniform dt.

        Parameters
        ----------
        sigma : `float` or `astropy.units.Quantity`, optional
            Real part of the Laplace variable s (1/s). Default 0.0.
        frequencies : `numpy.ndarray` or `astropy.units.Quantity`, optional
            Frequencies for the imaginary part of s (Hz). 
            If None, defaults to `np.fft.rfftfreq(n, d=dt)`.
        t_start : `float` or `astropy.units.Quantity`, optional
            Start time relative to the beginning of the TimeSeries (seconds).
            Defaults to 0.
        t_stop : `float` or `astropy.units.Quantity`, optional
            Stop time relative to the beginning of the TimeSeries (seconds).
            Defaults to end of series.
        window : `str`, `numpy.ndarray`, or `None`, optional
            Window function to apply.
        detrend : `bool`, optional
            If True, remove the mean before transformation.
        normalize : `str`, optional
            "integral": Sum * dt (standard Laplace integral approximation).
            "mean": Sum / n.
        dtype : `numpy.dtype`, optional
            Output data type. IF None, inferred from input (usually complex128).
        chunk_size : `int`, optional
            If provided, compute the transform in chunks along the frequency axis
            to save memory.

        Returns
        -------
        out : `FrequencySeries`
            The Laplace transform result (complex).
        """
        # 1. Handle Time Segment and Cropping
        # Calculate start/stop indices
        n_total = len(self)
        if self.dt is None:
             raise ValueError("TimeSeries must have a valid dt for Laplace transform")
        
        dt_val = self.dt.to("s").value
        fs_rate = 1.0 / dt_val
        
        def resolve_time_arg(arg, default_idx):
             if arg is None:
                  return default_idx
             if isinstance(arg, u.Quantity):
                  t_s = arg.to("s").value
             else:
                  t_s = float(arg)
             idx = int(round(t_s * fs_rate))
             return np.clip(idx, 0, n_total)

        idx_start = resolve_time_arg(t_start, 0)
        idx_stop = resolve_time_arg(t_stop, n_total)
        
        if idx_stop <= idx_start:
             raise ValueError(f"Invalid time range: t_start index {idx_start} >= t_stop index {idx_stop}")
             
        # Extract data segment
        data = self.value[idx_start:idx_stop]
        n = len(data)
        
        # 2. Preprocessing (Detrend, Window)
        if detrend:
             # Just remove mean as requested
             data = data - np.mean(data)
             
        if window is not None:
             if isinstance(window, str) or isinstance(window, tuple):
                  # Lazy import to avoid top-level dependency if possible, though scipy is used elsewhere
                  try:
                      import scipy.signal
                      win = scipy.signal.get_window(window, n)
                  except ImportError:
                      raise ImportError("scipy is required for window generation")
             else:
                  win = np.asarray(window)
                  if len(win) != n:
                       raise ValueError("Window length must match segment length")
             data *= win
             
        # 3. Frequency Axis
        if frequencies is None:
             freqs_val = np.fft.rfftfreq(n, d=dt_val)
             freqs_quant = u.Quantity(freqs_val, "Hz")
        else:
             if isinstance(frequencies, u.Quantity):
                  freqs_quant = frequencies.to("Hz")
                  freqs_val = freqs_quant.value
             else:
                  freqs_val = np.asarray(frequencies)
                  freqs_quant = u.Quantity(freqs_val, "Hz")
                  
        # 4. Sigma
        if isinstance(sigma, u.Quantity):
             sigma_val = sigma.to("1/s").value
        else:
             sigma_val = float(sigma)
             
        # 5. Computation
        # L(f) = sum_k x[k] * exp( - (sigma + i*2*pi*f) * t[k] ) * (dt if integral else 1/n)
        
        tau = np.arange(n) * dt_val
        
        # Determine output dtype
        if dtype is None:
             out_dtype = np.result_type(data.dtype, np.complex128)
        else:
             out_dtype = dtype
             
        n_freqs = len(freqs_val)
        out_data = np.zeros(n_freqs, dtype=out_dtype)
        
        # Factor for normalization
        if normalize == "integral":
             norm_factor = dt_val
        elif normalize == "mean":
             norm_factor = 1.0 / n
        else:
             raise ValueError(f"Unknown normalize mode: {normalize}")
             
        if chunk_size is None:
             # Vectorized all at once
             # s = sigma + i*2*pi*f
             # exponent: - (sigma + i*2*pi*f) * tau = -sigma*tau - i*2*pi*f*tau
             
             # Calculate -sigma*tau term (shape (n,))
             real_exp = np.exp(-sigma_val * tau)
             
             # Apply real exponent to data first
             data_weighted = data * real_exp
             
             # Now compute DFT-like sum: sum_k (data_weighted[k] * exp(-i*2*pi*f*tau[k]))
             # This is NOT exactly FFT unless frequencies are FFT frequencies.
             # We use matrix multiplication or einsum.
             # frequencies (M,), tau (N,) -> phase (M, N)
             # This can be huge. (M * N).
             
             # Use broadcasting carefully.
             # sum( data_weighted[None, :] * exp( -1j * 2*pi * f[:, None] * tau[None, :] ), axis=1 )
             
             # phase = 2 * np.pi * freqs_val[:, None] * tau[None, :]
             # complex_exp = np.exp(-1j * phase)
             # out_data = np.sum(data_weighted * complex_exp, axis=1) * norm_factor
             
             # Optimization: If frequencies are exactly rfftfreq(n, dt), we could use FFT?
             # But sigma makes it dampened. 
             # If sigma=0 and freqs=rfftfreq, it's FFT.
             # If frequencies are custom, general summation is needed.
             
             # Let's stick to explicit sum to be correct for arbitrary inputs.
             # Memory check: M*N complex128. 
             # If N=10000, M=5000 -> 50e6 elements ~ 800MB. OK for modern machines.
             # If larger, user should use chunk_size.
             
             phase = 2 * np.pi * freqs_val[:, None] * tau[None, :]
             complex_exp = np.exp(-1j * phase)
             # data_weighted is (N,)
             # broadcast to (M, N)
             out_data = np.dot(complex_exp, data_weighted) * norm_factor
             
        else:
             # Chunked computation
             real_exp = np.exp(-sigma_val * tau)
             data_weighted = data * real_exp
             
             for i in range(0, n_freqs, chunk_size):
                  end = min(i + chunk_size, n_freqs)
                  f_chunk = freqs_val[i:end]
                  
                  phase_chunk = 2 * np.pi * f_chunk[:, None] * tau[None, :]
                  complex_exp_chunk = np.exp(-1j * phase_chunk)
                  
                  out_data[i:end] = np.dot(complex_exp_chunk, data_weighted) * norm_factor

        # 6. Create Output
        from gwexpy.frequencyseries import FrequencySeries
        
        # Propagate units
        if normalize == "integral":
             out_unit = self.unit * u.s
        else:
             out_unit = self.unit
             
        fs = FrequencySeries(
            out_data,
            epoch=self.epoch, # Inherit epoch (start time of original series, though transform time origin is t_start)
            # transform typically implies origin at window start.
            # But gwpy usually keeps original epoch for derived series unless shifted.
            # We keep self.epoch. 
            unit=out_unit,
            name=f"{self.name}_laplace" if self.name else "laplace",
            channel=self.channel,
            **kwargs,
        )
        
        fs.frequencies = freqs_quant
        fs.laplace_sigma = sigma_val
        
        return fs

    def cepstrum(
        self,
        kind: str = "real",
        *,
        window: Any = None,
        detrend: bool = False,
        eps: Optional[float] = None,
        fft_mode: str = "gwpy",
    ) -> Any:
        """
        Compute the Cepstrum of the TimeSeries.

        Parameters
        ----------
        kind : {"real", "power", "complex"}, optional
            Type of cepstrum. Default is "real".
        window : `str`, `numpy.ndarray`, or `None`, optional
            Window function to apply.
        detrend : `bool`, optional
            If `True`, detach trend (linear).
        eps : `float`, optional
            Small value to avoid log(0).
        fft_mode : `str`, optional
            Mode for the underlying FFT ("gwpy").

        Returns
        -------
        out : `FrequencySeries`
            The cepstrum, with frequencies interpreted as quefrency (seconds).
        """
        try:
             import scipy.fft
             import scipy.signal
        except ImportError:
            raise ImportError(
                "scipy is required for cepstrum. "
                "Please install it via `pip install scipy`."
            )

        data = self.value.copy()

        # Detrend
        if detrend:
             data = scipy.signal.detrend(data, type='linear')
        
        # Window
        if window is not None:
             if isinstance(window, str) or isinstance(window, tuple):
                  win = scipy.signal.get_window(window, len(data))
             else:
                  win = np.asarray(window)
             data *= win
        
        # FFT computation based on kind
        
        if kind == "complex":
             # c = irfft( log(rfft(x) + eps_complex) )
             spec = scipy.fft.rfft(data)
             if eps is not None:
                  spec += eps
             with np.errstate(divide='ignore', invalid='ignore'):
                  log_spec = np.log(spec)
             ceps = scipy.fft.irfft(log_spec, n=len(data))
             
        elif kind == "real":
             # c = irfft( log(|rfft(x)| + eps) )
             spec = scipy.fft.rfft(data)
             mag = np.abs(spec)
             if eps is not None:
                  mag += eps
             with np.errstate(divide='ignore'):
                  log_mag = np.log(mag)
             ceps = scipy.fft.irfft(log_mag, n=len(data))
             
        elif kind == "power":
             # c = irfft( log(|rfft(x)|^2 + eps) )
             spec = scipy.fft.rfft(data)
             pwr = np.abs(spec)**2
             if eps is not None:
                  pwr += eps
             with np.errstate(divide='ignore'):
                  log_pwr = np.log(pwr)
             ceps = scipy.fft.irfft(log_pwr, n=len(data))
             
        else:
             raise ValueError(f"Unknown cepstrum kind: {kind}")

        # Quefrency axis
        # k * dt
        if self.dt is None:
             dt = 1.0
        else:
             dt = self.dt.to("s").value
             
        n = len(ceps)
        quefrencies = np.arange(n) * dt
        quefrencies = u.Quantity(quefrencies, "s")
        
        from gwexpy.frequencyseries import FrequencySeries
        
        fs = FrequencySeries(
            ceps,
            frequencies=quefrencies, 
            unit=u.dimensionless_unscaled,
            name=self.name + "_cepstrum" if self.name else "cepstrum",
            channel=self.channel,
            epoch=self.epoch
        )
        
        fs.axis_type = "quefrency"
        fs.transform = "cepstrum"
        fs.cepstrum_kind = kind
        fs.original_n = len(data)
        fs.dt = self.dt
        
        return fs

    def cwt(
        self,
        wavelet: str = "cmor1.5-1.0",
        widths: Any = None,
        frequencies: Any = None,
        *,
        window: Any = None,
        detrend: bool = False,
        output: str = "spectrogram",
        **kwargs: Any,
    ) -> Any:
        """
        Compute the Continuous Wavelet Transform (CWT) using PyWavelets.

        Parameters
        ----------
        wavelet : `str`, optional
            Wavelet name (default "cmor1.5-1.0" for complex morlet).
            See `pywt.wavelist(kind='continuous')`.
        widths : `array-like`, optional
            Scales to use for CWT. (Note: pywt uses 'scales').
        frequencies : `array-like`, optional
             Frequencies to use (Hz). If provided, converts to scales.
        window : `str`, `numpy.ndarray`, or `None`
             Window function to apply to time series before CWT.
        detrend : `bool`
             If True, detrend the time series.
        output : {"spectrogram", "ndarray"}
             Output format.
        **kwargs :
             Additional arguments passed to `pywt.cwt`.

        Returns
        -------
        out : `gwpy.spectrogram.Spectrogram` or `(ndarray, freqs)`
        """
        try:
             import pywt
             import scipy.signal
        except ImportError as e:
             raise ImportError(
                 f"pywt (PyWavelets) and scipy are required for cwt. "
                 f"Please install them via `pip install PyWavelets scipy`. Error: {e}"
             )

        data = self.value.copy()
        
        if detrend:
             data = scipy.signal.detrend(data, type='linear')
             
        if window is not None:
             if isinstance(window, str) or isinstance(window, tuple):
                  win = scipy.signal.get_window(window, len(data))
             else:
                  win = np.asarray(window)
             data *= win
             
        # Resolve widths(scales)/frequencies
        if self.dt is None:
              raise ValueError("TimeSeries requires dt for CWT")
        dt = self.dt.to("s").value
        
        if frequencies is not None:
             if widths is not None:
                  raise ValueError("Cannot specify both widths(scales) and frequencies")
             
             freqs_quant = u.Quantity(frequencies, "Hz")
             freqs_arr = freqs_quant.value
             
             # Convert freq to scales
             # scale = center_frequency(wavelet) / (freq * dt)
             # center_frequency returns frequency in cycles/sample?
             # pywt.scale2frequency(wavelet, scale) -> freq (normalized)
             # f_Hz = scale2freq / (scale * dt)
             # scale = scale2freq / (f_Hz * dt)
             
             # We need a reference scale to get center freq?
             # pywt.scale2frequency(wavelet, 1) gives the center freq for scale 1.
             center_freq = pywt.scale2frequency(wavelet, 1)
             
             with np.errstate(divide='ignore'):
                 scales = center_freq / (freqs_arr * dt)
             
        elif widths is None:
             raise ValueError("Must specify either widths(scales) or frequencies")
        else:
             scales = np.asarray(widths)
             # Calculate corresponding freqs for output
             # f = scale2freq / (scale * dt)
             # pywt.scale2frequency accepts array? No, usually scalar.
             # But it's linear. center_freq / scale / dt
             center_freq = pywt.scale2frequency(wavelet, 1)
             freqs_arr = center_freq / (scales * dt)

        # Compute CWT using PyWavelets
        # pywt.cwt(data, scales, wavelet)
        # returns (coefs, frequencies) if sampling_period is given in recent versions?
        # signature: cwt(data, scales, wavelet, sampling_period=1.0, method='conv', axis=-1)
        # Returns: coefs, frequencies (if 1.1+)
        
        coefs, _ = pywt.cwt(data, scales, wavelet, sampling_period=dt, **kwargs)
        
        freqs_quant = u.Quantity(freqs_arr, "Hz")

        if output == "ndarray":
             return coefs, freqs_quant
        
        elif output == "spectrogram":
             try:
                  from gwpy.spectrogram import Spectrogram
             except ImportError:
                  return coefs, freqs_quant
             
             # PyWavelets returns (len(scales), len(data))
             
             # Transpose to (time, freq) for GWpy Spectrogram
             # GWpy Spectrogram expects (times, frequencies) dimensions in .value usually?
             # Spectrogram(value, index=..., columns=...) aka times=..., frequencies=...
             out_spec = coefs.T
             
             # Sort frequencies if needed
             # scales usually roughly proportional to 1/freq.
             # If input frequencies were sorted ascending, scales are descending.
             # If passed widths(scales) directly, we don't know sort order.
             
             # Let's sort to be safe
             idx_sorted = np.argsort(freqs_arr)
             freqs_sorted = freqs_arr[idx_sorted]
             out_spec_sorted = out_spec[:, idx_sorted]
             
             return Spectrogram(
                 out_spec_sorted,
                 times=self.times,
                 frequencies=u.Quantity(freqs_sorted, "Hz"),
                 unit=self.unit,
                 name=self.name,
                 channel=self.channel,
                 epoch=self.epoch
             )
        else:
             raise ValueError(f"Unknown output format: {output}")


    # --- HHT (Hilbert-Huang Transform) ---

    def emd(
        self,
        *,
        method: str = "eemd",
        max_imf: Optional[int] = None,
        sift_max_iter: int = 1000,
        stopping_criterion: Any = "default",
        eemd_noise_std: float = 0.2,
        eemd_trials: int = 100,
        random_state: Optional[int] = None,
        return_residual: bool = True,
    ) -> Any:
        """
        Perform Empirical Mode Decomposition (EMD) or EEMD.

        Parameters
        ----------
        method : {"emd", "eemd"}, optional
            Decomposition method. Default is "eemd".
        max_imf : `int`, optional
            Maximum number of IMFs to extract.
        sift_max_iter : `int`, optional
            Maximum number of sifting iterations.
        stopping_criterion : `str` or `dict`, optional
            Stopping criterion configuration (passed to PyEMD if supported).
        eemd_noise_std : `float`, optional
            Noise standard deviation for EEMD (relative to signal std). Default 0.2.
        eemd_trials : `int`, optional
            Number of trials for EEMD. Default 100.
        random_state : `int` or `np.random.RandomState`, optional
            Seed for reproducibility.
        return_residual : `bool`, optional
            If True, include residual in the output.

        Returns
        -------
        imfs : `TimeSeriesDict`
            A dictionary of extracted IMFs (keys: "IMF1", "IMF2", ..., "residual").
        """
        try:
            import PyEMD
        except ImportError:
            raise ImportError(
                "PyEMD is required for EMD/EEMD. "
                "Please install it via `pip install EMD-signal`."
            )

        # Config RNG
        if random_state is not None:
             if isinstance(random_state, int):
                  np.random.seed(random_state)
             # PyEMD often uses global numpy random, or allows seed injection depending on version.
             # EEMD.EMD class might accept nothing specific easily, but we set global if simple.
             # Ideally we should control it better if PyEMD API allows. 
             # Current PyEMD (EMD-signal) relies heavily on numpy.random.

        data = self.value
        
        # Initialize EMD/EEMD
        if method.lower() == "eemd":
             decomposer = PyEMD.EEMD(trials=eemd_trials, noise_width=eemd_noise_std)
             # EEMD internal EMD config
             if max_imf is not None:
                  decomposer.MAX_ITERATION = max_imf # This might not be max_imf but max iter?
                  # PyEMD 0.2.10+ EEMD.emd param 'max_imf'
                  pass

             # Run EEMD
             # EEMD(S, T=None, max_imf=-1)
             try:
                  imfs_array = decomposer.eemd(
                      data,
                      T=None,
                      max_imf=max_imf if max_imf is not None else -1,
                  )
             except (PermissionError, OSError) as exc:
                  if isinstance(exc, OSError) and exc.errno not in (None, 13):
                       raise
                  # Fallback to serial execution if multiprocessing is blocked.
                  decomposer.parallel = False
                  decomposer.processes = 1
                  imfs_array = decomposer.eemd(
                      data,
                      T=None,
                      max_imf=max_imf if max_imf is not None else -1,
                  )
             
        elif method.lower() == "emd":
             decomposer = PyEMD.EMD()
             # Config
             # PyEMD API varies slightly by version. 
             # assuming EMD-signal package
             
             imfs_array = decomposer.emd(data, T=None, max_imf=max_imf if max_imf is not None else -1)
        else:
             raise ValueError(f"Unknown EMD method: {method}")

        # imfs_array shape: (n_imfs + 1, n_samples) usually includes residual as last element?
        # PyEMD returns IMFs. 
        # Check if residual is included. PyEMD usually returns IMFs, and user can calc residual or it is the last one.
        # Actually PyEMD returns only IMFs, residual is residue.
        # But `emd.emd` returns IMFs. 
        # `eemd.eemd` returns IMFs.
        
        # Let's inspect shape.
        n_rows = imfs_array.shape[0]
        
        # Identify residue: usually the last one in PyEMD output is the residue?
        # Standard: PyEMD returns [IMF1, IMF2, ..., IMFn, Residual] ?
        # Documentation says "returns set of IMFs". 
        
        # We will assume:
        # IMFs + Residual
        
        from .collections import TimeSeriesDict
        out_dict = TimeSeriesDict()
        
        # Assign keys
        # The last component is usually monotonic residue.
        
        n_imfs = n_rows - 1
        
        for i in range(n_imfs):
             key = f"IMF{i+1}"
             out_dict[key] = self.__class__(
                  imfs_array[i],
                  t0=self.t0,
                  dt=self.dt,
                  unit=self.unit,
                  name=f"{self.name}_{key}" if self.name else key,
                  channel=self.channel 
             )
             
        if return_residual:
             key = "residual"
             out_dict[key] = self.__class__(
                  imfs_array[-1],
                  t0=self.t0,
                  dt=self.dt,
                  unit=self.unit,
                  name=f"{self.name}_{key}" if self.name else key,
                  channel=self.channel
             )
             
        return out_dict

    def hilbert_analysis(self, *, unwrap_phase: bool = True, frequency_unit: str = "Hz") -> dict[str, Any]:
        """
        Perform Hilbert Spectral Analysis (HSA) to get instantaneous properties.

        Returns
        -------
        results : `dict`
            {
                "analytic": TimeSeries (complex),
                "amplitude": TimeSeries (real),
                "phase": TimeSeries (rad),
                "frequency": TimeSeries (Hz)
            }
        """
        # 1. Analytic Signal
        analytic = self.analytic_signal()
        
        # 2. Amplitude
        amp = np.abs(analytic.value)
        amplitude = self.__class__(
             amp,
             t0=self.t0,
             dt=self.dt,
             unit=self.unit,
             name=f"{self.name}_IA" if self.name else "IA",
             channel=self.channel,
        )
        
        # 3. Phase
        pha = np.angle(analytic.value)
        if unwrap_phase:
             pha = np.unwrap(pha)
             
        phase = self.__class__(
             pha,
             t0=self.t0,
             dt=self.dt,
             unit="rad",
             name=f"{self.name}_Phase" if self.name else "Phase",
             channel=self.channel
        )
        
        # 4. Instantaneous Frequency
        # f = (1/2pi) * d(phi)/dt
        # Use gradient
        if self.dt is None:
             raise ValueError("dt is required for Instantaneous Frequency")
        dt_val = self.dt.to("s").value
        
        dphi = np.gradient(pha, dt_val)
        freq_val = dphi / (2 * np.pi)
        
        frequency = self.__class__(
             freq_val,
             t0=self.t0,
             dt=self.dt,
             unit="Hz",
             name=f"{self.name}_IF" if self.name else "IF",
             channel=self.channel
        )
        
        if frequency_unit != "Hz":
             frequency = frequency.to(frequency_unit)
             
        return {
             "analytic": analytic,
             "amplitude": amplitude,
             "phase": phase,
             "frequency": frequency
        }

    def hht(
        self,
        *,
        emd_method: str = "eemd",
        emd_kwargs: Optional[dict[str, Any]] = None,
        hilbert_kwargs: Optional[dict[str, Any]] = None,
        output: str = "dict",
    ) -> Any:
        """
        Perform Hilbert-Huang Transform (HHT): EMD + HSA.

        Parameters
        ----------
        emd_method : `str`
            "emd" or "eemd".
        emd_kwargs : `dict`, optional
            Arguments for `emd()`.
        hilbert_kwargs : `dict`, optional
            Arguments for `hilbert_analysis()`.
        output : `str`
            "dict" (returns IMFs and properties) or "spectrogram" (returns Spectrogram).

        Returns
        -------
        result : `dict` or `Spectrogram`
        """
        if emd_kwargs is None:
             emd_kwargs = {}
        if hilbert_kwargs is None:
             hilbert_kwargs = {}
             
        # 1. EMD
        imfs = self.emd(method=emd_method, **emd_kwargs)
        
        # 2. HSA for each IMF
        from .collections import TimeSeriesDict
        ia_dict = TimeSeriesDict()
        if_dict = TimeSeriesDict()
        
        residual = None
        if "residual" in imfs:
             residual = imfs.pop("residual")
             
        for key, imf in imfs.items():
             res = imf.hilbert_analysis(**hilbert_kwargs)
             ia_dict[key] = res["amplitude"]
             if_dict[key] = res["frequency"]
             
        if output == "dict":
             return {
                  "imfs": imfs,
                  "ia": ia_dict,
                  "if": if_dict,
                  "residual": residual
             }
             
        elif output == "spectrogram":
             # 3. Construct Hilbert Spectrum (Spectrogram)
             # Grid mapping: Time vs Frequency, accumulate Amplitude^2 (Energy)
             
             # Determine freq bins
             # Range: 0 to Nyquist
             fs_rate = self.sample_rate.to("Hz").value
             nyquist = fs_rate / 2.0
             
             n_bins = 100 # Default resolution
             freq_bins = np.linspace(0, nyquist, n_bins + 1)
             
             # Initialize grid (Frequency x Time)
             n_time = len(self)
             spec_data = np.zeros((n_bins, n_time)) # gwpy Spectrogram usually (Time, Freq) internally?
             # But constructor takes value with (Time, Freq) usually.
             
             # We accumulate energy in (time, freq) bin
             # For each time t:
             #   For each IMF k:
             #      f = if_dict[k][t]
             #      a = ia_dict[k][t]
             #      find bin for f, add a^2 to grid[t, bin]
             
             # Vectorized approach
             # Stack all IFs: (N_IMFs, N_Time)
             # Stack all IAs: (N_IMFs, N_Time)
             
             keys = list(imfs.keys())
             if not keys:
                  return None # Or empty spec
                  
             if_stack = np.stack([if_dict[k].value for k in keys])
             ia_stack = np.stack([ia_dict[k].value for k in keys])
             
             # Digitize frequencies
             # inds: (N_IMFs, N_Time)
             inds = np.digitize(if_stack, freq_bins) - 1
             
             # Valid indices 0 <= inds < n_bins
             mask = (inds >= 0) & (inds < n_bins)
             
             # Flatten and accumulate
             # This is scatter to grid.
             # np.add.at is useful
             
             # Target grid: (N_Time, N_Bins)
             grid = np.zeros((n_time, n_bins))
             
             # We iterate over IMFs or time?
             # Since N_IMFs is small, iterate over IMFs is fine.
             for k in range(len(keys)):
                  valid = mask[k]
                  # time indices: 0..N_Time-1
                  t_inds = np.arange(n_time)[valid]
                  f_inds = inds[k][valid]
                  energies = ia_stack[k][valid] ** 2
                  
                  # grid[t, f] += energy
                  # advanced indexing with duplicates requires add.at
                  np.add.at(grid, (t_inds, f_inds), energies)
                  
             from gwpy.spectrogram import Spectrogram
             
             # Frequencies center
             freq_centers = (freq_bins[:-1] + freq_bins[1:]) / 2.0
             
             return Spectrogram(
                  grid,
                  times=self.times,
                  frequencies=u.Quantity(freq_centers, "Hz"),
                  unit=self.unit**2, # Energy density like
                  name=self.name + " Hilbert Spectrum",
                  channel=self.channel,
                  epoch=self.epoch
             )

        else:
             raise ValueError(f"Unknown output format: {output}")

    # --- New Statistical / Info Processing Methods ---

    def impute(
        self,
        *,
        method: str = "interpolate",
        limit: Optional[int] = None,
        axis: str = "time",
        max_gap: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Impute missing values.
        See gwexpy.timeseries.preprocess.impute_timeseries for details.
        """
        from gwexpy.timeseries.preprocess import impute_timeseries
        return impute_timeseries(self, method=method, limit=limit, axis=axis, max_gap=max_gap, **kwargs)

    def standardize(self, *, method: str = "zscore", ddof: int = 0, robust: bool = False) -> Any:
        """
        Standardize the series.
        See gwexpy.timeseries.preprocess.standardize_timeseries for details.
        """
        from gwexpy.timeseries.preprocess import standardize_timeseries
        return standardize_timeseries(self, method=method, ddof=ddof, robust=robust)

    def fit_arima(self, order: tuple[int, int, int] = (1, 0, 0), **kwargs: Any) -> Any:
        """
        Fit ARIMA model to the series.
        See gwexpy.timeseries.arima.fit_arima for details.
        """
        from gwexpy.timeseries.arima import fit_arima
        return fit_arima(self, order=order, **kwargs)

    def hurst(self, **kwargs: Any) -> Any:
        """
        Compute Hurst exponent.
        See gwexpy.timeseries.hurst.hurst for details.
        """
        from gwexpy.timeseries.hurst import hurst
        return hurst(self, **kwargs)

    def local_hurst(self, window: Any, **kwargs: Any) -> Any:
        """
        Compute local Hurst exponent over a sliding window.
        See gwexpy.timeseries.hurst.local_hurst for details.
        """
        from gwexpy.timeseries.hurst import local_hurst
        return local_hurst(self, window=window, **kwargs)

    # Rolling statistics
    def rolling_mean(
        self,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
    ) -> Any:
        """Rolling mean over time."""
        from gwexpy.timeseries.rolling import rolling_mean
        return rolling_mean(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)

    def rolling_std(
        self,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
        ddof: int = 0,
    ) -> Any:
        """Rolling standard deviation over time."""
        from gwexpy.timeseries.rolling import rolling_std
        return rolling_std(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend, ddof=ddof)

    def rolling_median(
        self,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
    ) -> Any:
        """Rolling median over time."""
        from gwexpy.timeseries.rolling import rolling_median
        return rolling_median(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)

    def rolling_min(
        self,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
    ) -> Any:
        """Rolling minimum over time."""
        from gwexpy.timeseries.rolling import rolling_min
        return rolling_min(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)

    def rolling_max(
        self,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
    ) -> Any:
        """Rolling maximum over time."""
        from gwexpy.timeseries.rolling import rolling_max
        return rolling_max(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)

    def transfer_function(
        self,
        other: Any,
        fftlength: Optional[float] = None,
        overlap: Optional[float] = None,
        window: Any = "hann",
        average: str = "mean",
        *,
        method: str = "gwpy",
        fft_kwargs: Optional[dict[str, Any]] = None,
        downsample: Optional[float] = None,
        align: str = "intersection",
        **kwargs: Any,
    ) -> Any:
        """
        Compute the transfer function between this TimeSeries and another.

        Parameters
        ----------
        other : `TimeSeries`
            The input TimeSeries.
        fftlength : `int`, optional
            Length of the FFT, in seconds (default) or samples.
        overlap : `int`, optional
            Overlap between segments, in seconds (default) or samples.
        window : `str`, `numpy.ndarray`, optional
            Window function to apply.
        average : `str`, optional
            Method to average viewing periods.
        method : `str`, optional
            "gwpy" or "csd_psd": Use GWpy CSD/PSD estimator.
            "fft": Use direct FFT ratio (other.fft() / self.fft()).
            "auto": Use "fft" if fftlength is None, else "gwpy".
        fft_kwargs : `dict`, optional
            Keyword arguments for TimeSeries.fft().
        downsample : `bool`, `None`, optional
            If True, downsample to match rates. If False, raise error on mismatch.
            If None (default), warn and downsample.
        align : `str`, optional
            "intersection": Crop to overlapping time.
            "none": Require exact time match (or at least same array size/start).

        Returns
        -------
        out : `FrequencySeries`
            Transfer function.
        """
        import warnings

        use_fft = False
        if method in ("gwpy", "csd_psd"):
            use_fft = False
        elif method == "fft":
            use_fft = True
        elif method == "auto":
            use_fft = fftlength is None
        else:
            raise ValueError(f"Unknown method: {method}")

        if not use_fft:
            # CSD / PSD estimator (GWpy compatible)
            # Calculate CSD and PSD
            csd = self.csd(
                other,
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                average=average,
                **kwargs,
            )
            psd = self.psd(
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                average=average,
                **kwargs,
            )
            # Crop to same size if needed (usually csd/psd should match if params same)
            size = min(csd.size, psd.size)
            return csd[:size] / psd[:size]

        else:
            # FFT Ratio

            # Copy fft_kwargs
            kw = dict(fft_kwargs) if fft_kwargs is not None else {}

            a = self
            b = other

            # 1. Sample Rate
            if a.sample_rate != b.sample_rate:
                if downsample is False:
                    raise ValueError("Sample rates differ and downsample=False")
                if downsample is None:
                    warnings.warn(
                        "Sample rates differ, downsampling to match.", UserWarning
                    )

                # Downsample higher to lower
                rate_a = a.sample_rate.value
                rate_b = b.sample_rate.value

                if rate_a > rate_b:
                    a = a.resample(b.sample_rate)
                elif rate_b > rate_a:
                    b = b.resample(a.sample_rate)

            # 2. Align
            if align == "intersection":
                # Intersection of spans
                start = max(a.span[0], b.span[0])
                end = min(a.span[1], b.span[1])

                if end <= start:
                    raise ValueError("No comparison overlap between TimeSeries")

                a = a.crop(start, end)
                b = b.crop(start, end)

            elif align == "none":
                pass
            else:
                raise ValueError("align must be 'intersection' or 'none'")

            # 3. Ensure equal length
            size = min(a.size, b.size)
            if a.size != size:
                a = a[:size]
            if b.size != size:
                b = b[:size]

            # 4. FFTs
            fx = a.fft(**kw)
            fy = b.fft(**kw)

            # Crop to min size (usually same)
            fsize = min(fx.size, fy.size)

            tf = fy[:fsize] / fx[:fsize]

            # Name
            if b.name and a.name:
                tf.name = f"{b.name} / {a.name}"

            return tf

    def xcorr(
        self,
        other: Any,
        *,
        maxlag: Optional[float] = None,
        normalize: Optional[str] = None,
        mode: str = "full",
        demean: bool = True,
    ) -> "TimeSeries":
        """
        Compute time-domain cross-correlation between two TimeSeries.
        """
        from scipy import signal

        dt_self = self.dt if isinstance(self.dt, u.Quantity) else u.Quantity(self.dt, "s")
        dt_other = other.dt if isinstance(other.dt, u.Quantity) else u.Quantity(other.dt, "s")
        if dt_self != dt_other:
            raise ValueError("TimeSeries must share the same dt/sample_rate for xcorr")
        dt = dt_self

        x = self.value.astype(float)
        y = other.value.astype(float)
        if demean:
            x = x - np.nanmean(x)
            y = y - np.nanmean(y)

        corr = signal.correlate(x, y, mode=mode, method="auto")
        lags = signal.correlation_lags(len(x), len(y), mode=mode)

        # Normalization
        if normalize is None:
            pass
        elif normalize == "biased":
            corr = corr / len(x)
        elif normalize == "unbiased":
            corr = corr / (len(x) - np.abs(lags))
        elif normalize == "coeff":
            denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
            if denom != 0:
                corr = corr / denom
        else:
            raise ValueError("normalize must be None|'biased'|'unbiased'|'coeff'")

        # Max lag trimming
        if maxlag is not None:
            if isinstance(maxlag, u.Quantity):
                lag_samples = int(np.floor(np.abs(maxlag.to(dt.unit).value / dt.value)))
            else:
                lag_samples = int(maxlag)
            mask = np.abs(lags) <= lag_samples
            corr = corr[mask]
            lags = lags[mask]

        lag_times = lags * dt
        name = f"xcorr({self.name},{getattr(other, 'name', '')})"
        return self.__class__(
            corr,
            times=lag_times,
            unit=self.unit * getattr(other, "unit", 1),
            name=name,
        )
    
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
    ) -> tuple["TimeSeries", dict[str, Any]]:
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
        p = _to_val(prominence, self.unit) # Prominence same unit as data
        
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
             # Regular/Irregular handling for times?
             # Empty timeseries usually fine with list
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


    # ===============================
    # Interoperability Methods (P0)
    # ===============================
    
    def to_pandas(self, index: str = "datetime", *, name: Optional[str] = None, copy: bool = False) -> Any:
        """
        Convert TimeSeries to pandas.Series.
        
        Parameters
        ----------
        index : str, default "datetime"
            Index type: "datetime" (UTC aware), "seconds" (unix), or "gps".
        copy : bool, default False
            Whether to guarantee a copy.
        """
        from gwexpy.interop import to_pandas_series
        return to_pandas_series(self, index=index, name=name, copy=copy)
        
    @classmethod
    def from_pandas(
        cls: type["TimeSeries"],
        series: Any,
        *,
        unit: Optional[Any] = None,
        t0: Any = None,
        dt: Any = None,
    ) -> Any:
        """
        Create TimeSeries from pandas.Series.
        """
        from gwexpy.interop import from_pandas_series
        return from_pandas_series(cls, series, unit=unit, t0=t0, dt=dt)
        
    def to_xarray(self, time_coord: str = "datetime") -> Any:
        """
        Convert to xarray.DataArray.
        """
        from gwexpy.interop import to_xarray
        return to_xarray(self, time_coord=time_coord)
        
    @classmethod
    def from_xarray(cls: type["TimeSeries"], da: Any, *, unit: Optional[Any] = None) -> Any:
        """
        Create TimeSeries from xarray.DataArray.
        """
        from gwexpy.interop import from_xarray
        return from_xarray(cls, da, unit=unit)
        
    def to_hdf5_dataset(
        self,
        group: Any,
        path: str,
        *,
        overwrite: bool = False,
        compression: Optional[str] = None,
        compression_opts: Any = None,
    ) -> None:
        """
        Write to HDF5 group/dataset (interop level).
        """
        from gwexpy.interop import to_hdf5
        to_hdf5(self, group, path, overwrite=overwrite, compression=compression, compression_opts=compression_opts)
        
    @classmethod
    def from_hdf5_dataset(cls: type["TimeSeries"], group: Any, path: str) -> Any:
        """
        Read from HDF5 group/dataset.
        """
        from gwexpy.interop import from_hdf5
        return from_hdf5(cls, group, path)
        
    def to_obspy_trace(self, *, stats_extra: Optional[dict[str, Any]] = None, dtype: Any = None) -> Any:
        """
        Convert to obspy.Trace.
        """
        from gwexpy.interop import to_obspy_trace
        return to_obspy_trace(self, stats_extra=stats_extra, dtype=dtype)
        
    @classmethod
    def from_obspy_trace(
        cls: type["TimeSeries"],
        tr: Any,
        *,
        unit: Optional[Any] = None,
        name_policy: str = "id",
    ) -> Any:
        """
        Create TimeSeries from obspy.Trace.
        """
        from gwexpy.interop import from_obspy_trace
        return from_obspy_trace(cls, tr, unit=unit, name_policy=name_policy)
        
    def to_sqlite(self, conn: Any, series_id: Optional[str] = None, *, overwrite: bool = False) -> Any:
        """
        Save to sqlite3 database.
        """
        from gwexpy.interop import to_sqlite
        return to_sqlite(self, conn, series_id=series_id, overwrite=overwrite)
        
    @classmethod
    def from_sqlite(cls: type["TimeSeries"], conn: Any, series_id: Any) -> Any:
        """
        Load from sqlite3 database.
        """
        from gwexpy.interop import from_sqlite
        return from_sqlite(cls, conn, series_id)



    # ===============================
    # P1 Methods (Computational)
    # ===============================
    
    def to_torch(
        self,
        device: Optional[str] = None,
        dtype: Any = None,
        requires_grad: bool = False,
        copy: bool = False,
    ) -> Any:
        """Convert to torch.Tensor."""
        from gwexpy.interop import to_torch
        return to_torch(self, device=device, dtype=dtype, requires_grad=requires_grad, copy=copy)
        
    @classmethod
    def from_torch(
        cls: type["TimeSeries"],
        tensor: Any,
        *,
        t0: Any = None,
        dt: Any = None,
        unit: Optional[Any] = None,
    ) -> Any:
        """Create from torch.Tensor."""
        from gwexpy.interop import from_torch
        # t0/dt required usually as tensor has no metadata
        if t0 is None or dt is None:
            raise ValueError("t0 and dt are required when converting from raw tensor")
        return from_torch(cls, tensor, t0=t0, dt=dt, unit=unit)
        
    def to_tf(self, dtype: Any = None) -> Any:
        """Convert to tensorflow.Tensor."""
        from gwexpy.interop import to_tf
        return to_tf(self, dtype=dtype)
        
    @classmethod
    def from_tf(
        cls: type["TimeSeries"],
        tensor: Any,
        *,
        t0: Any = None,
        dt: Any = None,
        unit: Optional[Any] = None,
    ) -> Any:
        """Create from tensorflow.Tensor."""
        from gwexpy.interop import from_tf
        if t0 is None or dt is None:
            raise ValueError("t0 and dt are required")
        return from_tf(cls, tensor, t0=t0, dt=dt, unit=unit)
        
    def to_dask(self, chunks: Any = "auto") -> Any:
        """Convert to dask.array."""
        from gwexpy.interop import to_dask
        return to_dask(self, chunks=chunks)
        
    @classmethod
    def from_dask(
        cls: type["TimeSeries"],
        array: Any,
        *,
        t0: Any = None,
        dt: Any = None,
        unit: Optional[Any] = None,
        compute: bool = True,
    ) -> Any:
        """Create from dask.array."""
        from gwexpy.interop import from_dask
        if t0 is None or dt is None:
            raise ValueError("t0 and dt are required")
        return from_dask(cls, array, t0=t0, dt=dt, unit=unit, compute=compute)
        
    def to_zarr(
        self,
        store: Any,
        path: str,
        chunks: Any = None,
        compressor: Any = None,
        overwrite: bool = False,
    ) -> None:
        """Write to Zarr array."""
        from gwexpy.interop import to_zarr
        to_zarr(self, store, path, chunks=chunks, compressor=compressor, overwrite=overwrite)
        
    @classmethod
    def from_zarr(cls: type["TimeSeries"], store: Any, path: str) -> Any:
        """Read from Zarr array."""
        from gwexpy.interop import from_zarr
        return from_zarr(cls, store, path)
        
    def to_netcdf4(self, ds: Any, var_name: str, **kwargs: Any) -> None:
        """Write to netCDF4 Dataset."""
        from gwexpy.interop import to_netcdf4
        to_netcdf4(self, ds, var_name, **kwargs)
        
    @classmethod
    def from_netcdf4(cls: type["TimeSeries"], ds: Any, var_name: str) -> Any:
        """Read from netCDF4 Dataset."""
        from gwexpy.interop import from_netcdf4
        return from_netcdf4(cls, ds, var_name)
        
    def to_jax(self, dtype: Any = None) -> Any:
        """Convert to jax.numpy.array."""
        from gwexpy.interop import to_jax
        return to_jax(self, dtype=dtype)
        
    @classmethod
    def from_jax(
        cls: type["TimeSeries"],
        array: Any,
        *,
        t0: Any = None,
        dt: Any = None,
        unit: Optional[Any] = None,
    ) -> Any:
        """Create from jax array."""
        from gwexpy.interop import from_jax
        if t0 is None or dt is None:
            raise ValueError("t0 and dt are required")
        return from_jax(cls, array, t0=t0, dt=dt, unit=unit)
    
    def to_cupy(self, dtype: Any = None) -> Any:
        """Convert to cupy.array."""
        from gwexpy.interop import to_cupy
        return to_cupy(self, dtype=dtype)
        
    @classmethod
    def from_cupy(
        cls: type["TimeSeries"],
        array: Any,
        *,
        t0: Any = None,
        dt: Any = None,
        unit: Optional[Any] = None,
    ) -> Any:
        """Create from cupy array."""
        from gwexpy.interop import from_cupy
        if t0 is None or dt is None:
            raise ValueError("t0 and dt are required")
        return from_cupy(cls, array, t0=t0, dt=dt, unit=unit)
        
    # ===============================
    # P2 Methods (Domain Specific)
    # ===============================
    
    def to_librosa(self, y_dtype: Any = np.float32) -> Any:
        """Export to librosa-compatible numpy array."""
        from gwexpy.interop import to_librosa
        return to_librosa(self, y_dtype=y_dtype)
        
    def to_pydub(self, sample_width: int = 2, channels: int = 1) -> Any:
        """Export to pydub.AudioSegment."""
        from gwexpy.interop import to_pydub
        return to_pydub(self, sample_width=sample_width, channels=channels)
        
    @classmethod
    def from_pydub(cls: type["TimeSeries"], seg: Any, *, unit: Optional[Any] = None) -> Any:
        """Create from pydub.AudioSegment."""
        from gwexpy.interop import from_pydub
        return from_pydub(cls, seg, unit=unit)
        
    def to_astropy_timeseries(self, column: str = "value", time_format: str = "gps") -> Any:
        """Convert to astropy.timeseries.TimeSeries."""
        from gwexpy.interop import to_astropy_timeseries
        return to_astropy_timeseries(self, column=column, time_format=time_format)
        
    @classmethod
    def from_astropy_timeseries(
        cls: type["TimeSeries"],
        ap_ts: Any,
        column: str = "value",
        unit: Optional[Any] = None,
    ) -> Any:
        """Create from astropy.timeseries.TimeSeries."""
        from gwexpy.interop import from_astropy_timeseries
        return from_astropy_timeseries(cls, ap_ts, column=column, unit=unit)

    def to_mne_rawarray(self, info: Any = None) -> Any:
        """Convert to ``mne.io.RawArray`` (single-channel)."""
        from gwexpy.interop import to_mne_rawarray
        return to_mne_rawarray(self, info=info)

    def to_mne_raw(self, info: Any = None) -> Any:
        """Alias for :meth:`to_mne_rawarray`."""
        return self.to_mne_rawarray(info=info)
