import inspect
from enum import Enum
import numpy as np
import numpy as np
from astropy import units as u
try:
    import scipy.signal
except ImportError:
    pass # scipy is optional dependency for gwpy but required here for hilbert


from gwpy.timeseries import TimeSeries as BaseTimeSeries
from gwpy.timeseries import TimeSeriesDict as BaseTimeSeriesDict
from gwpy.timeseries import TimeSeriesList as BaseTimeSeriesList

from gwexpy.types.seriesmatrix import SeriesMatrix
from gwexpy.types.metadata import MetaData, MetaDataMatrix
from gwexpy.frequencyseries.frequencyseries import FrequencySeriesMatrix

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


def _extract_axis_info(ts):
    """
    Extract axis information for a TimeSeries-like object.
    """
    axis = getattr(ts, "times", None)
    if axis is None:
        raise ValueError("times axis is required")

    regular = False
    dt = None
    try:
        dt = ts.dt
    except Exception:
        dt = None

    if dt is not None:
        try:
            if isinstance(dt, u.Quantity):
                val = dt.value
                finite = np.isfinite(val)
                zero = np.all(val == 0)
            else:
                val = float(dt)
                finite = np.isfinite(val)
                zero = (val == 0)
            regular = bool(finite and not zero)
        except Exception:
            regular = False

    t0 = None
    if regular:
        try:
            t0 = ts.t0
        except Exception:
            try:
                t0 = axis[0]
            except Exception:
                t0 = None
        if t0 is None:
            regular = False

    try:
        n = len(axis)
    except Exception:
        n = None
        regular = False

    return {"regular": regular, "dt": dt, "t0": t0, "n": n, "times": axis}


def _validate_common_axis(axis_infos, method_name):
    """
    Validate that a list of axis infos share a common axis.
    """
    if not axis_infos:
        return None, 0

    all_regular = all(info["regular"] for info in axis_infos)
    if all_regular:
        ref = axis_infos[0]
        ref_dt = ref["dt"]
        ref_t0 = ref["t0"]
        ref_n = ref["n"]
        for info in axis_infos[1:]:
            if info["n"] != ref_n:
                raise ValueError(
                    f"{method_name} requires common length; mismatch in length"
                )
            if info["dt"] != ref_dt:
                raise ValueError(f"{method_name} requires common dt; mismatch in dt")
            if info["t0"] != ref_t0:
                raise ValueError(f"{method_name} requires common t0; mismatch in t0")
        return ref["times"], ref_n

    ref_times = axis_infos[0]["times"]
    ref_unit = getattr(ref_times, "unit", None)
    ref_vals = ref_times.value if hasattr(ref_times, "value") else ref_times
    for info in axis_infos[1:]:
        times = info["times"]
        unit = getattr(times, "unit", None)
        if ref_unit is not None or unit is not None:
            if ref_unit != unit:
                raise ValueError(
                    f"{method_name} requires common times unit; mismatch in times"
                )
            lhs = times.value if hasattr(times, "value") else None
            rhs = ref_times.value if hasattr(ref_times, "value") else None
            if lhs is None or rhs is None:
                raise ValueError(
                    f"{method_name} requires comparable times arrays; mismatch in times"
                )
            if not np.array_equal(lhs, rhs):
                raise ValueError(
                    f"{method_name} requires identical times arrays; mismatch in times"
                )
        else:
            if not np.array_equal(times, ref_times):
                raise ValueError(
                    f"{method_name} requires identical times arrays; mismatch in times"
                )
    return ref_times, len(ref_vals)


def _extract_freq_axis_info(fs):
    """
    Extract frequency-axis information from a FrequencySeries-like object.
    """
    freqs = getattr(fs, "frequencies", None)
    if freqs is None:
        raise ValueError("frequencies axis is required")

    regular = False
    df = None
    try:
        df = fs.df
    except Exception:
        df = None

    if df is not None:
        try:
            if isinstance(df, u.Quantity):
                val = df.value
                finite = np.isfinite(val)
                zero = np.all(val == 0)
            else:
                val = float(df)
                finite = np.isfinite(val)
                zero = (val == 0)
            regular = bool(finite and not zero)
        except Exception:
            regular = False

    f0 = None
    if regular:
        try:
            f0 = fs.f0
        except Exception:
            try:
                f0 = freqs[0]
            except Exception:
                f0 = None
        if f0 is None:
            regular = False

    try:
        n = len(freqs)
    except Exception:
        n = None
        regular = False

    return {"regular": regular, "df": df, "f0": f0, "n": n, "freqs": freqs}


def _validate_common_frequency_axis(axis_infos, method_name):
    """
    Validate common frequency axis across FrequencySeries results.
    """
    if not axis_infos:
        return None, None, None, 0

    all_regular = all(info["regular"] for info in axis_infos)
    if all_regular:
        ref = axis_infos[0]
        ref_df = ref["df"]
        ref_f0 = ref["f0"]
        ref_n = ref["n"]
        for info in axis_infos[1:]:
            if info["n"] != ref_n:
                raise ValueError(
                    f"{method_name} requires common length; mismatch in length"
                )
            if info["df"] != ref_df:
                raise ValueError(f"{method_name} requires common df; mismatch in df")
            if info["f0"] != ref_f0:
                raise ValueError(f"{method_name} requires common f0; mismatch in f0")
        return ref["freqs"], ref_df, ref_f0, ref_n

    ref_freqs = axis_infos[0]["freqs"]
    ref_unit = getattr(ref_freqs, "unit", None)
    ref_vals = ref_freqs.value if hasattr(ref_freqs, "value") else ref_freqs
    for info in axis_infos[1:]:
        freqs = info["freqs"]
        unit = getattr(freqs, "unit", None)
        if ref_unit is not None or unit is not None:
            if ref_unit != unit:
                raise ValueError(
                    f"{method_name} requires common frequencies unit; mismatch in frequencies"
                )
            lhs = freqs.value if hasattr(freqs, "value") else None
            rhs = ref_freqs.value if hasattr(ref_freqs, "value") else None
            if lhs is None or rhs is None:
                raise ValueError(
                    f"{method_name} requires comparable frequency arrays; mismatch in frequencies"
                )
            if not np.array_equal(lhs, rhs):
                raise ValueError(
                    f"{method_name} requires identical frequency arrays; mismatch in frequencies"
                )
        else:
            if not np.array_equal(freqs, ref_freqs):
                raise ValueError(
                    f"{method_name} requires identical frequency arrays; mismatch in frequencies"
                )
    return ref_freqs, None, None, len(ref_vals)


def _validate_common_epoch(epochs, method_name):
    if not epochs:
        return None
    ref = epochs[0]
    for e in epochs[1:]:
        if e != ref:
            raise ValueError(f"{method_name} requires common epoch; mismatch in epoch")
    return ref


class TimeSeries(BaseTimeSeries):
    """Light wrapper of gwpy's TimeSeries for compatibility."""

    def asfreq(
        self,
        rule,
        method=None,
        fill_value=np.nan,
        *,
        origin='t0',
        offset=0 * u.s,
        align='ceil',
        tolerance=None,
        max_gap=None,
        copy=True,
    ):
        """
        Reindex the TimeSeries to a new fixed-interval grid associated with the given rule.
        """
        # 1. Parse rule to target dt (Quantity)
        if isinstance(rule, str):
            # Simple parser for '1s', '10ms' etc.
            # Using unit logic from astropy or similar
            try:
                target_dt = u.Quantity(rule)
            except TypeError:
                # If astropy fails to parse simple string directly, try splitting
                import re
                match = re.match(r"([0-9\.]+)([a-zA-Z]+)", rule)
                if match:
                    val, unit_str = match.groups()
                    target_dt = float(val) * u.Unit(unit_str)
                else:
                    raise ValueError(f"Could not parse rule: {rule}")
        elif isinstance(rule, u.Quantity):
            target_dt = rule
        else:
             # Fallback: assume seconds if it's a number? 
             # Requirement says: "support s/ms/us/ns/min/h/day", numeric not supported by default.
             raise TypeError("rule must be a string or astropy Quantity (time).")

        if not target_dt.unit.is_equivalent(u.s):
             raise ValueError("rule must be time-like")

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
             
             # Extract values directly as they are assumed to be seconds
             start_time_val = old_times_q[0].value
             stop_time_val = self.span[1].value if hasattr(self.span[1], 'value') else self.span[1]
        else:
             target_dt_in_time_unit = u.Quantity(target_dt, time_unit)
             start_time_val = u.Quantity(old_times_q[0], time_unit).value
             stop_time_val = u.Quantity(self.span[1], time_unit).value
             
        dt_val = target_dt_in_time_unit.value
        
        # 3. Determine Origin Base
        if origin == 't0':
            origin_val = start_time_val
        elif origin == 'gps0':
            origin_val = 0.0
        elif isinstance(origin, (u.Quantity, str)):
            # convert origin to same unit as time_unit (if possible) or seconds
            # If origin is a Time object etc? simpler to assume quantity or str
            origin_val = u.Quantity(origin).to(time_unit).value
        else:
            origin_val = 0.0 
            
        offset_val = u.Quantity(offset).to(time_unit).value
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
             return self.__class__([], t0=start_time, dt=target_dt, channel=self.channel, name=self.name, unit=self.unit)

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
                
                if max_gap is not None:
                     max_gap_val = u.Quantity(max_gap).to(time_unit).value
                     # We need to detect gaps in NEW grid that are far from OLD points?
                     # Or gaps in OLD grid?
                     # Standard behavior: if interval between source points > max_gap, fill with NaN.
                     # This is hard to do with simple np.interp.
                     pass

                # Simple linear interp
                # For complex data, interp real/imag separately
                if np.iscomplexobj(y):
                    new_data = np.interp(new_times_val, x, y.real) + 1j * np.interp(new_times_val, x, y.imag)
                else:
                    new_data = np.interp(new_times_val, x, y, left=np.nan, right=np.nan)
                
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

    def resample(self, rate, *args, **kwargs):
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

    def _resample_time_bin(
        self,
        rule,
        agg='mean',
        closed='left',
        label='left',
        origin='t0',
        offset=0 * u.s,
        align='floor',
        min_count=1,
        nan_policy='omit',
        inplace=False,
    ):
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
        pad=None,
        pad_mode='reflect',
        pad_value=0.0,
        nan_policy='raise',
        copy=True,
    ):
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
        
    def hilbert(self, *args, **kwargs):
        """Alias for analytic_signal."""
        return self.analytic_signal(*args, **kwargs)
        
    def envelope(self, *args, **kwargs):
        """Compute the envelope of the TimeSeries."""
        analytic = self.analytic_signal(*args, **kwargs)
        return abs(analytic)
        
    def instantaneous_phase(self, deg=False, unwrap=False, **kwargs):
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
        
    def unwrap_phase(self, deg=False, **kwargs):
        """Alias for instantaneous_phase(unwrap=True)."""
        return self.instantaneous_phase(deg=deg, unwrap=True, **kwargs)
        
    def instantaneous_frequency(self, unwrap=True, smooth=None, **kwargs):
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
        phase=None,
        f0=None,
        fdot=0.0,
        fddot=0.0,
        phase_epoch=None,
        phase0=0.0,
        prefer_dt=True,
    ):
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
        phase=None,
        f0=None,
        fdot=0.0,
        fddot=0.0,
        phase_epoch=None,
        phase0=0.0,
        singlesided=False,
        copy=True,
    ):
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
        phase=None,
        f0=None,
        fdot=0.0,
        fddot=0.0,
        phase_epoch=None,
        phase0=0.0,
        lowpass=None,
        lowpass_kwargs=None,
        output_rate=None,
        singlesided=False,
    ):
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
        phase=None,
        f0=None,
        fdot=0.0,
        fddot=0.0,
        phase_epoch=None,
        phase0=0.0,
        stride=1.0,
        singlesided=True,
        output='amp_phase',
        deg=True,
    ):
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
        nfft=None,
        *,
        mode="gwpy",
        pad_mode="zero",
        pad_left=0,
        pad_right=0,
        nfft_mode=None,
        other_length=None,
        **kwargs,
    ):
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
                return super().fft(nfft=nfft, **kwargs)

            # Fallback to super for gwpy mode even if explicit args (ignore extras)
            return super().fft(nfft=nfft, **kwargs)

        if mode != "transient":
            raise ValueError(f"Unknown mode: {mode}")

        # 2. Transient mode
        # Create padded array
        x = self.value

        # Padding
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

        # Set frequencies
        fs.frequencies = np.fft.rfftfreq(target_nfft, d=self.dt.value)

        return fs

    # --- New Statistical / Info Processing Methods ---

    def impute(self, *, method="interpolate", limit=None, axis="time", **kwargs):
        """
        Impute missing values.
        See gwexpy.timeseries.preprocess.impute_timeseries for details.
        """
        from gwexpy.timeseries.preprocess import impute_timeseries
        return impute_timeseries(self, method=method, limit=limit, axis=axis, **kwargs)

    def standardize(self, *, method="zscore", ddof=0, robust=False):
        """
        Standardize the series.
        See gwexpy.timeseries.preprocess.standardize_timeseries for details.
        """
        from gwexpy.timeseries.preprocess import standardize_timeseries
        return standardize_timeseries(self, method=method, ddof=ddof, robust=robust)

    def fit_arima(self, order=(1, 0, 0), **kwargs):
        """
        Fit ARIMA model to the series.
        See gwexpy.timeseries.arima.fit_arima for details.
        """
        from gwexpy.timeseries.arima import fit_arima
        return fit_arima(self, order=order, **kwargs)

    def hurst(self, **kwargs):
        """
        Compute Hurst exponent.
        See gwexpy.timeseries.hurst.hurst for details.
        """
        from gwexpy.timeseries.hurst import hurst
        return hurst(self, **kwargs)

    def local_hurst(self, window, **kwargs):
        """
        Compute local Hurst exponent over a sliding window.
        See gwexpy.timeseries.hurst.local_hurst for details.
        """
        from gwexpy.timeseries.hurst import local_hurst
        return local_hurst(self, window=window, **kwargs)

    def transfer_function(
        self,
        other,
        fftlength=None,
        overlap=None,
        window="hann",
        average="mean",
        *,
        method="gwpy",
        fft_kwargs=None,
        downsample=None,
        align="intersection",
        **kwargs,
    ):
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






    # ===============================
    # Interoperability Methods (P0)
    # ===============================
    
    def to_pandas(self, index="datetime", *, name=None, copy=False):
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
    def from_pandas(cls, series, *, unit=None, t0=None, dt=None):
        """
        Create TimeSeries from pandas.Series.
        """
        from gwexpy.interop import from_pandas_series
        return from_pandas_series(cls, series, unit=unit, t0=t0, dt=dt)
        
    def to_xarray(self, time_coord="datetime"):
        """
        Convert to xarray.DataArray.
        """
        from gwexpy.interop import to_xarray
        return to_xarray(self, time_coord=time_coord)
        
    @classmethod
    def from_xarray(cls, da, *, unit=None):
        """
        Create TimeSeries from xarray.DataArray.
        """
        from gwexpy.interop import from_xarray
        return from_xarray(cls, da, unit=unit)
        
    def to_hdf5_dataset(self, group, path, *, overwrite=False, compression=None, compression_opts=None):
        """
        Write to HDF5 group/dataset (interop level).
        """
        from gwexpy.interop import to_hdf5
        to_hdf5(self, group, path, overwrite=overwrite, compression=compression, compression_opts=compression_opts)
        
    @classmethod
    def from_hdf5_dataset(cls, group, path):
        """
        Read from HDF5 group/dataset.
        """
        from gwexpy.interop import from_hdf5
        return from_hdf5(cls, group, path)
        
    def to_obspy_trace(self, *, stats_extra=None, dtype=None):
        """
        Convert to obspy.Trace.
        """
        from gwexpy.interop import to_obspy_trace
        return to_obspy_trace(self, stats_extra=stats_extra, dtype=dtype)
        
    @classmethod
    def from_obspy_trace(cls, tr, *, unit=None, name_policy="id"):
        """
        Create TimeSeries from obspy.Trace.
        """
        from gwexpy.interop import from_obspy_trace
        return from_obspy_trace(cls, tr, unit=unit, name_policy=name_policy)
        
    def to_sqlite(self, conn, series_id=None, *, overwrite=False):
        """
        Save to sqlite3 database.
        """
        from gwexpy.interop import to_sqlite
        return to_sqlite(self, conn, series_id=series_id, overwrite=overwrite)
        
    @classmethod
    def from_sqlite(cls, conn, series_id):
        """
        Load from sqlite3 database.
        """
        from gwexpy.interop import from_sqlite
        return from_sqlite(cls, conn, series_id)



    # ===============================
    # P1 Methods (Computational)
    # ===============================
    
    def to_torch(self, device=None, dtype=None, requires_grad=False, copy=False):
        """Convert to torch.Tensor."""
        from gwexpy.interop import to_torch
        return to_torch(self, device=device, dtype=dtype, requires_grad=requires_grad, copy=copy)
        
    @classmethod
    def from_torch(cls, tensor, *, t0=None, dt=None, unit=None):
        """Create from torch.Tensor."""
        from gwexpy.interop import from_torch
        # t0/dt required usually as tensor has no metadata
        if t0 is None or dt is None:
            raise ValueError("t0 and dt are required when converting from raw tensor")
        return from_torch(cls, tensor, t0=t0, dt=dt, unit=unit)
        
    def to_tf(self, dtype=None):
        """Convert to tensorflow.Tensor."""
        from gwexpy.interop import to_tf
        return to_tf(self, dtype=dtype)
        
    @classmethod
    def from_tf(cls, tensor, *, t0=None, dt=None, unit=None):
        """Create from tensorflow.Tensor."""
        from gwexpy.interop import from_tf
        if t0 is None or dt is None:
            raise ValueError("t0 and dt are required")
        return from_tf(cls, tensor, t0=t0, dt=dt, unit=unit)
        
    def to_dask(self, chunks="auto"):
        """Convert to dask.array."""
        from gwexpy.interop import to_dask
        return to_dask(self, chunks=chunks)
        
    @classmethod
    def from_dask(cls, array, *, t0=None, dt=None, unit=None, compute=True):
        """Create from dask.array."""
        from gwexpy.interop import from_dask
        if t0 is None or dt is None:
            raise ValueError("t0 and dt are required")
        return from_dask(cls, array, t0=t0, dt=dt, unit=unit, compute=compute)
        
    def to_zarr(self, store, path, chunks=None, compressor=None, overwrite=False):
        """Write to Zarr array."""
        from gwexpy.interop import to_zarr
        to_zarr(self, store, path, chunks=chunks, compressor=compressor, overwrite=overwrite)
        
    @classmethod
    def from_zarr(cls, store, path):
        """Read from Zarr array."""
        from gwexpy.interop import from_zarr
        return from_zarr(cls, store, path)
        
    def to_netcdf4(self, ds, var_name, **kwargs):
        """Write to netCDF4 Dataset."""
        from gwexpy.interop import to_netcdf4
        to_netcdf4(self, ds, var_name, **kwargs)
        
    @classmethod
    def from_netcdf4(cls, ds, var_name):
        """Read from netCDF4 Dataset."""
        from gwexpy.interop import from_netcdf4
        return from_netcdf4(cls, ds, var_name)
        
    def to_jax(self, dtype=None):
        """Convert to jax.numpy.array."""
        from gwexpy.interop import to_jax
        return to_jax(self, dtype=dtype)
        
    @classmethod
    def from_jax(cls, array, *, t0=None, dt=None, unit=None):
        """Create from jax array."""
        from gwexpy.interop import from_jax
        if t0 is None or dt is None:
            raise ValueError("t0 and dt are required")
        return from_jax(cls, array, t0=t0, dt=dt, unit=unit)
    
    def to_cupy(self, dtype=None):
        """Convert to cupy.array."""
        from gwexpy.interop import to_cupy
        return to_cupy(self, dtype=dtype)
        
    @classmethod
    def from_cupy(cls, array, *, t0=None, dt=None, unit=None):
        """Create from cupy array."""
        from gwexpy.interop import from_cupy
        if t0 is None or dt is None:
            raise ValueError("t0 and dt are required")
        return from_cupy(cls, array, t0=t0, dt=dt, unit=unit)
        
    # ===============================
    # P2 Methods (Domain Specific)
    # ===============================
    
    def to_librosa(self, y_dtype=np.float32):
        """Export to librosa-compatible numpy array."""
        from gwexpy.interop import to_librosa
        return to_librosa(self, y_dtype=y_dtype)
        
    def to_pydub(self, sample_width=2, channels=1):
        """Export to pydub.AudioSegment."""
        from gwexpy.interop import to_pydub
        return to_pydub(self, sample_width=sample_width, channels=channels)
        
    @classmethod
    def from_pydub(cls, seg, *, unit=None):
        """Create from pydub.AudioSegment."""
        from gwexpy.interop import from_pydub
        return from_pydub(cls, seg, unit=unit)
        
    def to_astropy_timeseries(self, column="value", time_format="gps"):
        """Convert to astropy.timeseries.TimeSeries."""
        from gwexpy.interop import to_astropy_timeseries
        return to_astropy_timeseries(self, column=column, time_format=time_format)
        
    @classmethod
    def from_astropy_timeseries(cls, ap_ts, column="value", unit=None):
        """Create from astropy.timeseries.TimeSeries."""
        from gwexpy.interop import from_astropy_timeseries
        return from_astropy_timeseries(cls, ap_ts, column=column, unit=unit)


class TimeSeriesDict(BaseTimeSeriesDict):
    """Dictionary of TimeSeries objects."""

    def asfreq(self, rule, **kwargs):
        """
        Apply asfreq to each TimeSeries in the dict.
        Returns a new TimeSeriesDict.
        """
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.asfreq(rule, **kwargs)
        return new_dict
        
    def resample(self, rate, **kwargs):
        """
        Resample items in the TimeSeriesDict. 
        In-place operation (updates the dict contents).
        
        If rate is time-like, performs time-bin resampling.
        Otherwise performs signal processing resampling (gwpy's native behavior).
        """
        is_time_bin = False
        if isinstance(rate, str):
            is_time_bin = True
        elif isinstance(rate, u.Quantity) and rate.unit.physical_type == 'time':
            is_time_bin = True
            
        if is_time_bin:
            # Time-bin logic: replace items in-place
            # We can't strictly modify the objects in-place easily 
            # (asfreq/resample return new objects usually), 
            # so we replace the values in the dict.
            for key in list(self.keys()):
                 self[key] = self[key].resample(rate, **kwargs)
            return self
        else:
            # Native gwpy resample (signal processing)
            # gwpy's TimeSeriesDict.resample is in-place
            return super().resample(rate, **kwargs)
            
    def analytic_signal(self, *args, **kwargs):
        """Apply analytic_signal to each item."""
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.analytic_signal(*args, **kwargs)
        return new_dict
        
    def hilbert(self, *args, **kwargs):
        """Alias for analytic_signal."""
        return self.analytic_signal(*args, **kwargs)
        
    def envelope(self, *args, **kwargs):
        """Apply envelope to each item."""
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.envelope(*args, **kwargs)
        return new_dict
        
    def instantaneous_phase(self, *args, **kwargs):
        return new_dict
        
    # ===============================
    # P2 Methods (Domain Specific)
    # ===============================
    
    def to_mne_rawarray(self, info=None, picks=None):
        """Convert to mne.io.RawArray."""
        from gwexpy.interop import to_mne_rawarray
        return to_mne_rawarray(self, info=info, picks=picks)
        
    @classmethod
    def from_mne_raw(cls, raw, *, unit_map=None):
        """Create from mne.io.Raw."""
        from gwexpy.interop import from_mne_raw
        return from_mne_raw(cls, raw, unit_map=unit_map)
        
    def unwrap_phase(self, *args, **kwargs):
        """Apply unwrap_phase to each item."""
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.unwrap_phase(*args, **kwargs)
        return new_dict
        
    def instantaneous_frequency(self, *args, **kwargs):
        """Apply instantaneous_frequency to each item."""
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.instantaneous_frequency(*args, **kwargs)
        return new_dict

    def mix_down(self, *args, **kwargs):
        """Apply mix_down to each item."""
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.mix_down(*args, **kwargs)
        return new_dict
        
    def baseband(self, *args, **kwargs):
        """Apply baseband to each item."""
        new_dict = self.__class__()
        for key, ts in self.items():
             new_dict[key] = ts.baseband(*args, **kwargs)
        return new_dict
        
    def lock_in(self, *args, **kwargs):
        """
        Apply lock_in to each item.
        Returns TimeSeriesDict (if output='complex') or tuple of TimeSeriesDicts.
        """
        # We need to know the output structure (tuple vs single)
        # Peek first item
        if not self:
             return self.__class__()
             
        keys = list(self.keys())
        first_res = self[keys[0]].lock_in(*args, **kwargs)
        
        if isinstance(first_res, tuple):
             # Tuple return (e.g. mag, phase or i, q)
             # Assume logic dictates uniform return type
             dict_tuple = tuple(self.__class__() for _ in first_res)
             
             for key, ts in self.items():
                  res = ts.lock_in(*args, **kwargs)
                  for i, val in enumerate(res):
                       dict_tuple[i][key] = val
             return dict_tuple
        else:
             # Single return
             new_dict = self.__class__()
             new_dict[keys[0]] = first_res
             for key in keys[1:]:
                  new_dict[key] = self[key].lock_in(*args, **kwargs)
             return new_dict

    def csd_matrix(self, other=None, *, fftlength=None, overlap=None, window='hann', hermitian=True, include_diagonal=True, **kwargs):
        return csd_matrix_from_collection(
            self, other, 
            fftlength=fftlength, overlap=overlap, window=window, 
            hermitian=hermitian, include_diagonal=include_diagonal, 
            **kwargs
        )

    def coherence_matrix(self, other=None, *, fftlength=None, overlap=None, window='hann', symmetric=True, include_diagonal=True, diagonal_value=1.0, **kwargs):
        return coherence_matrix_from_collection(
            self, other,
            fftlength=fftlength, overlap=overlap, window=window,
            symmetric=symmetric, include_diagonal=include_diagonal,
            diagonal_value=diagonal_value,
            **kwargs
        )


    def asd(self, fftlength=4, overlap=2):
        from gwexpy.frequencyseries import FrequencySeries

        dict_cls = getattr(FrequencySeries, "DictClass", None)
        if dict_cls is None:
            from gwexpy.frequencyseries import FrequencySeriesDict as dict_cls  # pragma: no cover

        return dict_cls(
            {
                key: ts.asd(fftlength=fftlength, overlap=overlap).view(FrequencySeries)
                for key, ts in self.items()
            }
        )
        
    
    # ===============================
    # Interoperability Methods (P0)
    # ===============================
    
    def to_pandas(self, index="datetime", *, copy=False):
        """Convert to pandas.DataFrame."""
        from gwexpy.interop import to_pandas_dataframe
        return to_pandas_dataframe(self, index=index, copy=copy)
        
    @classmethod
    def from_pandas(cls, df, *, unit_map=None, t0=None, dt=None):
        """Create TimeSeriesDict from pandas.DataFrame."""
        from gwexpy.interop import from_pandas_dataframe
        return from_pandas_dataframe(cls, df, unit_map=unit_map, t0=t0, dt=dt)
    
    def impute(self, *, method="interpolate", limit=None, axis="time", **kwargs):
        """Apply impute to each item."""
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.impute(method=method, limit=limit, axis=axis, **kwargs)
        return new_dict

    def to_matrix(self, *, align="intersection", **kwargs):
        """
        Convert dictionary to TimeSeriesMatrix with alignment.
        """
        from gwexpy.timeseries.preprocess import align_timeseries_collection
        # Ensure consistent order (keys sorted) or specific
        # Dicts are ordered in modern python but keys() usually safe
        keys = list(self.keys())
        series_list = [self[k] for k in keys]
        
        vals, times, meta = align_timeseries_collection(series_list, how=align, **kwargs)
        
        # SeriesMatrix expects 3D usually (rows, cols, time) or checks last axis
        # vals: (samples, channels).
        # We create (channels, 1, samples).
        data = vals.T[:, None, :]
        
        matrix = TimeSeriesMatrix(
            data,
            times=times,
            # meta might contain channel_names from original list (names of TS objects)
            # But converting dict to matrix usually implies keys become channel names?
            # User requirement: "preserve labels"
            # TimeSeries from dict usually inherit name from key if created via read? 
            # Not always. We should force keys as names?
            # "Must preserve channel ordering from input."
        )
        # Assign channel names from keys
        matrix.channel_names = keys
        return matrix


try:
    from gwpy.types.index import SeriesType  # pragma: no cover - optional in gwpy
except ImportError:  # fallback for gwpy versions without SeriesType
    from enum import Enum

    class SeriesType(Enum):
        TIME = "time"
        FREQ = "freq"


class TimeSeriesList(BaseTimeSeriesList):
    """List of TimeSeries objects."""
    
    def csd_matrix(self, other=None, *, fftlength=None, overlap=None, window='hann', hermitian=True, include_diagonal=True, **kwargs):
        """
        Compute Cross Spectral Density Matrix.
        
        Parameters
        ----------
        other : TimeSeriesDict or TimeSeriesList, optional
            Other collection for cross-CSD.
        fftlength, overlap, window :
            See TimeSeries.csd() arguments.
        hermitian : bool, default=True
            If True and other is None, compute only upper triangle and conjugate fill lower.
        include_diagonal : bool, default=True
            Whether to compute diagonal elements.
            
        Returns
        -------
        FrequencySeriesMatrix
        """
        return csd_matrix_from_collection(
            self, other, 
            fftlength=fftlength, overlap=overlap, window=window, 
            hermitian=hermitian, include_diagonal=include_diagonal, 
            **kwargs
        )

    def coherence_matrix(self, other=None, *, fftlength=None, overlap=None, window='hann', symmetric=True, include_diagonal=True, diagonal_value=1.0, **kwargs):
        """
        Compute Coherence Matrix.
        
        Parameters
        ----------
        other : TimeSeriesDict or TimeSeriesList, optional
            Other collection.
        fftlength, overlap, window :
            See TimeSeries.coherence().
        symmetric : bool, default=True
            If True and other is None, compute only upper triangle and copy to lower.
        include_diagonal : bool, default=True
            Include diagonal.
        diagonal_value : float or None, default=1.0
            Value to fill diagonal if include_diagonal is True. If None, compute diagonal coherence.
            
        Returns
        -------
        FrequencySeriesMatrix
        """
        return coherence_matrix_from_collection(
            self, other,
            fftlength=fftlength, overlap=overlap, window=window,
            symmetric=symmetric, include_diagonal=include_diagonal,
            diagonal_value=diagonal_value,
            **kwargs
        )
    
    def impute(self, *, method="interpolate", limit=None, axis="time", **kwargs):
         new_list = self.__class__()
         for ts in self:
             new_list.append(ts.impute(method=method, limit=limit, axis=axis, **kwargs))
         return new_list

    def to_matrix(self, *, align="intersection", **kwargs):
        from gwexpy.timeseries.preprocess import align_timeseries_collection
        vals, times, meta = align_timeseries_collection(list(self), how=align, **kwargs)
        # Use names from metadata (from TS objects)
        names = meta.get("channel_names")
        
        data = vals.T[:, None, :]
        
        matrix = TimeSeriesMatrix(
            data,
            times=times,
        )
        if names:
             matrix.channel_names = names
        return matrix

class TimeSeriesMatrix(SeriesMatrix):
    """
    Matrix container for multiple TimeSeries objects.

    Provides dt, t0, times aliases and constructs FrequencySeriesMatrix via FFT.
    """

    series_class = TimeSeries
    series_type = SeriesType.TIME
    default_xunit = "s"
    default_yunit = None
    _default_plot_method = "plot"

    def __new__(
        cls,
        data=None,
        times=None,
        dt=None,
        t0=None,
        sample_rate=None,
        epoch=None,
        **kwargs,
    ):
        import warnings

        # 1. Enforce Mutual Exclusivity (GWpy rules)
        if epoch is not None and t0 is not None:
            raise ValueError("give only one of epoch or t0")
        if sample_rate is not None and dt is not None:
            raise ValueError("give only one of sample_rate or dt")

        # 2. Map time-specific args to SeriesMatrix generic args
        if times is not None:
            # If times (xindex) is provided, it takes priority.
            # GWpy semantics: Ignore dx, x0, epoch args if times is present.

            # Check if user provided explicit xindex in kwargs (redundant)
            existing_xindex = kwargs.pop("xindex", None)

            kwargs["xindex"] = times

            # Check for conflict in explicit args
            conflict = False
            if existing_xindex is not None:
                conflict = True
            if dt is not None or sample_rate is not None:
                conflict = True
            if t0 is not None or epoch is not None:
                conflict = True

            # Check for conflict in kwargs (x0, dx, epoch) AND pop them so they don't propagate
            # We must pop them to ensure they are not stored.
            if "dx" in kwargs:
                conflict = True
                kwargs.pop("dx")
            if "x0" in kwargs:
                conflict = True
                kwargs.pop("x0")
            if "epoch" in kwargs:
                # 'epoch' might be in kwargs if passed as **kwargs, though signature captures 'epoch'
                # If it's captured in signature, it's None or set. If in kwargs, it's redundant/conflict.
                conflict = True
                kwargs.pop("epoch")

            if conflict:
                warnings.warn(
                    "dt/sample_rate/t0/epoch/dx/x0/xindex given with times, ignoring",
                    UserWarning,
                )

            # Do NOT set dx, x0, or epoch in kwargs based on the ignored explicit args.

        else:
            # 3. Handle dt / sample_rate -> dx
            if dt is not None:
                kwargs["dx"] = dt
            elif sample_rate is not None:
                # Convert sample_rate to dx = 1/sample_rate
                # Ensure sample_rate is treated as a Quantity if it has units, or raw float.
                if isinstance(sample_rate, u.Quantity):
                    sr_quantity = sample_rate
                else:
                    sr_quantity = u.Quantity(sample_rate, "Hz")

                # dx = 1 / sample_rate
                kwargs["dx"] = (1.0 / sr_quantity).to(
                    kwargs.get("xunit", cls.default_xunit)
                )

            # 4. Handle t0 / epoch -> x0
            if t0 is not None:
                kwargs["x0"] = t0
            elif epoch is not None and "x0" not in kwargs:
                kwargs["x0"] = epoch

            # Default x0 when needed (SeriesMatrix builds index from x0, dx)
            # Only if times is None and (dx is provided) and x0 is missing
            if "dx" in kwargs and "x0" not in kwargs:
                if kwargs.get("xindex") is None:
                    kwargs["x0"] = 0

            # NOTE: We do NOT set kwargs["epoch"] = epoch here anymore (Round 5).
            # GWpy treats epoch as an alias for t0/x0, not inconsistent separate metadata.
            # SeriesMatrix might have its own epoch handling, but TimeSeriesMatrix relies on x0/xindex.

        # Default xunit
        if "xunit" not in kwargs:
            kwargs["xunit"] = cls.default_xunit

        return super().__new__(cls, data, **kwargs)

    # --- Properties mapping to SeriesMatrix attributes ---

    @property
    def dt(self):
        """Time spacing (dx)."""
        return self.dx

    @property
    def t0(self):
        """Start time (x0)."""
        return self.x0

    @property
    def times(self):
        """Time array (xindex)."""
        return self.xindex

    @property
    def span(self):
        """Time span (xspan)."""
        return self.xspan

    @property
    def sample_rate(self):
        """Sampling rate (1/dt)."""
        if self.dt is None:
            return None
        rate = 1.0 / self.dt
        if isinstance(rate, u.Quantity):
            return rate.to("Hz")
        return u.Quantity(rate, "Hz")

    @sample_rate.setter
    def sample_rate(self, value):
        if value is None:
            self.xindex = None
            return

        from gwpy.types.index import Index
        
        # Setter logic...
        # If we have data, we adjust dx? 
        # For matrix, we assume regular sampling if sample_rate is set.
        pass # Not changing existing setter logic here, just context match
        
    # --- Preprocessing & Decomposition ---
    
    def impute(self, **kwargs):
        """Impute missing values."""
        # Matrix-level impute or per-channel?
        # Provide naive implementation mapping over columns or using decomposition logic?
        # User requirement says "TimeSeriesDict/List/Matrix: implement impute(...) that applies per-channel"
        # Since logic isn't in preprocess.py for matrix iteration, we implement it here or call preprocess helper (which I didn't fully genericize).
        
        # We can treat each column as a TimeSeries (if regular).
        # Efficient way:
        new_val = self.value.copy()
        
        # Using impute_timeseries logic inline or map?
        # Let's map for simplicity/correctness reuse
        # value is 3D: (channels, 1, time) or similar
        val_3d = self.value
        n_rows, n_cols, n_samples = val_3d.shape
        
        method = kwargs.get("method", "interpolate")
        
        # We need to flatten loops or iterate
        new_val = val_3d.copy()
        for r in range(n_rows):
             for c in range(n_cols):
                  ts_data = new_val[r, c, :]
                  ts_tmp = self.series_class(ts_data, dt=self.dt, t0=self.t0)
                  imp = ts_tmp.impute(**kwargs)
                  new_val[r, c, :] = imp.value
            
        new_mat = self.copy()
        new_mat.value[:] = new_val
        return new_mat

    def standardize(self, *, axis="time", method="zscore", ddof=0):
        """
        Standardize the matrix.
        See gwexpy.timeseries.preprocess.standardize_matrix.
        """
        # Note: standardize_matrix assumes (channels, time) input correctly now (handles axis).
        return standardize_matrix(self, axis=axis, method=method, ddof=ddof)

    def whiten_channels(self, *, method="pca", eps=1e-12, n_components=None):
        """
        Whiten the matrix (channels/components).
        Returns (whitened_matrix, WhiteningModel).
        See gwexpy.timeseries.preprocess.whiten_matrix.
        """
        return whiten_matrix(self, method=method, eps=eps, n_components=n_components)
        
    def to_dict(self):
        """Convert to TimeSeriesDict."""
        # channel_names property should be available if SeriesMatrix logic works
        # If not, generate defaults
        names = getattr(self, "channel_names", None)
        if names is None:
             names = [str(i) for i in range(self.shape[0])]
             
        # Create dict
        # Assuming we can extract columns as TimeSeries
        d = TimeSeriesDict()
        
        # Flatten structure? Or iterate only rows?
        # If shape is (channels, 1, time), iterating axis 0 covers channels.
        # If shape is (1, channels, time), need logic.
        # We assume flat list of channels for dict.
        # Flatten non-time dimensions.
        flat_val = self.value.reshape(-1, self.shape[-1])
        
        # Use channel_names if length matches
        names = getattr(self, "channel_names", None)
        if names is None or len(names) != flat_val.shape[0]:
             names = [str(i) for i in range(flat_val.shape[0])]
        
        for i, name in enumerate(names):
             # Extract time series
             ts_data = flat_val[i]
             
             ts = self.series_class(ts_data, t0=self.t0, dt=self.dt, name=name)
             d[name] = ts
        return d

    def to_list(self):
        """Convert to TimeSeriesList."""
        d = self.to_dict()
        return TimeSeriesList(d.values())

    # Decomposition
    def pca_fit(self, **kwargs):
        """Fit PCA."""
        return pca_fit(self, **kwargs)

    def pca_transform(self, pca_res, **kwargs):
        """Transform using PCA."""
        return pca_transform(pca_res, self, **kwargs)
        
    def pca_inverse_transform(self, pca_res, scores):
        """Inverse transform PCA scores."""
        return pca_inverse_transform(pca_res, scores)

    def pca(self, return_model=False, **kwargs):
        """Fit and transform PCA."""
        res = self.pca_fit(**kwargs)
        scores = self.pca_transform(res, n_components=kwargs.get("n_components"))
        if return_model:
            return scores, res
        return scores

    def ica_fit(self, **kwargs):
        """Fit ICA."""
        return ica_fit(self, **kwargs)

    def ica_transform(self, ica_res):
        """Transform using ICA."""
        return ica_transform(ica_res, self)

    def ica_inverse_transform(self, ica_res, sources):
        """Inverse transform ICA sources."""
        return ica_inverse_transform(ica_res, sources)
        
    def ica(self, return_model=False, **kwargs):
        """Fit and transform ICA."""
        res = self.ica_fit(**kwargs)
        sources = self.ica_transform(res)
        if return_model:
            return sources, res
        return sources


        if not isinstance(value, u.Quantity):
            value = u.Quantity(value, "Hz")

        # Use existing unit or default to seconds
        # xunit property comes from SeriesMatrix
        xunit = self.xunit
        if xunit is None or xunit == u.dimensionless_unscaled:
            xunit = u.Unit("s")

        # Calculate new dx
        new_dx = (1.0 / value).to(xunit)

        # Rebuild xindex
        # Validate safe start value (must be Quantity for Index.define consistency)
        if self.xindex is not None and len(self.xindex) > 0:
            start_val = self.xindex[0]
            # If start_val is already a Quantity, use it (converted)
            # If it's a float/int (ndarray xindex), wrap it in xunit
            if isinstance(start_val, u.Quantity):
                start = start_val.to(xunit)
            else:
                start = u.Quantity(start_val, xunit)
        else:
            # If xindex is currently None or empty, default x0=0 with correct unit
            start = u.Quantity(0, xunit)

        self.xindex = Index.define(start, new_dx, self.shape[-1])

    # --- Element Access ---

    def __getitem__(self, item):
        """
        Return TimeSeries for single element access, or TimeSeriesMatrix for slicing.
        """
        # 1. Handle scalar access (returning TimeSeries element) directly
        if isinstance(item, tuple) and len(item) == 2:
            r, c = item
            is_scalar_r = isinstance(r, (int, np.integer, str))
            is_scalar_c = isinstance(c, (int, np.integer, str))

            if is_scalar_r and is_scalar_c:
                # Direct access to underlying numpy array and meta
                # Avoid super().__getitem__ which constructs Series

                # Resolve string keys to integers
                ri = self.row_index(r) if isinstance(r, str) else r
                ci = self.col_index(c) if isinstance(c, str) else c

                val = self._value[ri, ci]
                meta = self.meta[ri, ci]

                # Construct TimeSeries
                # GWpy semantics: prefer no-copy for times
                return self.series_class(
                    val,
                    times=self.times,
                    unit=meta.unit,
                    name=meta.name,
                    channel=meta.channel,
                )

        # 2. Handle slicing (returning TimeSeriesMatrix)
        # Call super().__getitem__ which returns SeriesMatrix (or view of it)
        ret = super().__getitem__(item)

        # If the result is a SeriesMatrix, ensure it is viewed as TimeSeriesMatrix
        if isinstance(ret, SeriesMatrix) and not isinstance(ret, TimeSeriesMatrix):
            return ret.view(TimeSeriesMatrix)

        return ret

    # --- Plotting ---

    def plot(self, **kwargs):
        """
        Plot the matrix data.
        """
        if "xscale" not in kwargs:
            kwargs["xscale"] = "auto-gps"
        return super().plot(**kwargs)

    def _apply_timeseries_method(self, method_name, *args, **kwargs):
        """
        Apply a TimeSeries method element-wise and rebuild a TimeSeriesMatrix.

        Parameters
        ----------
        method_name : str
            Name of the TimeSeries method to invoke.
        *args, **kwargs :
            Forwarded to the TimeSeries method. If ``inplace`` is supplied,
            the matrix will be updated in place after applying the method.

        Returns
        -------
        TimeSeriesMatrix
            New matrix unless ``inplace=True`` is requested.
        """
        N, M, _ = self.shape
        if N == 0 or M == 0:
            return self if kwargs.get("inplace", False) else self.copy()

        if not hasattr(self.series_class, method_name):
            raise NotImplementedError(
                f"Not implemented: TimeSeries has no method '{method_name}' in this GWpy version"
            )

        inplace_matrix = bool(kwargs.get("inplace", False))
        base_kwargs = dict(kwargs)
        base_kwargs.pop("inplace", None)

        supports_inplace = False
        ts_attr = getattr(self.series_class, method_name, None)
        if ts_attr is not None:
            try:
                sig = inspect.signature(ts_attr)
                supports_inplace = "inplace" in sig.parameters
            except (TypeError, ValueError):
                supports_inplace = False

        dtype = None
        axis_infos = []
        values = [[None for _ in range(M)] for _ in range(N)]
        meta_array = np.empty((N, M), dtype=object)

        for i in range(N):
            for j in range(M):
                ts = self[i, j]
                method = getattr(ts, method_name)
                call_kwargs = dict(base_kwargs)
                if supports_inplace:
                    call_kwargs["inplace"] = inplace_matrix
                ts_result = method(*args, **call_kwargs)
                if ts_result is None:
                    ts_result = ts

                axis_info = _extract_axis_info(ts_result)
                axis_infos.append(axis_info)
                axis_length = axis_info["n"]
                data_arr = np.asarray(ts_result.value)
                if data_arr.shape[-1] != axis_length:
                    raise ValueError(
                        f"{method_name} produced inconsistent data lengths"
                    )

                values[i][j] = data_arr
                meta_array[i, j] = MetaData(
                    unit=ts_result.unit,
                    name=ts_result.name,
                    channel=ts_result.channel,
                )
                dtype = (
                    data_arr.dtype
                    if dtype is None
                    else np.result_type(dtype, data_arr.dtype)
                )

        common_axis, axis_length = _validate_common_axis(axis_infos, method_name)

        out_shape = (N, M, axis_length)
        out_data = np.empty(out_shape, dtype=dtype)
        for i in range(N):
            for j in range(M):
                out_data[i, j, :] = values[i][j]

        meta_matrix = MetaDataMatrix(meta_array)

        if inplace_matrix:
            if self.shape != out_data.shape:
                self.resize(out_data.shape, refcheck=False)
            np.copyto(self.view(np.ndarray), out_data, casting="unsafe")
            self._value = self.view(np.ndarray)
            # Update meta if changed
            # (Inplace updates on meta are tricky if structure changed, but attributes like unit might change)
            # Minimal meta update in-place?
            # Rebuilding metadata matrix is complex in-place.
            return self
        
        # New Matrix
        # Reconstruct correct class
        new_mat = self.__class__(
             out_data,
             xindex=common_axis,
             xunit=common_axis.unit if isinstance(common_axis, u.Quantity) else None,
             # Pass generic args
        )
        new_mat._meta = meta_matrix
        return new_mat

    # Interoperability Delegation
    def to_neo_analogsignal(self, units=None):
        """Convert to neo.AnalogSignal."""
        from gwexpy.interop import to_neo_analogsignal
        return to_neo_analogsignal(self, units=units)
        
    @classmethod
    def from_neo_analogsignal(cls, sig):
        """Create from neo.AnalogSignal."""
        from gwexpy.interop import from_neo_analogsignal
        return from_neo_analogsignal(cls, sig)
        
    def to_mne_rawarray(self, info=None):
        """Convert to mne.io.RawArray."""
        from gwexpy.interop import to_mne_rawarray
        # Convert to dict first? or map?
        # to_mne_rawarray expects TimeSeriesDict or matrix logic.
        # But to_mne_rawarray in interop/mne_.py takes TimeSeriesDict.
        # We can implement a direct matrix path or convert to dict.
        # P2 spec: "from_mne_raw -> TimeSeriesDict".
        # Maybe to_mne is better on Dict. Matrix can use to_dict().
        # Implementing direct method for convenience.
        tsd = self.to_dict()
        return to_mne_rawarray(tsd, info=info) 




    def _coerce_other_timeseries_input(self, other, method_name):
        """
        Normalize 'other' input for bivariate spectral methods.
        """
        if isinstance(other, TimeSeriesMatrix):
            if other.shape[:2] != self.shape[:2]:
                raise ValueError(
                    f"shape mismatch: {self.shape[:2]} vs {other.shape[:2]}"
                )

            def _getter(i, j):
                return other[i, j]

            return _getter

        if isinstance(other, BaseTimeSeries):
            def _getter(i, j):
                return other

            return _getter

        raise TypeError(
            "other must be TimeSeriesMatrix or TimeSeries for bivariate spectral methods"
        )

    def _apply_bivariate_spectral_method(self, method_name, other, *args, **kwargs):
        """
        Apply a bivariate TimeSeries spectral method element-wise and return FrequencySeriesMatrix.
        """
        if not hasattr(self.series_class, method_name):
            raise NotImplementedError(
                f"Not implemented: TimeSeries has no method '{method_name}' in this GWpy version"
            )

        get_other = self._coerce_other_timeseries_input(other, method_name)

        N, M, _ = self.shape
        values = [[None for _ in range(M)] for _ in range(N)]
        meta_array = np.empty((N, M), dtype=object)
        freq_infos = []
        epochs = []
        dtype = None

        for i in range(N):
            for j in range(M):
                ts_a = self[i, j]
                ts_b = get_other(i, j)
                result = getattr(ts_a, method_name)(ts_b, *args, **kwargs)
                if not hasattr(result, "frequencies"):
                    raise TypeError(
                        f"{method_name} must return a FrequencySeries-like object"
                    )
                freq_info = _extract_freq_axis_info(result)
                freq_infos.append(freq_info)
                epochs.append(getattr(result, "epoch", None))

                data_arr = np.asarray(result.value)
                values[i][j] = data_arr
                name = getattr(result, "name", None) or getattr(ts_a, "name", None)
                channel = getattr(result, "channel", None)
                if channel is None or str(channel) == "":
                    channel = getattr(ts_a, "channel", None)
                meta_array[i, j] = MetaData(
                    unit=getattr(result, "unit", None),
                    name=name,
                    channel=channel,
                )
                dtype = (
                    data_arr.dtype
                    if dtype is None
                    else np.result_type(dtype, data_arr.dtype)
                )

        common_freqs, common_df, common_f0, n_freq = _validate_common_frequency_axis(
            freq_infos, method_name
        )
        common_epoch = _validate_common_epoch(epochs, method_name)

        out_data = np.empty((N, M, n_freq), dtype=dtype)
        for i in range(N):
            for j in range(M):
                out_data[i, j, :] = values[i][j]

        meta_matrix = MetaDataMatrix(meta_array)

        return FrequencySeriesMatrix(
            out_data,
            frequencies=common_freqs,
            meta=meta_matrix,
            rows=self.rows,
            cols=self.cols,
            name=getattr(self, "name", ""),
            epoch=common_epoch,
        )

    def _apply_univariate_spectral_method(self, method_name, *args, **kwargs):
        """
        Apply a univariate TimeSeries spectral method element-wise and return FrequencySeriesMatrix.
        """
        if not hasattr(self.series_class, method_name):
            raise NotImplementedError(
                f"Not implemented: TimeSeries has no method '{method_name}' in this GWpy version"
            )

        N, M, _ = self.shape
        values = [[None for _ in range(M)] for _ in range(N)]
        meta_array = np.empty((N, M), dtype=object)
        freq_infos = []
        epochs = []
        dtype = None

        for i in range(N):
            for j in range(M):
                ts = self[i, j]
                result = getattr(ts, method_name)(*args, **kwargs)
                if not hasattr(result, "frequencies"):
                    raise TypeError(
                        f"{method_name} must return a FrequencySeries-like object"
                    )
                freq_info = _extract_freq_axis_info(result)
                freq_infos.append(freq_info)
                epochs.append(getattr(result, "epoch", None))

                data_arr = np.asarray(result.value)
                values[i][j] = data_arr
                name = getattr(result, "name", None) or getattr(ts, "name", None)
                channel = getattr(result, "channel", None)
                if channel is None or str(channel) == "":
                    channel = getattr(ts, "channel", None)
                meta_array[i, j] = MetaData(
                    unit=getattr(result, "unit", None),
                    name=name,
                    channel=channel,
                )
                dtype = (
                    data_arr.dtype
                    if dtype is None
                    else np.result_type(dtype, data_arr.dtype)
                )

        common_freqs, common_df, common_f0, n_freq = _validate_common_frequency_axis(
            freq_infos, method_name
        )
        common_epoch = _validate_common_epoch(epochs, method_name)

        out_data = np.empty((N, M, n_freq), dtype=dtype)
        for i in range(N):
            for j in range(M):
                out_data[i, j, :] = values[i][j]

        meta_matrix = MetaDataMatrix(meta_array)

        return FrequencySeriesMatrix(
            out_data,
            frequencies=common_freqs,
            meta=meta_matrix,
            rows=self.rows,
            cols=self.cols,
            name=getattr(self, "name", ""),
            epoch=common_epoch,
        )

    def _run_spectral_method(self, method_name, **kwargs):
        """
        Helper for fft, psd, asd.
        """
        from gwexpy.frequencyseries.frequencyseries import FrequencySeriesMatrix

        N, M, K = self.shape

        # Run first element to determine frequency axis and output properties
        # Run first element to determine frequency axis and output properties
        # Use self[0, 0] to get the first TimeSeries element
        ts0 = self[0, 0]
        method = getattr(ts0, method_name)
        fs0 = method(**kwargs)

        # Prepare output array
        n_freq = len(fs0)
        out_shape = (N, M, n_freq)
        out_data = np.empty(out_shape, dtype=fs0.dtype)

        # Attributes
        out_units = np.empty((N, M), dtype=object)
        out_names = np.empty((N, M), dtype=object)
        out_channels = np.empty((N, M), dtype=object)

        # Loop over all elements
        for i in range(N):
            for j in range(M):
                if i == 0 and j == 0:
                    fs = fs0
                else:
                    # Use self[i, j] which now returns a proper TimeSeries
                    ts = self[i, j]
                    fs = getattr(ts, method_name)(**kwargs)

                out_data[i, j, :] = fs.value
                out_units[i, j] = fs.unit
                out_names[i, j] = fs.name
                out_channels[i, j] = fs.channel

        return FrequencySeriesMatrix(
            out_data,
            frequencies=fs0.frequencies,
            units=out_units,
            names=out_names,
            channels=out_channels,
            rows=self.rows,
            cols=self.cols,
            name=getattr(self, "name", ""),
            epoch=getattr(self, "epoch", None),
        )

    # --- Spectral Methods ---

    def fft(self, **kwargs):
        """
        Compute FFT of each element.
        Returns FrequencySeriesMatrix.
        """
        return self._run_spectral_method("fft", **kwargs)

    def psd(self, **kwargs):
        """
        Compute PSD of each element.
        Returns FrequencySeriesMatrix.
        """
        return self._run_spectral_method("psd", **kwargs)

    def asd(self, **kwargs):
        """
        Compute ASD of each element.
        Returns FrequencySeriesMatrix.
        """
        return self._run_spectral_method("asd", **kwargs)

    def _repr_string_(self):
        if self.size > 0:
            u = self.meta[0, 0].unit
        else:
            u = None
        return f"<TimeSeriesMatrix shape={self.shape}, dt={self.dt}, unit={u}>"


def _make_tsm_timeseries_wrapper(method_name):
    def _wrapper(self, *args, **kwargs):
        return self._apply_timeseries_method(method_name, *args, **kwargs)

    _wrapper.__name__ = method_name
    _wrapper.__qualname__ = f"{TimeSeriesMatrix.__name__}.{method_name}"
    _wrapper.__doc__ = f"Element-wise delegate to `TimeSeries.{method_name}`."
    return _wrapper


_TSM_TIME_DOMAIN_METHODS = [
    "detrend",
    "taper",
    "whiten",
    "filter",
    "lowpass",
    "highpass",
    "bandpass",
    "notch",
    "resample",
]

_TSM_MISSING_TIME_DOMAIN_METHODS = [
    m for m in _TSM_TIME_DOMAIN_METHODS if not hasattr(BaseTimeSeries, m)
]
# Not implemented: `TimeSeriesMatrix` does not define wrappers for methods that
# are missing from `gwpy.timeseries.TimeSeries` in the installed GWpy version.
for _m in _TSM_TIME_DOMAIN_METHODS:
    if _m in _TSM_MISSING_TIME_DOMAIN_METHODS:
        continue
    setattr(TimeSeriesMatrix, _m, _make_tsm_timeseries_wrapper(_m))


def _make_tsm_bivariate_wrapper(method_name):
    def _wrapper(self, other, *args, **kwargs):
        return self._apply_bivariate_spectral_method(method_name, other, *args, **kwargs)

    _wrapper.__name__ = method_name
    _wrapper.__qualname__ = f"{TimeSeriesMatrix.__name__}.{method_name}"
    _wrapper.__doc__ = f"Element-wise delegate to `TimeSeries.{method_name}` with another TimeSeries."
    return _wrapper


def _make_tsm_univariate_wrapper(method_name):
    def _wrapper(self, *args, **kwargs):
        return self._apply_univariate_spectral_method(method_name, *args, **kwargs)

    _wrapper.__name__ = method_name
    _wrapper.__qualname__ = f"{TimeSeriesMatrix.__name__}.{method_name}"
    _wrapper.__doc__ = f"Element-wise delegate to `TimeSeries.{method_name}`."
    return _wrapper


_TSM_BIVARIATE_METHODS = [
    "csd",
    "coherence",
    "transfer_function",
]

_TSM_UNIVARIATE_METHODS = [
    "auto_coherence",
]

for _m in _TSM_BIVARIATE_METHODS:
    if hasattr(BaseTimeSeries, _m):
        setattr(TimeSeriesMatrix, _m, _make_tsm_bivariate_wrapper(_m))

for _m in _TSM_UNIVARIATE_METHODS:
    if hasattr(BaseTimeSeries, _m):
        setattr(TimeSeriesMatrix, _m, _make_tsm_univariate_wrapper(_m))
