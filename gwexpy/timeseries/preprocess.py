
import numpy as np
import warnings
import scipy.linalg
from astropy import units as u

# Import low-level algorithms from signal.preprocessing
from gwexpy.signal.preprocessing import (
    WhiteningModel,
    StandardizationModel,
    whiten as _whiten_array,
    standardize as _standardize_array,
    impute as _impute_array,
)

try:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from .timeseries import TimeSeries
        from .matrix import TimeSeriesMatrix
except ImportError:
    pass


def align_timeseries_collection(
    series_list,
    *,
    how="intersection",
    fill_value=np.nan,
    method=None,
    tolerance=None,
):
    """
    Align a collection of TimeSeries to a common time axis.

    Parameters
    ----------
    series_list : list of TimeSeries
        Input series to align.
    how : str, optional
        "intersection" (default) or "union".
    fill_value : float, optional
        Value to fill missing data with when how="union". Default is np.nan.
    method : str, optional
        Interpolation method for resampling/alignment (e.g. 'linear', 'nearest', 'pad').
        Passed to TimeSeries.asfreq.
    tolerance : float, optional
        Tolerance for time comparison in seconds. Passed to TimeSeries.asfreq.

    Returns
    -------
    values : np.ndarray
        Shape (n_samples, n_channels).
    times : np.ndarray
        Common time axis (Time array).
    meta : dict
        Metadata including 'dt', 'epoch', 'channel_names', etc.
    """
    if not series_list:
        raise ValueError("No timeseries provided to align.")

    # 1. Determine common sample rate (minimum rate / maximum dt)
    dts = []
    has_time_dt = False
    for ts in series_list:
        if ts.dt is None:
            raise ValueError("align_timeseries_collection requires dt for all series")
        dt_q = ts.dt if isinstance(ts.dt, u.Quantity) else u.Quantity(ts.dt)
        dt_vals = np.asanyarray(dt_q.value)
        if np.any(dt_vals <= 0):
            raise ValueError("align_timeseries_collection requires dt > 0")
        dts.append(dt_q)
        if getattr(dt_q.unit, "physical_type", None) == "time":
            has_time_dt = True

    # 2. Determine time bounds
    # Check if we should operate in physical time (seconds)
    is_time_based = has_time_dt
    if not is_time_based:
        # If any series has time unit, force time-based
        for ts in series_list:
            if ts.times.unit is not None and ts.times.unit.physical_type == 'time':
                is_time_based = True
                break

    if is_time_based:
        common_time_unit = u.s
        dt_candidates = []
        for dt_q in dts:
            if dt_q.unit is None or dt_q.unit == u.dimensionless_unscaled:
                dt_candidates.append(u.Quantity(dt_q.value, u.s))
            elif dt_q.unit.physical_type == "dimensionless":
                dt_candidates.append(u.Quantity(dt_q.value, u.s))
            elif dt_q.unit.physical_type == "time":
                dt_candidates.append(dt_q.to(u.s))
            else:
                raise ValueError(
                    f"align_timeseries_collection requires time-like dt when time-based alignment is used: {dt_q}"
                )
        target_dt = max(dt_candidates)
    else:
        # Fallback to first unit (or dimensionless)
        ref_u = series_list[0].times.unit
        common_time_unit = ref_u if ref_u is not None else u.dimensionless_unscaled
        target_dt = max(dts)
        try:
            target_dt.to(common_time_unit)
        except u.UnitConversionError as exc:
            raise ValueError(
                f"Incompatible dt unit in collection: {target_dt.unit} vs {common_time_unit}"
            ) from exc
    
    # Helper to get start/end in common unit
    def get_span_val(ts):
        ts_u = ts.times.unit if ts.times.unit is not None else u.dimensionless_unscaled
        
        # t0 conversion
        if is_time_based and ts_u == u.dimensionless_unscaled:
             # Treat dimensionless as seconds
             t0 = ts.t0.value
        elif ts_u != common_time_unit:
             try:
                 t0 = ts.t0.to(common_time_unit).value
             except u.UnitConversionError:
                 raise ValueError(f"Incompatible time units in collection: {ts_u} vs {common_time_unit}")
        else:
             t0 = ts.t0.value
             
        # End conversion
        end_q = ts.span[1]
        
        # If end_q is dimensionless quantity and we are in time mode, treat as seconds value
        if is_time_based and (not hasattr(end_q, "unit") or end_q.unit == u.dimensionless_unscaled or end_q.unit is None):
              end = end_q.value if hasattr(end_q, "value") else end_q
        elif hasattr(end_q, "to"):
             try:
                 end = end_q.to(common_time_unit).value
             except u.UnitConversionError:
                 # Backup: if unit mismatch but one is None? already checked.
                 raise ValueError(f"Incompatible span unit: {end_q.unit} vs {common_time_unit}")    
        else:
             # float or similar
             end = end_q
             
        return t0, end

    starts = []
    ends = []
    for ts in series_list:
        s, e = get_span_val(ts)
        starts.append(s)
        ends.append(e)
    
    def float_min(x): return min(x)
    def float_max(x): return max(x)

    if how == "intersection":
        common_t0 = float_max(starts)
        common_end = float_min(ends)
        # Numerical tolerance for "empty" intersection
        if common_end < common_t0 and not np.isclose(common_end, common_t0):
             raise ValueError(f"No overlap found. common_t0={common_t0}, common_end={common_end}")
    elif how == "union":
        common_t0 = float_min(starts)
        common_end = float_max(ends)
    else:
        raise ValueError(f"Unknown alignment how='{how}'.")

    # 3. Create common time axis
    # Use common_time_unit for output
    out_unit = common_time_unit
    
    duration = common_end - common_t0
    target_dt_s = target_dt.to(common_time_unit).value
    
    if duration <= 0:
         n_samples = 0
    else:
         # Use ceil to ensure we cover the full range, matching asfreq behavior
         n_samples = int(np.ceil(duration / target_dt_s))
    
    # Create common times in Seconds
    common_times_s = common_t0 + np.arange(n_samples) * target_dt_s
    # Convert to output unit
    common_times = (common_times_s * common_time_unit).to(out_unit)
    
    if n_samples <= 0 and how == "intersection":
         # Fallback for empty
         pass

    # 4. Fill matrix
    n_channels = len(series_list)
    # Determine output dtype (promote if needed)
    # For now assume float or complex
    is_complex = any(ts.dtype.kind == 'c' for ts in series_list)
    dtype = np.complex128 if is_complex else np.float64
    
    values = np.full((n_samples, n_channels), fill_value, dtype=dtype)
    
    for i, ts in enumerate(series_list):
        # Align using asfreq
        # We specify the target grid by using origin=common_t0.
        # This ensures grid points match common_times exactly phase-wise.
        
        # Origin must be compatible with ts.times unit or convertible
        # If is_time_based=True, asfreq will coerce dimensionless ts to seconds.
        # So we should pass origin in seconds (common_time_unit).
        
        if is_time_based:
             # Regardless of ts.unit, we pass origin as Quantity(common_time_unit)
             origin_val = u.Quantity(common_t0, common_time_unit)
        elif ts.times.unit is None:
             origin_val = u.Quantity(common_t0, u.dimensionless_unscaled)
        else:
             origin_val = u.Quantity(common_t0, common_time_unit).to(ts.times.unit)
        
        # We process the whole series onto the grid defined by common_t0
        # asfreq returns the coverage of the original series but on the new grid.
        
        ts_aligned = ts.asfreq(
            target_dt,
            method=method,
            fill_value=fill_value,
            origin=origin_val,
            tolerance=tolerance,
            align='floor'
        )
        
        # Calculate offset of ts_aligned.t0 relative to common_t0
        # Both are on the grid defined by common_t0 and target_dt.
        t0_aligned_s = ts_aligned.t0.to(common_time_unit).value
        
        # Index offset
        # Since we aligned to the grid, the difference should be integer multiple of dt
        # We use floor(x + 0.5) to snap to nearest integer safely.
        offset = int(np.floor((t0_aligned_s - common_t0) / target_dt_s + 0.5))
        
        # Copy valid overlap into values buffer
        # Buffer range: [0, n_samples)
        # TS range: [offset, offset + len)
        
        # Overlap in buffer coordinates
        buf_start = max(0, offset)
        buf_end = min(n_samples, offset + len(ts_aligned))
        
        # Overlap in TS coordinates
        ts_start = max(0, -offset)
        ts_end = ts_start + (buf_end - buf_start)
        
        if buf_end > buf_start:
             values[buf_start:buf_end, i] = ts_aligned.value[ts_start:ts_end]
             
    meta = {
        "t0": u.Quantity(common_t0, common_time_unit).to(out_unit),
        "dt": target_dt,
        "channel_names": [ts.name for ts in series_list],
    }
    
    return values, common_times, meta


def impute_timeseries(ts, *, method="interpolate", limit=None, axis="time", max_gap=None, **kwargs):
    """
    Impute missing values in a TimeSeries.

    Parameters
    ----------
    ts : TimeSeries
        Series to impute.
    method : str
        'interpolate', 'ffill', 'bfill', 'mean', 'median'.
    limit : int
        Pandas-style limit (max consecutive NaNs to fill).
    axis : str
        Axis to operate on (default 'time').
    max_gap : float or Quantity, optional
        Maximum gap duration to simple fill. If a gap is larger than this, it is left as NaN.
        If float, assumes same unit as ts.times (usually seconds if dimensionless).
    
    Returns
    -------
    TimeSeries
        Imputed series.
    """
    val = ts.value.copy()
    nans = np.isnan(val)
    if not np.any(nans):
        return ts.copy()

    # Determine timing info if max_gap is requested
    has_gap_constraint = max_gap is not None
    gap_threshold = None
    times_val = None
    
    if has_gap_constraint:
         # Try to get times
         try:
             # Use times value directly (float, usually seconds or GPS)
             times_val = ts.times.value
             time_unit = ts.times.unit
             
             if hasattr(max_gap, 'to'): # Quantity
                  if time_unit:
                       gap_threshold = max_gap.to(time_unit).value
                  else:
                       # TimeSeries is dimensionless or unknown unit?
                       # Assume seconds if max_gap has time unit
                       if max_gap.unit.physical_type == 'time':
                           gap_threshold = max_gap.to(u.s).value
                           # We assume times_val is seconds if unit is None/dimensionless 
                           # (GWpy convention usually)
                       else:
                           gap_threshold = max_gap.value
             else:
                  gap_threshold = float(max_gap)
         except (AttributeError, TypeError, ValueError):
             # If no times mechanism, fallback or warn?
             # For TimeSeries, .times should exist.
             # If regular, dt * samples.
             if hasattr(ts, 'dt'):
                 dt = ts.dt
                 if hasattr(dt, 'value'): dt = dt.value
                 # Use index-based check if we can't get times array easily?
                 # Actually if we have dt, gap size in indices = max_gap / dt
                 pass
             times_val = None

    if method == "interpolate":
        valid = ~nans
        # Use time-based interpolation if possible, or index
        if times_val is not None:
             x = times_val
        else:
             x = np.arange(len(val))
             
        # Interpolate
        # Note: np.interp works on complex by treating as float? No.
        # If max_gap is set, we strictly avoid edge extrapolation (per requirements).
        # Otherwise preserve existing behavior (default extrapolation).
        left_val = np.nan if has_gap_constraint else None
        right_val = np.nan if has_gap_constraint else None

        if np.iscomplexobj(val):
             real_part = np.interp(x[nans], x[valid], val[valid].real, left=left_val, right=right_val)
             imag_part = np.interp(x[nans], x[valid], val[valid].imag, left=left_val, right=right_val)
             val[nans] = real_part + 1j * imag_part
        else:
             val[nans] = np.interp(x[nans], x[valid], val[valid], left=left_val, right=right_val)
             
        # Apply max_gap constraint
        if has_gap_constraint and times_val is not None:
             # Identify gaps in original VALID data
             # A "gap" is an interval between two valid points where NaNs existed.
             # If distance(valid[i+1], valid[i]) > threshold, then points in between should be NaN.
             
             valid_indices = np.where(valid)[0]
             if len(valid_indices) > 0:
                  valid_times = x[valid_indices]
                  diffs = np.diff(valid_times)
                  
                  # Find intervals larger than threshold
                  big_gaps = np.where(diffs > gap_threshold - 1e-12)[0] # Tolerance
                  
                  for idx in big_gaps:
                       t_start = valid_times[idx]
                       t_end = valid_times[idx+1]
                       
                       # Mask points in this interval (exclusive of boundaries)
                       # Logic: boundaries are valid, everything strictly between was interpolated
                       # so we revert it to NaN.
                       mask = (x > t_start) & (x < t_end)
                       val[mask] = np.nan
                       
    elif method == "ffill":
        from pandas import Series
        s = Series(val)
        # Pandas limit is by COUNT, not time gap.
        # So we use standard limit if provided.
        # If max_gap is provided, we post-filter?
        # ffill fills forward.
        # If we have max_gap, we shouldn't fill if time_curr - time_last_valid > max_gap.
        # Harder to map to pandas directly.
        # Manual implementation might be needed for max_gap + ffill?
        # P0 spec says: "max_gap responsibility is limited to selecting target intervals for interpolation"
        # Since interpolation fills the gap, if gap > max_gap, we just revert.
        
        val = s.ffill(limit=limit).values
        
        if has_gap_constraint and times_val is not None:
             # Check validity
             # For ffill: for each point, time - time_of_last_valid <= max_gap
             # But we can just use the generic logic:
             # Find original gaps. If gap > max_gap, revert ALL points in that gap to NaN?
             # Yes, "intervals longer than max_gap are not interpolated (remain as NaN)"
             # This applies regardless of method.
             
             valid_indices = np.where(~nans)[0]
             if len(valid_indices) > 0:
                  valid_times = times_val[valid_indices]
                  diffs = np.diff(valid_times)
                  big_gaps = np.where(diffs > gap_threshold - 1e-12)[0]
                  
                  for idx in big_gaps:
                       t_start = valid_times[idx]
                       t_end = valid_times[idx+1]
                       mask = (times_val > t_start) & (times_val < t_end)
                       # Only revert points that were originally NaN (to be safe? val was replaced)
                       # Yes, revert to NaN.
                       val[mask] = np.nan
                       
    elif method == "bfill":
        from pandas import Series
        s = Series(val)
        val = s.bfill(limit=limit).values
        
        # Apply max_gap generic rollback
        if has_gap_constraint and times_val is not None:
             valid_indices = np.where(~nans)[0]
             if len(valid_indices) > 0:
                  valid_times = times_val[valid_indices]
                  diffs = np.diff(valid_times)
                  big_gaps = np.where(diffs > gap_threshold - 1e-12)[0]
                  for idx in big_gaps:
                       t_start = valid_times[idx]
                       t_end = valid_times[idx+1]
                       mask = (times_val > t_start) & (times_val < t_end)
                       val[mask] = np.nan
    elif method == "mean":
        val[nans] = np.nanmean(val)
        # Mean fills everything.
        # Max gap? 
        # "intervals longer than max_gap are not interpolated"
        # Does 'mean' imputation imply bridging gaps? Usually it's global fill.
        # But if the requirement stands, we should not fill in large gaps even with global mean?
        # That seems consistent.
        if has_gap_constraint and times_val is not None:
             valid_indices = np.where(~nans)[0]
             if len(valid_indices) > 0:
                  valid_times = times_val[valid_indices]
                  diffs = np.diff(valid_times)
                  big_gaps = np.where(diffs > gap_threshold - 1e-12)[0]
                  for idx in big_gaps:
                       t_start = valid_times[idx]
                       t_end = valid_times[idx+1]
                       mask = (times_val > t_start) & (times_val < t_end)
                       val[mask] = np.nan
                       
    elif method == "median":
        val[nans] = np.nanmedian(val)
        if has_gap_constraint and times_val is not None:
             valid_indices = np.where(~nans)[0]
             if len(valid_indices) > 0:
                  valid_times = times_val[valid_indices]
                  diffs = np.diff(valid_times)
                  big_gaps = np.where(diffs > gap_threshold - 1e-12)[0]
                  for idx in big_gaps:
                       t_start = valid_times[idx]
                       t_end = valid_times[idx+1]
                       mask = (times_val > t_start) & (times_val < t_end)
                       val[mask] = np.nan
    else:
        raise ValueError(f"Unknown impute method '{method}'")

    new_ts = ts.copy()
    new_ts.value[:] = val
    return new_ts

def standardize_timeseries(ts, *, method="zscore", ddof=0, robust=False):
    """
    Standardize a TimeSeries.
    """
    val = ts.value
    if robust or method == "robust":
        med = np.nanmedian(val)
        mad = np.nanmedian(np.abs(val - med))
        scale = 1.4826 * mad
        if scale == 0:
            warnings.warn("MAD is zero, setting scale to 1.0 to avoid division by zero.")
            scale = 1.0
        method_used = "robust"
    elif method == "zscore":
        med = np.nanmean(val)
        scale = np.nanstd(val, ddof=ddof)
        if scale == 0:
            warnings.warn("Standard deviation is zero, setting scale to 1.0 to avoid division by zero.")
            scale = 1.0
        method_used = "zscore"
    else:
        raise ValueError(f"Unknown standardization method '{method}'. "
                         f"Supported methods are 'zscore', 'robust'.")

    # Handle dtype. If input is integer, standardization (float) will be trucated if copied directly.
    # Check if value dtype is integer-like
    if np.issubdtype(ts.value.dtype, np.integer):
         # Create a new TimeSeries with float data
         # We can't change dtype of existing one in-place easily if we want to return new object.
         # TimeSeries(data, ...) constructor
         val_float = ts.value.astype("float64")
         # We construct new_ts manually to preserve metadata
         new_ts = ts.__class__(
             val_float, 
             t0=ts.t0, 
             dt=ts.dt, 
             name=ts.name,
             unit=ts.unit, # Inherit unit or make dimensionless? Standardization makes it dimensionless usually.
             channel=getattr(ts, 'channel', None)
         )
         # If standardized, unit becomes dimensionless technically (sigma units). 
         # Unless we keep original unit? 
         # Gwpy might complain if unit mismatch?
         # Standard score is dimensionless.
         new_ts.unit = u.dimensionless_unscaled
    else:
         new_ts = ts.copy()
         # Should we force dimensionless unit?
         if hasattr(new_ts, 'override_unit'):
             new_ts.override_unit(u.dimensionless_unscaled)
         else:
             try:
                 new_ts.unit = u.dimensionless_unscaled
             except AttributeError:
                 pass
         
    new_ts.value[:] = (val - med) / scale
    
    model = StandardizationModel(mean=np.array([med]), scale=np.array([scale]), axis="time")
    return new_ts, model

def standardize_matrix(matrix, *, axis="time", method="zscore", ddof=0):
    """
    Standardize a TimeSeriesMatrix.
    """
    val = matrix.value.copy()
    # TimeSeriesMatrix layout: (channels, time) usually, or (rows, cols, time).
    # Time is always the last axis (-1).
    # "time" axis -> operating along time axis (normalize each channel).
    # "channel" axis -> operating along channel axis (normalize each sample).
    
    np_axis = -1 if axis == "time" else (0, 1) # Standardize across all dimensions
    
    if method == "robust":
        med = np.nanmedian(val, axis=np_axis, keepdims=True)
        mad = np.nanmedian(np.abs(val - med), axis=np_axis, keepdims=True)
        scale = 1.4826 * mad
    else:
        med = np.nanmean(val, axis=np_axis, keepdims=True)
        scale = np.nanstd(val, axis=np_axis, ddof=ddof, keepdims=True)
        
    scale[scale == 0] = 1.0
    
    # We return a new matrix with same metadata.
    if np.issubdtype(matrix.value.dtype, np.integer):
         val_float = matrix.value.astype("float64")
         # Reconstruct using new numpy array
         new_mat = matrix.__class__(
             val_float,
             t0=matrix.t0,
             dt=matrix.dt
         )
         # Copy other meta?
         if hasattr(new_mat, 'channel_names'):
              new_mat.channel_names = getattr(matrix, 'channel_names', None)
    else:
         new_mat = matrix.copy()
         
    # Update values
    new_mat.value[:] = (val - med) / scale
    
    # Standardize result is dimensionless?
    # Usually yes. But SeriesMatrix doesn't handle units uniformly yet,
    # so we simply return the standardized matrix.
    return new_mat
    
def whiten_matrix(matrix, *, method="pca", eps=1e-12, n_components=None):
    """
    Whiten a TimeSeriesMatrix.
    """
    # Matrix value is 3D (rows, cols, time) usually.
    # Reshape to (features, time) -> (time, features)
    X_features = matrix.value.reshape(-1, matrix.shape[-1])
    X_T = X_features.T # (time, features)
    
    mean = np.mean(X_T, axis=0)
    X_centered = X_T - mean
    
    cov = np.cov(X_centered, rowvar=False)
    # ... SVD logic ...
    U, S, Vt = np.linalg.svd(cov)
    
    S_inv_sqrt = np.diag(1.0 / np.sqrt(S + eps))
    
    if method == "pca":
        W = S_inv_sqrt @ U.T
    elif method == "zca":
        W = U @ S_inv_sqrt @ U.T
    else:
        raise ValueError("method must be 'pca' or 'zca'")
        
    if n_components is not None:
        if method == "zca":
             warnings.warn("n_components ignores channel mapping for ZCA if reduced.")
        W = W[:n_components, :]
        
    X_whitened = X_centered @ W.T
    
    # result X_whitened is (time, components)
    # create new matrix with (components, 1, time)
    
    cls = matrix.__class__
    
    try:
        if X_whitened.shape[1] != X_features.shape[0]: # Check against original features count
            # Dimension reduced
            new_data = X_whitened.T[:, None, :] # (components, 1, time)
            
            new_mat = cls(
                new_data,
                t0=matrix.t0,
                dt=matrix.dt,
            )
        else:
             # Dimension preserved - reshape back to original shape if possible
             # But whitening mixes features, so (rows, cols) structure is lost/scrambled?
             # PCA/ZCA usually destroys spatial structure unless ZCA implicitly preserves it.
             # If ZCA: map back to (rows, cols)? 
             # For safety, we flatten to (features, 1, time) or reuse shape if ZCA.
             if method == "zca":
                 new_val = X_whitened.T.reshape(matrix.shape)
                 new_mat = matrix.copy()
                 new_mat.value[:] = new_val
             else:
                 new_mat = matrix.copy()
                 # If shapes match features
                 # Flatten usage
                 pass
                 # Actually, better return flattened structure if whitening mixed them.
                 new_data = X_whitened.T[:, None, :]
                 new_mat = cls(new_data, t0=matrix.t0, dt=matrix.dt)
                 
    except (IndexError, ValueError):
         new_data = X_whitened.T[:, None, :]
         new_mat = cls(new_data, t0=matrix.t0, dt=matrix.dt)
        
    model = WhiteningModel(mean, W)
    return new_mat, model
