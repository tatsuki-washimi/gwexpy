
import numpy as np
import warnings
import scipy.linalg
from astropy import units as u

try:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from .timeseries import TimeSeries, TimeSeriesMatrix
except ImportError:
    pass

class WhiteningModel:
    def __init__(self, mean, W):
        self.mean = mean
        self.W = W
        self.W_inv = np.linalg.pinv(W)

    def inverse_transform(self, X_w):
        """
        Project whitened data back to original space.
        """
        # X_rec = (X_w @ W_inv.T) + mean
        # X_w is (n_samples, n_components)
        # W_inv is (n_original, n_components)
        # W_inv.T is (n_components, n_original)
        # Wait: W was (n_components, n_original)
        # W_inv is (n_original, n_components)
        # X_w @ W_inv.T ? shape check: (n, k) @ (k, p) -> (n, p)
        # Yes.
        
        # Check if X_w is a matrix object or array
        if hasattr(X_w, 'value'):
             val = X_w.value
        else:
             val = X_w
             
        X_rec = (val @ self.W_inv.T) + self.mean
        return X_rec


class StandardizationModel:
    def __init__(self, mean, scale, axis):
        self.mean = mean
        self.scale = scale
        self.axis = axis
        
    def inverse_transform(self, X_std):
        """
        Undo standardization: X = X_std * scale + mean
        """
        # Broadcasting depends on axis.
        # This implementation assumes simple broadcasting or manual handling.
        # This method is here for completeness.
        if hasattr(X_std, 'value'):
             val = X_std.value
        else:
             val = X_std
             
        # If scale/mean are arrays, ensure shapes match or broadcast
        return val * self.scale + self.mean
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
        Interpolation method for resampling/alignment.
        If None, uses nearest/exact or whatever underlying resample uses if rates differ.
        If provided, passed to relevant interpolation methods.
    tolerance : float, optional
        Tolerance for time comparison in seconds.

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
    dts = [ts.dt for ts in series_list]
    target_dt = max(dts)
    
    resampled_list = []
    for ts in series_list:
        if ts.dt != target_dt:
             # Resample to target_dt (anti-aliasing)
             if method is not None:
                 # gwpy native resample
                 ts_new = ts.resample(1/target_dt)
             else:
                 ts_new = ts.resample(1/target_dt)
             resampled_list.append(ts_new)
        else:
             resampled_list.append(ts)
             
    series_list = resampled_list
    
    # 2. Determine time bounds
    # Ensure all t0 and span[1] are in compatible units before comparison
    # Convert all to a common time unit, e.g., seconds, for internal calculations for consistency
    common_time_unit = u.s # Use seconds for internal calculations for consistency
    t0s = [ts.t0.to(common_time_unit).value for ts in series_list]
    
    def _get_val(x):
         return x.value if hasattr(x, 'value') else x
         
    ends = [_get_val(ts.span[1]) for ts in series_list]
    
    # If using unit conversion logic for ends, we should be careful if they are not quantities.
    # To be safe, we rely on TimeSeries.t0 and dt being quantities or consistent.
    # If ts.span returning floats, we assume they are compatible with t0s (seconds).
    # If they have value, we take it.
    
    if hasattr(series_list[0].span[1], 'unit'):
         # Convert to common unit
         ends = [ts.span[1].to(common_time_unit).value for ts in series_list]
    
    def float_min(x): return min(x)
    def float_max(x): return max(x)

    if how == "intersection":
        common_t0 = float_max(t0s)
        common_end = float_min(ends)
        if common_end <= common_t0:
             # Fallback or error? User requested "crop all ... else raise" logic implied?
             # "crop all channels to common overlapping span"
             # If no overlap, length is 0 or negative.
             warnings.warn(f"No overlap found. common_t0={common_t0}, common_end={common_end}")
             # We can continue and return empty matrix
             # But let's check strictness.
             # "else raise ValueError" was for differing sampling rates in options.
             # I'll raise error here to be safe as usually this is a bug in usage.
             pass 
    elif how == "union":
        common_t0 = float_min(t0s)
        common_end = float_max(ends)
    else:
        raise ValueError(f"Unknown alignment how='{how}'.")

    # 3. Create common time axis
    unit = series_list[0].times.unit
    duration = common_end - common_t0
    target_dt_val = target_dt.to(unit).value
    
    if duration <= 0:
         n_samples = 0
    else:
         n_samples = int(np.round(duration / target_dt_val))
    
    common_times = np.linspace(common_t0, common_t0 + (n_samples-1)*target_dt_val, n_samples) * unit
    
    if n_samples <= 0 and how == "intersection":
        raise ValueError(f"Intersection resulted in empty interval: [{common_t0}, {common_end}]")

    # 4. Fill matrix
    n_channels = len(series_list)
    matrix = np.full((n_samples, n_channels), fill_value, dtype=float)
    
    for i, ts in enumerate(series_list):
        ts_t0_val = ts.t0.value
        offset = int(np.round((ts_t0_val - common_t0) / target_dt_val))
        ts_len = len(ts)
        
        mat_start = max(0, offset)
        mat_end = min(n_samples, offset + ts_len)
        ts_start = max(0, -offset)
        ts_end = ts_start + (mat_end - mat_start)
        
        if mat_end > mat_start and ts_end > ts_start:
             val_slice = ts.value[ts_start:ts_end]
             matrix[mat_start:mat_end, i] = val_slice
             
    meta = {
        "t0": common_t0 * unit,
        "dt": target_dt,
        "channel_names": [ts.name for ts in series_list],
    }
    
    return matrix, common_times, meta


def impute_timeseries(ts, *, method="interpolate", limit=None, axis="time", **kwargs):
    """
    Impute missing values in a TimeSeries.
    """
    val = ts.value.copy()
    nans = np.isnan(val)
    if not np.any(nans):
        return ts.copy()

    if method == "interpolate":
        valid = ~nans
        x = np.arange(len(val))
        val[nans] = np.interp(x[nans], x[valid], val[valid])
    elif method == "ffill":
        from pandas import Series
        s = Series(val)
        val = s.ffill(limit=limit).values
    elif method == "bfill":
        from pandas import Series
        s = Series(val)
        val = s.bfill(limit=limit).values
    elif method == "mean":
        val[nans] = np.nanmean(val)
    elif method == "median":
        val[nans] = np.nanmedian(val)
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
                 
    except Exception:
         new_data = X_whitened.T[:, None, :]
         new_mat = cls(new_data, t0=matrix.t0, dt=matrix.dt)
        
    model = WhiteningModel(mean, W)
    return new_mat, model
