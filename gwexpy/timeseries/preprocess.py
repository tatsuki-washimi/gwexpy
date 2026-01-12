
import numpy as np
import warnings
from astropy import units as u

# Import low-level algorithms from signal.preprocessing
from gwexpy.signal.preprocessing import (
    WhiteningModel,
    StandardizationModel,
)
from .utils import _coerce_t0_gps

try:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from .timeseries import TimeSeries  # noqa: F401
        from .matrix import TimeSeriesMatrix  # noqa: F401
except ImportError:
    pass


def _limit_mask(nans, limit, *, direction="forward"):
    if limit is None:
        return np.zeros_like(nans, dtype=bool)
    limit = int(limit)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if limit == 0:
        return nans.copy()
    mask = np.zeros_like(nans, dtype=bool)
    run_start = None
    n = len(nans)
    for i in range(n):
        if nans[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                run_len = i - run_start
                if run_len > limit:
                    if direction == "backward":
                        mask[run_start:i - limit] = True
                    else:
                        mask[run_start + limit:i] = True
                run_start = None
    if run_start is not None:
        run_len = n - run_start
        if run_len > limit:
            if direction == "backward":
                mask[run_start:n - limit] = True
            else:
                mask[run_start + limit:n] = True
    return mask


def _ffill_numpy(val, limit=None):
    out = val.copy()
    have_last = False
    last_val = None
    run = 0
    for i in range(len(out)):
        if np.isnan(out[i]):
            if not have_last:
                continue
            if limit is None or run < limit:
                out[i] = last_val
            run += 1
        else:
            last_val = out[i]
            have_last = True
            run = 0
    return out


def _bfill_numpy(val, limit=None):
    out = val.copy()
    have_next = False
    next_val = None
    run = 0
    for i in range(len(out) - 1, -1, -1):
        if np.isnan(out[i]):
            if not have_next:
                continue
            if limit is None or run < limit:
                out[i] = next_val
            run += 1
        else:
            next_val = out[i]
            have_next = True
            run = 0
    return out


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

    All time spans are treated as semi-open intervals [start, end) (end exclusive).
    The common time grid uses ``n_samples = ceil((end - start) / dt)``, so samples
    are generated at ``start + k * dt`` for ``k = 0..n_samples-1`` and never
    include the end point, consistent with [start, end).

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

    Notes
    -----
    - ``how="intersection"`` computes the semi-open intersection of spans and
      raises ``ValueError`` if the intersection is empty.
    - ``how="union"`` spans the semi-open union of all series and uses
      ``fill_value`` outside each series' coverage.
    """
    if not series_list:
        raise ValueError("No timeseries provided to align.")

    # 1. Determine common sample rate (minimum rate / maximum dt)
    dts = []
    has_time_dt = False
    for ts in series_list:
        # Check regularity safely
        from .timeseries import TimeSeries
        if isinstance(ts, TimeSeries):
             is_regular = ts.is_regular
        else:
             # Fallback for BaseTimeSeries or other objects
             is_regular = getattr(ts, 'regular', True)

        if not is_regular:
             # For irregular series, we can't use .dt safely.
             # We estimate an average dt for alignment purposes.
             times_val = np.asarray(ts.times)
             if len(times_val) > 1:
                  avg_dt = (times_val[-1] - times_val[0]) / (len(times_val) - 1)
                  dt_q = u.Quantity(avg_dt, ts.times.unit or u.s)
             else:
                  # fallback if 1 point
                  dt_q = u.Quantity(1.0, ts.times.unit or u.s)
        else:
             if ts.dt is None:
                  raise ValueError("align_timeseries_collection requires dt for all series")
             # Ensure dt is a Quantity
             dt_q = ts.dt if isinstance(ts.dt, u.Quantity) else u.Quantity(ts.dt, u.dimensionless_unscaled)

        dt_vals = np.asanyarray(dt_q.value)
        if np.any(dt_vals <= 0):
            raise ValueError("align_timeseries_collection requires dt > 0")
        dts.append(dt_q)

        # Check physical type safely
        dt_unit = getattr(dt_q, "unit", None)
        if dt_unit is not None and getattr(dt_unit, "physical_type", None) == "time":
            has_time_dt = True

    # 2. Determine time bounds and common unit
    # Check if we should operate in physical time (seconds)
    is_time_based = has_time_dt
    if not is_time_based:
        # If any series has time unit in its axis, force time-based
        for ts in series_list:
            t_unit = getattr(ts.times, "unit", None)
            if t_unit is not None and getattr(t_unit, "physical_type", None) == 'time':
                is_time_based = True
                break
    if not is_time_based:
        # Treat dimensionless axes as GPS seconds by default
        all_dimless = True
        for dt_q in dts:
            unit = getattr(dt_q, "unit", None)
            phys = getattr(unit, "physical_type", None) if unit is not None else None
            if unit is not None and unit != u.dimensionless_unscaled and phys != "dimensionless":
                all_dimless = False
                break
        if all_dimless:
            for ts in series_list:
                t_unit = getattr(ts.times, "unit", None)
                phys = getattr(t_unit, "physical_type", None) if t_unit is not None else None
                if t_unit is not None and t_unit != u.dimensionless_unscaled and phys != "dimensionless":
                    all_dimless = False
                    break
        if all_dimless:
            is_time_based = True

    if is_time_based:
        # Determine base time unit. Prefer seconds but keep original if all are same.
        time_units = set()
        for dt_q in dts:
            if getattr(dt_q.unit, "physical_type", None) == "time":
                time_units.add(dt_q.unit)
        for ts in series_list:
            t_unit = getattr(ts.times, "unit", None)
            if t_unit is not None and getattr(t_unit, "physical_type", None) == 'time':
                time_units.add(t_unit)

        common_time_unit = u.s
        if time_units and (len(time_units) > 1 or list(time_units)[0] != u.s):
            warnings.warn(
                f"Converting time units {time_units} to GPS seconds for alignment."
            )

        dt_candidates = []
        for dt_q in dts:
            # Safer access to physical_type
            phys = getattr(dt_q.unit, "physical_type", None)
            if dt_q.unit == u.dimensionless_unscaled or phys == "dimensionless":
                # Interpret dimensionless as common_time_unit
                dt_candidates.append(u.Quantity(dt_q.value, common_time_unit))
            elif phys == "time":
                dt_candidates.append(dt_q.to(common_time_unit))
            else:
                raise ValueError(
                    f"align_timeseries_collection requires time-like dt when time-based alignment is used: {dt_q}"
                )
        target_dt = max(dt_candidates)
    else:
        # Fallback to first unit found in dts or times
        common_time_unit = u.dimensionless_unscaled
        for dt_q in dts:
            if dt_q.unit != u.dimensionless_unscaled:
                common_time_unit = dt_q.unit
                break
        if common_time_unit == u.dimensionless_unscaled:
            for ts in series_list:
                t_unit = getattr(ts.times, "unit", None)
                if t_unit is not None and t_unit != u.dimensionless_unscaled:
                    common_time_unit = t_unit
                    break

        # Convert all to common_time_unit, treating dimensionless specifically
        dt_candidates = []
        for dt_q in dts:
            if dt_q.unit == u.dimensionless_unscaled:
                 dt_candidates.append(u.Quantity(dt_q.value, common_time_unit))
            else:
                 try:
                     dt_candidates.append(dt_q.to(common_time_unit))
                 except u.UnitConversionError as exc:
                     raise ValueError(
                         f"Incompatible dt unit in collection: {dt_q.unit} vs {common_time_unit}"
                     ) from exc
        target_dt = max(dt_candidates)

    # Helper to get start/end in common unit
    def get_span_val(ts):
        ts_u = ts.times.unit if ts.times.unit is not None else u.dimensionless_unscaled

        # t0 conversion
        if is_time_based:
             t0_q = _coerce_t0_gps(ts.t0)
             if hasattr(t0_q, "to"):
                 t0 = t0_q.to(common_time_unit).value
             else:
                 t0 = float(t0_q)
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
              dt_q = getattr(ts, "dt", None)
              if dt_q is not None:
                   dt_unit = getattr(dt_q, "unit", None)
                   phys = getattr(dt_unit, "physical_type", None) if dt_unit is not None else None
                   if dt_unit is None or dt_unit == u.dimensionless_unscaled or phys == "dimensionless":
                        dt_base = dt_q.value if hasattr(dt_q, "value") else dt_q
                        dt_val = u.Quantity(dt_base, u.s).to(common_time_unit).value
                   else:
                        dt_val = u.Quantity(dt_q).to(common_time_unit).value
                   end = t0 + (len(ts) * dt_val)
              else:
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
        # Semi-open intersection is empty when end <= start.
        if common_end <= common_t0:
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

    if fill_value is None:
        fill_value = np.nan

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
        if hasattr(ts_aligned.t0, "to"):
             t0_aligned_s = ts_aligned.t0.to(common_time_unit).value
        else:
             t0_aligned_s = float(ts_aligned.t0)

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


def _impute_1d(val_1d, x, method, has_gap_constraint, gap_threshold, limit=None):
    """Internal 1D imputation core."""

    nans_1d = np.isnan(val_1d)
    if not np.any(nans_1d):
        return val_1d

    valid_1d = ~nans_1d
    if not np.any(valid_1d):
        return val_1d # All NaNs

    x_valid = x[valid_1d]
    y_valid = val_1d[valid_1d]
    apply_limit_mask = True

    # Boundary handling
    if has_gap_constraint:
        fill_value = (np.nan, np.nan)
    else:
        fill_value = "extrapolate"

    if method in ["linear", "nearest", "slinear", "quadratic", "cubic"]:
        from scipy.interpolate import interp1d
        if np.iscomplexobj(val_1d):
            f_real = interp1d(x_valid, y_valid.real, kind=method, bounds_error=False, fill_value=fill_value)
            f_imag = interp1d(x_valid, y_valid.imag, kind=method, bounds_error=False, fill_value=fill_value)
            val_1d[nans_1d] = f_real(x[nans_1d]) + 1j * f_imag(x[nans_1d])
        else:
            f = interp1d(x_valid, y_valid, kind=method, bounds_error=False, fill_value=fill_value)
            val_1d[nans_1d] = f(x[nans_1d])
    elif method == "ffill":
        try:
            import pandas as pd
        except ImportError:
            val_1d[:] = _ffill_numpy(val_1d, limit=limit)
        else:
            val_1d[:] = pd.Series(val_1d).ffill(limit=limit).values
        apply_limit_mask = False
    elif method == "bfill":
        try:
            import pandas as pd
        except ImportError:
            val_1d[:] = _bfill_numpy(val_1d, limit=limit)
        else:
            val_1d[:] = pd.Series(val_1d).bfill(limit=limit).values
        apply_limit_mask = False
    elif method == "mean":
        val_1d[nans_1d] = np.nanmean(val_1d)
    elif method == "median":
        val_1d[nans_1d] = np.nanmedian(val_1d)

    if has_gap_constraint and gap_threshold is not None:
        valid_indices = np.where(valid_1d)[0]
        if len(valid_indices) > 1:
            diffs = np.diff(x[valid_indices])
            big_gaps = np.where(diffs > gap_threshold - 1e-12)[0]
            for idx in big_gaps:
                t_start = x[valid_indices[idx]]
                t_end = x[valid_indices[idx+1]]
                mask = (x > t_start) & (x < t_end)
                val_1d[mask] = np.nan
        if len(valid_indices) > 0:
            val_1d[:valid_indices[0]] = np.nan
            val_1d[valid_indices[-1] + 1:] = np.nan

    if limit is not None and apply_limit_mask:
        limit_mask = _limit_mask(nans_1d, limit, direction="forward")
        if np.any(limit_mask):
            val_1d[limit_mask] = np.nan

    return val_1d

def impute_timeseries(ts, *, method="linear", limit=None, axis=-1, max_gap=None, **kwargs):
    """
    Impute missing values in a TimeSeries or array. Supports multi-dimensional data.
    """
    if hasattr(ts, 'value'):
        val = ts.value.copy()
        is_ts = True
    else:
        val = np.asarray(ts).copy()
        is_ts = False

    if method == "interpolate":
        method = "linear"

    if axis == "time":
        axis = -1
    axis = axis % val.ndim

    times_val = None
    time_unit = None
    if is_ts:
        try:
            times_val = ts.times.value
            time_unit = ts.times.unit
        except AttributeError:
            pass

    if times_val is None:
        times_val = np.arange(val.shape[axis])

    has_gap_constraint = max_gap is not None
    gap_threshold = None
    if has_gap_constraint:
        if hasattr(max_gap, 'to') and time_unit:
            gap_threshold = max_gap.to(time_unit).value
        elif hasattr(max_gap, 'to'):
            gap_threshold = max_gap.to(u.s).value if max_gap.unit.physical_type == 'time' else max_gap.value
        else:
            gap_threshold = float(max_gap)

    nans = np.isnan(val)
    if not np.any(nans):
        return ts.copy() if is_ts else val

    other_axes = [i for i in range(val.ndim) if i != axis]
    if other_axes:
        nans_reduced = np.any(nans, axis=tuple(other_axes))
        # Broadcast nans_reduced back to original shape for comparison
        reshape_dims = [1] * val.ndim
        reshape_dims[axis] = -1
        nans_broadcast = nans_reduced.reshape(reshape_dims)
        nans_common = np.all(nans == nans_broadcast)
    else:
        nans_common = True
        nans_reduced = nans

    if nans_common and method not in ["ffill", "bfill", "mean", "median"]:
        valid_mask = ~nans_reduced
        if not np.any(valid_mask):
             return ts.copy() if is_ts else val

        x_valid = times_val[valid_mask]
        slices = [slice(None)] * val.ndim
        slices[axis] = valid_mask
        y_valid = val[tuple(slices)]

        if has_gap_constraint:
            fill_value = (np.nan, np.nan)
        else:
            fill_value = "extrapolate"

        if method in ["linear", "nearest", "slinear", "quadratic", "cubic"]:
            from scipy.interpolate import interp1d
            if np.iscomplexobj(val):
                f_real = interp1d(x_valid, y_valid.real, kind=method, axis=axis, bounds_error=False, fill_value=fill_value)
                f_imag = interp1d(x_valid, y_valid.imag, kind=method, axis=axis, bounds_error=False, fill_value=fill_value)
                nan_mask_idx = np.where(nans_reduced)[0]
                nan_slices = [slice(None)] * val.ndim
                nan_slices[axis] = nan_mask_idx
                val[tuple(nan_slices)] = f_real(times_val[nan_mask_idx]) + 1j * f_imag(times_val[nan_mask_idx])
            else:
                f = interp1d(x_valid, y_valid, kind=method, axis=axis, bounds_error=False, fill_value=fill_value)
                nan_mask_idx = np.where(nans_reduced)[0]
                nan_slices = [slice(None)] * val.ndim
                nan_slices[axis] = nan_mask_idx
                val[tuple(nan_slices)] = f(times_val[nan_mask_idx])

        if has_gap_constraint:
            diffs = np.diff(x_valid)
            big_gaps = np.where(diffs > gap_threshold - 1e-12)[0]
            for idx in big_gaps:
                t_start = x_valid[idx]
                t_end = x_valid[idx+1]
                mask_idx = np.where((times_val > t_start) & (times_val < t_end))[0]
                revert_slc = [slice(None)] * val.ndim
                revert_slc[axis] = mask_idx
                val[tuple(revert_slc)] = np.nan
            if len(x_valid) > 0:
                lead_mask = np.where(times_val < x_valid[0])[0]
                tail_mask = np.where(times_val > x_valid[-1])[0]
                if lead_mask.size:
                    revert_slc = [slice(None)] * val.ndim
                    revert_slc[axis] = lead_mask
                    val[tuple(revert_slc)] = np.nan
                if tail_mask.size:
                    revert_slc = [slice(None)] * val.ndim
                    revert_slc[axis] = tail_mask
                    val[tuple(revert_slc)] = np.nan

        if limit is not None:
            limit_mask = _limit_mask(nans_reduced, limit, direction="forward")
            if np.any(limit_mask):
                limit_slc = [slice(None)] * val.ndim
                limit_slc[axis] = limit_mask
                val[tuple(limit_slc)] = np.nan
    else:
        it = np.ndindex(tuple(s for i, s in enumerate(val.shape) if i != axis))
        for idxs in it:
            slc = [slice(None)] * val.ndim
            j = 0
            for i in range(val.ndim):
                if i != axis:
                    slc[i] = idxs[j]
                    j += 1
            val[tuple(slc)] = _impute_1d(
                val[tuple(slc)],
                times_val,
                method,
                has_gap_constraint,
                gap_threshold,
                limit=limit,
            )

    if is_ts:
        target_dtype = val.dtype
        needs_cast = np.issubdtype(ts.value.dtype, np.integer) and target_dtype.kind in ("f", "c")
        if needs_cast:
            val = val.astype(np.result_type(val, np.float64))
            if getattr(ts, "dt", None) is None or (hasattr(ts, "is_regular") and not ts.is_regular):
                new_ts = ts.__class__(
                    val,
                    times=ts.times,
                    name=ts.name,
                    unit=ts.unit,
                    channel=getattr(ts, "channel", None),
                )
            else:
                new_ts = ts.__class__(
                    val,
                    t0=ts.t0,
                    dt=ts.dt,
                    name=ts.name,
                    unit=ts.unit,
                    channel=getattr(ts, "channel", None),
                )
        else:
            new_ts = ts.copy()
            new_ts.value[:] = val
        return new_ts
    return val

def standardize_timeseries(ts, *, method="zscore", ddof=0, robust=None):
    """
    Standardize a TimeSeries.
    """
    if robust is True:
        method = "robust"

    val = ts.value
    if robust or method == "robust":
        med = np.nanmedian(val)
        mad = np.nanmedian(np.abs(val - med))
        scale = 1.4826 * mad
        if scale == 0:
            warnings.warn("MAD is zero, setting scale to 1.0 to avoid division by zero.")
            scale = 1.0
    elif method == "zscore":
        med = np.nanmean(val)
        scale = np.nanstd(val, ddof=ddof)
        if scale == 0:
            warnings.warn("Standard deviation is zero, setting scale to 1.0 to avoid division by zero.")
            scale = 1.0
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

def standardize_matrix(matrix, *, axis="time", method="zscore", ddof=0, robust=None):
    """
    Standardize a TimeSeriesMatrix.
    """
    if robust is True:
        method = "robust"
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
