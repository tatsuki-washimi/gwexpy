"""
gwexpy.signal.preprocessing.imputation
---------------------------------------

Missing value imputation algorithms for signal processing.
"""

import numpy as np


def _coerce_times(times, n):
    x = np.asarray(times)
    if hasattr(x, "value"):
        x = np.asarray(x.value)
    if x.ndim != 1:
        raise ValueError("times must be a 1D array")
    if len(x) != n:
        raise ValueError("times must have the same length as values")
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    if np.any(np.diff(x_sorted) <= 0):
        raise ValueError("times must be strictly increasing")
    return x_sorted, sort_idx


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


def impute(values, *, method="interpolate", limit=None, times=None,
           max_gap=None, fill_value=np.nan):
    """
    Impute missing values in an array.

    Parameters
    ----------
    values : ndarray
        1D array with potential NaN values.
    method : str, optional
        Imputation method: 'interpolate', 'ffill', 'bfill', 'mean', 'median'.
        Default is 'interpolate'.
    limit : int, optional
        Maximum number of consecutive NaNs to fill.
    times : ndarray, optional
        Time array corresponding to values. Used for time-based interpolation
        and max_gap calculation.
    max_gap : float, optional
        Maximum gap duration to fill. If a gap is larger than this, it is left as NaN.
    fill_value : float, optional
        Value to use for fill operations that don't have a source value.

    Returns
    -------
    imputed : ndarray
        Array with imputed values.
    """
    val = np.asarray(values).copy()
    nans = np.isnan(val)

    if not np.any(nans):
        return val

    # Use times if provided, else indices
    sort_idx = None
    if times is not None:
        x, sort_idx = _coerce_times(times, len(val))
        if sort_idx is not None:
            val = val[sort_idx]
            nans = nans[sort_idx]
    else:
        x = np.arange(len(val))

    valid = ~nans
    if max_gap is not None:
        gap_threshold = float(max_gap.value) if hasattr(max_gap, "value") else float(max_gap)
    else:
        gap_threshold = None
    has_gap_constraint = gap_threshold is not None

    if method == "interpolate":
        left_val = np.nan if has_gap_constraint else None
        right_val = np.nan if has_gap_constraint else None

        if not np.any(valid):
            pass
        elif np.iscomplexobj(val):
            real_part = np.interp(x[nans], x[valid], val[valid].real,
                                  left=left_val, right=right_val)
            imag_part = np.interp(x[nans], x[valid], val[valid].imag,
                                  left=left_val, right=right_val)
            val[nans] = real_part + 1j * imag_part
        else:
            val[nans] = np.interp(x[nans], x[valid], val[valid],
                                  left=left_val, right=right_val)

    elif method == "ffill":
        try:
            from pandas import Series
        except ImportError:
            val = _ffill_numpy(val, limit=limit)
        else:
            s = Series(val)
            val = s.ffill(limit=limit).values

    elif method == "bfill":
        try:
            from pandas import Series
        except ImportError:
            val = _bfill_numpy(val, limit=limit)
        else:
            s = Series(val)
            val = s.bfill(limit=limit).values

    elif method == "mean":
        val[nans] = np.nanmean(val)

    elif method == "median":
        val[nans] = np.nanmedian(val)

    else:
        raise ValueError(f"Unknown impute method '{method}'")

    # Apply max_gap constraint
    valid_indices = np.where(~nans)[0]
    if has_gap_constraint and len(valid_indices) > 1:
        valid_times = x[valid_indices]
        diffs = np.diff(valid_times)
        big_gaps = np.where(diffs > gap_threshold - 1e-12)[0]

        for idx in big_gaps:
            t_start = valid_times[idx]
            t_end = valid_times[idx + 1]
            mask = (x > t_start) & (x < t_end)
            val[mask] = np.nan
    if has_gap_constraint and len(valid_indices) > 0:
        first_valid = valid_indices[0]
        last_valid = valid_indices[-1]
        if first_valid > 0:
            val[:first_valid] = np.nan
        if last_valid < len(val) - 1:
            val[last_valid + 1:] = np.nan

    if limit is not None and method not in ("ffill", "bfill"):
        limit_mask = _limit_mask(nans, limit, direction="forward")
        if np.any(limit_mask):
            val[limit_mask] = np.nan

    is_fill_nan = False
    try:
        is_fill_nan = np.isnan(fill_value)
    except (TypeError, ValueError):
        pass
    if not is_fill_nan:
        if method == "interpolate":
            if len(valid_indices) == 0:
                val[np.isnan(val)] = fill_value
            else:
                left_edge = valid_indices[0]
                right_edge = valid_indices[-1]
                edge_mask = (np.arange(len(val)) < left_edge) | (np.arange(len(val)) > right_edge)
                val[edge_mask & np.isnan(val)] = fill_value
        elif method in ("ffill", "bfill"):
            val[np.isnan(val)] = fill_value
        else:
            if np.all(np.isnan(val)):
                val[:] = fill_value

    if sort_idx is not None:
        out = np.empty_like(val)
        out[sort_idx] = val
        val = out

    return val


__all__ = ["impute"]
