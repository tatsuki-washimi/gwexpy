"""
gwexpy.signal.preprocessing.imputation
---------------------------------------

Missing value imputation algorithms for signal processing.
"""

import numpy as np


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
    if times is not None:
        x = np.asarray(times)
    else:
        x = np.arange(len(val))

    valid = ~nans
    gap_threshold = max_gap if max_gap is not None else None
    has_gap_constraint = gap_threshold is not None

    if method == "interpolate":
        left_val = np.nan if has_gap_constraint else None
        right_val = np.nan if has_gap_constraint else None

        if np.iscomplexobj(val):
            real_part = np.interp(x[nans], x[valid], val[valid].real,
                                  left=left_val, right=right_val)
            imag_part = np.interp(x[nans], x[valid], val[valid].imag,
                                  left=left_val, right=right_val)
            val[nans] = real_part + 1j * imag_part
        else:
            val[nans] = np.interp(x[nans], x[valid], val[valid],
                                  left=left_val, right=right_val)

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

    # Apply max_gap constraint
    if has_gap_constraint:
        valid_indices = np.where(~nans)[0]
        if len(valid_indices) > 1:
            valid_times = x[valid_indices]
            diffs = np.diff(valid_times)
            big_gaps = np.where(diffs > gap_threshold - 1e-12)[0]

            for idx in big_gaps:
                t_start = valid_times[idx]
                t_end = valid_times[idx + 1]
                mask = (x > t_start) & (x < t_end)
                val[mask] = np.nan

    return val


__all__ = ["impute"]
