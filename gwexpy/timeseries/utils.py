"""
gwexpy.timeseries.utils
-----------------------

Utility functions for time series axis validation and extraction.
"""

import numpy as np
from astropy import units as u

__all__ = [
    "_extract_axis_info",
    "_validate_common_axis",
    "_extract_freq_axis_info",
    "_validate_common_frequency_axis",
    "_validate_common_epoch",
    "SeriesType",
]


def _extract_axis_info(ts):
    """
    Extract axis information for a TimeSeries-like object.

    Parameters
    ----------
    ts : TimeSeries-like
        Object with a `times` attribute.

    Returns
    -------
    dict
        Dictionary with keys: 'regular', 'dt', 't0', 'n', 'times'.
    """
    axis = getattr(ts, "times", None)
    if axis is None:
        raise ValueError("times axis is required")

    regular = False
    dt = None
    try:
        dt = ts.dt
    except AttributeError:
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
        except (TypeError, ValueError):
            regular = False

    t0 = None
    if regular:
        try:
            t0 = ts.t0
        except AttributeError:
            try:
                t0 = axis[0]
            except (IndexError, TypeError):
                t0 = None
        if t0 is None:
            regular = False

    try:
        n = len(axis)
    except TypeError:
        n = None
        regular = False

    return {"regular": regular, "dt": dt, "t0": t0, "n": n, "times": axis}


def _validate_common_axis(axis_infos, method_name):
    """
    Validate that a list of axis infos share a common axis.

    Parameters
    ----------
    axis_infos : list of dict
        List of axis info dictionaries from _extract_axis_info.
    method_name : str
        Name of the calling method (for error messages).

    Returns
    -------
    times : array-like
        Common time axis.
    n : int
        Number of samples.
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

    Parameters
    ----------
    fs : FrequencySeries-like
        Object with a `frequencies` attribute.

    Returns
    -------
    dict
        Dictionary with keys: 'regular', 'df', 'f0', 'n', 'freqs'.
    """
    freqs = getattr(fs, "frequencies", None)
    if freqs is None:
        raise ValueError("frequencies axis is required")

    regular = False
    df = None
    try:
        df = fs.df
    except AttributeError:
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
        except (TypeError, ValueError):
            regular = False

    f0 = None
    if regular:
        try:
            f0 = fs.f0
        except AttributeError:
            try:
                f0 = freqs[0]
            except (IndexError, TypeError):
                f0 = None
        if f0 is None:
            regular = False

    try:
        n = len(freqs)
    except TypeError:
        n = None
        regular = False

    return {"regular": regular, "df": df, "f0": f0, "n": n, "freqs": freqs}


def _validate_common_frequency_axis(axis_infos, method_name):
    """
    Validate common frequency axis across FrequencySeries results.

    Parameters
    ----------
    axis_infos : list of dict
        List of axis info dictionaries from _extract_freq_axis_info.
    method_name : str
        Name of the calling method (for error messages).

    Returns
    -------
    freqs : array-like
        Common frequency axis.
    df : Quantity or None
        Frequency spacing.
    f0 : Quantity or None
        Starting frequency.
    n : int
        Number of frequency bins.
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
    """
    Validate that all epochs are identical.

    Parameters
    ----------
    epochs : list
        List of epoch values.
    method_name : str
        Name of the calling method (for error messages).

    Returns
    -------
    epoch
        The common epoch value, or None if list is empty.
    """
    if not epochs:
        return None
    ref = epochs[0]
    for e in epochs[1:]:
        if e != ref:
            raise ValueError(f"{method_name} requires common epoch; mismatch in epoch")
    return ref


try:
    from gwpy.types.index import SeriesType  # pragma: no cover - optional in gwpy
except ImportError:  # fallback for gwpy versions without SeriesType
    from enum import Enum

    class SeriesType(Enum):
        TIME = "time"
        FREQ = "freq"
