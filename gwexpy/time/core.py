from __future__ import annotations

import importlib
from typing import Any

import astropy.units as u
import numpy as np
from astropy.time import Time
from gwpy.time import from_gps as _gwpy_from_gps
from gwpy.time import tconvert as _gwpy_tconvert
from gwpy.time import to_gps as _gwpy_to_gps

try:
    pd: Any = importlib.import_module("pandas")
except ImportError:
    pd = None

__all__ = ["to_gps", "from_gps", "tconvert"]

_VALID_DTYPES = frozenset({None, float, "float", "quantity"})


def _validate_dtype(dtype):
    if dtype not in _VALID_DTYPES:
        raise ValueError(
            f"Invalid dtype {dtype!r} for to_gps(). "
            "Expected one of: None, float, 'float', 'quantity'."
        )


def _apply_dtype(value, dtype):
    if dtype is None:
        return value
    if dtype in (float, "float"):
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=float)
        return float(value)
    # dtype == "quantity"
    if isinstance(value, np.ndarray):
        return u.Quantity(np.asarray(value, dtype=float), unit=u.s)
    return u.Quantity(float(value), unit=u.s)


def _is_array(obj):
    if isinstance(obj, (str, bytes)):
        return False
    if pd is not None and isinstance(obj, (pd.Series, pd.Index, pd.DatetimeIndex)):
        return True
    if isinstance(obj, np.ndarray):
        return obj.ndim > 0
    return isinstance(obj, (list, tuple))


def _is_numeric_array(arr):
    if arr.dtype.kind in ("i", "u", "f"):
        return True
    if arr.dtype.kind == "M":
        return False
    try:
        arr.astype(float)
        return True
    except (TypeError, ValueError):
        return False


def _normalize_time_input(t):
    if pd is not None:
        if isinstance(t, pd.Timestamp):
            return t.to_pydatetime()
        if isinstance(t, (pd.Series, pd.Index, pd.DatetimeIndex)):
            return t.to_numpy()

    if type(t).__name__ == "UTCDateTime" and hasattr(t, "datetime"):
        return t.datetime

    if _is_array(t):
        try:
            if len(t) == 0:
                return t
        except TypeError:
            return t
        first = t[0]
        if pd is not None and isinstance(first, pd.Timestamp):
            return [x.to_pydatetime() for x in t]
        if type(first).__name__ == "UTCDateTime" and hasattr(first, "datetime"):
            return [x.datetime for x in t]

    return t


def to_gps(t, *args, dtype=None, **kwargs):
    """Convert a given time or array of times to GPS seconds.

    This is a vectorized extension of `gwpy.time.to_gps`. It supports
    single values (strings, datetime, etc.) as well as arrays, pandas Series,
    and lists.

    Parameters
    ----------
    t : str, datetime.datetime, astropy.time.Time, or array-like
        The input time(s) to convert. Supported formats include UTC strings,
        datetime objects, pandas Timestamps, or arrays of these types.
    *args
        Additional positional arguments passed to `gwpy.time.to_gps`.
    dtype : {None, float, "float", "quantity"}, optional
        Controls the return type.

        - ``None`` (default): preserves existing return types without
          modification. Scalars may return ``LIGOTimeGPS`` (via GWpy);
          arrays return ``numpy.ndarray`` of float64.
        - ``float`` or ``"float"``: scalars return Python ``float``; arrays
          return ``numpy.ndarray`` of float64.
        - ``"quantity"``: returns ``astropy.units.Quantity`` in seconds for
          both scalars and arrays. Use this when comparing against or doing
          arithmetic with ``TimeSeries.times``.
    **kwargs
        Additional keyword arguments passed to `gwpy.time.to_gps`.

    Returns
    -------
    LIGOTimeGPS, float, numpy.ndarray, or astropy.units.Quantity
        The equivalent time in GPS seconds. The exact type depends on the
        input shape and the *dtype* argument (see above).

    Raises
    ------
    ValueError
        If *dtype* is not one of the recognised values.

    Examples
    --------
    Default (GWpy-compatible) behaviour::

        >>> to_gps("2026-03-04 06:00:00")
        LIGOTimeGPS(1741057218, 0)

    Interoperable with ``TimeSeries.times``::

        >>> g = to_gps("2026-03-04 06:00:00", dtype="quantity")
        >>> ts.times > g   # works without TypeError
        >>> ts.times - g   # works without TypeError

    """
    _validate_dtype(dtype)
    t_norm = _normalize_time_input(t)

    if isinstance(t_norm, Time):
        return _apply_dtype(t_norm.gps, dtype)

    if not _is_array(t_norm):
        if isinstance(t_norm, np.datetime64):
            t_norm = t_norm.item()
        return _apply_dtype(_gwpy_to_gps(t_norm, *args, **kwargs), dtype)

    try:
        # If the input is already a Quantity, convert to seconds first so that
        # np.asarray() stripping the unit does not silently misinterpret the
        # values (e.g. [1000, 2000] ms would become [1000, 2000] s otherwise).
        if isinstance(t_norm, u.Quantity):
            t_norm = t_norm.to(u.s).value
        arr = np.asarray(t_norm)
        if _is_numeric_array(arr):
            return _apply_dtype(arr.astype(float), dtype)
        return _apply_dtype(Time(t_norm, *args, **kwargs).gps, dtype)
    except (ValueError, TypeError):
        return _apply_dtype(
            np.array([_gwpy_to_gps(x, *args, **kwargs) for x in t_norm], dtype=float),
            dtype,
        )


def from_gps(gps, *args, **kwargs):
    """Convert a given GPS time or array of GPS times to datetime objects.

    This is a vectorized extension of `gwpy.time.from_gps`. It supports
    single scalar GPS times as well as arrays, pandas Series, and lists.

    Parameters
    ----------
    gps : float, int, astropy.time.Time, or array-like
        The input GPS time(s) to convert.
    *args
        Additional positional arguments passed to `gwpy.time.from_gps`.
    **kwargs
        Additional keyword arguments passed to `gwpy.time.from_gps`.

    Returns
    -------
    datetime.datetime or numpy.ndarray
        The equivalent UTC datetime object. Returns a datetime for scalar inputs
        and a numpy.ndarray of datetime objects for array-like inputs.

    """
    gps_norm = _normalize_time_input(gps)

    if isinstance(gps_norm, Time):
        return gps_norm.to_datetime()

    if not _is_array(gps_norm):
        return _gwpy_from_gps(gps_norm, *args, **kwargs)

    try:
        arr = np.asarray(gps_norm)
        if arr.dtype.kind not in ("i", "u", "f"):
            arr = arr.astype(float)
        return Time(arr, format="gps", *args, **kwargs).to_datetime()
    except (ValueError, TypeError):
        return np.array([_gwpy_from_gps(x, *args, **kwargs) for x in gps_norm])


def tconvert(t="now", *args, **kwargs):
    """Convert a time between GPS seconds and UTC datetime.

    This function automatically detects the type of the input `t`. If `t` is
    numeric (or an array of numbers), it is assumed to be a GPS time and is
    converted to a datetime (like `from_gps`). If `t` is a string, datetime,
    or an array of those types, it is converted to GPS seconds (like `to_gps`).

    Parameters
    ----------
    t : numeric, str, datetime.datetime, array-like, optional
        The input time(s) to convert. Defaults to "now".
    *args
        Additional positional arguments passed to the underlying converter.
    **kwargs
        Additional keyword arguments passed to the underlying converter.

    Returns
    -------
    float, datetime.datetime, or numpy.ndarray
        The converted time. The return type depends on the input type.

    """
    t_norm = _normalize_time_input(t)

    if not _is_array(t_norm):
        return _gwpy_tconvert(t_norm, *args, **kwargs)

    try:
        arr = np.asarray(t_norm)
        is_numeric = _is_numeric_array(arr)
    except (ValueError, TypeError):
        is_numeric = False

    if is_numeric:
        return from_gps(t_norm, *args, **kwargs)
    return to_gps(t_norm, *args, **kwargs)
