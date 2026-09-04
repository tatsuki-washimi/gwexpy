from __future__ import annotations

import datetime as _dt
import importlib
import inspect as _inspect
from decimal import ROUND_HALF_EVEN, Decimal
from typing import Any

import numpy as np
from astropy import units as u
from astropy.time import Time
from gwpy.time import from_gps as _gwpy_from_gps
from gwpy.time import tconvert as _gwpy_tconvert
from gwpy.time import to_gps as _gwpy_to_gps

try:
    pd: Any = importlib.import_module("pandas")
except ImportError:
    pd = None

__all__ = ["to_gps", "from_gps", "tconvert"]


class _TConvertDefault(str):
    """Omission marker that preserves the canonical displayed default."""


class _TConvertAliasOmitted:
    """Stable omission marker for the GWexpy-only ``t=`` alias."""

    def __repr__(self):
        return "<omitted>"


_TCONVERT_DEFAULT = _TConvertDefault("now")
_TCONVERT_ALIAS_OMITTED = _TConvertAliasOmitted()


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


def _is_date_component_sequence(obj):
    """Return whether *obj* has GWpy's scalar date-component call shape."""
    if not isinstance(obj, (list, tuple)) or not 3 <= len(obj) <= 7:
        return False
    try:
        values = np.asarray(obj)
    except (TypeError, ValueError):
        return False
    if values.ndim != 1 or any(np.ndim(item) != 0 for item in obj):
        return False
    if values.dtype.kind == "c":
        return True
    if not _is_numeric_array(values):
        return False

    # A list/tuple with three or more numeric values is ambiguous: GWpy
    # interprets it as ``datetime.datetime(*value)``, while GWexpy also
    # accepts numeric vectors as an extension.  Route only date-shaped values
    # to GWpy.  In particular, GPS vectors such as ``[0., 1., 2., 3.]`` (the
    # form emitted by some dataframe adapters) must not be mistaken for a
    # year/month/day tuple.  Keep a broad component bound so invalid dates such
    # as ``(2017, 13, 1)`` still reach GWpy and retain its failure semantics.
    try:
        year = float(values[0])
        if not np.isfinite(year) or not 1 <= year <= 9999:
            return False
        if len(values) >= 2 and abs(float(values[1])) > 31:
            return False
        # Day-of-month is at most 31, but allow the full day-of-year range so
        # invalid calendar dates still reach GWpy instead of silently becoming
        # a numeric vector.
        if len(values) >= 3 and abs(float(values[2])) > 366:
            return False
        # Once the year, month-like component, and day-like component identify
        # a date-shaped call, retain GWpy's validation for all later
        # components.  Do not reject hour/minute/second/microsecond values here:
        # values such as ``(2017, 1, 1, 25)`` must reach ``datetime`` and raise
        # the same ``ValueError`` as GWpy instead of falling back to a GPS
        # vector conversion.
    except (OverflowError, TypeError, ValueError):
        return False
    return True


_VALID_DTYPES = frozenset({None, float, "float", "quantity"})


def _validate_dtype(dtype):
    if dtype not in _VALID_DTYPES:
        raise ValueError(
            f"Invalid dtype {dtype!r} for to_gps(). "
            "Expected one of: None, float, 'float', 'quantity'."
        )


def _as_float_seconds(result):
    if isinstance(result, np.ndarray):
        return result.astype(float)

    arr = np.asarray(result, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return arr


def _format_gps_result(result, dtype):
    if dtype is None:
        return result
    if dtype is float or dtype == "float":
        return _as_float_seconds(result)
    if dtype == "quantity":
        return _as_float_seconds(result) * u.s
    raise ValueError("dtype must be None, float, 'float', or 'quantity'")


def _datetime64_scalar_to_gps(value, *args, **kwargs):
    """Convert one datetime64 without discarding represented nanoseconds."""
    if np.isnat(value):
        raise ValueError("NaT is not a valid datetime64 instant")
    gps_decimal = Time(
        value,
        format="datetime64",
        scale="utc",
    ).to_value("gps", "decimal")
    # Astropy 8 can return a Decimal a tiny fraction of a nanosecond below the
    # represented datetime64 instant. LIGOTimeGPS truncates that value and
    # would therefore lose one whole nanosecond. Quantize at the destination
    # type's resolution while retaining Astropy's leap-second conversion.
    gps_decimal = gps_decimal.quantize(Decimal("0.000000001"), rounding=ROUND_HALF_EVEN)
    return _gwpy_to_gps(gps_decimal, *args, **kwargs)


def _datetime64_array_to_gps(values, *args, **kwargs):
    """Return exact LIGOTimeGPS elements for a datetime64 array."""
    if np.isnat(values).any():
        raise ValueError("NaT is not a valid datetime64 instant")
    result = np.empty(values.shape, dtype=object)
    for index in np.ndindex(values.shape):
        result[index] = _datetime64_scalar_to_gps(values[index], *args, **kwargs)
    return result


def _as_astropy_gps_input(value):
    """Replace LIGOTimeGPS-like values with exact Decimal GPS seconds."""

    def convert(item):
        if hasattr(item, "gpsSeconds") and hasattr(item, "gpsNanoSeconds"):
            return Decimal(int(item.gpsSeconds)) + Decimal(
                int(item.gpsNanoSeconds)
            ).scaleb(-9)
        return item

    if _is_array(value):
        values = np.asarray(value)
        if values.dtype.kind != "O":
            return values
        result = np.empty(values.shape, dtype=object)
        for index in np.ndindex(values.shape):
            result[index] = convert(values[index])
        return result
    return convert(value)


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
        Output mode for the converted GPS seconds. ``None`` preserves exact
        ``LIGOTimeGPS`` elements for NumPy ``datetime64`` inputs; other inputs
        preserve the existing GWpy-compatible behavior. ``float`` or
        ``"float"`` returns Python ``float`` for scalar inputs and
        ``numpy.ndarray`` for array-like inputs. ``"quantity"`` returns an
        ``astropy.units.Quantity`` in seconds, which can be compared with
        GWpy/GWExpy time axes.
    **kwargs
        Additional keyword arguments passed to `gwpy.time.to_gps`.

    Returns
    -------
    object
        The equivalent time in GPS seconds. With ``dtype=None``, NumPy
        ``datetime64`` scalars return ``LIGOTimeGPS`` and arrays return object
        arrays of exact ``LIGOTimeGPS`` elements. With ``dtype=float`` or
        ``dtype="float"``, returns numeric seconds. With ``dtype="quantity"``,
        returns seconds as an ``astropy.units.Quantity``.

    """
    _validate_dtype(dtype)
    t_norm = _normalize_time_input(t)

    if isinstance(t_norm, Time):
        if t_norm.isscalar and dtype is None:
            result = _gwpy_to_gps(t_norm, *args, **kwargs)
        else:
            result = t_norm.gps
    elif _is_date_component_sequence(t_norm):
        result = _gwpy_to_gps(t_norm, *args, **kwargs)
    elif not _is_array(t_norm):
        if isinstance(t_norm, np.datetime64):
            result = _datetime64_scalar_to_gps(t_norm, *args, **kwargs)
        else:
            result = _gwpy_to_gps(t_norm, *args, **kwargs)
    else:
        arr = np.asarray(t_norm)
        if arr.dtype.kind == "M":
            result = _datetime64_array_to_gps(arr, *args, **kwargs)
        else:
            try:
                unit = getattr(t_norm, "unit", None)
                if unit is not None and hasattr(t_norm, "to_value"):
                    if unit.is_equivalent(u.s):
                        result = np.asarray(t_norm.to_value(u.s), dtype=float)
                    elif unit.is_equivalent(u.dimensionless_unscaled):
                        result = np.asarray(
                            t_norm.to_value(u.dimensionless_unscaled),
                            dtype=float,
                        )
                    else:
                        raise u.UnitConversionError(
                            f"{unit!r} is not convertible to seconds for GPS conversion"
                        )
                elif _is_numeric_array(arr):
                    result = arr.astype(float)
                else:
                    result = Time(t_norm, *args, **kwargs).gps
            except u.UnitConversionError:
                raise
            except (ValueError, TypeError):
                result = np.array([_gwpy_to_gps(x, *args, **kwargs) for x in t_norm])

    return _format_gps_result(result, dtype)


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
        return gps_norm.utc.to_datetime(
            timezone=_dt.UTC,
            leap_second_strict="raise",
        )

    if not args and not kwargs:
        if not _is_array(gps_norm):
            return _gwpy_from_gps(gps_norm)

        values = np.asarray(gps_norm)
        result = np.empty(values.shape, dtype=object)
        for index in np.ndindex(values.shape):
            result[index] = _gwpy_from_gps(values[index])
        return result

    try:
        values = _as_astropy_gps_input(gps_norm)
        times = Time(values, format="gps", *args, **kwargs)
    except (ValueError, TypeError) as exc:
        raise ValueError("GPS values could not be converted to UTC datetimes") from exc
    return times.utc.to_datetime(
        timezone=_dt.UTC,
        leap_second_strict="raise",
    )


def tconvert(
    gpsordate=_TCONVERT_DEFAULT,
    *args,
    t=_TCONVERT_ALIAS_OMITTED,
    **kwargs,
):
    """Convert a time between GPS seconds and UTC datetime.

    This function automatically detects the type of the input. If it is
    numeric (or an array of numbers), it is assumed to be a GPS time and is
    converted to a datetime (like `from_gps`). If the input is a string,
    datetime, or an array of those types, it is converted to GPS seconds
    (like `to_gps`).

    Parameters
    ----------
    gpsordate : numeric, str, datetime.datetime, array-like, optional
        The input time(s) to convert. Defaults to "now".
    *args
        Additional positional arguments passed to the underlying converter.
    t : numeric, str, datetime.datetime, array-like, optional
        GWexpy compatibility alias for ``gpsordate``. Supplying both names is
        an error.
    **kwargs
        Additional keyword arguments passed to the underlying converter.

    Returns
    -------
    float, datetime.datetime, or numpy.ndarray
        The converted time. The return type depends on the input type.

    """
    canonical_given = gpsordate is not _TCONVERT_DEFAULT
    alias_given = t is not _TCONVERT_ALIAS_OMITTED
    if canonical_given and alias_given:
        raise TypeError("cannot specify both 'gpsordate' and 't'")
    if alias_given:
        gpsordate = t
    elif not canonical_given:
        gpsordate = "now"

    t_norm = _normalize_time_input(gpsordate)

    if isinstance(t_norm, np.datetime64):
        return to_gps(t_norm, *args, **kwargs)

    if _is_date_component_sequence(t_norm):
        return _gwpy_tconvert(t_norm, *args, **kwargs)

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


_tconvert_runtime_signature = _inspect.signature(tconvert)
_tconvert_runtime_parameters = tuple(_tconvert_runtime_signature.parameters.values())
_tconvert_gwpy_parameter = next(
    iter(_inspect.signature(_gwpy_tconvert).parameters.values())
)
setattr(
    tconvert,
    "__signature__",
    _tconvert_runtime_signature.replace(
        parameters=(
            _tconvert_gwpy_parameter,
            *_tconvert_runtime_parameters[1:],
        ),
    ),
)
