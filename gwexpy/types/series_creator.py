from astropy import units as u
import numpy as np
from gwexpy.timeseries import TimeSeries
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.time import to_gps

_TIME_KEYS = ("t0", "dt", "times", "sample_rate", "epoch")
_FREQ_KEYS = ("f0", "df", "frequencies")


def _is_time_unit(unit):
    try:
        return u.Unit(unit).is_equivalent(u.s)
    except Exception:
        return False


def _is_freq_unit(unit):
    try:
        with_hz = getattr(u.equivalencies, "with_Hertz", None)
        if with_hz is None:
            return u.Unit(unit).is_equivalent(u.Hz)
        return u.Unit(unit).is_equivalent(u.Hz, equivalencies=with_hz())
    except Exception:
        return False


def _is_angular_frequency(unit):
    try:
        unit = u.Unit(unit)
    except Exception:
        return False
    return "angular" in str(unit.physical_type)


def _normalize_quantity(data, unit):
    if isinstance(data, u.Quantity):
        if unit is not None:
            return data.to(u.Unit(unit))
        return data
    if unit is not None:
        return u.Quantity(data, unit)
    return data


def _maybe_normalize_times(value):
    if isinstance(value, u.Quantity):
        return value
    try:
        return to_gps(value)
    except Exception:
        return value


def _to_hz(value):
    if not isinstance(value, u.Quantity):
        return value
    with_hz = getattr(u.equivalencies, "with_Hertz", None)
    try:
        if with_hz is not None:
            return value.to(u.Hz, equivalencies=with_hz())
        return value.to(u.Hz)
    except u.UnitConversionError:
        if _is_angular_frequency(value.unit):
            return u.Quantity(value.value / (2 * np.pi), u.Hz)
        raise


def _normalize_frequency_axis(value):
    if isinstance(value, u.Quantity) and (_is_angular_frequency(value.unit) or _is_freq_unit(value.unit)):
        return _to_hz(value)
    return value


def as_series(data, unit=None, **kwargs):
    xindex = kwargs.pop("xindex", None)
    value_unit = u.Unit(unit) if unit is not None else None
    data_is_quantity = isinstance(data, u.Quantity)
    data_unit = data.unit if data_is_quantity else None

    data_as_axis = False
    if data_is_quantity and value_unit is not None:
        try:
            data.to(value_unit)
        except u.UnitConversionError:
            if _is_time_unit(data_unit) or _is_freq_unit(data_unit) or _is_angular_frequency(data_unit):
                data_as_axis = True
            else:
                raise

    if data_as_axis:
        axis_values = np.asarray(data.value)
        data_q = u.Quantity(np.zeros_like(axis_values, dtype=float), value_unit) if value_unit is not None else np.zeros_like(axis_values, dtype=float)
    else:
        data_q = _normalize_quantity(data, unit)

    series_kind = None
    if any(key in kwargs for key in _TIME_KEYS):
        series_kind = "time"
    elif any(key in kwargs for key in _FREQ_KEYS):
        series_kind = "freq"

    if series_kind is None and xindex is not None:
        xunit = getattr(xindex, "unit", None)
        if xunit is not None:
            if _is_time_unit(xunit):
                series_kind = "time"
            elif _is_freq_unit(xunit) or _is_angular_frequency(xunit):
                series_kind = "freq"

    if series_kind is None and data_unit is not None:
        if _is_time_unit(data_unit):
            series_kind = "time"
        elif _is_freq_unit(data_unit) or _is_angular_frequency(data_unit):
            series_kind = "freq"

    if series_kind is None and value_unit is not None:
        if _is_time_unit(value_unit):
            series_kind = "time"
        elif _is_freq_unit(value_unit) or _is_angular_frequency(value_unit):
            series_kind = "freq"

    if series_kind is None:
        raise ValueError(
            "Could not infer series type. Provide time/frequency units or axis keywords."
        )

    if series_kind == "freq":
        if data_as_axis:
            axis = xindex if xindex is not None else data
            kwargs.setdefault("frequencies", _normalize_frequency_axis(axis))
        else:
            if isinstance(data_q, u.Quantity) and (_is_angular_frequency(data_q.unit) or _is_freq_unit(data_q.unit)):
                data_q = _to_hz(data_q)
            if xindex is not None:
                kwargs.setdefault("frequencies", _normalize_frequency_axis(xindex))
        return FrequencySeries(data_q, **kwargs)

    if data_as_axis:
        axis = xindex if xindex is not None else data
        kwargs.setdefault("times", _maybe_normalize_times(axis))
        if "name" not in kwargs:
            kwargs["name"] = "time"
    else:
        if xindex is not None:
            kwargs.setdefault("times", _maybe_normalize_times(xindex))
        if "times" in kwargs:
            kwargs["times"] = _maybe_normalize_times(kwargs["times"])
        if "t0" in kwargs:
            kwargs["t0"] = _maybe_normalize_times(kwargs["t0"])

    return TimeSeries(data_q, **kwargs)
