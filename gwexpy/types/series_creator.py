from __future__ import annotations
from typing import Any

import numpy as np
from astropy import units as u


def _is_time_unit(unit: Any) -> bool:
    try:
        return u.Unit(unit).is_equivalent(u.s)
    except (ValueError, TypeError):
        return False


def _is_freq_unit(unit: Any) -> bool:
    try:
        with_hz = getattr(u.equivalencies, "with_Hertz", None)
        if with_hz is None:
            return u.Unit(unit).is_equivalent(u.Hz)
        return u.Unit(unit).is_equivalent(u.Hz, equivalencies=with_hz())
    except (ValueError, TypeError):
        return False


def _is_angular_frequency(unit: Any) -> bool:
    try:
        unit = u.Unit(unit)
    except (ValueError, TypeError):
        return False
    return "angular" in str(unit.physical_type)


def _to_quantity_1d(values: Any) -> u.Quantity:
    if not isinstance(values, u.Quantity):
        raise TypeError("as_series expects a 1D astropy.units.Quantity")
    q = values
    if q.ndim != 1:
        raise ValueError("as_series expects a 1D axis")
    return q


def _to_hz(q: u.Quantity) -> u.Quantity:
    with_hz = getattr(u.equivalencies, "with_Hertz", None)
    try:
        if with_hz is not None:
            return q.to(u.Hz, equivalencies=with_hz())
        return q.to(u.Hz)
    except u.UnitConversionError:
        if _is_angular_frequency(q.unit):
            return u.Quantity(q.value / (2 * np.pi), u.Hz)
        raise


def _to_angular_frequency(hz: u.Quantity) -> u.Quantity:
    return hz.to(1 / u.s) * (2 * np.pi * u.rad)


def as_series(axis, unit=None, *, name=None):
    """
    Convert a 1D axis (``gwpy.types.index.Index`` or ``astropy.units.Quantity``)
    to a ``TimeSeries`` or ``FrequencySeries``.

    The created series is an identity mapping: the series values represent the
    axis values (optionally converted to ``unit``), while the series axis is
    taken from the input.

    Parameters
    ----------
    axis
        1D ``gwpy.types.index.Index`` or 1D ``astropy.units.Quantity``.
    unit
        Optional target unit for the *values* (not the axis). Must be time-like
        for time axes, and frequency-like (Hz) or angular frequency (rad/s) for
        frequency axes.
    name
        Optional series name.
    """
    from gwexpy.time import to_gps
    axis_unit = getattr(axis, "unit", None)

    if isinstance(axis, u.Quantity):
        axis_q = _to_quantity_1d(axis)
    elif axis_unit is not None:
        try:
            axis_q = _to_quantity_1d(u.Quantity(np.asarray(axis), u.Unit(axis_unit)))
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Could not convert axis with unit '{axis_unit}' to Quantity: {e}. "
                "The input values may be incompatible with the specified unit."
            ) from e
    else:
        # Try to convert to GPS if it looks like time (datetime, strings, etc)
        try:
            axis_gps = to_gps(axis)
            # If to_gps returned a numeric array from a non-numeric input, we treat it as seconds.
            # But if it was already numeric and had no unit, we should maybe be more strict?
            # For now, following the user request to support datetime arrays.
            if isinstance(axis, (np.ndarray, list)) and len(axis) > 0 and not isinstance(axis[0], (int, float, np.number)):
                 axis_q = _to_quantity_1d(u.Quantity(np.asarray(axis_gps), u.s))
            else:
                 # If it was already numeric and reached here, it means it had no .unit.
                 # We still try to wrap it in case to_gps did something useful,
                 # but this is where the "unclear error" usually happened.
                 axis_q = _to_quantity_1d(u.Quantity(np.asarray(axis_gps), u.s))
        except (ValueError, TypeError, AttributeError) as e:
            raise TypeError(
                f"as_series expects a 1D axis-like input (Quantity, Index with unit, or datetime array). "
                f"Current input type '{type(axis)}' could not be quantified: {e}"
            ) from e

    if _is_time_unit(axis_q.unit):
        value_unit = u.Unit(unit) if unit is not None else u.Unit(axis_q.unit)
        if not _is_time_unit(value_unit):
            raise ValueError("unit must be time-like for a time axis")

        from gwexpy.timeseries import TimeSeries
        values_q = axis_q.to(value_unit)
        times_axis = axis_q
        return TimeSeries(values_q.value, times=times_axis, unit=value_unit, name=name)

    if _is_freq_unit(axis_q.unit) or _is_angular_frequency(axis_q.unit):
        if unit is None:
            value_unit = u.Unit(axis_q.unit)
        else:
            value_unit = u.Unit(unit)
            if not (_is_freq_unit(value_unit) or _is_angular_frequency(value_unit)):
                raise ValueError("unit must be frequency-like (Hz) or angular frequency (rad/s) for a frequency axis")

        axis_hz = _to_hz(axis_q)
        if _is_angular_frequency(value_unit):
            values_q = _to_angular_frequency(axis_hz).to(value_unit)
        else:
            values_q = axis_hz.to(value_unit)

        from gwexpy.frequencyseries import FrequencySeries
        freq_axis = axis if axis_unit is not None and not _is_angular_frequency(axis_q.unit) else axis_hz
        return FrequencySeries(values_q.value, frequencies=freq_axis, unit=value_unit, name=name)

    raise ValueError("axis unit must be time-like (s) or frequency-like (Hz / rad/s)")
