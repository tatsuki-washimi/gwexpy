"""Exact GPS-nanosecond helpers for :mod:`gwexpy.timeseries`."""

from __future__ import annotations

import math
from contextvars import ContextVar
from decimal import Decimal, InvalidOperation
from operator import index
from typing import Any

import numpy as np
from astropy import units as u
from gwpy.time import LIGOTimeGPS

_EXACT_BUFFER_APPEND_DEPTH: ContextVar[int] = ContextVar(
    "_EXACT_BUFFER_APPEND_DEPTH", default=0
)


def _integer_gps_ns(value: Any, *, default_unit: u.UnitBase = u.s) -> int:
    """Return one time value as an integral number of GPS nanoseconds.

    The conversion operates on an input quantity's value and unit scale
    directly.  It deliberately does not first convert through float seconds,
    which loses small offsets at large GPS epochs.
    """
    if isinstance(value, LIGOTimeGPS):
        return index(value.ns())

    unit = default_unit
    raw_value = value
    if isinstance(value, u.Quantity):
        unit = value.unit
        raw_value = value.value
        if not unit.is_equivalent(u.s):
            raise ValueError("epoch must use a time-compatible unit")

    raw_array = np.asarray(raw_value)
    if raw_array.ndim != 0:
        raise TypeError("epoch must be a scalar integer number of GPS nanoseconds")
    try:
        numeric = Decimal(str(raw_array.item()))
        seconds_scale = Decimal(str(unit.decompose(bases=[u.s]).scale))
        nanoseconds = numeric * seconds_scale * Decimal(1_000_000_000)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise TypeError("epoch must be an integer number of GPS nanoseconds") from exc

    if not nanoseconds.is_finite() or nanoseconds != nanoseconds.to_integral_value():
        raise ValueError("epoch must be an integer number of GPS nanoseconds")
    return int(nanoseconds)


def _integral_dt_gps_ns(value: Any) -> int:
    """Return an integral time interval in nanoseconds or raise ``ValueError``."""
    try:
        return _integer_gps_ns(value)
    except ValueError as exact_error:
        raw_value = value.value if isinstance(value, u.Quantity) else value
        raw_scalar = np.asarray(raw_value)
        if raw_scalar.ndim != 0 or not isinstance(
            raw_scalar.item(), (float, np.floating)
        ):
            raise
        try:
            if isinstance(value, u.Quantity):
                nanoseconds = float(value.to_value(u.ns))
            else:
                nanoseconds = float(u.Quantity(value, u.s).to_value(u.ns))
        except (TypeError, ValueError):
            raise exact_error from None

        if not math.isfinite(nanoseconds):
            raise exact_error from None
        nearest = round(nanoseconds)
        tolerance = math.ulp(nanoseconds)
        if abs(nanoseconds - nearest) <= tolerance:
            return nearest
        raise exact_error from None


def _restore_exact_time_authority(source: Any, result: Any) -> Any:
    """Restore private exact epoch and cadence state on a derived series."""
    exact_t0_ns = getattr(source, "_gwex_t0_gps_ns", None)
    if exact_t0_ns is None:
        return result

    result._gwex_t0_gps_ns = exact_t0_ns
    try:
        result._gwex_dt_gps_ns = _integral_dt_gps_ns(result.dt)
    except (TypeError, ValueError):
        result.__dict__.pop("_gwex_dt_gps_ns", None)
    return result
