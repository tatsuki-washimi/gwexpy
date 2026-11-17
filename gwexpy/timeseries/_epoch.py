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
from gwpy.types import Series

from gwexpy.time._reference import (
    TIME_REFERENCE_ABSOLUTE,
    inherit_time_reference,
    resolve_time_inputs,
)

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


def _prepare_time_reference(
    data: Any,
    args: list[Any],
    kwargs: dict[str, Any],
    *,
    t0: Any,
    times_given: bool,
    exact_t0_ns: int | None,
) -> str | None:
    """Resolve the timing-reference state requested by constructor arguments.

    ``args`` and ``kwargs`` are the arguments that will be forwarded to GWpy;
    a ``NaN`` ``t0``/``epoch``/``x0`` is removed from them in place so that it
    never reaches (and poisons) the axis.  Returns ``"absolute"`` or
    ``"relative"`` when the arguments decide the state, or ``None`` when the
    state is inherited from ``data`` (a series that already has an axis).

    Raises
    ------
    ValueError
        If ``t0``, ``epoch`` or ``x0`` is ``+Inf`` or ``-Inf``.

    """
    if exact_t0_ns is not None:
        return TIME_REFERENCE_ABSOLUTE

    state, clear = resolve_time_inputs(
        {"t0": t0, "epoch": kwargs.get("epoch"), "x0": kwargs.get("x0")},
        explicit_axis=times_given or kwargs.get("xindex") is not None,
        inherit=isinstance(data, Series),
    )
    for name in clear:
        if name == "t0" and len(args) >= 2:
            args[1] = None
        else:
            kwargs[name] = None
    return state


def _restore_exact_time_authority(source: Any, result: Any) -> Any:
    """Restore private exact epoch and cadence state on a derived series.

    Also carries the source's timing-reference state (``time_reference``): a
    result rebuilt with an explicit ``x0`` would otherwise read as absolute.
    """
    inherit_time_reference(result, source)
    exact_t0_ns = getattr(source, "_gwex_t0_gps_ns", None)
    if exact_t0_ns is None:
        return result

    result._gwex_t0_gps_ns = exact_t0_ns
    try:
        result._gwex_dt_gps_ns = _integral_dt_gps_ns(result.dt)
    except (TypeError, ValueError):
        result.__dict__.pop("_gwex_dt_gps_ns", None)
    return result
