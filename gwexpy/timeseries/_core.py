"""Core TimeSeries class definition and basic operations.

This module contains the base TimeSeries class with essential functionality:
- Basic operations (tail, crop, append)
- Regularity checking (is_regular, _check_regular)
- Peak finding
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Union

try:
    from typing import TypeAlias
except ImportError:
    from typing import TypeAlias

import numpy as np
from astropy import units as u
from astropy.time import Time
from gwpy.timeseries import TimeSeries as BaseTimeSeries
from numpy.typing import ArrayLike

from gwexpy.types.mixin import RegularityMixin

from ._epoch import _EXACT_BUFFER_APPEND_DEPTH

QuantityLike: TypeAlias = Union[ArrayLike, u.Quantity]


def _crop_bound_to_float(value: Any | None) -> float | None:
    """Normalize one crop bound without quantizing an already-float GPS time."""
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    from gwexpy.time import to_gps

    result = to_gps(value)
    if isinstance(result, (np.ndarray, list)) and np.ndim(result) > 0:
        result = result[0]
    return float(result)


def _is_gwexpy_only_crop_bound(value: object) -> bool:
    """Return whether a crop bound uses an explicit GWexpy time extension."""
    if isinstance(value, (date, Time)):
        return True
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return True
        return False
    if isinstance(value, (tuple, list)) and 3 <= len(value) <= 7:
        return all(
            isinstance(item, (int, float, np.number))
            and not isinstance(item, (bool, np.bool_))
            for item in value
        )
    value_type = type(value)
    return value_type.__name__ == "UTCDateTime" and value_type.__module__.startswith(
        "obspy."
    )


def _crop_bound_to_axis(value: Any, axis_unit: u.UnitBase) -> float:
    """Convert an explicit absolute-time extension into an axis-unit value."""
    gps_seconds = _crop_bound_to_float(value)
    assert gps_seconds is not None
    if not axis_unit.is_equivalent(u.s):
        raise u.UnitConversionError(f"{axis_unit!r} is not a time unit")
    seconds_per_axis_unit = Decimal(str(axis_unit.decompose(bases=[u.s]).scale))
    return float(Decimal(str(gps_seconds)) / seconds_per_axis_unit)


def _regular_crop_slice(
    start: float | None, end: float | None, *, t0: float, dt: float, size: int
) -> slice:
    """Return the positional crop slice used by ``TimeSeriesMatrix``."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"regular crop requires a positive finite dt, got {dt!r}")

    def _index(bound: float | None, default: int) -> int:
        if bound is None:
            return default
        position = (bound - t0) / dt
        nearest = round(position)
        index = nearest if bound == t0 + nearest * dt else np.floor(position)
        return int(np.clip(index, 0, size))

    return slice(_index(start, 0), _index(end, size))


if TYPE_CHECKING:
    from gwexpy.timeseries.timeseries import TimeSeries


class TimeSeriesCore(RegularityMixin, BaseTimeSeries):
    """Core Ti meSeries class with basic operations.

    This is the base class that other mixins will extend.
    Inherits from gwpy.timeseries.TimeSeries for compatibility.
    """

    # ===============================
    # Properties
    # ===============================

    # ===============================
    # Basic Operations
    # ===============================

    def tail(self, n: int | None = 5) -> TimeSeriesCore:
        """Return the last `n` samples of this series."""
        if n is None:
            return self
        n = int(n)
        if n <= 0:
            return self[:0]
        return self[-n:]

    def crop(
        self,
        start: Any | None = None,
        end: Any | None = None,
        *,
        copy: bool = False,
    ) -> TimeSeriesCore:
        """Crop this series to the given GPS start and end times.

        Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).
        """
        if start is not None and _is_gwexpy_only_crop_bound(start):
            start = _crop_bound_to_axis(start, self.xunit)
        if end is not None and _is_gwexpy_only_crop_bound(end):
            end = _crop_bound_to_axis(end, self.xunit)
        return super().crop(start=start, end=end, copy=copy)

    def append(
        self,
        other: TimeSeries | BaseTimeSeries | ArrayLike,
        *,
        inplace: bool = True,
        gap: Any = None,
        pad: Any = None,
        resize: bool = True,
    ) -> TimeSeriesCore:
        """Append another `TimeSeries`, returning a GWexpy `TimeSeries`."""
        exact_t0_ns = getattr(self, "_gwex_t0_gps_ns", None)
        exact_dt_ns = getattr(self, "_gwex_dt_gps_ns", None)
        preserve_buffer_epoch = (
            not resize and exact_t0_ns is not None and exact_dt_ns is not None
        )
        token = None
        if preserve_buffer_epoch:
            token = _EXACT_BUFFER_APPEND_DEPTH.set(_EXACT_BUFFER_APPEND_DEPTH.get() + 1)
        try:
            res = super().append(
                other,
                inplace=inplace,
                pad=pad,
                gap=gap,
                resize=resize,
            )
        finally:
            if token is not None:
                _EXACT_BUFFER_APPEND_DEPTH.reset(token)

        if inplace:
            result = self
        elif isinstance(res, self.__class__):
            result = res
        else:
            result = self.__class__(
                res.value,
                times=res.times,
                unit=res.unit,
                name=res.name,
                channel=getattr(res, "channel", None),
            )

        if preserve_buffer_epoch:
            assert exact_t0_ns is not None
            assert exact_dt_ns is not None
            current_t0_ns = getattr(result, "_gwex_t0_gps_ns", exact_t0_ns)
            appended_rows = np.shape(other)[0]
            result._gwex_t0_gps_ns = current_t0_ns + appended_rows * exact_dt_ns
            result._gwex_dt_gps_ns = exact_dt_ns
        return result

    # find_peaks is inherited from SignalAnalysisMixin in the final TimeSeries class
