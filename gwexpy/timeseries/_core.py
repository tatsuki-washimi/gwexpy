"""Core TimeSeries class definition and basic operations.

This module contains the base TimeSeries class with essential functionality:
- Basic operations (tail, crop, append)
- Regularity checking (is_regular, _check_regular)
- Peak finding
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

try:
    from typing import TypeAlias
except ImportError:
    from typing import TypeAlias

import numpy as np
from astropy import units as u
from gwpy.timeseries import TimeSeries as BaseTimeSeries
from numpy.typing import ArrayLike

from gwexpy.types.mixin import RegularityMixin

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


def _regular_crop_slice(
    start: float | None, end: float | None, *, t0: float, dt: float, size: int
) -> slice:
    """Return the fail-closed positional crop slice for a regular sample axis."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"regular crop requires a positive finite dt, got {dt!r}")

    def _index(bound: float | None, default: int) -> int:
        if bound is None:
            return default
        position = (bound - t0) / dt
        nearest = round(position)
        cancellation = (
            4 * np.spacing(max(1.0, abs(t0), abs(bound))) / abs(dt)
        )
        index = nearest if abs(position - nearest) <= cancellation else np.floor(position)
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
        self, start: Any | None = None, end: Any | None = None, copy: bool = False
    ) -> TimeSeriesCore:
        """Crop this series to the given GPS start and end times.

        Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).
        """
        start = _crop_bound_to_float(start)
        end = _crop_bound_to_float(end)
        try:
            dt_quantity = self.dt
            t0_quantity = self.t0
            dt = float(dt_quantity.value)
            t0 = float(t0_quantity.value)
            sample_slice = _regular_crop_slice(
                start, end, t0=t0, dt=dt, size=len(self)
            )
            result = self[sample_slice]
        except (AttributeError, TypeError, u.UnitConversionError):
            # Preserve GWpy's irregular-axis behavior; the positional contract
            # applies only to regular time series.
            result = super().crop(start=start, end=end, copy=copy)
        else:
            if copy:
                result = result.copy()
            # A TimeSeries slice derives these values from its xindex.  Retain
            # the source dt and calculate t0 from an integer sample index so a
            # materialized large-GPS index cannot move the selected epoch.
            result._dx = dt_quantity.copy()
            result._x0 = u.Quantity(
                float(t0 + sample_slice.start * dt), t0_quantity.unit
            )
        return result

    def append(
        self,
        other: TimeSeries | BaseTimeSeries | ArrayLike,
        *,
        inplace: bool = True,
        pad: Any = None,
        gap: Any = None,
        resize: bool = True,
    ) -> TimeSeriesCore:
        """Append another `TimeSeries`, returning a GWexpy `TimeSeries`."""
        res = super().append(other, inplace=inplace, pad=pad, gap=gap, resize=resize)
        if inplace:
            return self
        if isinstance(res, self.__class__):
            return res
        return self.__class__(
            res.value,
            times=res.times,
            unit=res.unit,
            name=res.name,
            channel=getattr(res, "channel", None),
        )

    # find_peaks is inherited from SignalAnalysisMixin in the final TimeSeries class
