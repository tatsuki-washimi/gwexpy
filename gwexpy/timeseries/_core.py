"""
Core TimeSeries class definition and basic operations.

This module contains the base TimeSeries class with essential functionality:
- Basic operations (tail, crop, append)
- Regularity checking (is_regular, _check_regular)
- Peak finding
"""

from __future__ import annotations

from collections.abc import Iterable
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

if TYPE_CHECKING:
    from gwexpy.timeseries.timeseries import TimeSeries


class TimeSeriesCore(RegularityMixin, BaseTimeSeries):
    """
    Core Ti meSeries class with basic operations.

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
        """
        Crop this series to the given GPS start and end times.
        Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).
        """
        from gwexpy.time import to_gps

        # Convert inputs to GPS if provided
        if start is not None:
            start = to_gps(start)
            if isinstance(start, (np.ndarray, list)) and np.ndim(start) > 0:
                start = start[0]
            start = float(start)
        if end is not None:
            end = to_gps(end)
            if isinstance(end, (np.ndarray, list)) and np.ndim(end) > 0:
                end = end[0]
            end = float(end)

        return super().crop(start=start, end=end, copy=copy)

    def append(
        self,
        other: TimeSeries | BaseTimeSeries | ArrayLike,
        *,
        inplace: bool = True,
        pad: Any = None,
        gap: Any = None,
        resize: bool = True,
    ) -> TimeSeriesCore:
        """
        Append another TimeSeries (GWpy-compatible), returning gwexpy TimeSeries.
        """
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
