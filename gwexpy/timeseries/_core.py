"""
Core TimeSeries class definition and basic operations.

This module contains the base TimeSeries class with essential functionality:
- Basic operations (tail, crop, append)
- Regularity checking (is_regular, _check_regular)
- Peak finding
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import numpy as np
from astropy import units as u
from gwpy.timeseries import TimeSeries as BaseTimeSeries
from numpy.typing import ArrayLike

from gwexpy.types.mixin import RegularityMixin

QuantityLike: TypeAlias = ArrayLike | u.Quantity

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

    def find_peaks(
        self,
        height: QuantityLike | None = None,
        threshold: QuantityLike | None = None,
        distance: QuantityLike | None = None,
        prominence: QuantityLike | None = None,
        width: QuantityLike | Iterable[QuantityLike] | None = None,
        wlen: int | None = None,
        rel_height: float = 0.5,
        plateau_size: QuantityLike | None = None,
    ) -> tuple[TimeSeriesCore, dict[str, Any]]:
        """
        Find peaks in the TimeSeries.

        Wraps `scipy.signal.find_peaks`.

        Returns
        -------
        peaks : TimeSeries
             A TimeSeries containing only the peak values at their corresponding times.
        properties : dict
             Properties returned by scipy.signal.find_peaks.
        """
        from scipy.signal import find_peaks

        # Handle unit quantities
        val = self.value

        def _to_val(x: Any, unit: u.UnitBase | None = None) -> Any:
            if hasattr(x, "value"):
                if unit and hasattr(x, "to"):
                    return x.to(unit).value
                return x.value
            return x

        # Height/Threshold: relative to data units
        h = _to_val(height, self.unit)
        t = _to_val(threshold, self.unit)
        p = _to_val(prominence, self.unit)  # Prominence same unit as data

        # Distance/Width: time or samples
        # Scipy uses samples.
        dist: QuantityLike | None = distance
        wid: QuantityLike | Iterable[QuantityLike] | None = width

        if self.dt is not None:
            fs = self.sample_rate.to("Hz").value
            # If distance is time quantity
            if dist is not None and hasattr(dist, "to"):
                dist = int(dist.to("s").value * fs)

            # If width is quantity (or tuple of quantities)
            if (
                wid is not None
                and isinstance(wid, Iterable)
                and not isinstance(wid, (str, bytes))
            ):
                new_wid: list[float] = []
                for w in wid:
                    if hasattr(w, "to"):
                        new_wid.append(float(w.to("s").value * fs))
                    elif hasattr(w, "value"):
                        new_wid.append(float(w.value))
                    else:
                        new_wid.append(float(w))  # type: ignore[arg-type]
                wid = tuple(new_wid) if isinstance(wid, tuple) else new_wid
            elif wid is not None and hasattr(wid, "to"):
                wid = float(wid.to("s").value * fs)

        # Call scipy
        peaks_indices, props = find_peaks(
            val,
            height=h,
            threshold=t,
            distance=dist,
            prominence=p,
            width=wid,
            wlen=wlen,
            rel_height=rel_height,
            plateau_size=plateau_size,
        )

        if len(peaks_indices) == 0:
            # Return empty
            return self.__class__(
                [], times=[], unit=self.unit, name=self.name, channel=self.channel
            ), props

        peak_times = self.times[peaks_indices]
        peak_vals = val[peaks_indices]

        out = self.__class__(
            peak_vals,
            times=peak_times,
            unit=self.unit,
            name=f"{self.name}_peaks" if self.name else "peaks",
            channel=self.channel,
        )
        return out, props
