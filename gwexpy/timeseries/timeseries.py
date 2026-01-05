"""
gwexpy.timeseries.timeseries - Extended TimeSeries class

This module provides the main TimeSeries class for gwexpy, which extends
gwpy's TimeSeries with additional functionality.

The implementation is modularized across several files:
- _core.py: Core class definition and basic operations
- _spectral.py: Spectral transforms (FFT, CWT, etc.)
- _timeseries_legacy.py: Remaining methods (signal, resampling, analysis, interop)

This module integrates all Mixins into a single TimeSeries class.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from typing import Optional, Any

# Import Mixins
from ._spectral import TimeSeriesSpectralMixin
from ._signal import TimeSeriesSignalMixin
from ._resampling import TimeSeriesResamplingMixin
from ._analysis import TimeSeriesAnalysisMixin
from ._interop import TimeSeriesInteropMixin
from ._statistics import StatisticsMixin
from gwexpy.fitting.mixin import FittingMixin
from gwexpy.types.mixin import (
    RegularityMixin,
    PhaseMethodsMixin,
    SignalAnalysisMixin
)

# Import legacy for remaining methods
from ._timeseries_legacy import TimeSeries as _LegacyTimeSeries


class TimeSeries(
    TimeSeriesInteropMixin,    # Interoperability (highest priority)
    TimeSeriesAnalysisMixin,   # Analysis
    TimeSeriesResamplingMixin, # Resampling
    TimeSeriesSignalMixin,     # Signal processing
    SignalAnalysisMixin,       # Generic Signal Analysis (smooth, find_peaks)
    TimeSeriesSpectralMixin,   # Spectral transforms
    StatisticsMixin,           # Statistical analysis & correlation
    FittingMixin,              # Fitting functionality
    PhaseMethodsMixin,         # Phase/Angle methods (radian, degree, phase, angle)
    RegularityMixin,           # Regularity checking (is_regular, _check_regular)
    _LegacyTimeSeries,         # All legacy methods including BaseTimeSeries
):




    """
    Extended TimeSeries with all gwexpy functionality.

    This class combines functionality from multiple modules:
    - Core operations: is_regular, _check_regular, tail, crop, append, find_peaks
    - Spectral transforms: fft, psd, cwt, laplace, etc.
    - Signal processing: analytic_signal, mix_down, xcorr, etc.
    - Analysis: impute, standardize, rolling_*, etc.
    - Interoperability: to_pandas, to_torch, to_xarray, etc.

    Inherits from gwpy.timeseries.TimeSeries for full compatibility.
    """

    def _get_meta_for_constructor(self):
        """Helper for SignalAnalysisMixin to reconstruct object."""
        return {
            "t0": self.t0,
            "dt": self.dt,
        }

    def plot(self, **kwargs: Any):
        """Plot this TimeSeries. Delegates to gwexpy.plot.Plot."""
        from gwexpy.plot import Plot
        return Plot(self, **kwargs)

    def __new__(cls, data, *args, **kwargs):
        from gwexpy.timeseries.utils import _coerce_t0_gps

        should_coerce = True
        xunit = kwargs.get("xunit", None)
        if xunit is not None:
            try:
                should_coerce = u.Unit(xunit).is_equivalent(u.s)
            except (ValueError, TypeError):
                should_coerce = False
        else:
            dt = kwargs.get("dt", None)
            if isinstance(dt, u.Quantity):
                phys = getattr(dt.unit, "physical_type", None)
                if dt.unit != u.dimensionless_unscaled and phys != "time":
                    should_coerce = False

        if should_coerce:
            if "t0" in kwargs and kwargs["t0"] is not None:
                kwargs["t0"] = _coerce_t0_gps(kwargs["t0"])
            if "epoch" in kwargs and kwargs["epoch"] is not None:
                kwargs["epoch"] = _coerce_t0_gps(kwargs["epoch"])
        return super().__new__(cls, data, *args, **kwargs)

    # ===============================
    # Override methods from _core.py
    # (These take precedence over _LegacyTimeSeries versions)
    # ===============================

    def tail(self, n: int = 5) -> "TimeSeries":
        """Return the last `n` samples of this series."""
        if n is None:
            return self
        n = int(n)
        if n <= 0:
            return self[:0]
        return self[-n:]

    def crop(self, start: Any = None, end: Any = None, copy: bool = False) -> "TimeSeries":
        """
        Crop this series to the given GPS start and end times.
        Accepts any time format supported by gwexpy.time.to_gps.
        """
        from gwexpy.time import to_gps
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

        from gwpy.timeseries import TimeSeries as BaseTimeSeries
        return BaseTimeSeries.crop(self, start=start, end=end, copy=copy)

    def append(
        self,
        other: Any,
        inplace: bool = True,
        pad: Any = None,
        gap: Any = None,
        resize: bool = True,
    ) -> "TimeSeries":
        """
        Append another TimeSeries (GWpy-compatible), returning gwexpy TimeSeries.
        """
        from gwpy.timeseries import TimeSeries as BaseTimeSeries
        res = BaseTimeSeries.append(self, other, inplace=inplace, pad=pad, gap=gap, resize=resize)
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
        height: Any = None,
        threshold: Any = None,
        distance: Any = None,
        prominence: Any = None,
        width: Any = None,
        wlen: Optional[int] = None,
        rel_height: float = 0.5,
        plateau_size: Any = None,
    ) -> tuple["TimeSeries", dict[str, Any]]:
        """
        Find peaks in the TimeSeries.

        Wraps `scipy.signal.find_peaks`.

        Returns
        -------
        peaks : `TimeSeries`
            A new TimeSeries containing only the peak values, indexed by their times.
        props : `dict`
            Dictionary of peak properties returned by `scipy.signal.find_peaks`.
        """
        from scipy.signal import find_peaks

        val = self.value

        def _to_val(x, unit=None):
             if hasattr(x, "value"):
                  if unit and hasattr(x, "to"):
                      return x.to(unit).value
                  return x.value
             return x

        h = _to_val(height, self.unit)
        t = _to_val(threshold, self.unit)
        p = _to_val(prominence, self.unit)

        dist = distance
        wid = width

        if self.dt is not None:
             fs = self.sample_rate.to("Hz").value
             if hasattr(dist, "to"):
                  dist = int(dist.to("s").value * fs)

             if np.iterable(wid):
                  new_wid = []
                  for w in wid:
                       if hasattr(w, "to"):
                            new_wid.append(w.to("s").value * fs)
                       else:
                            new_wid.append(w)
                  wid = tuple(new_wid) if isinstance(wid, tuple) else new_wid
             elif hasattr(wid, "to"):
                  wid = wid.to("s").value * fs

        peaks_indices, props = find_peaks(
             val,
             height=h,
             threshold=t,
             distance=dist,
             prominence=p,
             width=wid,
             wlen=wlen,
             rel_height=rel_height,
             plateau_size=plateau_size
        )

        if len(peaks_indices) == 0:
             return self.__class__([], times=[], unit=self.unit, name=self.name, channel=self.channel), props

        peak_times = self.times[peaks_indices]
        peak_vals = val[peaks_indices]

        out = self.__class__(
             peak_vals,
             times=peak_times,
             unit=self.unit,
             name=f"{self.name}_peaks" if self.name else "peaks",
             channel=self.channel
        )
        return out, props


    def to_simpeg(self, location=None, rx_type="PointElectricField", orientation='x', **kwargs) -> Any:
        """
        Convert to SimPEG Data object.

        Parameters
        ----------
        location : array_like, optional
            Rx location (x, y, z). Default is [0, 0, 0].
        rx_type : str, optional
            Receiver class name. Default "PointElectricField".
        orientation : str, optional
            Receiver orientation ('x', 'y', 'z'). Default 'x'.

        Returns
        -------
        simpeg.data.Data
        """
        from gwexpy.interop import to_simpeg
        return to_simpeg(self, location=location, rx_type=rx_type, orientation=orientation, **kwargs)

    @classmethod
    def from_simpeg(cls, data_obj: Any, **kwargs: Any) -> Any:
        """
        Create TimeSeries from SimPEG Data object.

        Parameters
        ----------
        data_obj : simpeg.data.Data
            Input SimPEG Data.

        Returns
        -------
        TimeSeries
        """
        from gwexpy.interop import from_simpeg
        return from_simpeg(cls, data_obj, **kwargs)

    @classmethod
    def from_control(cls, response: Any, **kwargs) -> Any:
        """
        Create TimeSeries from python-control TimeResponseData.

        Parameters
        ----------
        response : control.TimeResponseData
            The simulation result from python-control.
        **kwargs : dict
            Additional arguments passed to the constructor.

        Returns
        -------
        TimeSeries or TimeSeriesDict
            The converted time-domain data.
        """
        from gwexpy.interop import from_control_response
        return from_control_response(cls, response, **kwargs)

    # =========================================================================
    # ARIMA / Modeling Methods
    # =========================================================================

    def arima(
        self,
        order: tuple = (1, 0, 0),
        *,
        seasonal_order: Optional[tuple] = None,
        auto: bool = False,
        **kwargs
    ):
        """
        Fit an ARIMA or SARIMAX model to this TimeSeries.

        This method wraps `statsmodels.tsa.arima.model.ARIMA` (or SARIMAX).
        If `auto=True`, it uses `pmdarima` to automatically find the best parameters.

        Parameters
        ----------
        order : tuple, default=(1, 0, 0)
            The (p,d,q) order of the model.
        seasonal_order : tuple, optional
            The (P,D,Q,s) seasonal order.
        auto : bool, default=False
            If True, perform Auto-ARIMA search (requires pmdarima).
        **kwargs
            Additional arguments passed to `fit_arima`.

        Returns
        -------
        ArimaResult
            Object containing the fitted model, with methods .predict(), .forecast(), .plot().
        """
        from .arima import fit_arima
        return fit_arima(self, order=order, seasonal_order=seasonal_order, auto=auto, **kwargs)

    def ar(self, p: int = 1, **kwargs):
        """
        Fit an AutoRegressive AR(p) model.
        Shortcut for .arima(order=(p, 0, 0)).
        """
        return self.arima(order=(p, 0, 0), **kwargs)

    def ma(self, q: int = 1, **kwargs):
        """
        Fit a Moving Average MA(q) model.
        Shortcut for .arima(order=(0, 0, q)).
        """
        return self.arima(order=(0, 0, q), **kwargs)

    def arma(self, p: int = 1, q: int = 1, **kwargs):
        """
        Fit an ARMA(p, q) model.
        Shortcut for .arima(order=(p, 0, q)).
        """
        return self.arima(order=(p, 0, q), **kwargs)

__all__ = ["TimeSeries"]
