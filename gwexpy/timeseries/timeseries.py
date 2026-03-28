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

from typing import TYPE_CHECKING, Any, SupportsIndex, cast

import numpy as np
from astropy import units as u
from numpy.typing import ArrayLike

from gwexpy.fitting.mixin import FittingMixin
from gwexpy.types.mixin import PhaseMethodsMixin, RegularityMixin, SignalAnalysisMixin
from gwexpy.types.mixin._plot_mixin import PlotMixin

from ._analysis import TimeSeriesAnalysisMixin
from ._interop import TimeSeriesInteropMixin
from ._resampling import TimeSeriesResamplingMixin
from ._signal import TimeSeriesSignalMixin

# Import Core Base
from ._core import TimeSeriesCore

# Import Mixins
from ._spectral import TimeSeriesSpectralMixin
from ._statistics import StatisticsMixin

# Import legacy for remaining methods
from ._timeseries_legacy import TimeSeries as _LegacyTimeSeries

if TYPE_CHECKING:
    from gwpy.timeseries import TimeSeries as GwpyTimeSeries

    from gwexpy.timeseries import TimeSeriesDict


class TimeSeries(
    PlotMixin,  # Centralized plot() via deferred import
    TimeSeriesInteropMixin,  # Interoperability (highest priority)
    TimeSeriesAnalysisMixin,  # Analysis
    TimeSeriesResamplingMixin,  # Resampling
    TimeSeriesSignalMixin,  # Signal processing
    SignalAnalysisMixin,  # Generic Signal Analysis (smooth, find_peaks)
    TimeSeriesSpectralMixin,  # Spectral transforms
    StatisticsMixin,  # Statistical analysis & correlation
    FittingMixin,  # Fitting functionality
    PhaseMethodsMixin,  # Phase/Angle methods (radian, degree, phase, angle)
    TimeSeriesCore,  # Core operations (tail, crop, append, find_peaks, RegularityMixin)
):
    """
    Extended TimeSeries with all gwexpy functionality.

    This class combines functionality from multiple modules:
    - Core operations: is_regular, _check_regular, tail, crop, append, find_peaks
    - Spectral transforms: fft, psd, cwt, laplace, etc.
    - Signal processing: hilbert, mix_down, xcorr, etc.
    - Analysis: impute, standardize, rolling_*, etc.
    - Interoperability: to_pandas, to_torch, to_xarray, etc.

    Inherits from gwpy.timeseries.TimeSeries for full compatibility.
    """

    def _get_meta_for_constructor(self) -> dict[str, Any]:
        """Helper for SignalAnalysisMixin to reconstruct object."""
        return {
            "t0": self.t0,
            "dt": self.dt,
        }

    def __new__(cls, data: ArrayLike, *args: Any, **kwargs: Any) -> TimeSeries:
        """
        Create a new TimeSeries.

        This constructor extends the standard gwpy.timeseries.TimeSeries constructor
        by adding support for automatic GPS time coercion for 't0' and 'epoch' parameters.

        Parameters
        ----------
        data : array_like
            The data values for the series.
        *args
            Additional positional arguments passed to the parent constructor.
        **kwargs
            Additional keyword arguments passed to the parent constructor.
        """
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
            # Determine target unit for t0/epoch normalization
            target_unit = u.s
            dt = kwargs.get("dt")
            if isinstance(dt, u.Quantity):
                target_unit = dt.unit
            else:
                xunit = kwargs.get("xunit")
                if xunit is not None:
                    try:
                        target_unit = u.Unit(xunit)
                    except (ValueError, TypeError):
                        pass

            if "t0" in kwargs and kwargs["t0"] is not None:
                t0_q = _coerce_t0_gps(kwargs["t0"])
                if t0_q is not None:
                    try:
                        # For GWpy 4.0 compatibility: convert to float value in target_unit.
                        # Using a Quantity with a different unit than the axis (dt)
                        # can trigger incorrect internal conversions in some GWpy versions.
                        kwargs["t0"] = float(t0_q.to(target_unit).value)
                    except (u.UnitConversionError, AttributeError, TypeError):
                        kwargs["t0"] = t0_q

            if "epoch" in kwargs and kwargs["epoch"] is not None:
                epoch_q = _coerce_t0_gps(kwargs["epoch"])
                if epoch_q is not None:
                    try:
                        kwargs["epoch"] = float(epoch_q.to(target_unit).value)
                    except (u.UnitConversionError, AttributeError, TypeError):
                        kwargs["epoch"] = epoch_q
        return super().__new__(cls, data, *args, **kwargs)

    def __array_finalize__(self, obj: Any) -> None:
        """
        Finalize the array after creation (slicing, view casting).

        Ensures that attributes starting with `_gwex_` are propagated
        from the parent object to the new view/instance.
        """
        super().__array_finalize__(obj)
        if obj is None:
            return

        # Propagate custom _gwex_ attributes
        for key, val in getattr(obj, "__dict__", {}).items():
            if key.startswith("_gwex_") and key not in self.__dict__:
                self.__dict__[key] = val

    def __reduce_ex__(self, protocol: SupportsIndex):
        from gwexpy.io.pickle_compat import timeseries_reduce_args

        return timeseries_reduce_args(self)


    # Basic operations (tail, crop, append, find_peaks) are inherited from TimeSeriesCore

    def to_simpeg(
        self,
        location: ArrayLike | None = None,
        rx_type: str = "PointElectricField",
        orientation: str = "x",
        **kwargs: Any,
    ) -> Any:
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

        return to_simpeg(
            self, location=location, rx_type=rx_type, orientation=orientation, **kwargs
        )

    @classmethod
    def from_simpeg(cls, data_obj: Any, **kwargs: Any) -> TimeSeries:
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
    def from_control(cls, response: Any, **kwargs: Any) -> TimeSeries | TimeSeriesDict:
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
        order: tuple[int, int, int] = (1, 0, 0),
        *,
        seasonal_order: tuple[int, int, int, int] | None = None,
        auto: bool = False,
        **kwargs: Any,
    ) -> Any:
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

        return fit_arima(
            self, order=order, seasonal_order=seasonal_order, auto=auto, **kwargs
        )

    def ar(self, p: int = 1, **kwargs: Any) -> Any:
        """
        Fit an AutoRegressive AR(p) model.
        Shortcut for .arima(order=(p, 0, 0)).
        """
        return self.arima(order=(p, 0, 0), **kwargs)

    def ma(self, q: int = 1, **kwargs: Any) -> Any:
        """
        Fit a Moving Average MA(q) model.
        Shortcut for .arima(order=(0, 0, q)).
        """
        return self.arima(order=(0, 0, q), **kwargs)

    def arma(self, p: int = 1, q: int = 1, **kwargs: Any) -> Any:
        """
        Fit an ARMA(p, q) model.
        Shortcut for .arima(order=(p, 0, q)).
        """
        return self.arima(order=(p, 0, q), **kwargs)


__all__ = ["TimeSeries"]
