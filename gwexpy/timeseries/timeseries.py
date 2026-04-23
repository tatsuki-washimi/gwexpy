"""Extended TimeSeries class for gwexpy.

This module provides the main TimeSeries class for gwexpy, which extends
gwpy's TimeSeries with additional functionality.

The implementation is modularized across several files:
- _core.py: Core class definition and basic operations
- _spectral.py: Spectral transforms (FFT, CWT, etc.)
- _timeseries_legacy.py: Remaining methods (signal, resampling, analysis, interop)

This module integrates all Mixins into a single TimeSeries class.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsIndex

from astropy import units as u
from numpy.typing import ArrayLike

from gwexpy.fitting.mixin import FittingMixin
from gwexpy.types.mixin import PhaseMethodsMixin, SignalAnalysisMixin
from gwexpy.types.mixin._plot_mixin import PlotMixin

from ._analysis import TimeSeriesAnalysisMixin

# Import Core Base
from ._core import TimeSeriesCore
from ._gwf_io import (
    _GWF_BACKENDS,
    _extract_gwf_read_args,
    _format_gwf_import_error,
    _resolve_gwf_format,
)
from ._interop import TimeSeriesInteropMixin
from ._resampling import TimeSeriesResamplingMixin
from ._signal import TimeSeriesSignalMixin

# Import Mixins
from ._spectral import TimeSeriesSpectralMixin
from ._statistics import StatisticsMixin

# Import legacy for remaining methods

if TYPE_CHECKING:

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
    """A data array holding some metadata to represent a time-series.

    `TimeSeries` is the primary object used to represent time-domain
    data in `gwexpy`. It extends the standard `gwpy.timeseries.TimeSeries`
    by incorporating additional mixins for plotting, signal analysis,
    regularity checks, numerical fitting, statistical methods, and
    enhanced interoperability.

    Parameters
    ----------
    data : array-like
        Input data array.

    unit : `~astropy.units.Unit`, optional
        Physical unit of these data.

    t0 : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional, default: `0`
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine.

    dt : `float`, `~astropy.units.Quantity`, optional, default: `1`
        Time resolution for these data.

    sample_rate : `float`, `~astropy.units.Quantity`, optional, default: `1`
        Sample rate for these data.

    times : `array-like`
        The complete array of times indexing the data.
        This argument takes precedence over `t0` and `dt` so should
        be given in place of these if relevant, not alongside.

    name : `str`, optional
        Descriptive title for this array.

    channel : `~gwpy.detector.Channel`, `str`, optional
        Source data stream for these data.

    dtype : `~numpy.dtype`, optional
        Input data type.

    copy : `bool`, optional, default: `False`
        Choose to copy the input data to new memory.

    subok : `bool`, optional, default: `True`
        Allow passing of sub-classes by the array generator.

    Notes
    -----
    In addition to the standard GWpy functionality, this class provides
    advanced features such as time-domain differentiation/integration,
    rolling statistics, and seamless interoperability with PyTorch,
    Xarray, and Polars.

    Key methods:

    .. autosummary::

       ~TimeSeries.plot
       ~TimeSeries.resample
       ~TimeSeries.filter
       ~TimeSeries.fft
       ~TimeSeries.psd
       ~TimeSeries.spectrogram

    Examples
    --------
    >>> from gwexpy.timeseries import TimeSeries
    >>> import numpy as np
    >>> data = np.array([0.1, -1.2, 0.5])
    >>> ts = TimeSeries(data, sample_rate=1000, unit='V')
    >>> ts
    <TimeSeries([ 0.1, -1.2,  0.5],
                unit=Unit("V"),
                t0=<Quantity 0. s>,
                dt=<Quantity 0.001 s>,
                name=None,
                channel=None)>

    """

    @classmethod
    def read(cls, source, *args, **kwargs):  # type: ignore[override]
        """Read a `TimeSeries` from a supported source.

        This override adds explicit `.gwf` handling for deterministic behavior
        when `.read()` is called with positional channel selectors.
        """
        gwf_format = _resolve_gwf_format(source, kwargs.get("format"))
        if gwf_format is not None:
            from gwpy.io.gwf.core import get_channel_names
            from gwpy.timeseries.io.gwf.core import read_timeseriesdict

            from gwexpy.interop._registry import ConverterRegistry

            channels, start, end, gwf_kwargs = _extract_gwf_read_args(
                args,
                kwargs,
                allow_multiple_channels=False,
            )
            backend = gwf_kwargs.pop("backend", _GWF_BACKENDS[gwf_format])
            try:
                if channels is None:
                    channels = get_channel_names(source, backend=backend)
                    if not channels:
                        raise ValueError(f"No channels found in GWF source: {source}")
                channel = channels[0]
                tsd = read_timeseriesdict(
                    source,
                    [channel],
                    start=start,
                    end=end,
                    backend=backend,
                    series_class=ConverterRegistry.get_constructor("TimeSeries"),
                    **gwf_kwargs,
                )
            except ImportError as exc:
                raise _format_gwf_import_error(gwf_format, exc)
            except TypeError as exc:
                raise ValueError(f"Invalid input for GWF read: {exc}") from exc
            if not tsd:
                raise ValueError(f"No data found in {gwf_format} source: {source}")
            return cls(next(iter(tsd.values())))

        return super().read(source, *args, **kwargs)

    def _get_meta_for_constructor(self) -> dict[str, Any]:
        """Reconstruct the object for SignalAnalysisMixin."""
        return {
            "t0": self.t0,
            "dt": self.dt,
        }

    def __new__(cls, data: ArrayLike, *args: Any, **kwargs: Any) -> TimeSeries:
        """Create a new TimeSeries.

        This constructor extends the standard `gwpy.timeseries.TimeSeries` constructor
        by adding support for automatic GPS time coercion for `t0` and `epoch` parameters.
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
        """Finalize the array after creation (slicing, view casting).

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
        """Provide pickle serialization support."""
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
        """Convert to SimPEG Data object.

        Parameters
        ----------
        location : array_like, optional
            Rx location (x, y, z). Default is [0, 0, 0].
        rx_type : str, optional
            Receiver class name. Default "PointElectricField".
        orientation : str, optional
            Receiver orientation ('x', 'y', 'z'). Default 'x'.
        **kwargs : Any
            Additional arguments passed to SimPEG converter.

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
        """Create TimeSeries from SimPEG Data object.

        Parameters
        ----------
        data_obj : simpeg.data.Data
            Input SimPEG Data.
        **kwargs : Any
            Additional arguments passed to constructor.

        Returns
        -------
        TimeSeries

        """
        from gwexpy.interop import from_simpeg

        return from_simpeg(cls, data_obj, **kwargs)

    @classmethod
    def from_control(cls, response: Any, **kwargs: Any) -> TimeSeries | TimeSeriesDict:
        """Create TimeSeries from python-control TimeResponseData.

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
        """Fit an ARIMA or SARIMAX model to this TimeSeries.

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
        """Fit an AutoRegressive AR(p) model.

        Shortcut for .arima(order=(p, 0, 0)).
        """
        return self.arima(order=(p, 0, 0), **kwargs)

    def ma(self, q: int = 1, **kwargs: Any) -> Any:
        """Fit a Moving Average MA(q) model.

        Shortcut for .arima(order=(0, 0, q)).
        """
        return self.arima(order=(0, 0, q), **kwargs)

    def arma(self, p: int = 1, q: int = 1, **kwargs: Any) -> Any:
        """Fit an ARMA(p, q) model.

        Shortcut for .arima(order=(p, 0, q)).
        """
        return self.arima(order=(p, 0, q), **kwargs)


__all__ = ["TimeSeries"]
