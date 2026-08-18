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

from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, SupportsIndex

import numpy as np
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
    _GWFParallelContractError,
    _resolve_gwf_format,
    _source_for_gwf_channel_listing,
    _validate_gwf_parallel_source,
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

    t0_ns : int, optional
        Total GPS nanoseconds for an exact epoch. This is keyword-only and
        accepts values from 0 through ``2**63 - 1``.

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

        This override adds explicit CSV and `.gwf` handling for deterministic
        behavior when `.read()` is called through the public API.
        """
        from gwexpy._bootstrap import ensure_io_registered

        ensure_io_registered()
        from .collections import _parse_public_read_format

        fmt, args = _parse_public_read_format(cls, args, kwargs)
        if fmt is not None:
            kwargs["format"] = fmt
        if fmt is None:
            from .collections import _hdf5_path_sources

            if _hdf5_path_sources(source) is not None:
                raise ValueError(
                    "TimeSeries.read() requires explicit format='hdf5' for "
                    "generic HDF5 sources; TimeSeries does not auto-detect "
                    "hdf.ndscope."
                )
        source_path = Path(source) if isinstance(source, (str, Path)) else None
        if fmt in {"nc", "netcdf4"}:
            from .io.netcdf4_ import read_timeseries_netcdf4

            reader_kwargs = dict(kwargs)
            reader_kwargs.pop("format", None)
            return cls(read_timeseries_netcdf4(source, **reader_kwargs))
        if fmt == "zarr":
            if args:
                raise TypeError(
                    "TimeSeries.read(..., format='zarr') does not accept "
                    "positional reader arguments; use channels=..., start=..., "
                    "and end=... keyword arguments."
                )
            from .io.zarr_ import read_timeseries_zarr

            reader_kwargs = dict(kwargs)
            reader_kwargs.pop("format", None)
            return cls(read_timeseries_zarr(source, **reader_kwargs))
        if fmt == "csv" or (
            fmt is None
            and source_path is not None
            and source_path.suffix.lower() == ".csv"
        ):
            from .io.csv_enhanced import read_timeseries_csv

            return read_timeseries_csv(source, **kwargs)

        gwf_format = _resolve_gwf_format(source, kwargs.get("format"))
        if gwf_format is not None:
            from gwpy.io.gwf.core import get_channel_names

            channels, start, end, gwf_kwargs = _extract_gwf_read_args(
                args,
                kwargs,
                allow_multiple_channels=False,
            )
            backend = gwf_kwargs.pop("backend", _GWF_BACKENDS[gwf_format])
            _validate_gwf_parallel_source(source, gwf_kwargs)
            try:
                if channels is None:
                    channel_source = _source_for_gwf_channel_listing(source)
                    channels = get_channel_names(channel_source, backend=backend)
                    if not channels:
                        raise ValueError(f"No channels found in GWF source: {source}")
                channel = channels[0]
                from .collections import TimeSeriesDict

                tsd = TimeSeriesDict.read(
                    source,
                    [channel],
                    start=start,
                    end=end,
                    backend=backend,
                    format=gwf_format,
                    **gwf_kwargs,
                )
            except ImportError as exc:
                raise _format_gwf_import_error(gwf_format, exc)
            except _GWFParallelContractError:
                raise
            except TypeError as exc:
                raise ValueError(f"Invalid input for GWF read: {exc}") from exc
            if not tsd:
                raise ValueError(f"No data found in {gwf_format} source: {source}")
            series = next(iter(tsd.values()))
            return series if isinstance(series, cls) else cls(series)

        return super().read(source, *args, **kwargs)

    def write(self, target, *args, **kwargs):  # type: ignore[override]
        """Write a `TimeSeries` to a supported target.

        This override preserves minimal metadata for direct CSV round-trips.
        """
        from gwexpy._bootstrap import ensure_io_registered

        ensure_io_registered()
        fmt = kwargs.get("format")
        target_path = Path(target) if isinstance(target, (str, Path)) else None
        if fmt == "csv" or (
            fmt is None
            and target_path is not None
            and target_path.suffix.lower() == ".csv"
        ):
            from .io.csv_enhanced import write_timeseries_csv

            write_kwargs = dict(kwargs)
            write_kwargs.pop("format", None)
            return write_timeseries_csv(self, target, **write_kwargs)
        return super().write(target, *args, **kwargs)

    def _get_meta_for_constructor(self) -> dict[str, Any]:
        """Reconstruct the object for SignalAnalysisMixin."""
        return {
            "t0": self.t0,
            "dt": self.dt,
        }

    def __new__(
        cls, data: ArrayLike, *args: Any, t0_ns: Any = None, **kwargs: Any
    ) -> TimeSeries:
        """Create a new TimeSeries.

        This constructor extends the standard `gwpy.timeseries.TimeSeries` constructor
        by adding support for automatic GPS time coercion for `t0` and `epoch` parameters.
        """
        from gwexpy.timeseries.utils import (
            _coerce_t0_gps,
            _gps_ns_to_ligo,
            _t0_gps_ns_state,
            _validate_t0_gps_ns,
        )

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

        state_ns: int | None = None
        state_precision: str | None = None
        t0_value = kwargs.get("t0")
        epoch_value = kwargs.get("epoch")
        supplied_epoch = next(
            (value for value in (t0_value, epoch_value) if value is not None), None
        )
        if t0_ns is not None:
            state_ns = _validate_t0_gps_ns(t0_ns)
            state_precision = "exact"
            for name, value in (("t0", t0_value), ("epoch", epoch_value)):
                if value is None:
                    continue
                supplied_ns, _ = _t0_gps_ns_state(value)
                if supplied_ns != state_ns:
                    raise ValueError(f"t0_ns and {name} must agree to the nanosecond")
            kwargs["t0"] = _gps_ns_to_ligo(state_ns)
            kwargs.pop("epoch", None)
        elif should_coerce and supplied_epoch is not None:
            state_ns, state_precision = _t0_gps_ns_state(supplied_epoch)

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
        new = super().__new__(cls, data, *args, **kwargs)
        new._gwex_t0_gps_ns = state_ns
        new._gwex_t0_gps_precision = state_precision
        return new

    @property
    def t0_gps_ns(self) -> int | None:
        """Return the tracked total GPS origin in nanoseconds, if available."""
        return getattr(self, "_gwex_t0_gps_ns", None)

    @staticmethod
    def _clear_t0_gps_state(result: Any) -> None:
        if isinstance(result, TimeSeries):
            result._gwex_t0_gps_ns = None
            result._gwex_t0_gps_precision = None

    def __getitem__(self, item: Any) -> Any:
        """Index the series and conservatively propagate GPS nanosecond state."""
        result = super().__getitem__(item)
        if not isinstance(result, TimeSeries):
            return result

        if len(result) == 0:
            self._clear_t0_gps_state(result)
            return result

        if not isinstance(item, slice):
            self._clear_t0_gps_state(result)
            return result

        start, _, step = item.indices(len(self))
        if step != 1:
            self._clear_t0_gps_state(result)
            return result

        if getattr(self, "is_regular", False):
            try:
                cached_xindex = getattr(result, "_xindex", None)
                exact_gps_state = (
                    getattr(self, "_gwex_t0_gps_precision", None) == "exact"
                )
                if exact_gps_state:
                    if cached_xindex is not None:
                        result._xindex = cached_xindex.copy()
                        source_xindex = result._xindex
                        result_start = 0
                    else:
                        source_xindex = self.xindex
                        result_start = start
                    result._x0 = source_xindex[result_start].copy()
                    if len(source_xindex) > 1:
                        result._dx = source_xindex[1] - source_xindex[0]
                    else:
                        result._dx = self.dx.copy()
                else:
                    result._dx = self.dx.copy()
                    result._x0 = u.Quantity(
                        float(self.x0.value + start * self.dx.value), self.x0.unit
                    )
            except (AttributeError, TypeError, ValueError, u.UnitConversionError):
                self._clear_t0_gps_state(result)
                return result

        origin_ns = getattr(self, "_gwex_t0_gps_ns", None)
        precision = getattr(self, "_gwex_t0_gps_precision", None)
        if origin_ns is None or not getattr(self, "is_regular", False):
            self._clear_t0_gps_state(result)
            return result

        try:
            dt = self.dt.to(u.s).value
            if np.ndim(dt) != 0 or not np.isfinite(dt) or dt <= 0:
                raise ValueError
            offset_ns = Fraction(start) * Fraction(str(float(dt))) * 1_000_000_000
        except (AttributeError, TypeError, ValueError, u.UnitConversionError):
            self._clear_t0_gps_state(result)
            return result

        if offset_ns.denominator == 1:
            result._gwex_t0_gps_ns = int(origin_ns) + offset_ns.numerator
            result._gwex_t0_gps_precision = precision
        else:
            from gwexpy.timeseries.utils import _round_fraction_ties_even

            result._gwex_t0_gps_ns = int(origin_ns) + _round_fraction_ties_even(
                offset_ns
            )
            result._gwex_t0_gps_precision = "quantized"
        return result

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
