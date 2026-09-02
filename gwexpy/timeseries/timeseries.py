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

from contextvars import ContextVar
from datetime import date
from operator import index
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, SupportsIndex, cast

import numpy as np
from astropy import units as u
from numpy.typing import ArrayLike

from gwexpy.fitting.mixin import FittingMixin
from gwexpy.types.mixin import PhaseMethodsMixin, SignalAnalysisMixin
from gwexpy.types.mixin._plot_mixin import PlotMixin

from ._analysis import TimeSeriesAnalysisMixin

# Import Core Base
from ._core import TimeSeriesCore
from ._epoch import _integer_gps_ns, _integral_dt_gps_ns
from ._gwf_io import (
    _GWF_BACKENDS,
    _GWF_PARALLEL_HELP,
    _copy_gwf_custom_attributes,
    _extract_gwf_read_args,
    _format_gwf_import_error,
    _gwf_parallel_read_signature,
    _GWFParallelContractError,
    _normalize_gwf_parallel_kwargs,
    _prepare_gwf_parallel_source,
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


_SUPPRESS_EXACT_FINALIZE_FROM: ContextVar[frozenset[int]] = ContextVar(
    "_SUPPRESS_EXACT_FINALIZE_FROM", default=frozenset()
)
_EXACT_STATE_KEYS = frozenset({"_gwex_t0_gps_ns", "_gwex_dt_gps_ns"})


def _is_gwexpy_only_epoch(value: object) -> bool:
    """Return whether ``value`` needs GWexpy's explicit time normalizer."""
    # ``datetime`` deliberately arrives through its ``date`` base class.
    # NumPy ``datetime64`` is excluded: GWpy 4 accepts it (despite producing an
    # unusual axis), so default construction must pass it through unchanged.
    if isinstance(value, date):
        return True
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return True
    if isinstance(value, (tuple, list)) and 3 <= len(value) <= 7:
        return all(
            isinstance(item, (int, float, np.number))
            and not isinstance(item, (bool, np.bool_))
            for item in value
        )
    return False


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

    t0_ns : `int`, optional, keyword-only
        Exact GPS epoch in integer nanoseconds. This is the authoritative
        epoch representation for :attr:`t0_gps_ns`; it cannot be combined
        with ``t0``, ``epoch``, ``x0``, ``xindex``, or ``times``.

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

        GWF accepts the compatible ``parallel=`` and ``nproc=`` keywords; see
        the generated signature and the GWF I/O guide for their constraints.
        """
        from gwexpy._bootstrap import register_all

        register_all()

        fmt = kwargs.get("format")
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

        source, gwf_format = _prepare_gwf_parallel_source(
            source, kwargs.get("format"), kwargs
        )
        if gwf_format is not None:
            channels, start, end, gwf_kwargs = _extract_gwf_read_args(
                args,
                kwargs,
                allow_multiple_channels=False,
            )
            source = _validate_gwf_parallel_source(source, gwf_kwargs)
            _, parallel_workers = _normalize_gwf_parallel_kwargs(
                dict(gwf_kwargs),
                number_of_spans=len(source) if isinstance(source, (list, tuple)) else 1,
            )
            backend = gwf_kwargs.pop("backend", _GWF_BACKENDS[gwf_format])
            try:
                from gwpy.io.gwf.core import get_channel_names

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
                if parallel_workers > 1:
                    # A multi-worker backend exception has already crossed the
                    # parent-owned executor boundary.  Keep its public type,
                    # args, and provenance intact rather than treating it as a
                    # serial optional-dependency discovery failure.
                    raise
                raise _format_gwf_import_error(gwf_format, exc)
            except _GWFParallelContractError:
                raise
            except TypeError as exc:
                if parallel_workers > 1:
                    raise
                raise ValueError(f"Invalid input for GWF read: {exc}") from exc
            if not tsd:
                raise ValueError(f"No data found in {gwf_format} source: {source}")
            series = next(iter(tsd.values()))
            result = cls(series)
            _copy_gwf_custom_attributes(series, result, only_missing=False)
            return result

        return super().read(source, *args, **kwargs)

    def write(self, target, *args, **kwargs):  # type: ignore[override]
        """Write a `TimeSeries` to a supported target.

        This override preserves minimal metadata for direct CSV round-trips.
        """
        from gwexpy._bootstrap import register_all

        register_all()

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
        exact_t0_ns = getattr(self, "_gwex_t0_gps_ns", None)
        if exact_t0_ns is not None:
            return {
                "t0_ns": exact_t0_ns,
                "dt": self.dt,
            }
        return {
            "t0": self.t0,
            "dt": self.dt,
        }

    def __new__(
        cls, data: ArrayLike, *args: Any, t0_ns: int | None = None, **kwargs: Any
    ) -> TimeSeries:
        """Create a new TimeSeries.

        This constructor extends the standard `gwpy.timeseries.TimeSeries` constructor
        by adding support for automatic GPS time coercion for ``t0`` and
        ``epoch`` parameters, plus an exact integer-nanosecond ``t0_ns``
        authority.
        """
        from gwexpy.timeseries.utils import _coerce_t0_gps

        exact_t0_ns: int | None = None
        if t0_ns is not None:
            if isinstance(t0_ns, (bool, np.bool_)):
                raise TypeError("t0_ns must be an integer number of GPS nanoseconds")
            try:
                exact_t0_ns = index(t0_ns)
            except TypeError as exc:
                raise TypeError(
                    "t0_ns must be an integer number of GPS nanoseconds"
                ) from exc

            # ``t0`` can be positional in GWpy's constructor, whose first two
            # positional parameters after data are ``unit`` and ``t0``.
            # A second positional argument is therefore a competing epoch
            # authority.  Do not try to compare a float/time-like value with
            # an exact integer: callers must select one authority explicitly.
            conflicting = {"t0", "epoch", "x0", "xindex", "times"}.intersection(kwargs)
            if len(args) >= 2 or conflicting:
                names = ", ".join(sorted(conflicting))
                if len(args) >= 2:
                    names = ", ".join(filter(None, (names, "positional t0")))
                raise TypeError(
                    f"t0_ns cannot be combined with another epoch authority ({names})"
                )

        positional_dt = args[2] if len(args) >= 3 else None
        dt = positional_dt if len(args) >= 3 else kwargs.get("dt")
        should_coerce = True
        xunit = kwargs.get("xunit", None)
        if xunit is not None:
            try:
                should_coerce = u.Unit(xunit).is_equivalent(u.s)
            except (ValueError, TypeError):
                should_coerce = False
        else:
            if isinstance(dt, u.Quantity):
                phys = getattr(dt.unit, "physical_type", None)
                if dt.unit != u.dimensionless_unscaled and phys != "time":
                    should_coerce = False

        def _target_axis_unit() -> u.UnitBase:
            explicit_xunit: u.UnitBase | None = None
            if xunit is not None:
                try:
                    explicit_xunit = u.Unit(xunit)
                except (ValueError, TypeError):
                    pass
                else:
                    if explicit_xunit.is_equivalent(u.s):
                        return explicit_xunit
            if isinstance(dt, u.Quantity):
                return dt.unit
            if explicit_xunit is not None:
                return explicit_xunit
            return u.s

        if should_coerce:
            # Determine target unit for t0/epoch normalization
            target_unit = _target_axis_unit()

            def normalize_epoch(value: object) -> object:
                if isinstance(value, (tuple, list)):
                    from gwpy.time import to_gps as gwpy_to_gps

                    epoch_q = u.Quantity(float(gwpy_to_gps(value)), u.s)
                else:
                    epoch_q = _coerce_t0_gps(value)
                if epoch_q is None:
                    return value
                try:
                    # GWpy stores a bare x0 in the axis unit selected by dt/xunit.
                    return float(epoch_q.to(target_unit).value)
                except (u.UnitConversionError, AttributeError, TypeError):
                    return epoch_q

            positional_t0 = args[1] if len(args) >= 2 else None
            keyword_t0 = kwargs.get("t0")
            duplicate_t0 = len(args) >= 2 and "t0" in kwargs
            effective_t0 = positional_t0 if len(args) >= 2 else keyword_t0
            epoch = kwargs.get("epoch")

            # Let GWpy own duplicate/conflicting-argument failures unchanged.
            if not duplicate_t0 and not (
                effective_t0 is not None and epoch is not None
            ):
                if effective_t0 is not None and _is_gwexpy_only_epoch(effective_t0):
                    normalized = normalize_epoch(effective_t0)
                    if len(args) >= 2:
                        parent_args = list(args)
                        parent_args[1] = normalized
                        args = tuple(parent_args)
                    else:
                        kwargs["t0"] = normalized
                elif epoch is not None and _is_gwexpy_only_epoch(epoch):
                    kwargs["epoch"] = normalize_epoch(epoch)
        if exact_t0_ns is not None:
            # GWpy's public axis is float/Quantity based.  Retain that view
            # for compatibility, while keeping the supplied integer as the
            # only exact authority.  Normalise into the actual axis unit so
            # GWpy does not reinterpret seconds as (for example) nanoseconds.
            target_unit = _target_axis_unit()
            kwargs["t0"] = float(u.Quantity(exact_t0_ns, u.ns).to_value(target_unit))

        new = super().__new__(cls, data, *args, **kwargs)
        if exact_t0_ns is not None:
            new._gwex_t0_gps_ns = exact_t0_ns
            try:
                new._gwex_dt_gps_ns = _integral_dt_gps_ns(new.dt)
            except (TypeError, ValueError):
                # Construction remains compatible for non-integral sampling
                # periods; operations requiring an exact derived epoch fail
                # closed when they encounter one.
                pass
        return new

    @property
    def t0_gps_ns(self) -> int:
        """GPS epoch as exact integer nanoseconds.

        Objects constructed with ``t0_ns=`` return the original integer
        authority without a float conversion.  Legacy objects constructed
        with GWpy's float-compatible epoch inputs retain their historical
        behaviour and are normalised through ``LIGOTimeGPS`` on demand.
        """
        exact = getattr(self, "_gwex_t0_gps_ns", None)
        if exact is not None:
            return int(exact)

        from gwpy.time import LIGOTimeGPS

        value = self.t0.to_value(u.s) if hasattr(self.t0, "to_value") else self.t0
        return LIGOTimeGPS(float(value)).ns()

    @property
    def t0(self) -> Any:
        """GWpy-compatible epoch view, synchronized with exact metadata."""
        from gwpy.timeseries import TimeSeries as BaseTimeSeries

        return BaseTimeSeries.t0.__get__(self, type(self))

    @t0.setter
    def t0(self, value: Any) -> None:
        self._set_exact_epoch(value, alias="t0")

    @t0.deleter
    def t0(self) -> None:
        from gwpy.timeseries import TimeSeries as BaseTimeSeries

        BaseTimeSeries.t0.__delete__(self)
        self.__dict__.pop("_gwex_t0_gps_ns", None)

    @property
    def x0(self) -> Any:
        """GWpy-compatible axis-origin alias, synchronized with ``t0_ns``."""
        from gwpy.timeseries import TimeSeries as BaseTimeSeries

        return BaseTimeSeries.x0.__get__(self, type(self))

    @x0.setter
    def x0(self, value: Any) -> None:
        self._set_exact_epoch(value, alias="x0")

    @x0.deleter
    def x0(self) -> None:
        from gwpy.timeseries import TimeSeries as BaseTimeSeries

        BaseTimeSeries.x0.__delete__(self)
        self.__dict__.pop("_gwex_t0_gps_ns", None)

    def _set_exact_epoch(self, value: Any, *, alias: str) -> None:
        """Set a GWpy epoch alias without desynchronizing exact authority."""
        from gwpy.timeseries import TimeSeries as BaseTimeSeries

        exact_t0_ns = getattr(self, "_gwex_t0_gps_ns", None)
        if exact_t0_ns is not None:
            new_t0_ns = _integer_gps_ns(value)
        else:
            new_t0_ns = None

        if alias == "t0":
            BaseTimeSeries.t0.__set__(self, value)
        else:
            BaseTimeSeries.x0.__set__(self, value)
        if new_t0_ns is not None:
            self._gwex_t0_gps_ns = new_t0_ns

    def copy(self, order: Literal["C", "F", "A", "K"] = "C") -> TimeSeries:
        """Copy this series without reconstructing its exact epoch from ``t0``."""
        from gwpy.timeseries import TimeSeries as BaseTimeSeries

        result = BaseTimeSeries.copy(self, order=order)
        exact_t0_ns = getattr(self, "_gwex_t0_gps_ns", None)
        if exact_t0_ns is not None:
            result._gwex_t0_gps_ns = exact_t0_ns
        exact_dt_ns = getattr(self, "_gwex_dt_gps_ns", None)
        if exact_dt_ns is not None:
            result._gwex_dt_gps_ns = exact_dt_ns
        return result

    def __getitem__(self, item: Any) -> Any:
        """Preserve an exact epoch authority when a regular slice advances it."""
        exact = getattr(self, "_gwex_t0_gps_ns", None)
        exact_dt = getattr(self, "_gwex_dt_gps_ns", None)
        slice_item = item[0] if isinstance(item, tuple) and len(item) == 1 else item
        if exact is not None and isinstance(slice_item, slice):
            suppressed = _SUPPRESS_EXACT_FINALIZE_FROM.get()
            token = _SUPPRESS_EXACT_FINALIZE_FROM.set(suppressed | {id(self)})
            try:
                result = super().__getitem__(item)
            finally:
                _SUPPRESS_EXACT_FINALIZE_FROM.reset(token)
        else:
            result = super().__getitem__(item)
        if (
            exact is None
            or not isinstance(slice_item, slice)
            or not isinstance(result, type(self))
        ):
            return result

        try:
            dt_ns = exact_dt
            if dt_ns is None:
                dt_ns = _integral_dt_gps_ns(self.dt)
        except (AttributeError, TypeError, ValueError, OverflowError):
            return result

        start, _, step = slice_item.indices(len(self))
        result._gwex_t0_gps_ns = int(exact) + start * dt_ns
        result._gwex_dt_gps_ns = dt_ns * step
        return result

    def __array_finalize__(self, obj: Any) -> None:
        """Finalize the array after creation (slicing, view casting).

        Ensures that attributes starting with `_gwex_` are propagated
        from the parent object to the new view/instance.
        """
        super().__array_finalize__(obj)
        if obj is None:
            return

        # Propagate custom _gwex_ attributes.
        suppress_exact = id(obj) in _SUPPRESS_EXACT_FINALIZE_FROM.get()
        for key, val in getattr(obj, "__dict__", {}).items():
            if suppress_exact and key in _EXACT_STATE_KEYS:
                continue
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


_timeseries_read = cast(Any, TimeSeries.read).__func__
_timeseries_read.__signature__ = _gwf_parallel_read_signature(_timeseries_read)
_timeseries_read.__doc__ = f"{_timeseries_read.__doc__}{_GWF_PARALLEL_HELP}"

__all__ = ["TimeSeries"]
