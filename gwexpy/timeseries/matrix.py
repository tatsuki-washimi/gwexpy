from __future__ import annotations

from typing import Optional, Any
import numpy as np
from astropy import units as u

try:
    import scipy.signal  # noqa: F401 - availability check
except ImportError:
    pass

from gwpy.timeseries import TimeSeries as BaseTimeSeries

from gwexpy.types.seriesmatrix import SeriesMatrix
from gwexpy.types.mixin import PhaseMethodsMixin

from .utils import SeriesType
from .timeseries import TimeSeries
from .collections import TimeSeriesDict, TimeSeriesList

from .matrix_core import (
    TimeSeriesMatrixCoreMixin
)
from .matrix_analysis import TimeSeriesMatrixAnalysisMixin
from .matrix_spectral import TimeSeriesMatrixSpectralMixin
from .matrix_interop import TimeSeriesMatrixInteropMixin


class TimeSeriesMatrix(
    PhaseMethodsMixin,
    TimeSeriesMatrixCoreMixin,
    TimeSeriesMatrixAnalysisMixin,
    TimeSeriesMatrixSpectralMixin,
    TimeSeriesMatrixInteropMixin,
    SeriesMatrix
):
    """
    Matrix container for multiple TimeSeries objects.

    Provides dt, t0, times aliases and constructs FrequencySeriesMatrix via FFT.
    """

    series_class = TimeSeries
    dict_class = TimeSeriesDict
    list_class = TimeSeriesList
    series_type = SeriesType.TIME
    default_xunit = "s"
    default_yunit = None
    _default_plot_method = "plot"

    def __new__(
        cls,
        data: Any = None,
        times: Any = None,
        dt: Any = None,
        t0: Any = None,
        sample_rate: Any = None,
        epoch: Any = None,
        **kwargs: Any,
    ) -> "TimeSeriesMatrix":
        import warnings
        from gwexpy.timeseries.utils import _coerce_t0_gps

        channel_names = kwargs.pop("channel_names", None)
        should_coerce = True
        xunit = kwargs.get("xunit", None)
        if xunit is not None:
            try:
                should_coerce = u.Unit(xunit).is_equivalent(u.s)
            except (ValueError, TypeError):
                should_coerce = False
        elif isinstance(dt, u.Quantity):
            phys = getattr(dt.unit, "physical_type", None)
            if dt.unit != u.dimensionless_unscaled and phys != "time":
                should_coerce = False

        # 1. Enforce Mutual Exclusivity (GWpy rules)
        if epoch is not None and t0 is not None:
            raise ValueError("give only one of epoch or t0")
        if sample_rate is not None and dt is not None:
            raise ValueError("give only one of sample_rate or dt")

        # 2. Map time-specific args to SeriesMatrix generic args
        if times is not None:
            # If times (xindex) is provided, it takes priority.
            # GWpy semantics: Ignore dx, x0 args if times is present.
            # epoch is preserved as metadata.
            existing_xindex = kwargs.pop("xindex", None)
            kwargs["xindex"] = times
            
            if epoch is not None:
                kwargs["epoch"] = epoch
            elif "epoch" not in kwargs:
                # Default to times[0] if possible
                try:
                    kwargs["epoch"] = times[0]
                except (IndexError, TypeError):
                    pass

            conflict = False
            if existing_xindex is not None:
                conflict = True
            if dt is not None or sample_rate is not None:
                conflict = True
            if t0 is not None:
                conflict = True

            if "dx" in kwargs:
                conflict = True
                kwargs.pop("dx")
            if "x0" in kwargs:
                conflict = True
                kwargs.pop("x0")

            if conflict:
                warnings.warn(
                    "dt/sample_rate/t0/dx/x0/xindex given with times, ignoring",
                    UserWarning,
                )
        else:
            # 3. Handle dt / sample_rate -> dx
            if dt is not None:
                kwargs["dx"] = dt
            elif sample_rate is not None:
                if isinstance(sample_rate, u.Quantity):
                    sr_quantity = sample_rate
                else:
                    sr_quantity = u.Quantity(sample_rate, "Hz")

                kwargs["dx"] = (1.0 / sr_quantity).to(
                    kwargs.get("xunit", cls.default_xunit)
                )

            # 4. Handle t0 / epoch -> x0 and epoch
            if t0 is not None:
                kwargs["x0"] = _coerce_t0_gps(t0) if should_coerce else t0
                if "epoch" not in kwargs:
                    kwargs["epoch"] = t0
            elif epoch is not None and "x0" not in kwargs:
                kwargs["x0"] = _coerce_t0_gps(epoch) if should_coerce else epoch
                if "epoch" not in kwargs:
                    kwargs["epoch"] = epoch

            if "dx" in kwargs and "x0" not in kwargs:
                if kwargs.get("xindex") is None:
                    kwargs["x0"] = 0

        # Default xunit
        if "xunit" not in kwargs:
            kwargs["xunit"] = cls.default_xunit

        if channel_names is not None:
            if "names" not in kwargs:
                cn = np.asarray(channel_names)
                
                # Intelligent reshaping based on data shape
                try:
                    if hasattr(data, "shape"):
                         dshape = data.shape
                    else:
                         dshape = np.shape(data)
                    
                    if len(dshape) >= 2:
                        N, M = dshape[:2]
                        if cn.size == N * M:
                            kwargs["names"] = cn.reshape(N, M)
                        elif cn.size == N:
                             kwargs["names"] = cn.reshape(N, 1)
                        else:
                             # Default to 1D, which broadcasts to (..., M) if size matches M
                             kwargs["names"] = cn
                    else:
                        kwargs["names"] = cn
                except Exception:
                     if cn.ndim == 1:
                          kwargs["names"] = cn.reshape(-1, 1)
                     else:
                          kwargs["names"] = cn

        obj = super().__new__(cls, data, **kwargs)


        return obj

    def plot(self, **kwargs: Any) -> Any:
        """Plot the matrix data."""
        from gwexpy.plot import Plot
        return Plot(self, **kwargs)


# --- Dynamic Registration of TimeSeries Methods ---

_TSM_TIME_DOMAIN_METHODS = [
    "detrend",
    "taper",
    "whiten",
    "filter",
    "lowpass",
    "highpass",
    "bandpass",
    "notch",
    "resample",
]

_TSM_MISSING_TIME_DOMAIN_METHODS = [
    m for m in _TSM_TIME_DOMAIN_METHODS if not hasattr(BaseTimeSeries, m)
]

for _m in _TSM_TIME_DOMAIN_METHODS:
    if _m in _TSM_MISSING_TIME_DOMAIN_METHODS:
        continue
    # Use the helper from core, but wrap it to add metadata if needed
    # Actually, the helpers in CoreMixin were slightly different (docstring etc.)
    # Let's redefine them here to maintain exact behavior
    def _make_wrapper(name):
        def _wrapper(self, *args, **kwargs):
            return self._apply_timeseries_method(name, *args, **kwargs)
        _wrapper.__name__ = name
        _wrapper.__qualname__ = f"TimeSeriesMatrix.{name}"
        _wrapper.__doc__ = f"Element-wise delegate to `TimeSeries.{name}`."
        return _wrapper
    
    setattr(TimeSeriesMatrix, _m, _make_wrapper(_m))


_TSM_BIVARIATE_METHODS = [
    "csd",
    "coherence",
    "transfer_function",
]

_TSM_UNIVARIATE_METHODS = [
    "auto_coherence",
]

for _m in _TSM_BIVARIATE_METHODS:
    if hasattr(BaseTimeSeries, _m):
        def _make_biv_wrapper(name):
            def _wrapper(self, other, *args, **kwargs):
                return self._apply_bivariate_spectral_method(name, other, *args, **kwargs)
            _wrapper.__name__ = name
            _wrapper.__qualname__ = f"TimeSeriesMatrix.{name}"
            _wrapper.__doc__ = f"Element-wise delegate to `TimeSeries.{name}` with another TimeSeries."
            return _wrapper
        setattr(TimeSeriesMatrix, _m, _make_biv_wrapper(_m))

for _m in _TSM_UNIVARIATE_METHODS:
    if hasattr(BaseTimeSeries, _m):
        def _make_univ_wrapper(name):
            def _wrapper(self, *args, **kwargs):
                return self._apply_univariate_spectral_method(name, *args, **kwargs)
            _wrapper.__name__ = name
            _wrapper.__qualname__ = f"TimeSeriesMatrix.{name}"
            _wrapper.__doc__ = f"Element-wise delegate to `TimeSeries.{name}`."
            return _wrapper
        setattr(TimeSeriesMatrix, _m, _make_univ_wrapper(_m))
