from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

# --- Optional Dependencies ---
try:
    import statsmodels.api as sm  # noqa: F401
    from statsmodels.tsa.arima.model import ARIMA

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        SARIMAX = None
    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
    except ImportError:
        ConvergenceWarning = None
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    ARIMA = None
    SARIMAX = None
    ConvergenceWarning = None

try:
    import pmdarima as pm

    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False

try:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from .timeseries import TimeSeries
except ImportError:
    pass


@dataclass
class ArimaForecastResult:
    forecast_ts: TimeSeries
    intervals: dict[str, TimeSeries]


class ArimaResult:
    """
    Result object for ARIMA/SARIMAX modeling on TimeSeries.
    Wraps statsmodels results and provides TimeSeries-aware methods.
    """

    def __init__(self, res, t0, dt, unit, name=None, channel=None, original_data=None):
        self.res = res
        self.t0 = t0
        self.dt = dt
        self.unit = unit
        self.name = name
        self.channel = channel
        self.original_data = original_data  # Keep reference for plotting

    def _time_to_index(self, t, default=0):
        """Helper to convert GPS time to relative integer index."""
        if t is None:
            return default
        if isinstance(t, (int, np.integer)):
            return int(t)
        if isinstance(t, float):
            # GPS time provided
            return int(round((t - self.t0) / self.dt))
        return default

    def predict(self, start=None, end=None, *, dynamic=False) -> TimeSeries:
        """
        In-sample prediction using GPS times or steps.

        Parameters
        ----------
        start : float (GPS) or int, optional
            Start time or index. Defaults to beginning of data.
        end : float (GPS) or int, optional
            End time or index. Defaults to end of data.
        dynamic : bool, default=False
            Use dynamic prediction (recursive).

        Returns
        -------
        TimeSeries
            Predicted values.
        """
        from .timeseries import TimeSeries

        start_idx = self._time_to_index(start, default=0)
        end_idx = self._time_to_index(end, default=None)

        # statsmodels predict uses 0-based indexing
        pred = self.res.predict(start=start_idx, end=end_idx, dynamic=dynamic)

        new_t0 = self.t0 + start_idx * self.dt

        # Handle case where predict returns a single value
        if np.ndim(pred) == 0:
            pred = [pred]

        return TimeSeries(
            pred, t0=new_t0, dt=self.dt, unit=self.unit, name=f"{self.name}_pred"
        )

    def forecast(
        self, steps: int, *, alpha: float = 0.05
    ) -> tuple[TimeSeries, dict[str, TimeSeries]]:
        """
        Out-of-sample forecasts.

        Parameters
        ----------
        steps : int
            Number of steps to forecast into the future.
        alpha : float, default=0.05
            Significance level (default 0.05 means 95% confidence intervals).

        Returns
        -------
        forecast : TimeSeries
            The mean forecast.
        intervals : dict
            Dict with 'lower' and 'upper' TimeSeries for confidence intervals.
        """
        from .timeseries import TimeSeries

        # statsmodels get_forecast handles out-of-sample
        pred_res = self.res.get_forecast(steps=steps)
        point = pred_res.predicted_mean
        conf = pred_res.conf_int(alpha=alpha)

        # Determine start time of forecast (immediately after training data)
        n_obs = (
            int(self.res.nobs)
            if hasattr(self.res, "nobs")
            else len(self.res.fittedvalues)
        )
        forecast_t0 = self.t0 + n_obs * self.dt

        forecast_ts = TimeSeries(
            point,
            t0=forecast_t0,
            dt=self.dt,
            unit=self.unit,
            name=f"{self.name}_forecast",
        )

        lower = TimeSeries(
            conf[:, 0], t0=forecast_t0, dt=self.dt, unit=self.unit, name="lower_ci"
        )
        upper = TimeSeries(
            conf[:, 1], t0=forecast_t0, dt=self.dt, unit=self.unit, name="upper_ci"
        )

        return forecast_ts, {"lower": lower, "upper": upper}

    def residuals(self) -> TimeSeries:
        """Return the model residuals as a TimeSeries."""
        from .timeseries import TimeSeries

        return TimeSeries(
            self.res.resid,
            t0=self.t0,
            dt=self.dt,
            unit=self.unit,
            name=f"{self.name}_resid",
        )

    def params_dict(self) -> dict:
        """
        Return model parameters as a dictionary.

        Returns
        -------
        dict
            Dictionary containing 'params' (parameter values), 'param_names'
            (parameter names), and 'aic', 'bic' if available.
        """
        result = {
            "params": self.res.params,
            "param_names": list(self.res.params.index)
            if hasattr(self.res.params, "index")
            else None,
        }

        # Add fit statistics if available
        if hasattr(self.res, "aic"):
            result["aic"] = self.res.aic
        if hasattr(self.res, "bic"):
            result["bic"] = self.res.bic

        return result

    def summary(self):
        """Return the statsmodels summary object."""
        return self.res.summary()

    def plot(self, forecast_steps=None, alpha=0.05, ax=None, **kwargs):
        """
        Plot original data, in-sample fit, and (optional) out-of-sample forecast.

        Parameters
        ----------
        forecast_steps : int, optional
            If provided, plot forecast for this many steps into the future.
        alpha : float, default=0.05
            Significance level for confidence intervals.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        **kwargs
            Passed to plot methods.
        """
        from matplotlib import pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        # 1. Original Data
        if self.original_data is not None:
            ax.plot(
                self.original_data,
                label="Original",
                color="gray",
                alpha=0.6,
                linewidth=1.5,
            )

        # 2. In-sample Prediction (Fit)
        # Note: ARIMA fits usually have bad predictions for the very first few samples (diffs)
        pred = self.predict(start=0, dynamic=False)
        # Shift slightly for visibility if needed, but usually exact overlay is best
        ax.plot(pred, label="Model Fit", color="tab:blue", linestyle="--", alpha=0.8)

        # 3. Forecast
        if forecast_steps is not None and forecast_steps > 0:
            fc, conf = self.forecast(steps=forecast_steps, alpha=alpha)

            from ..plot import plot_mmm

            plot_mmm(
                fc,
                conf["lower"],
                conf["upper"],
                ax=ax,
                color="tab:orange",
                linewidth=2,
                label="Forecast",
                alpha_fill=0.2,
                label_fill=f"{int((1 - alpha) * 100)}% CI",
            )

        ax.set_xlabel("GPS Time [s]")
        ax.set_ylabel(f"Amplitude [{self.unit}]")
        ax.legend(loc="best")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_title(f"ARIMA Model Results: {self.name or 'TimeSeries'}")

        return ax


def fit_arima(
    timeseries: TimeSeries,
    order: tuple[int, int, int] = (1, 0, 0),
    *,
    seasonal_order: tuple[int, int, int, int] | None = None,
    trend: str | None = "c",
    auto: bool = False,
    auto_kwargs: dict | None = None,
    method: str | None = None,
    method_kwargs: dict | None = None,
    nan_policy: str = "raise",
    impute_kwargs: dict | None = None,
    **kwargs,
) -> ArimaResult:
    """
    Fit an ARIMA or SARIMAX model to a TimeSeries.

    Parameters
    ----------
    timeseries : TimeSeries
        The input data.
    order : tuple, default=(1, 0, 0)
        The (p,d,q) order of the model. Ignored if auto=True.
    seasonal_order : tuple, optional
        The (P,D,Q,s) order of the seasonal component.
    trend : str, optional
        Trend parameter ('c', 't', 'ct', etc). Default 'c' (constant).
    auto : bool, default=False
        If True, use pmdarima (Auto-ARIMA) to find the best order.
    auto_kwargs : dict, optional
        Arguments passed to pmdarima.auto_arima (e.g., {'max_p': 5}).

    Notes
    -----
    Common upstream warnings from pmdarima/statsmodels are suppressed to keep
    notebook output clean (e.g., convergence and verbose/disp deprecations).
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError(
            "statsmodels is required for ARIMA. `pip install statsmodels`"
        )

    if timeseries.dt is None:
        raise ValueError("TimeSeries must have a regular sample rate (dt) for ARIMA.")

    y = timeseries.value
    t0 = timeseries.t0

    # --- NaN Handling ---
    if nan_policy == "impute":
        # Basic imputation if requested
        if np.any(np.isnan(y)):
            from .preprocess import impute_timeseries

            ikw = (impute_kwargs or {}).copy()
            ts_imp = impute_timeseries(timeseries, **ikw)
            y = np.asarray(ts_imp.value)
            if np.any(np.isnan(y)):
                raise ValueError(
                    "NaNs remain after imputation; adjust impute_kwargs or nan_policy."
                )

    elif nan_policy == "raise":
        if np.any(np.isnan(y)):
            raise ValueError("NaNs in data and nan_policy='raise'")

    y = y.astype(float)

    # --- Auto ARIMA (pmdarima) ---
    if auto:
        if not PMDARIMA_AVAILABLE:
            raise ImportError(
                "pmdarima is required for auto=True. `pip install pmdarima`"
            )

        akwargs = (auto_kwargs or {}).copy()
        # Merge extra kwargs into akwargs
        for k, v in kwargs.items():
            if k not in akwargs:
                akwargs[k] = v

        # Ensure seasonal and trend are set without conflict
        if "seasonal" not in akwargs:
            akwargs["seasonal"] = seasonal_order is not None
        if "trend" not in akwargs:
            akwargs["trend"] = trend

        # Adjust start values if max values are smaller (pmdarima requirement)
        # Defaults are start_p=2, start_q=2, start_P=1, start_Q=1
        for m_key, s_key in [
            ("max_p", "start_p"),
            ("max_q", "start_q"),
            ("max_P", "start_P"),
            ("max_Q", "start_Q"),
        ]:
            if m_key in akwargs:
                default_start = 2 if s_key in ["start_p", "start_q"] else 1
                current_start = akwargs.get(s_key, default_start)
                if akwargs[m_key] < current_start:
                    akwargs[s_key] = akwargs[m_key]

        with warnings.catch_warnings():
            if ConvergenceWarning is not None:
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            warnings.filterwarnings(
                "ignore", category=FutureWarning, module=r"sklearn.*"
            )
            model_auto = pm.auto_arima(y, **akwargs)

        # pmdarima wraps a statsmodels result. We extract it to wrap in our ArimaResult
        res = model_auto.arima_res_

        return ArimaResult(
            res,
            t0=t0,
            dt=timeseries.dt,
            unit=timeseries.unit,
            name=f"{timeseries.name}_auto",
            channel=getattr(timeseries, "channel", None),
            original_data=timeseries,
        )

    # --- Manual ARIMA / SARIMAX ---
    fit_kwargs = (method_kwargs or {}).copy()
    # Merge extra kwargs into fit_kwargs
    for k, v in kwargs.items():
        if k not in fit_kwargs:
            fit_kwargs[k] = v

    if method:
        fit_kwargs["method"] = method

    # Use SARIMAX class for everything if available, as it's more robust in statsmodels v0.12+
    if seasonal_order is None and ARIMA is not None:
        # Use standard ARIMA if no seasonality needed
        model = ARIMA(y, order=order, trend=trend)
    else:
        if SARIMAX is None:
            raise ImportError("SARIMAX not available in statsmodels")
        model = SARIMAX(y, order=order, seasonal_order=seasonal_order, trend=trend)

    with warnings.catch_warnings():
        if ConvergenceWarning is not None:
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r".*verbose.*",
            module=r"statsmodels.*",
        )
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r".*disp.*",
            module=r"statsmodels.*",
        )
        res = model.fit(**fit_kwargs)

    return ArimaResult(
        res,
        t0=t0,
        dt=timeseries.dt,
        unit=timeseries.unit,
        name=timeseries.name,
        channel=getattr(timeseries, "channel", None),
        original_data=timeseries,
    )
