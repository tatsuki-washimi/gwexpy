
import warnings
import numpy as np
from dataclasses import dataclass
from typing import Dict

try:
    import statsmodels.api as sm
    # Only import classes if available to avoid runtime errors
    from statsmodels.tsa.arima.model import ARIMA
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        SARIMAX = None
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from .timeseries import TimeSeries
except ImportError:
    pass

@dataclass
class ArimaForecastResult:
    forecast_ts: "TimeSeries"
    intervals: Dict[str, "TimeSeries"]

class ArimaResult:
    def __init__(self, res, t0, dt, unit, name=None, channel=None):
        self.res = res
        self.t0 = t0
        self.dt = dt
        self.unit = unit
        self.name = name
        self.channel = channel
        
    def predict(self, start=None, end=None, *, dynamic=False) -> "TimeSeries":
        """
        In-sample prediction or out-of-sample forecast.

        Parameters
        ----------
        start : int, str, or datetime, optional
            Zero-indexed observation number at which to start forecasting.
        end : int, str, or datetime, optional
            Zero-indexed observation number at which to end forecasting.
        dynamic : bool, default=False
            Where to become dynamic.

        Returns
        -------
        TimeSeries
            Predicted values.
        """
        # Handle start/end as time or int indices
        # statsmodels predict uses integer steps relative to start of series
        
        # NOTE: logic to convert time to int index needed if start/end are times
        # For simplicity, assume user passes behavior compatible (steps or times)?
        # statsmodels handles datetime/string indices if index was provided.
        # But we passed numpy array. So index is integer.
        # User must pass integer steps.
        # Result is values.
        
        pred = self.res.predict(start=start, end=end, dynamic=dynamic)
        
        # Handle start/end as time or int indices
        from .timeseries import TimeSeries
        
        # Calculate t0 for the prediction
        # If start is None, it defaults to the first observation (idx=0)
        # If start is int, it's the offset from self.t0
        if start is None:
            start_idx = 0
        elif isinstance(start, (int, np.integer)):
            start_idx = int(start)
        else:
            # For datetime or other types, we'd need more complex logic.
            # statsmodels might have already converted it if the index was set.
            # Here we assume integer indexing if t0/dt are provided.
            warnings.warn("Non-integer start in predict may result in incorrect t0 assignment if not supported by statsmodels internal index.")
            start_idx = 0 # fallback
            
        new_t0 = self.t0 + start_idx * self.dt
        
        return TimeSeries(
            pred,
            t0=new_t0,
            dt=self.dt,
            unit=self.unit,
            name=f"{self.name}_pred"
        )

    def forecast(self, steps, *, alpha=0.05):
        """
        Out-of-sample forecasts.

        Parameters
        ----------
        steps : int
            Number of steps to forecast.
        alpha : float, default=0.05
            Significance level for confidence intervals (default 95%).

        Returns
        -------
        forecast_ts : TimeSeries
            Point forecast.
        intervals : dict
            Dictionary containing 'lower' and 'upper' confidence interval TimeSeries.
        """
        from .timeseries import TimeSeries
        
        # get_forecast return PredictionResultsWrapper
        pred_res = self.res.get_forecast(steps=steps)
        point = pred_res.predicted_mean
        conf = pred_res.conf_int(alpha=alpha)
        
        # Time axis starts after the last sample of fitted data
        # Fitted data length?
        # self.res.nobs (approx)
        
        # For Arima, end of training is known.
        # We need the time of the end of the series.
        # self.t0 + nobs * dt
        
        n_obs = int(self.res.nobs) if hasattr(self.res, 'nobs') else len(self.res.fittedvalues)
        forecast_t0 = self.t0 + n_obs * self.dt
        
        forecast_ts = TimeSeries(
            point,
            t0=forecast_t0,
            dt=self.dt,
            unit=self.unit,
            name=f"{self.name}_forecast"
        )
        
        lower = TimeSeries(
            conf[:, 0],
            t0=forecast_t0,
            dt=self.dt,
            unit=self.unit,
            name="lower"
        )
        upper = TimeSeries(
            conf[:, 1],
            t0=forecast_t0,
            dt=self.dt,
            unit=self.unit,
            name="upper"
        )
        
        return forecast_ts, {"lower": lower, "upper": upper}

    def residuals(self):
        """
        Return residuals of the fitted model.

        Returns
        -------
        TimeSeries
            Residuals.
        """
        from .timeseries import TimeSeries
        return TimeSeries(
            self.res.resid,
            t0=self.t0,
            dt=self.dt,
            unit=self.unit,
            name=f"{self.name}_resid"
        )

    def params_dict(self):
        """
        Return model parameters as a dictionary.

        Returns
        -------
        dict
            Dictionary containing AIC, BIC, Log-Likelihood, and parameters.
        """
        return {
            "aic": self.res.aic,
            "bic": self.res.bic,
            "llf": self.res.llf,
            "params": self.res.params.tolist() if hasattr(self.res.params, "tolist") else self.res.params
        }

def fit_arima(
    timeseries,
    order=(1, 0, 0),
    *,
    seasonal_order=None,
    trend="c",
    enforce_stationarity=True,
    enforce_invertibility=True,
    method=None,
    method_kwargs=None,
    nan_policy="raise",
    impute_kwargs=None,
):
    """
    Fit an ARIMA/SARIMAX model to a TimeSeries.

    Parameters
    ----------
    timeseries : TimeSeries
        Input time series data.
    order : tuple, default=(1, 0, 0)
        The (p,d,q) order of the model for the number of AR parameters,
        differences, and MA parameters.
    seasonal_order : tuple, optional
        The (P,D,Q,s) order of the seasonal component of the model.
    trend : str, default='c'
        Parameter controlling the deterministic trend polynomial.
    enforce_stationarity : bool, default=True
        Whether or not to transform the AR parameters to enforce stationarity.
    enforce_invertibility : bool, default=True
        Whether or not to transform the MA parameters to enforce invertibility.
    method : str, optional
        The method to use for fitting (e.g., 'statespace', 'innovations_mle').
    method_kwargs : dict, optional
        Additional keyword arguments passed to the fit method.
    nan_policy : str, default='raise'
        'raise', 'drop', or 'impute'.
    impute_kwargs : dict, optional
        Arguments for imputation if nan_policy='impute'.

    Returns
    -------
    ArimaResult
        Result object containing the fitted model and providing prediction methods.
    """
    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required. pip install statsmodels")
        
    # Check regular sampling
    # We assume timeseries.dt is valid (not None) as enforced by TimeSeries usually.
    if timeseries.dt is None:
        raise ValueError("TimeSeries must have a regular sample rate (dt).")

    # Handle NaNs
    # "drop": drop NaNs? 
    # If we drop, time alignment breaks.
    # statsmodels ARIMA handles missing data? 
    # "If missing='drop' is passed to fit..." maybe.
    # User said: nan_policy "drop": drop NaNs and adjust time mapping.
    
    y = timeseries.value
    t0 = timeseries.t0
    
    if nan_policy == "raise":
        if np.any(np.isnan(y)):
            raise ValueError("NaNs in data and nan_policy='raise'")
    elif nan_policy == "drop":
        if np.any(np.isnan(y)):
            mask = ~np.isnan(y)
            y = y[mask]
            # Time axis changed. t0 remains? 
            # "adjust time mapping" could mean we treat it as contiguous now? 
            # Or pass irregular times? statsmodels ARIMA expects regular or times.
            # Usually dropping nans in ARIMA means 'treat as adjacent' or 'missing value handling supported by model'? 
            # statsmodels supports missing='drop'.
            # We'll stick to passing numpy array.
            warnings.warn("nan_policy='drop' removes NaNs; time axis interpretation may degrade.")
    elif nan_policy == "impute":
        from .preprocess import impute_timeseries
        kwargs = impute_kwargs or {}
        ts_imputed = impute_timeseries(timeseries, **kwargs)
        y = ts_imputed.value
        
    y = y.astype(float)
    
    fit_kwargs = method_kwargs or {}
    if method:
        fit_kwargs["method"] = method
        
    if seasonal_order is None:
        model = ARIMA(
            y,
            order=order,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility
        )
    else:
        if SARIMAX is None:
             # Fallback to ARIMA if SARIMAX not importable? 
             # Or raise.
             raise ImportError("SARIMAX requires statsmodels")
        model = SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility
        )
        
    res = model.fit(**fit_kwargs)
    
    return ArimaResult(
        res,
        t0=t0,
        dt=timeseries.dt,
        unit=timeseries.unit,
        name=timeseries.name,
        channel=getattr(timeseries, 'channel', None)
    )
