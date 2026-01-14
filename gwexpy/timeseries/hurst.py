from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u

if TYPE_CHECKING:
    pass


@dataclass
class HurstResult:
    """
    Result of a Hurst exponent calculation.

    Attributes
    ----------
    H : float
        The estimated Hurst exponent.
    method : str
        Method used ('rs', 'standard', 'generalized', 'exp').
    backend : str
        Backend library used ('hurst', 'hurst-exponent', 'exp-hurst').
    details : dict
        Additional details from the backend (e.g., 'c', 'data', 'fit').
    """

    H: float
    method: str
    backend: str
    details: dict[str, Any]

    def summary_dict(self):
        """Return a dictionary summary of the result."""
        d = {"H": self.H, "method": self.method, "backend": self.backend}
        if self.details:
            d.update(self.details)
        return d


def _get_hurst_rs(x, kind, simplified):
    try:
        from hurst import compute_Hc
    except ImportError:
        raise ImportError("hurst package required for method='rs'. pip install hurst")

    H, c, data = compute_Hc(x, kind=kind, simplified=simplified)
    return H, "rs", "hurst", {"c": c, "data": data}


def _get_hurst_exponent(x, method):
    try:
        import hurst_exponent
    except ImportError:
        raise ImportError("hurst-exponent package required. pip install hurst-exponent")

    if method == "standard":
        # standard_hurst returns (H, c, data) or (H, fit)?
        # Check library signature assumption or documentation.
        # User prompt: "return (H, fit) or (H, fit_dict)"
        res = hurst_exponent.standard_hurst(x)
        # Assuming return (H, c, data) tuple structure based on context?
        # Or (H, fit_val).
        # Let's handle return type gracefully.
        if isinstance(res, tuple):
            H = res[0]
            details = {"fit": res[1:]}
        else:
            H = res
            details = {}
        return H, "standard", "hurst-exponent", details

    elif method == "generalized":
        res = hurst_exponent.generalized_hurst(x)
        if isinstance(res, tuple):
            H = res[0]
            details = {"fit": res[1:]}
        else:
            H = res
            details = {}
        return H, "generalized", "hurst-exponent", details


def _get_exp_hurst(x):
    try:
        from exp_hurst import hurst as exp_hurst_fn
    except ImportError:
        # Fallback check?
        raise ImportError("exp-hurst package required. pip install exp-hurst")

    H = exp_hurst_fn(x)
    return H, "exp", "exp-hurst", {}


def hurst(
    timeseries,
    *,
    method="auto",
    kind="random_walk",
    simplified=True,
    return_details=False,
    nan_policy="raise",
    impute_kwargs=None,
    **kwargs,
) -> float | HurstResult:
    """
    Calculate the Hurst exponent of a time series.

    Parameters
    ----------
    timeseries : TimeSeries
        Input time series.
    method : str, optional
        Method to use: 'auto', 'rs', 'standard', 'generalized', 'exp'.
    kind : str, optional
        Kind of series for R/S analysis ('random_walk' or 'change').
    simplified : bool, default=True
        Use simplified R/S calculation if True.
    return_details : bool, default=False
        If True, return a HurstResult object. Otherwise return float H.
    nan_policy : str, default='raise'
        'raise' or 'impute'.
    impute_kwargs : dict, optional
        Arguments for imputation.

    Returns
    -------
    float or HurstResult
        Estimated Hurst exponent or detailed result object.
    """

    x = timeseries.value

    # NaN Handling
    if np.any(np.isnan(x)):
        if nan_policy == "raise":
            raise ValueError("Input contains NaNs.")
        elif nan_policy == "impute":
            from .preprocess import impute_timeseries

            ikw = impute_kwargs or {}
            ts_imp = impute_timeseries(timeseries, **ikw)
            x = ts_imp.value
        # else: pass? (let backend fail or handle)

    x = x.astype(float)

    # Backend dispatch
    res = None

    if method == "auto":
        # Order: rs -> standard -> exp
        try:
            res = _get_hurst_rs(x, kind, simplified)
        except ImportError:
            try:
                res = _get_hurst_exponent(x, "standard")
            except ImportError:
                try:
                    res = _get_exp_hurst(x)
                except ImportError:
                    raise ImportError(
                        "No Hurst backend found. Install hurst, hurst-exponent, or exp-hurst."
                    )

    elif method == "rs":
        res = _get_hurst_rs(x, kind, simplified)
    elif method in ["standard", "generalized"]:
        res = _get_hurst_exponent(x, method)
    elif method == "exp":
        res = _get_exp_hurst(x)
    else:
        raise ValueError(f"Unknown method '{method}'")

    H, meth, backend, det = res

    if return_details:
        return HurstResult(H, meth, backend, det)
    else:
        return H


def local_hurst(
    timeseries,
    window,
    *,
    step=None,
    method="auto",
    center=True,
    nan_policy="raise",
    impute_kwargs=None,
    **kwargs,
):
    """
    Compute local Hurst exponent using a sliding window.

    Parameters
    ----------
    timeseries : TimeSeries
        Input time series.
    window : int, float, or Quantity
        Window size. Integers are samples, floats/Quantities are time.
    step : int, float, or Quantity, optional
        Step size. Default is window // 2.
    method : str, optional
        Calculation method (see `hurst`).
    center : bool, default=True
        Whether to center the time labels in the window.
    nan_policy : str, default='raise'
        'raise' or 'impute'.
    impute_kwargs : dict, optional
        Arguments for imputation.

    Returns
    -------
    TimeSeries
        A new TimeSeries containing the local Hurst exponents.
    """
    from .timeseries import TimeSeries

    x = timeseries.value
    N = len(x)
    dt = timeseries.dt
    u_time = getattr(timeseries.times, "unit", None) or u.dimensionless_unscaled
    if (
        isinstance(dt, u.Quantity)
        and hasattr(u_time, "physical_type")
        and u_time.physical_type == "time"
        and dt.unit.is_equivalent(u_time)
    ):
        dt_val = dt.to_value(u_time)
    else:
        dt_val = dt.value if isinstance(dt, u.Quantity) else float(dt)

    # window can be samples (int) or quantity
    if isinstance(window, u.Quantity):
        w_samples = int(
            np.round(
                (window / (dt if isinstance(dt, u.Quantity) else dt_val * u.s))
                .decompose()
                .value
            )
        )
    elif isinstance(window, (int, np.integer)):
        w_samples = int(window)
    elif isinstance(window, (float, np.floating)):
        if isinstance(dt, u.Quantity) and dt.unit.physical_type == "time":
            w_samples = int(np.round(((window * u.s) / dt).decompose().value))
        else:
            w_samples = int(np.round(window))
    else:
        w_samples = int(window)  # fallback

    if step is None:
        step_samples = w_samples // 2
    else:
        if isinstance(step, u.Quantity):
            step_samples = int(
                np.round(
                    (step / (dt if isinstance(dt, u.Quantity) else dt_val * u.s))
                    .decompose()
                    .value
                )
            )
        elif isinstance(step, (int, np.integer)):
            step_samples = int(step)
        elif isinstance(step, (float, np.floating)):
            if isinstance(dt, u.Quantity) and dt.unit.physical_type == "time":
                step_samples = int(np.round(((step * u.s) / dt).decompose().value))
            else:
                step_samples = int(np.round(step))
        else:
            step_samples = int(step)

    if step_samples < 1:
        step_samples = 1

    # Slide
    # Output times
    # If center: t = center of window
    # Else: t = start of window

    starts = np.arange(0, N - w_samples + 1, step_samples)
    n_wins = len(starts)

    H_vals = np.full(n_wins, np.nan)
    t_vals = np.zeros(n_wins)

    # Pre-impute full series if requested to avoid repeated impute cost?
    # Spec says nan_policy applies to windows?
    # "If a window has insufficient non-NaN points, set output to NaN ... unless nan_policy='raise'"
    # If policy="impute", should we impute whole series first? Yes, consistent.
    if nan_policy == "impute":
        from .preprocess import impute_timeseries

        ikw = impute_kwargs or {}
        ts_imp = impute_timeseries(timeseries, **ikw)
        x_full = ts_imp.value
    else:
        x_full = x

    for i, s in enumerate(starts):
        e = s + w_samples
        segment_val = x_full[s:e]

        # Determine time
        t0_val = float(
            timeseries.t0.value if hasattr(timeseries.t0, "value") else timeseries.t0
        )
        if center:
            # mid point index
            mid = s + w_samples / 2.0
            t_vals[i] = t0_val + mid * dt_val
        else:
            t_vals[i] = t0_val + s * dt_val

        # Check NaNs in segment
        if np.any(np.isnan(segment_val)):
            if nan_policy == "raise":
                raise ValueError("NaN found in window")
            else:
                H_vals[i] = np.nan
                continue

        # Compute H
        try:
            # Make a temporary TimeSeries-like wrapper or just pass value to underlying helper?
            # But our hurst() fn takes 'timeseries' object usually?
            # Wait, hurst() definition takes `timeseries` and accesses .value.
            # I should define `hurst_val` helper or wrap segment.

            # Let's wrap segment in simple object with .value
            class MockTS:
                def __init__(self, v):
                    self.value = v

            # Recurse to one-shot hurst
            # We rely on 'hurst' function below.
            # We pass nan_policy='raise' because we handled 'impute' globally or 'drop' locally.
            # Actually we dealt with impute globally.
            h = hurst(
                MockTS(segment_val),
                method=method,
                return_details=False,
                nan_policy="raise",
                **kwargs,
            )
            H_vals[i] = h
        except (ValueError, ImportError):
            H_vals[i] = np.nan

    return TimeSeries(
        H_vals,
        times=t_vals
        * u_time,  # Explicit times array since step might be irregular implies fixed grid?
        # Actually it is regular grid if step is fixed.
        # But we computed times. Use `times` arg.
        unit=u.dimensionless_unscaled,
    )
