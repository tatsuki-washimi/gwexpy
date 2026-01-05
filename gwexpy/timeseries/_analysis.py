"""
Statistical analysis methods for TimeSeries.

This module provides statistical analysis functionality as a mixin class:
- Imputation: impute
- Standardization: standardize
- Time series modeling: fit_arima, hurst, local_hurst
- Rolling statistics: rolling_mean, rolling_std, rolling_median, rolling_min, rolling_max
"""

from __future__ import annotations

from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from gwexpy.timeseries.timeseries import TimeSeries


class TimeSeriesAnalysisMixin:
    """
    Mixin class providing statistical analysis methods for TimeSeries.

    This mixin is designed to be combined with TimeSeriesCore to create
    the full TimeSeries class.
    """

    # ===============================
    # Imputation and Preprocessing
    # ===============================

    def impute(
        self,
        *,
        method: str = "linear",
        limit: Optional[int] = None,
        axis: str = "time",
        max_gap: Optional[float] = None,
        **kwargs: Any,
    ) -> "TimeSeries":
        """
        Impute missing values.

        Parameters
        ----------
        method : str
            Interpolation method: 'linear', 'nearest', 'ffill', 'bfill', etc.
        limit : int, optional
            Maximum number of consecutive NaNs to fill.
        axis : str
            Axis along which to interpolate.
        max_gap : float, optional
            Maximum gap (in time units) to interpolate across.
        **kwargs
            Additional arguments passed to the imputation function.

        Returns
        -------
        TimeSeries
            Imputed series.

        See Also
        --------
        gwexpy.timeseries.preprocess.impute_timeseries
        """
        from gwexpy.timeseries.preprocess import impute_timeseries
        return impute_timeseries(self, method=method, limit=limit, axis=axis, max_gap=max_gap, **kwargs)

    def standardize(self, *, method: str = "zscore", ddof: int = 0, robust: bool = False) -> "TimeSeries":
        """
        Standardize the series.

        Parameters
        ----------
        method : str
            Standardization method: 'zscore', 'minmax', etc.
        ddof : int
            Delta degrees of freedom for std calculation.
        robust : bool
            If True, use median/IQR instead of mean/std.

        Returns
        -------
        TimeSeries
            Standardized series.

        See Also
        --------
        gwexpy.timeseries.preprocess.standardize_timeseries
        """
        from gwexpy.timeseries.preprocess import standardize_timeseries
        return standardize_timeseries(self, method=method, ddof=ddof, robust=robust)

    # ===============================
    # Time Series Modeling
    # ===============================

    def fit_arima(self, order: tuple = (1, 0, 0), **kwargs: Any) -> Any:
        """
        Fit ARIMA model to the series.

        Parameters
        ----------
        order : tuple
            (p, d, q) order of the ARIMA model.
        **kwargs
            Additional arguments passed to the ARIMA fitting function.

        Returns
        -------
        ARIMAResult
            Fitted model result.

        See Also
        --------
        gwexpy.timeseries.arima.fit_arima
        """
        from gwexpy.timeseries.arima import fit_arima
        return fit_arima(self, order=order, **kwargs)

    def hurst(self, **kwargs: Any) -> Any:
        """
        Compute Hurst exponent.

        The Hurst exponent is a measure of long-term memory of a time series.
        H < 0.5: anti-persistent (mean-reverting)
        H = 0.5: random walk
        H > 0.5: persistent (trending)

        Returns
        -------
        float
            Hurst exponent.

        See Also
        --------
        gwexpy.timeseries.hurst.hurst
        """
        from gwexpy.timeseries.hurst import hurst
        return hurst(self, **kwargs)

    def local_hurst(self, window: Any, **kwargs: Any) -> "TimeSeries":
        """
        Compute local Hurst exponent over a sliding window.

        Parameters
        ----------
        window : str, float, or Quantity
            Window size for local computation.
        **kwargs
            Additional arguments.

        Returns
        -------
        TimeSeries
            Local Hurst exponent series.

        See Also
        --------
        gwexpy.timeseries.hurst.local_hurst
        """
        from gwexpy.timeseries.hurst import local_hurst
        return local_hurst(self, window=window, **kwargs)

    # ===============================
    # Rolling Statistics
    # ===============================

    def rolling_mean(
        self,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
        ignore_nan: Optional[bool] = None,
    ) -> "TimeSeries":
        """
        Rolling mean over time.

        Parameters
        ----------
        window : str, float, or Quantity
            Window size.
        center : bool
            If True, center the window on each point.
        min_count : int
            Minimum number of observations required.
        nan_policy : str
            How to handle NaN values.
        backend : str
            Computation backend.
        ignore_nan : bool, optional
            If True, ignore NaNs (overrides nan_policy).

        Returns
        -------
        TimeSeries
            Rolling mean.
        """
        from gwexpy.timeseries.rolling import rolling_mean
        return rolling_mean(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend, ignore_nan=ignore_nan)

    def rolling_std(
        self,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
        ddof: int = 0,
        ignore_nan: Optional[bool] = None,
    ) -> "TimeSeries":
        """
        Rolling standard deviation over time.

        Parameters
        ----------
        window : str, float, or Quantity
            Window size.
        center : bool
            If True, center the window on each point.
        min_count : int
            Minimum number of observations required.
        nan_policy : str
            How to handle NaN values.
        backend : str
            Computation backend.
        ddof : int
            Delta degrees of freedom.
        ignore_nan : bool, optional
            If True, ignore NaNs (overrides nan_policy).

        Returns
        -------
        TimeSeries
            Rolling standard deviation.
        """
        from gwexpy.timeseries.rolling import rolling_std
        return rolling_std(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend, ddof=ddof, ignore_nan=ignore_nan)

    def rolling_median(
        self,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
        ignore_nan: Optional[bool] = None,
    ) -> "TimeSeries":
        """
        Rolling median over time.

        Parameters
        ----------
        window : str, float, or Quantity
            Window size.
        center : bool
            If True, center the window on each point.
        min_count : int
            Minimum number of observations required.
        nan_policy : str
            How to handle NaN values.
        backend : str
            Computation backend.
        ignore_nan : bool, optional
            If True, ignore NaNs (overrides nan_policy).

        Returns
        -------
        TimeSeries
            Rolling median.
        """
        from gwexpy.timeseries.rolling import rolling_median
        return rolling_median(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend, ignore_nan=ignore_nan)

    def rolling_min(
        self,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
        ignore_nan: Optional[bool] = None,
    ) -> "TimeSeries":
        """
        Rolling minimum over time.

        Parameters
        ----------
        window : str, float, or Quantity
            Window size.
        center : bool
            If True, center the window on each point.
        min_count : int
            Minimum number of observations required.
        nan_policy : str
            How to handle NaN values.
        backend : str
            Computation backend.
        ignore_nan : bool, optional
            If True, ignore NaNs (overrides nan_policy).

        Returns
        -------
        TimeSeries
            Rolling minimum.
        """
        from gwexpy.timeseries.rolling import rolling_min
        return rolling_min(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend, ignore_nan=ignore_nan)

    def rolling_max(
        self,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
        ignore_nan: Optional[bool] = None,
    ) -> "TimeSeries":
        """
        Rolling maximum over time.

        Parameters
        ----------
        window : str, float, or Quantity
            Window size.
        center : bool
            If True, center the window on each point.
        min_count : int
            Minimum number of observations required.
        nan_policy : str
            How to handle NaN values.
        backend : str
            Computation backend.
        ignore_nan : bool, optional
            If True, ignore NaNs (overrides nan_policy).

        Returns
        -------
        TimeSeries
            Rolling maximum.
        """
        from gwexpy.timeseries.rolling import rolling_max
        return rolling_max(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend, ignore_nan=ignore_nan)
