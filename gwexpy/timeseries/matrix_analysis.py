from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import logging
import numpy as np
from astropy import units as u

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .matrix import TimeSeriesMatrix

from .decomposition import (
    ica_fit,
    ica_inverse_transform,
    ica_transform,
    pca_fit,
    pca_inverse_transform,
    pca_transform,
)
from .preprocess import impute_timeseries, standardize_matrix, whiten_matrix


def _calc_correlation_direct(ts, target, meth):
    """Picklable helper for correlation_vector."""
    try:
        # Ensure ts is a gwexpy TimeSeries to have .correlation() method
        if not hasattr(ts, "correlation"):
            from .timeseries import TimeSeries as GWExTimeSeries

            sr = getattr(ts, "sample_rate", None)
            ts = GWExTimeSeries(ts.value, t0=ts.t0, sample_rate=sr, name=ts.name)
        return ts.correlation(target, method=meth)
    except Exception:
        logger.debug("Correlation calculation failed for %s", getattr(ts, "name", "unknown"), exc_info=True)
        return np.nan

class TimeSeriesMatrixAnalysisMixin:
    """Analysis and preprocessing methods for TimeSeriesMatrix."""

    def _resolve_axis(self: Any, axis):
        """Convert string axis to integer axis."""
        if axis == "time":
            return self._x_axis_norm
        elif axis == "channel":
            return 0
        return axis

    def skewness(
        self: Any, axis: Any = "time", nan_policy: str = "propagate", **kwargs: Any
    ) -> np.ndarray:
        """
        Compute the skewness of the matrix along the specified axis.

        Parameters
        ----------
        axis : str or int
            'time' (default), 'channel', or integer axis.
        nan_policy : str
            How to handle NaNs: 'propagate', 'raise', or 'omit'.
        """
        from scipy import stats

        ax = self._resolve_axis(axis)
        return stats.skew(self.value, axis=ax, nan_policy=nan_policy, **kwargs)

    def kurtosis(
        self: Any,
        axis: Any = "time",
        fisher: bool = True,
        nan_policy: str = "propagate",
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Compute the kurtosis of the matrix along the specified axis.

        Parameters
        ----------
        axis : str or int
            'time' (default), 'channel', or integer axis.
        fisher : bool
            If True, Fisher's definition (normal ==> 0.0).
        nan_policy : str
            How to handle NaNs: 'propagate', 'raise', or 'omit'.
        """
        from scipy import stats

        ax = self._resolve_axis(axis)
        return stats.kurtosis(
            self.value, axis=ax, fisher=fisher, nan_policy=nan_policy, **kwargs
        )

    def mean(
        self: Any,
        axis: Any = "time",
        dtype: Any = None,
        out: Any = None,
        keepdims: bool = False,
        *,
        where: Any = True,
        ignore_nan: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute mean along the specified axis."""
        ax = self._resolve_axis(axis)
        func = np.nanmean if ignore_nan else np.mean
        return func(
            self.value,
            axis=ax,
            dtype=dtype,
            out=out,
            keepdims=keepdims,
            where=where,
            **kwargs,
        )

    def std(
        self: Any,
        axis: Any = "time",
        dtype: Any = None,
        out: Any = None,
        ddof: int = 0,
        keepdims: bool = False,
        *,
        where: Any = True,
        ignore_nan: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute standard deviation along the specified axis."""
        ax = self._resolve_axis(axis)
        func = np.nanstd if ignore_nan else np.std
        return func(
            self.value,
            axis=ax,
            dtype=dtype,
            out=out,
            ddof=ddof,
            keepdims=keepdims,
            where=where,
            **kwargs,
        )

    def rms(
        self: Any,
        axis: Any = "time",
        keepdims: bool = False,
        ignore_nan: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute root-mean-square along the specified axis."""
        ax = self._resolve_axis(axis)
        func = np.nanmean if ignore_nan else np.mean
        # RMS does not support all kwargs in mean (like keepdims? No, mean supports keepdims)
        # But base rms passes argument 'keepdims' to mean.
        return np.sqrt(
            func(
                np.square(self.value), axis=ax, keepdims=keepdims, **kwargs
            )
        )

    def min(
        self: Any,
        axis: Any = "time",
        out: Any = None,
        keepdims: bool = False,
        initial: Any = None,
        where: Any = True,
        ignore_nan: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute minimum along the specified axis."""
        ax = self._resolve_axis(axis)
        func = np.nanmin if ignore_nan else np.min
        return func(
            self.value,
            axis=ax,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
            **kwargs,
        )

    def max(
        self: Any,
        axis: Any = "time",
        out: Any = None,
        keepdims: bool = False,
        initial: Any = None,
        where: Any = True,
        ignore_nan: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute maximum along the specified axis."""
        ax = self._resolve_axis(axis)
        func = np.nanmax if ignore_nan else np.max
        return func(
            self.value,
            axis=ax,
            out=out,
            keepdims=keepdims,
            initial=initial,
            where=where,
            **kwargs,
        )

    def _vectorized_detrend(self: Any, detrend: str = "linear", **kwargs: Any) -> Any:
        """
        Vectorized implementation of detrend.
        """
        from scipy.signal import detrend as scipy_detrend

        inplace = kwargs.pop("inplace", False)

        # Scipy detrend supports axis
        # Note: GWpy uses 'constant' or 'linear'. Scipy uses 'constant' or 'linear'.
        new_data = scipy_detrend(self.value, axis=-1, type=detrend, **kwargs)

        if inplace:
            self.value[:] = new_data
            return self

        new_mat = self.copy()
        new_mat.value[:] = new_data
        return new_mat

    def _vectorized_taper(self: Any, side: str = "leftright", **kwargs: Any) -> Any:
        """
        Semi-vectorized implementation of taper to ensure consistency with GWpy.
        """
        # GWpy's taper implementation is complex (zero-crossing detection etc.)
        # To ensure exact match, we apply it per channel but avoid TimeSeriesMatrix overhead.
        from gwpy.timeseries import TimeSeries as BaseTimeSeries

        data = np.asarray(self.value)
        out_data = np.empty_like(data)

        N, M = self.shape[:2]
        for i in range(N):
            for j in range(M):
                # Use a minimal BaseTimeSeries to get GWpy's taper logic
                ts = BaseTimeSeries(data[i, j], dt=self.dt)
                out_data[i, j] = ts.taper(side=side, **kwargs).value

        inplace = kwargs.get("inplace", False)
        if inplace:
            self.value[:] = out_data
            return self

        new_mat = self.copy()
        new_mat.value[:] = out_data
        return new_mat

    def hilbert(self: Any, **kwargs: Any) -> Any:
        """
        Compute the analytic signal (Hilbert transform).
        """
        return self._apply_timeseries_method("hilbert", **kwargs)

    def _vectorized_filter(self: Any, *filt: Any, **kwargs: Any) -> Any:
        """
        Vectorized implementation of filter (and bandpass, lowpass, etc.).
        """
        from scipy.signal import filtfilt, sosfiltfilt

        data = np.asarray(self.value)
        inplace = kwargs.pop("inplace", False)

        # Handle SOS or BA filter
        if (
            len(filt) == 1
            and np.asarray(filt[0]).ndim == 2
            and np.asarray(filt[0]).shape[1] == 6
        ):
            new_data = sosfiltfilt(filt[0], data, axis=-1, **kwargs)
        elif len(filt) == 2:
            new_data = filtfilt(filt[0], filt[1], data, axis=-1, **kwargs)
        else:
            return self._apply_timeseries_method(
                "filter", *filt, inplace=inplace, **kwargs
            )

        if inplace:
            self.value[:] = new_data
            return self

        new_mat = self.copy()
        new_mat.value[:] = new_data
        return new_mat

    def _vectorized_hilbert(self: Any, **kwargs: Any) -> Any:
        """
        Vectorized implementation of Hilbert transform (analytic signal).
        """
        from scipy.signal import hilbert

        data = np.asarray(self.value)
        h_data = hilbert(data, axis=-1)

        new_mat = self.copy().astype(complex)
        new_mat.value[:] = h_data
        return new_mat

    def _vectorized_radian(self: Any, unwrap: bool = False) -> Any:
        """
        Vectorized implementation of radian (phase angle via np.angle).
        """
        phi = np.angle(self.value)

        if unwrap:
            phi = np.unwrap(phi, axis=-1)

        new_mat = self.copy()
        new_mat.value[:] = phi
        new_mat.unit = u.rad
        return new_mat

    def _vectorized_degree(self: Any, unwrap: bool = False) -> Any:
        """
        Vectorized implementation of degree (phase angle via np.angle).
        """
        phi = np.angle(self.value, deg=True)

        if unwrap:
            phi = np.unwrap(phi, axis=-1, period=360.0)

        new_mat = self.copy()
        new_mat.value[:] = phi
        new_mat.unit = u.deg
        return new_mat

    def radian(self: Any, unwrap: bool = False) -> Any:
        """
        Calculate the phase angle of the matrix in radians.

        Computes np.angle(self.value) directly. Works for both real and complex
        matrices. For real data, returns 0 or π depending on sign.

        Parameters
        ----------
        unwrap : bool, optional
            If True, unwrap the phase. Default is False.

        Returns
        -------
        TimeSeriesMatrix
            Phase angle in radians.
        """
        return self._vectorized_radian(unwrap=unwrap)

    def degree(self: Any, unwrap: bool = False) -> Any:
        """
        Calculate the phase angle of the matrix in degrees.

        Computes np.angle(self.value) directly. Works for both real and complex
        matrices. For real data, returns 0 or 180 depending on sign.

        Parameters
        ----------
        unwrap : bool, optional
            If True, unwrap the phase. Default is False.

        Returns
        -------
        TimeSeriesMatrix
            Phase angle in degrees.
        """
        return self._vectorized_degree(unwrap=unwrap)

    def resample(self: Any, rate: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Resample the TimeSeriesMatrix.

        If 'rate' is a time-string (e.g. '1s') or time Quantity, performs time-bin aggregation.
        Otherwise, performs signal processing resampling.
        """
        is_time_bin = False
        if isinstance(rate, str):
            is_time_bin = True
        elif isinstance(rate, u.Quantity):
            if rate.unit.physical_type == "time":
                is_time_bin = True

        if is_time_bin:
            return self._apply_timeseries_method("resample", rate, *args, **kwargs)
        else:
            # Signal processing resampling (GWpy)
            self._check_regular("Signal processing resample")
            return super().resample(rate, *args, **kwargs)  # type: ignore[misc]

    def impute(
        self: Any,
        *,
        method: str = "linear",
        limit: int | None = None,
        axis: str = "time",
        max_gap: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Impute missing values in the matrix."""
        new_val = impute_timeseries(
            self.value, method=method, limit=limit, axis=axis, max_gap=max_gap, **kwargs
        )
        new_mat = self.copy()
        new_mat.value[:] = new_val
        return new_mat

    def standardize(
        self: Any,
        *,
        axis: str = "time",
        method: str = "zscore",
        ddof: int = 0,
        **kwargs: Any,
    ) -> Any:
        """
        Standardize the matrix.
        See gwexpy.timeseries.preprocess.standardize_matrix.
        """
        return standardize_matrix(cast("TimeSeriesMatrix", self), axis=cast(Literal['time', 'channel'], axis), method=method, ddof=ddof, **kwargs)

    def whiten_channels(
        self: Any,
        *,
        method: str = "pca",
        eps: float = 1e-12,
        n_components: int | None = None,
        return_model: bool = True,
    ) -> Any:
        """
        Whiten the matrix (channels/components).
        Returns (whitened_matrix, WhiteningModel) by default.
        Set return_model=False to return only the whitened matrix.
        See gwexpy.timeseries.preprocess.whiten_matrix.
        """
        mat, model = whiten_matrix(
            cast("TimeSeriesMatrix", self), method=cast(Literal['pca', 'zca'], method), eps=eps, n_components=n_components
        )
        if return_model:
            return mat, model
        return mat

    def rolling_mean(
        self: Any,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
        ignore_nan: bool | None = None,
    ) -> Any:
        """Rolling mean along the time axis."""
        from gwexpy.timeseries.rolling import rolling_mean

        return rolling_mean(
            self,
            window,
            center=center,
            min_count=min_count,
            nan_policy=nan_policy,
            backend=backend,
            ignore_nan=ignore_nan,
        )

    def rolling_std(
        self: Any,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
        ddof: int = 0,
        ignore_nan: bool | None = None,
    ) -> Any:
        """Rolling standard deviation along the time axis."""
        from gwexpy.timeseries.rolling import rolling_std

        return rolling_std(
            self,
            window,
            center=center,
            min_count=min_count,
            nan_policy=nan_policy,
            backend=backend,
            ddof=ddof,
            ignore_nan=ignore_nan,
        )

    def rolling_median(
        self: Any,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
        ignore_nan: bool | None = None,
    ) -> Any:
        """Rolling median along the time axis."""
        from gwexpy.timeseries.rolling import rolling_median

        return rolling_median(
            self,
            window,
            center=center,
            min_count=min_count,
            nan_policy=nan_policy,
            backend=backend,
            ignore_nan=ignore_nan,
        )

    def rolling_min(
        self: Any,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
        ignore_nan: bool | None = None,
    ) -> Any:
        """Rolling minimum along the time axis."""
        from gwexpy.timeseries.rolling import rolling_min

        return rolling_min(
            self,
            window,
            center=center,
            min_count=min_count,
            nan_policy=nan_policy,
            backend=backend,
            ignore_nan=ignore_nan,
        )

    def rolling_max(
        self: Any,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
        ignore_nan: bool | None = None,
    ) -> Any:
        """Rolling maximum along the time axis."""
        from gwexpy.timeseries.rolling import rolling_max

        return rolling_max(
            self,
            window,
            center=center,
            min_count=min_count,
            nan_policy=nan_policy,
            backend=backend,
            ignore_nan=ignore_nan,
        )

    def crop(self: Any, start: Any = None, end: Any = None, copy: bool = False) -> Any:
        """
        Crop this matrix to the given GPS start and end times.
        Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).
        """
        from gwexpy.time import to_gps

        def _to_float(val):
            if val is None:
                return None
            gps = to_gps(val)
            # LIGOTimeGPS has .gpsSeconds and .gpsNanoSeconds, or can be cast to float
            if hasattr(gps, "gpsSeconds"):
                return float(gps)
            return float(gps)

        start_float = _to_float(start)
        end_float = _to_float(end)
        return super().crop(start=start_float, end=end_float, copy=copy)  # type: ignore[misc]

    def pca_fit(self, **kwargs: Any) -> Any:
        """Fit PCA."""
        return pca_fit(self, **kwargs)

    def pca_transform(self, pca_res: Any, **kwargs: Any) -> Any:
        """Transform using PCA."""
        return pca_transform(pca_res, self, **kwargs)

    def pca_inverse_transform(self, pca_res: Any, scores: Any) -> Any:
        """Inverse transform PCA scores."""
        return pca_inverse_transform(pca_res, scores)

    def pca(self: Any, return_model: bool = False, **kwargs: Any) -> Any:
        """Fit and transform PCA."""
        res = self.pca_fit(**kwargs)
        scores = self.pca_transform(res, n_components=kwargs.get("n_components"))
        if return_model:
            return scores, res
        return scores

    def ica_fit(self, **kwargs: Any) -> Any:
        """Fit ICA."""
        return ica_fit(self, **kwargs)

    def ica_transform(self, ica_res: Any) -> Any:
        """Transform using ICA."""
        return ica_transform(ica_res, self)

    def ica_inverse_transform(self, ica_res: Any, sources: Any) -> Any:
        """Inverse transform ICA sources."""
        return ica_inverse_transform(ica_res, sources)

    def ica(self: Any, return_model: bool = False, **kwargs: Any) -> Any:
        """Fit and transform ICA."""
        res = self.ica_fit(**kwargs)
        sources = self.ica_transform(res)
        if return_model:
            return sources, res
        return sources

    def correlation(
        self: Any, other: Any = None, method: str = "pearson", **kwargs: Any
    ) -> Any:
        """
        Calculate correlation coefficients.

        - If `other` is None: Returns (N, M) x (N, M) pairwise correlation ndarray.
        - If `other` is a TimeSeries: Returns correlation with target as a DataFrame (via correlation_vector).
        - If `other` is a TimeSeriesMatrix: Element-wise correlation (if shapes match).
        """
        if other is None:
            # Pairwise correlation between all channels
            # Reshape (N, M, T) -> (N*M, T)
            data = self.value.reshape(-1, self.shape[-1])
            if method == "pearson":
                return np.corrcoef(data)
            # Other methods might need loops or specialized vectorized impls

        if hasattr(other, "ndim") and other.ndim == 1:
            # Target TimeSeries
            return self.correlation_vector(other, method=method, **kwargs)

        return self._apply_timeseries_method(
            "correlation", other, method=method, **kwargs
        )

    def mic(self: Any, other: Any, **kwargs: Any) -> Any:
        """
        Calculate Maximal Information Coefficient (MIC).
        """
        return self.correlation(other, method="mic", **kwargs)

    def distance_correlation(self: Any, other: Any, **kwargs: Any) -> Any:
        """
        Calculate Distance Correlation.
        """
        return self.correlation(other, method="distance", **kwargs)

    def pcc(self: Any, other: Any, **kwargs: Any) -> Any:
        """
        Calculate Pearson Correlation Coefficient.
        """
        return self.correlation(other, method="pearson", **kwargs)

    def ktau(self: Any, other: Any, **kwargs: Any) -> Any:
        """
        Calculate Kendall's Rank Correlation Coefficient.
        """
        return self.correlation(other, method="kendall", **kwargs)

    def correlation_vector(self: Any, target_timeseries: Any, method: str = "mic", nproc: int | None = None) -> Any:
        """
        Calculate correlation between a target TimeSeries and all channels in this Matrix.
        """
        import os
        from concurrent.futures import ProcessPoolExecutor

        import pandas as pd

        if nproc is None:
            nproc = os.cpu_count() or 1

        N, M, _ = self.shape
        results = []

        def _run_serial():
            for i in range(N):
                for j in range(M):
                    ts = self[i, j]
                    score = _calc_correlation_direct(ts, target_timeseries, method)
                    results.append(
                        {"row": i, "col": j, "channel": ts.name, "score": score}
                    )

        if nproc <= 1:
            _run_serial()
        else:
            try:
                with ProcessPoolExecutor(max_workers=nproc) as executor:
                    futures = {}
                    for i in range(N):
                        for j in range(M):
                            ts = self[i, j]
                            fut = executor.submit(
                                _calc_correlation_direct, ts, target_timeseries, method
                            )
                            futures[fut] = (i, j, ts.name)

                    for fut in futures:
                        i, j, name = futures[fut]
                        try:
                            score = fut.result()
                        except Exception:
                            logger.debug("Failed to retrieve parallel correlation result for %s", name, exc_info=True)
                            score = np.nan
                        results.append(
                            {"row": i, "col": j, "channel": name, "score": score}
                        )
            except (OSError, PermissionError, RuntimeError):
                _run_serial()

        df = pd.DataFrame(results)
        df = df.sort_values("score", ascending=False, key=abs).reset_index(drop=True)
        return df
