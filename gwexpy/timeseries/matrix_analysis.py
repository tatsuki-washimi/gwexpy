from __future__ import annotations

from typing import Any, Optional

import numpy as np
from astropy import units as u

from .preprocess import impute_timeseries, standardize_matrix, whiten_matrix
from .decomposition import (
    pca_fit, pca_transform, pca_inverse_transform,
    ica_fit, ica_transform, ica_inverse_transform
)


class TimeSeriesMatrixAnalysisMixin:
    """Analysis and preprocessing methods for TimeSeriesMatrix."""

    def radian(self, unwrap: bool = False, **kwargs: Any) -> Any:
        """
        Calculate the instantaneous phase of the matrix in radians.

        Parameters
        ----------
        unwrap : bool, optional
            If True, unwrap the phase.
        **kwargs
            Passed to analytic_signal.

        Returns
        -------
        TimeSeriesMatrix
            The phase of the matrix elements, in radians.
        """
        return self._apply_timeseries_method("radian", unwrap=unwrap, **kwargs)

    def degree(self, unwrap: bool = False, **kwargs: Any) -> Any:
        """
        Calculate the instantaneous phase of the matrix in degrees.

        Parameters
        ----------
        unwrap : bool, optional
            If True, unwrap the phase.
        **kwargs
            Passed to analytic_signal.

        Returns
        -------
        TimeSeriesMatrix
            The phase of the matrix elements, in degrees.
        """
        return self._apply_timeseries_method("degree", unwrap=unwrap, **kwargs)

    def resample(self, rate: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Resample the TimeSeriesMatrix.

        If 'rate' is a time-string (e.g. '1s') or time Quantity, performs time-bin aggregation.
        Otherwise, performs signal processing resampling.
        """
        is_time_bin = False
        if isinstance(rate, str):
            is_time_bin = True
        elif isinstance(rate, u.Quantity):
            if rate.unit.physical_type == 'time':
                is_time_bin = True

        if is_time_bin:
            return self._apply_timeseries_method("resample", rate, *args, **kwargs)
        else:
            # Signal processing resampling (GWpy)
            self._check_regular("Signal processing resample")
            return super().resample(rate, *args, **kwargs)

    def impute(
        self,
        *,
        method: str = "linear",
        limit: Optional[int] = None,
        axis: str = "time",
        max_gap: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Impute missing values in the matrix."""
        new_val = impute_timeseries(self.value, method=method, limit=limit, axis=axis, max_gap=max_gap, **kwargs)
        new_mat = self.copy()
        new_mat.value[:] = new_val
        return new_mat

    def standardize(self, *, axis: str = "time", method: str = "zscore", ddof: int = 0, **kwargs: Any) -> Any:
        """
        Standardize the matrix.
        See gwexpy.timeseries.preprocess.standardize_matrix.
        """
        return standardize_matrix(self, axis=axis, method=method, ddof=ddof, **kwargs)

    def whiten_channels(
        self,
        *,
        method: str = "pca",
        eps: float = 1e-12,
        n_components: Optional[int] = None,
        return_model: bool = True,
    ) -> Any:
        """
        Whiten the matrix (channels/components).
        Returns (whitened_matrix, WhiteningModel) by default.
        Set return_model=False to return only the whitened matrix.
        See gwexpy.timeseries.preprocess.whiten_matrix.
        """
        mat, model = whiten_matrix(self, method=method, eps=eps, n_components=n_components)
        if return_model:
            return mat, model
        return mat

    def rolling_mean(
        self,
        window: Any,
        *,
        center: bool = False,
        min_count: int = 1,
        nan_policy: str = "omit",
        backend: str = "auto",
        ignore_nan: Optional[bool] = None,
    ) -> Any:
        """Rolling mean along the time axis."""
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
    ) -> Any:
        """Rolling standard deviation along the time axis."""
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
    ) -> Any:
        """Rolling median along the time axis."""
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
    ) -> Any:
        """Rolling minimum along the time axis."""
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
    ) -> Any:
        """Rolling maximum along the time axis."""
        from gwexpy.timeseries.rolling import rolling_max
        return rolling_max(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend, ignore_nan=ignore_nan)

    def crop(self, start: Any = None, end: Any = None, copy: bool = False) -> Any:
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
            if hasattr(gps, 'gpsSeconds'):
                return float(gps)
            return float(gps)
        
        start_float = _to_float(start)
        end_float = _to_float(end)
        return super().crop(start=start_float, end=end_float, copy=copy)

    def pca_fit(self, **kwargs: Any) -> Any:
        """Fit PCA."""
        return pca_fit(self, **kwargs)

    def pca_transform(self, pca_res: Any, **kwargs: Any) -> Any:
        """Transform using PCA."""
        return pca_transform(pca_res, self, **kwargs)

    def pca_inverse_transform(self, pca_res: Any, scores: Any) -> Any:
        """Inverse transform PCA scores."""
        return pca_inverse_transform(pca_res, scores)

    def pca(self, return_model: bool = False, **kwargs: Any) -> Any:
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

    def ica(self, return_model: bool = False, **kwargs: Any) -> Any:
        """Fit and transform ICA."""
        res = self.ica_fit(**kwargs)
        sources = self.ica_transform(res)
        if return_model:
            return sources, res
        return sources

    def correlation_vector(self, target_timeseries, method='mic', nproc=None):
        """
        Calculate correlation between a target TimeSeries and all channels in this Matrix.
        """
        import pandas as pd
        import os
        from concurrent.futures import ProcessPoolExecutor

        if nproc is None:
            nproc = os.cpu_count() or 1

        N, M, _ = self.shape
        results = []

        def _calc_direct(ts, target, meth):
            try:
                return ts.correlation(target, method=meth)
            except Exception:
                return np.nan

        if nproc == 1:
            for i in range(N):
                for j in range(M):
                    ts = self[i, j]
                    score = _calc_direct(ts, target_timeseries, method)
                    results.append({
                        "row": i, "col": j, "channel": ts.name, "score": score
                    })
        else:
            with ProcessPoolExecutor(max_workers=nproc) as executor:
                futures = {}
                for i in range(N):
                    for j in range(M):
                        ts = self[i, j]
                        fut = executor.submit(_calc_direct, ts, target_timeseries, method)
                        futures[fut] = (i, j, ts.name)

                for fut in futures:
                    i, j, name = futures[fut]
                    try:
                        score = fut.result()
                    except Exception:
                        score = np.nan
                    results.append({
                        "row": i, "col": j, "channel": name, "score": score
                    })

        df = pd.DataFrame(results)
        df = df.sort_values("score", ascending=False, key=abs).reset_index(drop=True)
        return df
