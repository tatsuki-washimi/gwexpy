from __future__ import annotations

import inspect
import numpy as np
from astropy import units as u
from typing import Optional, Any
try:
    import scipy.signal
except ImportError:
    pass # scipy is optional dependency for gwpy but required here for hilbert


from gwpy.timeseries import TimeSeries as BaseTimeSeries

from gwexpy.types.seriesmatrix import SeriesMatrix
from gwexpy.types.metadata import MetaData, MetaDataMatrix

# --- Monkey Patch TimeSeriesDict ---


# New Imports
from .preprocess import (
    impute_timeseries, standardize_matrix, whiten_matrix
)
from .decomposition import (
    pca_fit, pca_transform, pca_inverse_transform,
    ica_fit, ica_transform, ica_inverse_transform
)


from .utils import *
from .timeseries import TimeSeries
from .collections import TimeSeriesDict, TimeSeriesList
class TimeSeriesMatrix(SeriesMatrix):
    """
    Matrix container for multiple TimeSeries objects.

    Provides dt, t0, times aliases and constructs FrequencySeriesMatrix via FFT.
    """

    series_class = TimeSeries
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
            # GWpy semantics: Ignore dx, x0, epoch args if times is present.

            # Check if user provided explicit xindex in kwargs (redundant)
            existing_xindex = kwargs.pop("xindex", None)

            kwargs["xindex"] = times

            # Check for conflict in explicit args
            conflict = False
            if existing_xindex is not None:
                conflict = True
            if dt is not None or sample_rate is not None:
                conflict = True
            if t0 is not None or epoch is not None:
                conflict = True

            # Check for conflict in kwargs (x0, dx, epoch) AND pop them so they don't propagate
            # We must pop them to ensure they are not stored.
            if "dx" in kwargs:
                conflict = True
                kwargs.pop("dx")
            if "x0" in kwargs:
                conflict = True
                kwargs.pop("x0")
            if "epoch" in kwargs:
                # 'epoch' might be in kwargs if passed as **kwargs, though signature captures 'epoch'
                # If it's captured in signature, it's None or set. If in kwargs, it's redundant/conflict.
                conflict = True
                kwargs.pop("epoch")

            if conflict:
                warnings.warn(
                    "dt/sample_rate/t0/epoch/dx/x0/xindex given with times, ignoring",
                    UserWarning,
                )

            # Do NOT set dx, x0, or epoch in kwargs based on the ignored explicit args.

        else:
            # 3. Handle dt / sample_rate -> dx
            if dt is not None:
                kwargs["dx"] = dt
            elif sample_rate is not None:
                # Convert sample_rate to dx = 1/sample_rate
                # Ensure sample_rate is treated as a Quantity if it has units, or raw float.
                if isinstance(sample_rate, u.Quantity):
                    sr_quantity = sample_rate
                else:
                    sr_quantity = u.Quantity(sample_rate, "Hz")

                # dx = 1 / sample_rate
                kwargs["dx"] = (1.0 / sr_quantity).to(
                    kwargs.get("xunit", cls.default_xunit)
                )

            # 4. Handle t0 / epoch -> x0
            if t0 is not None:
                kwargs["x0"] = _coerce_t0_gps(t0) if should_coerce else t0
            elif epoch is not None and "x0" not in kwargs:
                kwargs["x0"] = _coerce_t0_gps(epoch) if should_coerce else epoch

            # Default x0 when needed (SeriesMatrix builds index from x0, dx)
            # Only if times is None and (dx is provided) and x0 is missing
            if "dx" in kwargs and "x0" not in kwargs:
                if kwargs.get("xindex") is None:
                    kwargs["x0"] = 0

            # NOTE: We do NOT set kwargs["epoch"] = epoch here anymore (Round 5).
            # GWpy treats epoch as an alias for t0/x0, not inconsistent separate metadata.
            # SeriesMatrix might have its own epoch handling, but TimeSeriesMatrix relies on x0/xindex.

        # Default xunit
        if "xunit" not in kwargs:
            kwargs["xunit"] = cls.default_xunit

        if channel_names is not None:
            # SeriesMatrix uses 'names' to populate metadata.
            if "names" not in kwargs:
                # Heuristic: If channel_names is 1D, assume it maps to Rows (N).
                # Numpy broadcasting defaults 1D array to Row Vector (1, M) behavior for 2D.
                # But TimeSeriesMatrix channels are usually Rows (N, 1).
                # So we must reshape to Column Vector (N, 1).
                cn = np.asarray(channel_names)
                if cn.ndim == 1:
                     kwargs["names"] = cn.reshape(-1, 1)
                else:
                     kwargs["names"] = cn

        obj = super().__new__(cls, data, **kwargs)
        
        # Keep attribute for potential backward compatibility / user access
        if channel_names is not None:
             obj.channel_names = list(channel_names)
        return obj

    # --- Properties mapping to SeriesMatrix attributes ---

    @property
    def dt(self) -> Any:
        """Time spacing (dx)."""
        return self.dx

    @property
    def t0(self) -> Any:
        """Start time (x0)."""
        return self.x0

    @property
    def times(self) -> Any:
        """Time array (xindex)."""
        return self.xindex

    @property
    def span(self) -> Any:
        """Time span (xspan)."""
        return self.xspan

    @property
    def sample_rate(self) -> Any:
        """Sampling rate (1/dt)."""
        if self.dt is None:
            return None
        rate = 1.0 / self.dt
        if isinstance(rate, u.Quantity):
            return rate.to("Hz")
        return u.Quantity(rate, "Hz")

    @sample_rate.setter
    def sample_rate(self, value: Any) -> None:
        if value is None:
            self.xindex = None
            return

        from gwpy.types.index import Index

        rate = value if isinstance(value, u.Quantity) else u.Quantity(value, "Hz")
        # Update dt/dx
        new_dt = (1 / rate).to(self.xunit or u.s)
        self.dx = new_dt

        # Rebuild xindex to preserve start and length
        length = self.shape[-1]
        if self.xindex is not None and len(self.xindex) > 0:
            start = self.xindex[0]
            if not isinstance(start, u.Quantity):
                start = u.Quantity(start, self.xunit or new_dt.unit or u.s)
        else:
            start = u.Quantity(0, self.xunit or new_dt.unit or u.s)

        self.xindex = Index.define(start, new_dt, length)

    @property
    def is_regular(self) -> bool:
        """Return True if this TimeSeriesMatrix has a regular sample rate."""
        if self.times is None:
            return True
        if hasattr(self.times, "regular"):
            return self.times.regular
        if len(self.times) < 2:
            return True
        # np.asarray(self.times) extracts values from Index/ndarray
        times_val = np.asarray(self.times)
        diffs = np.diff(times_val)
        return np.allclose(diffs, diffs[0], atol=1e-12, rtol=1e-10)

    def _check_regular(self, method_name: Optional[str] = None):
        """Helper to ensure the matrix has a regular sample rate."""
        if not self.is_regular:
            method = method_name or "This method"
            raise ValueError(
                f"{method} requires a regular sample rate (constant dt). "
                "Consider using .asfreq() or .interpolate() to regularized the matrix first."
            )

    # --- Preprocessing & Decomposition ---

    def resample(self, rate: Any, *args: Any, **kwargs: Any) -> "TimeSeriesMatrix":
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
            # We need to apply _resample_time_bin (which is currently logic-only in TimeSeries)
            # Actually, TimeSeries._resample_time_bin is implemented using bincount which works on arrays.
            # However, it expects a 1D 'self'.
            # We can either genericize _resample_time_bin or loop.
            # For matrices, looping is safe but genericizing to work with any ndarray (axis=-1) is better.

            # For now, let's use the robust mapping behavior of _apply_timeseries_method
            # but handle the rate logic.
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
        # Use vectorized impute_timeseries from preprocess.py
        # TimeSeriesMatrix.value is (channels, 1, time) or (rows, cols, time)
        # We operate along 'time' (axis=-1) or specific axis.
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

    def crop(self, start: Any = None, end: Any = None, copy: bool = False) -> "TimeSeriesMatrix":
        """
        Crop this matrix to the given GPS start and end times.
        Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).
        """
        from gwexpy.time import to_gps
        if start is not None:
             start_gps = to_gps(start)
             if isinstance(start_gps, (np.ndarray, list)) and np.ndim(start_gps) > 0:
                 start_gps = start_gps[0]
             start = float(start_gps)
        if end is not None:
             end_gps = to_gps(end)
             if isinstance(end_gps, (np.ndarray, list)) and np.ndim(end_gps) > 0:
                 end_gps = end_gps[0]
             end = float(end_gps)
        return super().crop(start=start, end=end, copy=copy)

    def to_dict(self) -> "TimeSeriesDict":

        """Convert to TimeSeriesDict."""
        # channel_names property should be available if SeriesMatrix logic works
        # If not, generate defaults
        names = getattr(self, "channel_names", None)
        if names is None:
             names = [str(i) for i in range(self.shape[0])]

        # Create dict
        # Assuming we can extract columns as TimeSeries
        d = TimeSeriesDict()

        # Flatten structure? Or iterate only rows?
        # If shape is (channels, 1, time), iterating axis 0 covers channels.
        # If shape is (1, channels, time), need logic.
        # We assume flat list of channels for dict.
        # Flatten non-time dimensions.
        flat_val = self.value.reshape(-1, self.shape[-1])

        # Use channel_names if length matches
        names = getattr(self, "channel_names", None)
        if names is None or len(names) != flat_val.shape[0]:
             names = [str(i) for i in range(flat_val.shape[0])]

        for i, name in enumerate(names):
             # Extract time series
             ts_data = flat_val[i]

             ts = self.series_class(ts_data, t0=self.t0, dt=self.dt, name=name)
             d[name] = ts
        return d

    # === Interoperability ===

    def to_neo(self, units=None) -> Any:
        """
        Convert to neo.AnalogSignal.
        
        Returns
        -------
        neo.core.AnalogSignal
        """
        from gwexpy.interop import to_neo
        return to_neo(self, units=units)

    @classmethod
    def from_neo(cls, sig: Any) -> "TimeSeriesMatrix":
        """
        Create TimeSeriesMatrix from neo.AnalogSignal.
        
        Parameters
        ----------
        sig : neo.core.AnalogSignal
            Input signal.
            
        Returns
        -------
        TimeSeriesMatrix
        """
        from gwexpy.interop import from_neo
        return from_neo(cls, sig)

    def to_list(self) -> "TimeSeriesList":
        """Convert to TimeSeriesList."""
        d = self.to_dict()
        return TimeSeriesList(d.values())

    # Decomposition
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

    # --- Interop helpers (Matrix-level) ---

    def to_torch(
        self,
        device: Optional[str] = None,
        dtype: Any = None,
        requires_grad: bool = False,
        copy: bool = False,
    ) -> Any:
        """
        Convert matrix values to a torch.Tensor (shape preserved).
        """
        from gwexpy.interop import to_torch
        return to_torch(self, device=device, dtype=dtype, requires_grad=requires_grad, copy=copy)

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

    # --- Correlation ---

    def correlation_vector(self, target_timeseries, method='mic', nproc=None):
        """
        Calculate correlation between a target TimeSeries and all channels in this Matrix.
        
        Args:
            target_timeseries (TimeSeries): The target signal (e.g., DARM).
            method (str): 'pearson', 'kendall', 'mic'.
            nproc (int, optional): Number of parallel processes. 
                                   If None, uses os.cpu_count() (or 1 if cannot determine).
        
        Returns:
            pandas.DataFrame: Ranking of channels by correlation score.
                              Columns: ['row', 'col', 'channel', 'score']
        """
        import pandas as pd
        import os
        from concurrent.futures import ProcessPoolExecutor

        if nproc is None:
            nproc = os.cpu_count() or 1
            
        N, M, _ = self.shape
        results = []

        # Helper function to prevent pickling issues with bound methods if any
        def _calc_corr(ts_data, ts_opts, target, meth):
            # Reconstruct TS to avoid pickling large amounts of data unnecessarily? 
            # Actually we pass the data anyway.
            # Using the TimeSeries object directly.
            from gwexpy.timeseries import TimeSeries
            # ts_opts: dict of t0, dt, name, etc
            ts = TimeSeries(ts_data, **ts_opts)
            return ts.correlation(target, method=meth)

        # However, passing 'ts' object directly is easiest if it pickles well.
        # TimeSeries pickles fine.
        
        def _calc_direct(ts, target, meth):
            try:
                return ts.correlation(target, method=meth)
            except Exception:
                return np.nan

        # Collect tasks
        # Note: If memory is an issue with large Matrices, we might want to be careful.
        # But 'target_timeseries' is shared.
        
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


        if not isinstance(value, u.Quantity):
            value = u.Quantity(value, "Hz")

        # Use existing unit or default to seconds
        # xunit property comes from SeriesMatrix
        xunit = self.xunit
        if xunit is None or xunit == u.dimensionless_unscaled:
            xunit = u.Unit("s")

        # Calculate new dx
        new_dx = (1.0 / value).to(xunit)

        # Rebuild xindex
        # Validate safe start value (must be Quantity for Index.define consistency)
        if self.xindex is not None and len(self.xindex) > 0:
            start_val = self.xindex[0]
            # If start_val is already a Quantity, use it (converted)
            # If it's a float/int (ndarray xindex), wrap it in xunit
            if isinstance(start_val, u.Quantity):
                start = start_val.to(xunit)
            else:
                start = u.Quantity(start_val, xunit)
        else:
            # If xindex is currently None or empty, default x0=0 with correct unit
            start = u.Quantity(0, xunit)

        self.xindex = Index.define(start, new_dx, self.shape[-1])

    # --- Element Access ---

    def __getitem__(self, item: Any) -> Any:
        """
        Return TimeSeries for single element access, or TimeSeriesMatrix for slicing.
        """
        # 1. Handle scalar access (returning TimeSeries element) directly
        if isinstance(item, tuple) and len(item) == 2:
            r, c = item
            is_scalar_r = isinstance(r, (int, np.integer, str))
            is_scalar_c = isinstance(c, (int, np.integer, str))

            if is_scalar_r and is_scalar_c:
                # Direct access to underlying numpy array and meta
                # Avoid super().__getitem__ which constructs Series

                # Resolve string keys to integers
                ri = self.row_index(r) if isinstance(r, str) else r
                ci = self.col_index(c) if isinstance(c, str) else c

                val = self._value[ri, ci]
                meta = self.meta[ri, ci]

                # Construct TimeSeries
                # GWpy semantics: prefer no-copy for times
                return self.series_class(
                    val,
                    times=self.times,
                    unit=meta.unit,
                    name=meta.name,
                    channel=meta.channel,
                )

        # 2. Handle slicing (returning TimeSeriesMatrix)
        # Call super().__getitem__ which returns SeriesMatrix (or view of it)
        ret = super().__getitem__(item)

        # If the result is a SeriesMatrix, ensure it is viewed as TimeSeriesMatrix
        if isinstance(ret, SeriesMatrix) and not isinstance(ret, TimeSeriesMatrix):
            return ret.view(TimeSeriesMatrix)

        return ret

    # --- Plotting ---

    def plot(self, **kwargs: Any) -> Any:
        """
        Plot the matrix data.
        """
        if "xscale" not in kwargs:
            kwargs["xscale"] = "auto-gps"
        return super().plot(**kwargs)

    def _apply_timeseries_method(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Apply a TimeSeries method element-wise and rebuild a TimeSeriesMatrix.

        Parameters
        ----------
        method_name : str
            Name of the TimeSeries method to invoke.
        *args, **kwargs :
            Forwarded to the TimeSeries method. If ``inplace`` is supplied,
            the matrix will be updated in place after applying the method.

        Returns
        -------
        TimeSeriesMatrix
            New matrix unless ``inplace=True`` is requested.
        """
        N, M, _ = self.shape
        if N == 0 or M == 0:
            return self if kwargs.get("inplace", False) else self.copy()

        if not hasattr(self.series_class, method_name):
            raise NotImplementedError(
                f"Not implemented: TimeSeries has no method '{method_name}' in this GWpy version"
            )

        inplace_matrix = bool(kwargs.get("inplace", False))
        base_kwargs = dict(kwargs)
        base_kwargs.pop("inplace", None)

        supports_inplace = False
        ts_attr = getattr(self.series_class, method_name, None)
        if ts_attr is not None:
            try:
                sig = inspect.signature(ts_attr)
                supports_inplace = "inplace" in sig.parameters
            except (TypeError, ValueError):
                supports_inplace = False

        dtype = None
        axis_infos = []
        values = [[None for _ in range(M)] for _ in range(N)]
        meta_array = np.empty((N, M), dtype=object)

        for i in range(N):
            for j in range(M):
                ts = self[i, j]
                method = getattr(ts, method_name)
                call_kwargs = dict(base_kwargs)
                if supports_inplace:
                    call_kwargs["inplace"] = inplace_matrix
                ts_result = method(*args, **call_kwargs)
                if ts_result is None:
                    ts_result = ts

                axis_info = _extract_axis_info(ts_result)
                axis_infos.append(axis_info)
                axis_length = axis_info["n"]
                data_arr = np.asarray(ts_result.value)
                if data_arr.shape[-1] != axis_length:
                    raise ValueError(
                        f"{method_name} produced inconsistent data lengths"
                    )

                values[i][j] = data_arr
                meta_array[i, j] = MetaData(
                    unit=ts_result.unit,
                    name=ts_result.name,
                    channel=ts_result.channel,
                )
                dtype = (
                    data_arr.dtype
                    if dtype is None
                    else np.result_type(dtype, data_arr.dtype)
                )

        common_axis, axis_length = _validate_common_axis(axis_infos, method_name)

        out_shape = (N, M, axis_length)
        out_data = np.empty(out_shape, dtype=dtype)
        for i in range(N):
            for j in range(M):
                out_data[i, j, :] = values[i][j]

        meta_matrix = MetaDataMatrix(meta_array)

        if inplace_matrix:
            if self.shape != out_data.shape:
                self.resize(out_data.shape, refcheck=False)
            np.copyto(self.view(np.ndarray), out_data, casting="unsafe")
            self._value = self.view(np.ndarray)
            # Update meta if changed
            # (Inplace updates on meta are tricky if structure changed, but attributes like unit might change)
            # Minimal meta update in-place?
            # Rebuilding metadata matrix is complex in-place.
            return self

        # New Matrix
        # Reconstruct correct class
        new_mat = self.__class__(
             out_data,
             xindex=common_axis,
             xunit=common_axis.unit if isinstance(common_axis, u.Quantity) else None,
             # Pass generic args
        )
        new_mat._meta = meta_matrix
        return new_mat

    # Interoperability Delegation
    def to_neo(self, units: Optional[str] = None) -> Any:
        """Convert to neo.AnalogSignal."""
        from gwexpy.interop import to_neo
        return to_neo(self, units=units)

    @classmethod
    def from_neo(cls: type["TimeSeriesMatrix"], sig: Any) -> Any:
        """Create from neo.AnalogSignal."""
        from gwexpy.interop import from_neo
        return from_neo(cls, sig)

    def to_mne(self, info: Any = None) -> Any:
        """Convert to mne.io.RawArray."""
        from gwexpy.interop import to_mne_rawarray
        # Convert to dict first? or map?
        # to_mne_rawarray expects TimeSeriesDict or matrix logic.
        # But to_mne_rawarray in interop/mne_.py takes TimeSeriesDict.
        # We can implement a direct matrix path or convert to dict.
        # P2 spec: "from_mne_raw -> TimeSeriesDict".
        # Maybe to_mne is better on Dict. Matrix can use to_dict().
        # Implementing direct method for convenience.
        tsd = self.to_dict()
        return to_mne_rawarray(tsd, info=info)




    def _coerce_other_timeseries_input(self, other: Any, method_name: str) -> Any:
        """
        Normalize 'other' input for bivariate spectral methods.
        """
        if isinstance(other, TimeSeriesMatrix):
            if other.shape[:2] != self.shape[:2]:
                raise ValueError(
                    f"shape mismatch: {self.shape[:2]} vs {other.shape[:2]}"
                )

            def _getter(i, j):
                return other[i, j]

            return _getter

        if isinstance(other, BaseTimeSeries):
            def _getter(i, j):
                return other

            return _getter

        raise TypeError(
            "other must be TimeSeriesMatrix or TimeSeries for bivariate spectral methods"
        )

    def _apply_bivariate_spectral_method(self, method_name: str, other: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Apply a bivariate TimeSeries spectral method element-wise and return FrequencySeriesMatrix.
        """
        from gwexpy.frequencyseries.frequencyseries import FrequencySeriesMatrix

        if not hasattr(self.series_class, method_name):
            raise NotImplementedError(
                f"Not implemented: TimeSeries has no method '{method_name}' in this GWpy version"
            )

        get_other = self._coerce_other_timeseries_input(other, method_name)

        N, M, _ = self.shape
        values = [[None for _ in range(M)] for _ in range(N)]
        meta_array = np.empty((N, M), dtype=object)
        freq_infos = []
        epochs = []
        dtype = None

        for i in range(N):
            for j in range(M):
                ts_a = self[i, j]
                ts_b = get_other(i, j)
                result = getattr(ts_a, method_name)(ts_b, *args, **kwargs)
                if not hasattr(result, "frequencies"):
                    raise TypeError(
                        f"{method_name} must return a FrequencySeries-like object"
                    )
                freq_info = _extract_freq_axis_info(result)
                freq_infos.append(freq_info)
                epochs.append(getattr(result, "epoch", None))

                data_arr = np.asarray(result.value)
                values[i][j] = data_arr
                name = getattr(result, "name", None) or getattr(ts_a, "name", None)
                channel = getattr(result, "channel", None)
                if channel is None or str(channel) == "":
                    channel = getattr(ts_a, "channel", None)
                meta_array[i, j] = MetaData(
                    unit=getattr(result, "unit", None),
                    name=name,
                    channel=channel,
                )
                dtype = (
                    data_arr.dtype
                    if dtype is None
                    else np.result_type(dtype, data_arr.dtype)
                )

        common_freqs, common_df, common_f0, n_freq = _validate_common_frequency_axis(
            freq_infos, method_name
        )
        common_epoch = _validate_common_epoch(epochs, method_name)

        out_data = np.empty((N, M, n_freq), dtype=dtype)
        for i in range(N):
            for j in range(M):
                out_data[i, j, :] = values[i][j]

        meta_matrix = MetaDataMatrix(meta_array)

        return FrequencySeriesMatrix(
            out_data,
            frequencies=common_freqs,
            meta=meta_matrix,
            rows=self.rows,
            cols=self.cols,
            name=getattr(self, "name", ""),
            epoch=common_epoch,
        )

    def _apply_univariate_spectral_method(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Apply a univariate TimeSeries spectral method element-wise and return FrequencySeriesMatrix.
        """
        from gwexpy.frequencyseries.frequencyseries import FrequencySeriesMatrix

        if not hasattr(self.series_class, method_name):
            raise NotImplementedError(
                f"Not implemented: TimeSeries has no method '{method_name}' in this GWpy version"
            )

        N, M, _ = self.shape
        values = [[None for _ in range(M)] for _ in range(N)]
        meta_array = np.empty((N, M), dtype=object)
        freq_infos = []
        epochs = []
        dtype = None

        for i in range(N):
            for j in range(M):
                ts = self[i, j]
                result = getattr(ts, method_name)(*args, **kwargs)
                if not hasattr(result, "frequencies"):
                    raise TypeError(
                        f"{method_name} must return a FrequencySeries-like object"
                    )
                freq_info = _extract_freq_axis_info(result)
                freq_infos.append(freq_info)
                epochs.append(getattr(result, "epoch", None))

                data_arr = np.asarray(result.value)
                values[i][j] = data_arr
                name = getattr(result, "name", None) or getattr(ts, "name", None)
                channel = getattr(result, "channel", None)
                if channel is None or str(channel) == "":
                    channel = getattr(ts, "channel", None)
                meta_array[i, j] = MetaData(
                    unit=getattr(result, "unit", None),
                    name=name,
                    channel=channel,
                )
                dtype = (
                    data_arr.dtype
                    if dtype is None
                    else np.result_type(dtype, data_arr.dtype)
                )

        common_freqs, common_df, common_f0, n_freq = _validate_common_frequency_axis(
            freq_infos, method_name
        )
        common_epoch = _validate_common_epoch(epochs, method_name)

        out_data = np.empty((N, M, n_freq), dtype=dtype)
        for i in range(N):
            for j in range(M):
                out_data[i, j, :] = values[i][j]

        meta_matrix = MetaDataMatrix(meta_array)

        return FrequencySeriesMatrix(
            out_data,
            frequencies=common_freqs,
            meta=meta_matrix,
            rows=self.rows,
            cols=self.cols,
            name=getattr(self, "name", ""),
            epoch=common_epoch,
        )

    def _apply_spectrogram_method(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """
        Apply a TimeSeries spectrogram method element-wise and return SpectrogramMatrix.
        """
        from gwexpy.spectrogram import SpectrogramMatrix

        if not hasattr(self.series_class, method_name):
            raise NotImplementedError(
                f"Not implemented: TimeSeries has no method '{method_name}' in this GWpy version"
            )

        N, M, _ = self.shape
        if N == 0 or M == 0:
            return SpectrogramMatrix(np.empty((N, M, 0, 0)))

        values = [[None for _ in range(M)] for _ in range(N)]
        meta_array = np.empty((N, M), dtype=object)
        time_infos = []
        freq_infos = []
        epochs = []
        dtype = None
        unit_ref = None
        name_ref = None

        for i in range(N):
            for j in range(M):
                ts = self[i, j]
                result = getattr(ts, method_name)(*args, **kwargs)
                if not hasattr(result, "times") or not hasattr(result, "frequencies"):
                    raise TypeError(
                        f"{method_name} must return a Spectrogram-like object"
                    )

                time_info = _extract_axis_info(result)
                freq_info = _extract_freq_axis_info(result)
                time_infos.append(time_info)
                freq_infos.append(freq_info)
                epochs.append(getattr(result, "epoch", None))

                data_arr = np.asarray(result.value)
                if data_arr.ndim != 2:
                    raise ValueError(
                        f"{method_name} must return 2D spectrogram data"
                    )
                values[i][j] = data_arr
                meta_array[i, j] = MetaData(
                    unit=getattr(result, "unit", None),
                    name=getattr(result, "name", None),
                    channel=getattr(result, "channel", None),
                )

                unit = getattr(result, "unit", None)
                if unit_ref is None:
                    unit_ref = unit
                elif unit != unit_ref:
                    raise ValueError(
                        f"{method_name} requires common unit; mismatch in unit"
                    )

                if name_ref is None:
                    name_ref = getattr(result, "name", None)

                dtype = (
                    data_arr.dtype
                    if dtype is None
                    else np.result_type(dtype, data_arr.dtype)
                )

        common_times, n_time = _validate_common_axis(time_infos, method_name)
        common_freqs, _, _, n_freq = _validate_common_frequency_axis(
            freq_infos, method_name
        )
        _validate_common_epoch(epochs, method_name)

        out_data = np.empty((N, M, n_time, n_freq), dtype=dtype)
        for i in range(N):
            for j in range(M):
                if values[i][j].shape != (n_time, n_freq):
                    raise ValueError(
                        f"{method_name} produced inconsistent spectrogram shapes"
                    )
                out_data[i, j, :, :] = values[i][j]

        meta_matrix = MetaDataMatrix(meta_array)

        return SpectrogramMatrix(
            out_data,
            times=common_times,
            frequencies=common_freqs,
            unit=unit_ref,
            name=getattr(self, "name", None) or name_ref,
            rows=self.rows,
            cols=self.cols,
            meta=meta_matrix,
        )

    def _run_spectral_method(self, method_name: str, **kwargs: Any) -> Any:
        """
        Helper for fft, psd, asd.
        """
        from gwexpy.frequencyseries.frequencyseries import FrequencySeriesMatrix

        N, M, K = self.shape

        # Run first element to determine frequency axis and output properties
        # Run first element to determine frequency axis and output properties
        # Use self[0, 0] to get the first TimeSeries element
        ts0 = self[0, 0]
        method = getattr(ts0, method_name)
        fs0 = method(**kwargs)

        # Prepare output array
        n_freq = len(fs0)
        out_shape = (N, M, n_freq)
        out_data = np.empty(out_shape, dtype=fs0.dtype)

        # Attributes
        out_units = np.empty((N, M), dtype=object)
        out_names = np.empty((N, M), dtype=object)
        out_channels = np.empty((N, M), dtype=object)

        # Loop over all elements
        for i in range(N):
            for j in range(M):
                if i == 0 and j == 0:
                    fs = fs0
                else:
                    # Use self[i, j] which now returns a proper TimeSeries
                    ts = self[i, j]
                    fs = getattr(ts, method_name)(**kwargs)

                out_data[i, j, :] = fs.value
                out_units[i, j] = fs.unit
                out_names[i, j] = fs.name
                out_channels[i, j] = fs.channel

        return FrequencySeriesMatrix(
            out_data,
            frequencies=fs0.frequencies,
            units=out_units,
            names=out_names,
            channels=out_channels,
            rows=self.rows,
            cols=self.cols,
            name=getattr(self, "name", ""),
            epoch=getattr(self, "epoch", None),
        )

    # --- Spectral Methods ---


    def lock_in(self, **kwargs: Any) -> Any:
        """
        Apply lock-in amplification element-wise.

        Returns
        -------
        TimeSeriesMatrix or tuple of TimeSeriesMatrix
            If output='amp_phase' (default) or 'iq', returns (matrix1, matrix2).
            If output='complex', returns a single complex TimeSeriesMatrix.
        """
        output = kwargs.get("output", "amp_phase")
        expect_tuple = output in ["amp_phase", "iq"]

        N, M, _ = self.shape
        if N == 0 or M == 0:
             if expect_tuple:
                 return self.copy(), self.copy()
             return self.copy()

        vals1 = [[None for _ in range(M)] for _ in range(N)]
        vals2 = [[None for _ in range(M)] for _ in range(N)] if expect_tuple else None

        meta1 = np.empty((N, M), dtype=object)
        meta2 = np.empty((N, M), dtype=object) if expect_tuple else None

        ax_infos = []
        method_name = "lock_in"
        dtype1 = None
        dtype2 = None

        for i in range(N):
            for j in range(M):
                ts = self[i, j]
                res = ts.lock_in(**kwargs)

                if expect_tuple:
                    r1, r2 = res
                    vals1[i][j] = np.asarray(r1.value)
                    meta1[i, j] = MetaData(unit=str(r1.unit), name=r1.name, channel=r1.channel)
                    dtype1 = np.result_type(dtype1, r1.value.dtype) if dtype1 else r1.value.dtype

                    vals2[i][j] = np.asarray(r2.value)
                    meta2[i, j] = MetaData(unit=str(r2.unit), name=r2.name, channel=r2.channel)
                    dtype2 = np.result_type(dtype2, r2.value.dtype) if dtype2 else r2.value.dtype

                    ax_infos.append(_extract_axis_info(r1))
                else:
                    # Single return
                    vals1[i][j] = np.asarray(res.value)
                    meta1[i, j] = MetaData(unit=str(res.unit), name=res.name, channel=res.channel)
                    dtype1 = np.result_type(dtype1, res.value.dtype) if dtype1 else res.value.dtype
                    ax_infos.append(_extract_axis_info(res))


        # Validate common axis
        common_axis, axis_len = _validate_common_axis(ax_infos, method_name)

        def _build(v, d, m):
            out_shape = (N, M, axis_len)
            out = np.empty(out_shape, dtype=d)
            for r in range(N):
                for c in range(M):
                    out[r, c, :] = v[r][c]
            new_mat = self.__class__(
                 out,
                 xindex=common_axis,
                 xunit=common_axis.unit if isinstance(common_axis, u.Quantity) else None,
            )
            new_mat.meta = MetaDataMatrix(m)
            return new_mat

        m1 = _build(vals1, dtype1, meta1)
        if expect_tuple:
            m2 = _build(vals2, dtype2, meta2)
            return m1, m2
        return m1

    def fft(self, **kwargs: Any) -> Any:
        """
        Compute FFT of each element.
        Returns FrequencySeriesMatrix.
        """
        return self._run_spectral_method("fft", **kwargs)

    def psd(self, **kwargs: Any) -> Any:
        """
        Compute PSD of each element.
        Returns FrequencySeriesMatrix.
        """
        return self._run_spectral_method("psd", **kwargs)

    def asd(self, **kwargs: Any) -> Any:
        """
        Compute ASD of each element.
        Returns FrequencySeriesMatrix.
        """
        return self._run_spectral_method("asd", **kwargs)

    def spectrogram(self, *args: Any, **kwargs: Any) -> Any:
        """
        Compute spectrogram of each element.
        Returns SpectrogramMatrix.
        """
        return self._apply_spectrogram_method("spectrogram", *args, **kwargs)

    def spectrogram2(self, *args: Any, **kwargs: Any) -> Any:
        """
        Compute spectrogram2 of each element.
        Returns SpectrogramMatrix.
        """
        return self._apply_spectrogram_method("spectrogram2", *args, **kwargs)

    def q_transform(self, *args: Any, **kwargs: Any) -> Any:
        """
        Compute Q-transform of each element.
        Returns SpectrogramMatrix.
        """
        return self._apply_spectrogram_method("q_transform", *args, **kwargs)

    def _repr_string_(self) -> str:
        if self.size > 0:
            u = self.meta[0, 0].unit
        else:
            u = None
        return f"<TimeSeriesMatrix shape={self.shape}, dt={self.dt}, unit={u}>"


def _make_tsm_timeseries_wrapper(method_name):
    def _wrapper(self, *args, **kwargs):
        return self._apply_timeseries_method(method_name, *args, **kwargs)

    _wrapper.__name__ = method_name
    _wrapper.__qualname__ = f"{TimeSeriesMatrix.__name__}.{method_name}"
    _wrapper.__doc__ = f"Element-wise delegate to `TimeSeries.{method_name}`."
    return _wrapper


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
# Not implemented: `TimeSeriesMatrix` does not define wrappers for methods that
# are missing from `gwpy.timeseries.TimeSeries` in the installed GWpy version.
for _m in _TSM_TIME_DOMAIN_METHODS:
    if _m in _TSM_MISSING_TIME_DOMAIN_METHODS:
        continue
    setattr(TimeSeriesMatrix, _m, _make_tsm_timeseries_wrapper(_m))


def _make_tsm_bivariate_wrapper(method_name):
    def _wrapper(self, other, *args, **kwargs):
        return self._apply_bivariate_spectral_method(method_name, other, *args, **kwargs)

    _wrapper.__name__ = method_name
    _wrapper.__qualname__ = f"{TimeSeriesMatrix.__name__}.{method_name}"
    _wrapper.__doc__ = f"Element-wise delegate to `TimeSeries.{method_name}` with another TimeSeries."
    return _wrapper


def _make_tsm_univariate_wrapper(method_name):
    def _wrapper(self, *args, **kwargs):
        return self._apply_univariate_spectral_method(method_name, *args, **kwargs)

    _wrapper.__name__ = method_name
    _wrapper.__qualname__ = f"{TimeSeriesMatrix.__name__}.{method_name}"
    _wrapper.__doc__ = f"Element-wise delegate to `TimeSeries.{method_name}`."
    return _wrapper


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
        setattr(TimeSeriesMatrix, _m, _make_tsm_bivariate_wrapper(_m))

for _m in _TSM_UNIVARIATE_METHODS:
    if hasattr(BaseTimeSeries, _m):
        setattr(TimeSeriesMatrix, _m, _make_tsm_univariate_wrapper(_m))
