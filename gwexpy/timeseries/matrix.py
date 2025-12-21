import inspect
from enum import Enum
import numpy as np
from astropy import units as u
from typing import Optional, Union, Any, List, Iterable
try:
    import scipy.signal
except ImportError:
    pass # scipy is optional dependency for gwpy but required here for hilbert


from gwpy.timeseries import TimeSeries as BaseTimeSeries
from gwpy.timeseries import TimeSeriesDict as BaseTimeSeriesDict
from gwpy.timeseries import TimeSeriesList as BaseTimeSeriesList

from gwexpy.types.seriesmatrix import SeriesMatrix
from gwexpy.types.metadata import MetaData, MetaDataMatrix

# --- Monkey Patch TimeSeriesDict ---


# New Imports
from .preprocess import (
    impute_timeseries, standardize_timeseries, align_timeseries_collection, 
    standardize_matrix, whiten_matrix
)
from .arima import fit_arima
from .hurst import hurst, local_hurst
from .decomposition import (
    pca_fit, pca_transform, pca_inverse_transform, 
    ica_fit, ica_transform, ica_inverse_transform
)
from .spectral import csd_matrix_from_collection, coherence_matrix_from_collection


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
        data=None,
        times=None,
        dt=None,
        t0=None,
        sample_rate=None,
        epoch=None,
        **kwargs,
    ):
        import warnings

        channel_names = kwargs.pop("channel_names", None)

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
                kwargs["x0"] = t0
            elif epoch is not None and "x0" not in kwargs:
                kwargs["x0"] = epoch

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

        obj = super().__new__(cls, data, **kwargs)
        if channel_names is not None:
            obj.channel_names = list(channel_names)
        return obj

    # --- Properties mapping to SeriesMatrix attributes ---

    @property
    def dt(self):
        """Time spacing (dx)."""
        return self.dx

    @property
    def t0(self):
        """Start time (x0)."""
        return self.x0

    @property
    def times(self):
        """Time array (xindex)."""
        return self.xindex

    @property
    def span(self):
        """Time span (xspan)."""
        return self.xspan

    @property
    def sample_rate(self):
        """Sampling rate (1/dt)."""
        if self.dt is None:
            return None
        rate = 1.0 / self.dt
        if isinstance(rate, u.Quantity):
            return rate.to("Hz")
        return u.Quantity(rate, "Hz")

    @sample_rate.setter
    def sample_rate(self, value):
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
        
    # --- Preprocessing & Decomposition ---
    
    def impute(self, *, method="interpolate", limit=None, axis="time", max_gap=None, **kwargs):
        """Impute missing values."""
        # Matrix-level impute or per-channel?
        # Provide naive implementation mapping over columns or using decomposition logic?
        # User requirement says "TimeSeriesDict/List/Matrix: implement impute(...) that applies per-channel"
        # Since logic isn't in preprocess.py for matrix iteration, we implement it here or call preprocess helper (which I didn't fully genericize).
        
        # We can treat each column as a TimeSeries (if regular).
        # Efficient way:
        new_val = self.value.copy()
        
        # Using impute_timeseries logic inline or map?
        # Let's map for simplicity/correctness reuse
        # value is 3D: (channels, 1, time) or similar
        val_3d = self.value
        n_rows, n_cols, n_samples = val_3d.shape
        
        # We need to flatten loops or iterate
        new_val = val_3d.copy()
        for r in range(n_rows):
             for c in range(n_cols):
                  ts_data = new_val[r, c, :]
                  ts_tmp = self.series_class(ts_data, dt=self.dt, t0=self.t0)
                  # Passing explicit args
                  imp = ts_tmp.impute(method=method, limit=limit, max_gap=max_gap, axis=axis, **kwargs)
                  new_val[r, c, :] = imp.value
            
        new_mat = self.copy()
        new_mat.value[:] = new_val
        return new_mat

    def standardize(self, *, axis="time", method="zscore", ddof=0):
        """
        Standardize the matrix.
        See gwexpy.timeseries.preprocess.standardize_matrix.
        """
        # Note: standardize_matrix assumes (channels, time) input correctly now (handles axis).
        return standardize_matrix(self, axis=axis, method=method, ddof=ddof)

    def whiten_channels(self, *, method="pca", eps=1e-12, n_components=None, return_model=True):
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

    def rolling_mean(self, window, *, center=False, min_count=1, nan_policy="omit", backend="auto"):
        """Rolling mean along the time axis."""
        from gwexpy.timeseries.rolling import rolling_mean
        return rolling_mean(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)

    def rolling_std(self, window, *, center=False, min_count=1, nan_policy="omit", backend="auto", ddof=0):
        """Rolling standard deviation along the time axis."""
        from gwexpy.timeseries.rolling import rolling_std
        return rolling_std(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend, ddof=ddof)

    def rolling_median(self, window, *, center=False, min_count=1, nan_policy="omit", backend="auto"):
        """Rolling median along the time axis."""
        from gwexpy.timeseries.rolling import rolling_median
        return rolling_median(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)

    def rolling_min(self, window, *, center=False, min_count=1, nan_policy="omit", backend="auto"):
        """Rolling minimum along the time axis."""
        from gwexpy.timeseries.rolling import rolling_min
        return rolling_min(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)

    def rolling_max(self, window, *, center=False, min_count=1, nan_policy="omit", backend="auto"):
        """Rolling maximum along the time axis."""
        from gwexpy.timeseries.rolling import rolling_max
        return rolling_max(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)
        
    def crop(self, start=None, end=None, copy=False):
        """
        Crop this matrix to the given GPS start and end times.
        Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).
        """
        from gwexpy.time import to_gps
        if start is not None:
             start = float(to_gps(start))
        if end is not None:
             end = float(to_gps(end))
        return super().crop(start=start, end=end, copy=copy)

    def to_dict(self):

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

    def to_list(self):
        """Convert to TimeSeriesList."""
        d = self.to_dict()
        return TimeSeriesList(d.values())

    # Decomposition
    def pca_fit(self, **kwargs):
        """Fit PCA."""
        return pca_fit(self, **kwargs)

    def pca_transform(self, pca_res, **kwargs):
        """Transform using PCA."""
        return pca_transform(pca_res, self, **kwargs)
        
    def pca_inverse_transform(self, pca_res, scores):
        """Inverse transform PCA scores."""
        return pca_inverse_transform(pca_res, scores)

    def pca(self, return_model=False, **kwargs):
        """Fit and transform PCA."""
        res = self.pca_fit(**kwargs)
        scores = self.pca_transform(res, n_components=kwargs.get("n_components"))
        if return_model:
            return scores, res
        return scores

    # --- Interop helpers (Matrix-level) ---

    def to_torch(self, device=None, dtype=None, requires_grad=False, copy=False):
        """
        Convert matrix values to a torch.Tensor (shape preserved).
        """
        from gwexpy.interop import to_torch
        return to_torch(self, device=device, dtype=dtype, requires_grad=requires_grad, copy=copy)

    def ica_fit(self, **kwargs):
        """Fit ICA."""
        return ica_fit(self, **kwargs)

    def ica_transform(self, ica_res):
        """Transform using ICA."""
        return ica_transform(ica_res, self)

    def ica_inverse_transform(self, ica_res, sources):
        """Inverse transform ICA sources."""
        return ica_inverse_transform(ica_res, sources)
        
    def ica(self, return_model=False, **kwargs):
        """Fit and transform ICA."""
        res = self.ica_fit(**kwargs)
        sources = self.ica_transform(res)
        if return_model:
            return sources, res
        return sources


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

    def __getitem__(self, item):
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

    def plot(self, **kwargs):
        """
        Plot the matrix data.
        """
        if "xscale" not in kwargs:
            kwargs["xscale"] = "auto-gps"
        return super().plot(**kwargs)

    def _apply_timeseries_method(self, method_name, *args, **kwargs):
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
    def to_neo_analogsignal(self, units=None):
        """Convert to neo.AnalogSignal."""
        from gwexpy.interop import to_neo_analogsignal
        return to_neo_analogsignal(self, units=units)
        
    @classmethod
    def from_neo_analogsignal(cls, sig):
        """Create from neo.AnalogSignal."""
        from gwexpy.interop import from_neo_analogsignal
        return from_neo_analogsignal(cls, sig)
        
    def to_mne_rawarray(self, info=None):
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

    def to_mne_raw(self, info=None):
        """Alias for :meth:`to_mne_rawarray`."""
        return self.to_mne_rawarray(info=info)




    def _coerce_other_timeseries_input(self, other, method_name):
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

    def _apply_bivariate_spectral_method(self, method_name, other, *args, **kwargs):
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

    def _apply_univariate_spectral_method(self, method_name, *args, **kwargs):
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

    def _apply_spectrogram_method(self, method_name, *args, **kwargs):
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

    def _run_spectral_method(self, method_name, **kwargs):
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


    def lock_in(self, **kwargs):
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

    def fft(self, **kwargs):
        """
        Compute FFT of each element.
        Returns FrequencySeriesMatrix.
        """
        return self._run_spectral_method("fft", **kwargs)

    def psd(self, **kwargs):
        """
        Compute PSD of each element.
        Returns FrequencySeriesMatrix.
        """
        return self._run_spectral_method("psd", **kwargs)

    def asd(self, **kwargs):
        """
        Compute ASD of each element.
        Returns FrequencySeriesMatrix.
        """
        return self._run_spectral_method("asd", **kwargs)

    def spectrogram(self, *args, **kwargs):
        """
        Compute spectrogram of each element.
        Returns SpectrogramMatrix.
        """
        return self._apply_spectrogram_method("spectrogram", *args, **kwargs)

    def spectrogram2(self, *args, **kwargs):
        """
        Compute spectrogram2 of each element.
        Returns SpectrogramMatrix.
        """
        return self._apply_spectrogram_method("spectrogram2", *args, **kwargs)

    def q_transform(self, *args, **kwargs):
        """
        Compute Q-transform of each element.
        Returns SpectrogramMatrix.
        """
        return self._apply_spectrogram_method("q_transform", *args, **kwargs)

    def _repr_string_(self):
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
