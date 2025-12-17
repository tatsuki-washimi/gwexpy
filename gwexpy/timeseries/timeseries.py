import inspect
from enum import Enum
import numpy as np
from astropy import units as u

from gwpy.timeseries import TimeSeries as BaseTimeSeries
from gwpy.timeseries import TimeSeriesDict as BaseTimeSeriesDict
from gwpy.timeseries import TimeSeriesList as BaseTimeSeriesList

from gwexpy.types.seriesmatrix import SeriesMatrix
from gwexpy.types.metadata import MetaData, MetaDataMatrix
from gwexpy.frequencyseries.frequencyseries import FrequencySeriesMatrix


def _extract_axis_info(ts):
    """
    Extract axis information for a TimeSeries-like object.
    """
    axis = getattr(ts, "times", None)
    if axis is None:
        raise ValueError("times axis is required")

    regular = False
    dt = None
    try:
        dt = ts.dt
    except Exception:
        dt = None

    if dt is not None:
        try:
            if isinstance(dt, u.Quantity):
                val = dt.value
                finite = np.isfinite(val)
                zero = np.all(val == 0)
            else:
                val = float(dt)
                finite = np.isfinite(val)
                zero = (val == 0)
            regular = bool(finite and not zero)
        except Exception:
            regular = False

    t0 = None
    if regular:
        try:
            t0 = ts.t0
        except Exception:
            try:
                t0 = axis[0]
            except Exception:
                t0 = None
        if t0 is None:
            regular = False

    try:
        n = len(axis)
    except Exception:
        n = None
        regular = False

    return {"regular": regular, "dt": dt, "t0": t0, "n": n, "times": axis}


def _validate_common_axis(axis_infos, method_name):
    """
    Validate that a list of axis infos share a common axis.
    """
    if not axis_infos:
        return None, 0

    all_regular = all(info["regular"] for info in axis_infos)
    if all_regular:
        ref = axis_infos[0]
        ref_dt = ref["dt"]
        ref_t0 = ref["t0"]
        ref_n = ref["n"]
        for info in axis_infos[1:]:
            if info["n"] != ref_n:
                raise ValueError(
                    f"{method_name} requires common length; mismatch in length"
                )
            if info["dt"] != ref_dt:
                raise ValueError(f"{method_name} requires common dt; mismatch in dt")
            if info["t0"] != ref_t0:
                raise ValueError(f"{method_name} requires common t0; mismatch in t0")
        return ref["times"], ref_n

    ref_times = axis_infos[0]["times"]
    ref_unit = getattr(ref_times, "unit", None)
    ref_vals = ref_times.value if hasattr(ref_times, "value") else ref_times
    for info in axis_infos[1:]:
        times = info["times"]
        unit = getattr(times, "unit", None)
        if ref_unit is not None or unit is not None:
            if ref_unit != unit:
                raise ValueError(
                    f"{method_name} requires common times unit; mismatch in times"
                )
            lhs = times.value if hasattr(times, "value") else None
            rhs = ref_times.value if hasattr(ref_times, "value") else None
            if lhs is None or rhs is None:
                raise ValueError(
                    f"{method_name} requires comparable times arrays; mismatch in times"
                )
            if not np.array_equal(lhs, rhs):
                raise ValueError(
                    f"{method_name} requires identical times arrays; mismatch in times"
                )
        else:
            if not np.array_equal(times, ref_times):
                raise ValueError(
                    f"{method_name} requires identical times arrays; mismatch in times"
                )
    return ref_times, len(ref_vals)


def _extract_freq_axis_info(fs):
    """
    Extract frequency-axis information from a FrequencySeries-like object.
    """
    freqs = getattr(fs, "frequencies", None)
    if freqs is None:
        raise ValueError("frequencies axis is required")

    regular = False
    df = None
    try:
        df = fs.df
    except Exception:
        df = None

    if df is not None:
        try:
            if isinstance(df, u.Quantity):
                val = df.value
                finite = np.isfinite(val)
                zero = np.all(val == 0)
            else:
                val = float(df)
                finite = np.isfinite(val)
                zero = (val == 0)
            regular = bool(finite and not zero)
        except Exception:
            regular = False

    f0 = None
    if regular:
        try:
            f0 = fs.f0
        except Exception:
            try:
                f0 = freqs[0]
            except Exception:
                f0 = None
        if f0 is None:
            regular = False

    try:
        n = len(freqs)
    except Exception:
        n = None
        regular = False

    return {"regular": regular, "df": df, "f0": f0, "n": n, "freqs": freqs}


def _validate_common_frequency_axis(axis_infos, method_name):
    """
    Validate common frequency axis across FrequencySeries results.
    """
    if not axis_infos:
        return None, None, None, 0

    all_regular = all(info["regular"] for info in axis_infos)
    if all_regular:
        ref = axis_infos[0]
        ref_df = ref["df"]
        ref_f0 = ref["f0"]
        ref_n = ref["n"]
        for info in axis_infos[1:]:
            if info["n"] != ref_n:
                raise ValueError(
                    f"{method_name} requires common length; mismatch in length"
                )
            if info["df"] != ref_df:
                raise ValueError(f"{method_name} requires common df; mismatch in df")
            if info["f0"] != ref_f0:
                raise ValueError(f"{method_name} requires common f0; mismatch in f0")
        return ref["freqs"], ref_df, ref_f0, ref_n

    ref_freqs = axis_infos[0]["freqs"]
    ref_unit = getattr(ref_freqs, "unit", None)
    ref_vals = ref_freqs.value if hasattr(ref_freqs, "value") else ref_freqs
    for info in axis_infos[1:]:
        freqs = info["freqs"]
        unit = getattr(freqs, "unit", None)
        if ref_unit is not None or unit is not None:
            if ref_unit != unit:
                raise ValueError(
                    f"{method_name} requires common frequencies unit; mismatch in frequencies"
                )
            lhs = freqs.value if hasattr(freqs, "value") else None
            rhs = ref_freqs.value if hasattr(ref_freqs, "value") else None
            if lhs is None or rhs is None:
                raise ValueError(
                    f"{method_name} requires comparable frequency arrays; mismatch in frequencies"
                )
            if not np.array_equal(lhs, rhs):
                raise ValueError(
                    f"{method_name} requires identical frequency arrays; mismatch in frequencies"
                )
        else:
            if not np.array_equal(freqs, ref_freqs):
                raise ValueError(
                    f"{method_name} requires identical frequency arrays; mismatch in frequencies"
                )
    return ref_freqs, None, None, len(ref_vals)


def _validate_common_epoch(epochs, method_name):
    if not epochs:
        return None
    ref = epochs[0]
    for e in epochs[1:]:
        if e != ref:
            raise ValueError(f"{method_name} requires common epoch; mismatch in epoch")
    return ref


class TimeSeries(BaseTimeSeries):
    """Light wrapper of gwpy's TimeSeries for compatibility."""

    pass


class TimeSeriesList(BaseTimeSeriesList):
    """List of TimeSeries objects."""

    pass


class TimeSeriesDict(BaseTimeSeriesDict):
    """Dictionary of TimeSeries objects."""

    def asd(self, fftlength=4, overlap=2):
        from gwexpy.frequencyseries import FrequencySeries

        dict_cls = getattr(FrequencySeries, "DictClass", None)
        if dict_cls is None:
            from gwexpy.frequencyseries import FrequencySeriesDict as dict_cls  # pragma: no cover

        return dict_cls(
            {
                key: ts.asd(fftlength=fftlength, overlap=overlap).view(FrequencySeries)
                for key, ts in self.items()
            }
        )


try:
    from gwpy.types.index import SeriesType  # pragma: no cover - optional in gwpy
except ImportError:  # fallback for gwpy versions without SeriesType

    class SeriesType(Enum):
        TIME = "time"
        FREQ = "freq"


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

        return super().__new__(cls, data, **kwargs)

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
            self.meta = meta_matrix
            self.xindex = common_axis
            return self

        result = TimeSeriesMatrix(
            out_data,
            times=common_axis,
            meta=meta_matrix,
            rows=self.rows,
            cols=self.cols,
            name=getattr(self, "name", ""),
        )
        result.epoch = getattr(self, "epoch", getattr(result, "epoch", None))
        return result

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
