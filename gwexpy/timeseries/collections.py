from astropy import units as u
from typing import Optional, Any
try:
    import scipy.signal
except ImportError:
    pass # scipy is optional dependency for gwpy but required here for hilbert


from gwpy.timeseries import TimeSeries as BaseTimeSeries
from gwpy.timeseries import TimeSeriesDict as BaseTimeSeriesDict
from gwpy.timeseries import TimeSeriesList as BaseTimeSeriesList


# --- Monkey Patch TimeSeriesDict ---


# New Imports
from .spectral import csd_matrix_from_collection, coherence_matrix_from_collection


from .utils import *
class TimeSeriesDict(BaseTimeSeriesDict):
    """Dictionary of TimeSeries objects."""

    def asfreq(self, rule, **kwargs):
        """
        Apply asfreq to each TimeSeries in the dict.
        Returns a new TimeSeriesDict.
        """
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.asfreq(rule, **kwargs)
        return new_dict

    def resample(self, rate, **kwargs):
        """
        Resample items in the TimeSeriesDict.
        In-place operation (updates the dict contents).

        If rate is time-like, performs time-bin resampling.
        Otherwise performs signal processing resampling (gwpy's native behavior).
        """
        is_time_bin = False
        if isinstance(rate, str):
            is_time_bin = True
        elif isinstance(rate, u.Quantity) and rate.unit.physical_type == 'time':
            is_time_bin = True

        if is_time_bin:
            # Time-bin logic: replace items in-place
            # We can't strictly modify the objects in-place easily
            # (asfreq/resample return new objects usually),
            # so we replace the values in the dict.
            for key in list(self.keys()):
                 self[key] = self[key].resample(rate, **kwargs)
            return self
        else:
            # Native gwpy resample (signal processing)
            # gwpy's TimeSeriesDict.resample is in-place
            return super().resample(rate, **kwargs)

    def analytic_signal(self, *args, **kwargs):
        """Apply analytic_signal to each item."""
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.analytic_signal(*args, **kwargs)
        return new_dict

    def hilbert(self, *args, **kwargs):
        """Alias for analytic_signal."""
        return self.analytic_signal(*args, **kwargs)

    def envelope(self, *args, **kwargs):
        """Apply envelope to each item."""
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.envelope(*args, **kwargs)
        return new_dict

    def instantaneous_phase(self, *args, **kwargs):
        """Apply instantaneous_phase to each item."""
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.instantaneous_phase(*args, **kwargs)
        return new_dict

        return new_dict

    def hht(self, *args, **kwargs):
        """
        Apply hht (Hilbert-Huang Transform) to each item.
        Returns a dict of results (either dicts or Spectrograms).
        """
        results = {}
        for key, ts in self.items():
            results[key] = ts.hht(*args, **kwargs)
        return results

    # ===============================
    # P2 Methods (Domain Specific)
    # ===============================

    def to_mne_rawarray(self, info=None, picks=None):
        """Convert to mne.io.RawArray."""
        from gwexpy.interop import to_mne_rawarray
        return to_mne_rawarray(self, info=info, picks=picks)

    def to_mne_raw(self, info=None, picks=None):
        """Alias for :meth:`to_mne_rawarray`."""
        return self.to_mne_rawarray(info=info, picks=picks)

    @classmethod
    def from_mne_raw(cls, raw, *, unit_map=None):
        """Create from mne.io.Raw."""
        from gwexpy.interop import from_mne_raw
        return from_mne_raw(cls, raw, unit_map=unit_map)

    def unwrap_phase(self, *args, **kwargs):
        """Apply unwrap_phase to each item."""
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.unwrap_phase(*args, **kwargs)
        return new_dict

    def instantaneous_frequency(self, *args, **kwargs):
        """Apply instantaneous_frequency to each item."""
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.instantaneous_frequency(*args, **kwargs)
        return new_dict

    def mix_down(self, *args, **kwargs):
        """Apply mix_down to each item."""
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.mix_down(*args, **kwargs)
        return new_dict

    def baseband(self, *args, **kwargs):
        """Apply baseband to each item."""
        new_dict = self.__class__()
        for key, ts in self.items():
             new_dict[key] = ts.baseband(*args, **kwargs)
        return new_dict

    def lock_in(self, *args, **kwargs):
        """
        Apply lock_in to each item.
        Returns TimeSeriesDict (if output='complex') or tuple of TimeSeriesDicts.
        """
        # We need to know the output structure (tuple vs single)
        # Peek first item
        if not self:
             return self.__class__()

        keys = list(self.keys())
        first_res = self[keys[0]].lock_in(*args, **kwargs)

        if isinstance(first_res, tuple):
             # Tuple return (e.g. mag, phase or i, q)
             # Assume logic dictates uniform return type
             dict_tuple = tuple(self.__class__() for _ in first_res)

             for key, ts in self.items():
                  res = ts.lock_in(*args, **kwargs)
                  for i, val in enumerate(res):
                       dict_tuple[i][key] = val
             return dict_tuple
        else:
             # Single return
             new_dict = self.__class__()
             new_dict[keys[0]] = first_res
             for key in keys[1:]:
                  new_dict[key] = self[key].lock_in(*args, **kwargs)
             return new_dict

    def csd_matrix(self, other=None, *, fftlength=None, overlap=None, window='hann', hermitian=True, include_diagonal=True, **kwargs):
        return csd_matrix_from_collection(
            self, other,
            fftlength=fftlength, overlap=overlap, window=window,
            hermitian=hermitian, include_diagonal=include_diagonal,
            **kwargs
        )

    def coherence_matrix(self, other=None, *, fftlength=None, overlap=None, window='hann', symmetric=True, include_diagonal=True, diagonal_value=1.0, **kwargs):
        return coherence_matrix_from_collection(
            self, other,
            fftlength=fftlength, overlap=overlap, window=window,
            symmetric=symmetric, include_diagonal=include_diagonal,
            diagonal_value=diagonal_value,
            **kwargs
        )

    def csd(self, other=None, *, fftlength=None, overlap=None, window="hann", hermitian=True, include_diagonal=True, **kwargs):
        """
        Compute CSD for each element or as a matrix depending on `other`.
        """
        if other is self:
            other = None
        if other is None or (isinstance(other, str) and other.lower() == "self"):
            return self.csd_matrix(
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                hermitian=hermitian,
                include_diagonal=include_diagonal,
                **kwargs,
            )

        if isinstance(other, BaseTimeSeries):
            from gwexpy.frequencyseries import FrequencySeriesDict
            new_dict = FrequencySeriesDict()
            for key, ts in self.items():
                new_dict[key] = ts.csd(
                    other, fftlength=fftlength, overlap=overlap, window=window, **kwargs
                )
            return new_dict

        if isinstance(other, (BaseTimeSeriesList, BaseTimeSeriesDict, list, dict)):
            return csd_matrix_from_collection(
                self,
                other,
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                hermitian=hermitian,
                include_diagonal=include_diagonal,
                **kwargs,
            )

        raise TypeError(
            "other must be TimeSeries, TimeSeriesList/Dict, or None/'self'"
        )

    def coherence(self, other=None, *, fftlength=None, overlap=None, window="hann", symmetric=True, include_diagonal=True, diagonal_value=1.0, **kwargs):
        """
        Compute coherence for each element or as a matrix depending on `other`.
        """
        if other is self:
            other = None
        if other is None or (isinstance(other, str) and other.lower() == "self"):
            return self.coherence_matrix(
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                symmetric=symmetric,
                include_diagonal=include_diagonal,
                diagonal_value=diagonal_value,
                **kwargs,
            )

        if isinstance(other, BaseTimeSeries):
            from gwexpy.frequencyseries import FrequencySeriesDict
            new_dict = FrequencySeriesDict()
            for key, ts in self.items():
                new_dict[key] = ts.coherence(
                    other, fftlength=fftlength, overlap=overlap, window=window, **kwargs
                )
            return new_dict

        if isinstance(other, (BaseTimeSeriesList, BaseTimeSeriesDict, list, dict)):
            return coherence_matrix_from_collection(
                self,
                other,
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                symmetric=symmetric,
                include_diagonal=include_diagonal,
                diagonal_value=diagonal_value,
                **kwargs,
            )

        raise TypeError(
            "other must be TimeSeries, TimeSeriesList/Dict, or None/'self'"
        )


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

    def spectrogram(self, *args, **kwargs):
        """
        Compute spectrogram for each TimeSeries in the dict.
        Returns a SpectrogramDict.
        """
        from gwexpy.spectrogram import SpectrogramDict
        new_dict = SpectrogramDict()
        for key, ts in self.items():
            new_dict[key] = ts.spectrogram(*args, **kwargs)
        return new_dict

    def spectrogram2(self, *args, **kwargs):
        """
        Compute spectrogram2 for each TimeSeries in the dict.
        Returns a SpectrogramDict.
        """
        from gwexpy.spectrogram import SpectrogramDict
        new_dict = SpectrogramDict()
        for key, ts in self.items():
            new_dict[key] = ts.spectrogram2(*args, **kwargs)
        return new_dict

    def q_transform(self, *args, **kwargs):
        """
        Compute Q-transform for each TimeSeries in the dict.
        Returns a SpectrogramDict.
        """
        from gwexpy.spectrogram import SpectrogramDict
        new_dict = SpectrogramDict()
        for key, ts in self.items():
            new_dict[key] = ts.q_transform(*args, **kwargs)
        return new_dict


    # ===============================
    # Interoperability Methods (P0)
    # ===============================

    def to_pandas(self, index="datetime", *, copy=False):
        """Convert to pandas.DataFrame."""
        from gwexpy.interop import to_pandas_dataframe
        return to_pandas_dataframe(self, index=index, copy=copy)

    @classmethod
    def from_pandas(cls, df, *, unit_map=None, t0=None, dt=None):
        """Create TimeSeriesDict from pandas.DataFrame."""
        from gwexpy.interop import from_pandas_dataframe
        return from_pandas_dataframe(cls, df, unit_map=unit_map, t0=t0, dt=dt)

    def to_polars(self, time_column="time", time_unit="datetime"):
        """Convert to polars.DataFrame."""
        from gwexpy.interop import to_polars_dict
        return to_polars_dict(self, time_column=time_column, time_unit=time_unit)

    @classmethod
    def from_polars(cls, df, *, time_column="time", unit_map=None):
        """Create TimeSeriesDict from polars.DataFrame."""
        from gwexpy.interop import from_polars_dict
        return from_polars_dict(cls, df, time_column=time_column, unit_map=unit_map)

    def to_tmultigraph(self, name: Optional[str] = None) -> Any:
        """Convert to ROOT TMultiGraph."""
        from gwexpy.interop import to_tmultigraph
        return to_tmultigraph(self, name=name)

    def write(self, target: str, *args: Any, **kwargs: Any) -> Any:
        fmt = kwargs.get("format")
        if fmt == "root" or (isinstance(target, str) and target.endswith(".root")):
             from gwexpy.interop.root_ import write_root_file
             return write_root_file(self, target, **kwargs)
        return super().write(target, *args, **kwargs)


    def impute(self, *, method="interpolate", limit=None, axis="time", max_gap=None, **kwargs):
        """Apply impute to each item."""
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.impute(method=method, limit=limit, axis=axis, max_gap=max_gap, **kwargs)
        return new_dict

    def rolling_mean(self, window, *, center=False, min_count=1, nan_policy="omit", backend="auto"):
        """Apply rolling mean to each item."""
        from gwexpy.timeseries.rolling import rolling_mean
        return rolling_mean(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)

    def rolling_std(self, window, *, center=False, min_count=1, nan_policy="omit", backend="auto", ddof=0):
        """Apply rolling std to each item."""
        from gwexpy.timeseries.rolling import rolling_std
        return rolling_std(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend, ddof=ddof)

    def rolling_median(self, window, *, center=False, min_count=1, nan_policy="omit", backend="auto"):
        """Apply rolling median to each item."""
        from gwexpy.timeseries.rolling import rolling_median
        return rolling_median(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)

    def rolling_min(self, window, *, center=False, min_count=1, nan_policy="omit", backend="auto"):
        """Apply rolling min to each item."""
        from gwexpy.timeseries.rolling import rolling_min
        return rolling_min(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)

    def rolling_max(self, window, *, center=False, min_count=1, nan_policy="omit", backend="auto"):
        """Apply rolling max to each item."""
        from gwexpy.timeseries.rolling import rolling_max
        return rolling_max(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)

    def to_matrix(self, *, align="intersection", **kwargs):
        """
        Convert dictionary to TimeSeriesMatrix with alignment.
        """
        from gwexpy.timeseries.preprocess import align_timeseries_collection
        # Ensure consistent order (keys sorted) or specific
        # Dicts are ordered in modern python but keys() usually safe
        keys = list(self.keys())
        series_list = [self[k] for k in keys]

        vals, times, meta = align_timeseries_collection(series_list, how=align, **kwargs)

        # SeriesMatrix expects 3D usually (rows, cols, time) or checks last axis
        # vals: (samples, channels).
        # We create (channels, 1, samples).
        data = vals.T[:, None, :]

        from .matrix import TimeSeriesMatrix
        matrix = TimeSeriesMatrix(
            data,
            times=times,
            # meta might contain channel_names from original list (names of TS objects)
            # But converting dict to matrix usually implies keys become channel names?
            # User requirement: "preserve labels"
            # TimeSeries from dict usually inherit name from key if created via read?
            # Not always. We should force keys as names?
            # "Must preserve channel ordering from input."
        )
        # Assign channel names from keys
        matrix.channel_names = keys
        return matrix


    # ===============================
    # Batch Processing Methods (P1)
    # ===============================

    # --- Waveform Operations ---

    def crop(self, start=None, end=None, copy=False) -> "TimeSeriesDict":
        """
        Crop each TimeSeries in the dict.
        Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).
        Returns a new TimeSeriesDict.
        """
        from gwexpy.time import to_gps
        # Convert inputs to GPS if provided
        if start is not None:
            start = float(to_gps(start))
        if end is not None:
            end = float(to_gps(end))

        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.crop(start=start, end=end, copy=copy)
        return new_dict

    def append(self, other, copy=True, **kwargs) -> "TimeSeriesDict":
        """
        Append another mapping of TimeSeries or a single TimeSeries to each item.
        """
        if isinstance(other, BaseTimeSeries):
             for ts in self.values():
                 ts.append(other, **kwargs)
             return self
        return super().append(other, copy=copy, **kwargs)

    def prepend(self, *args, **kwargs) -> "TimeSeriesDict":
        """
        Prepend to each TimeSeries in the dict (in-place).
        Returns self.
        """
        for ts in self.values():
            ts.prepend(*args, **kwargs)
        return self

    # def update(self, *args, **kwargs) -> "TimeSeriesDict":
    #     """
    #     Update each TimeSeries in the dict (in-place).
    #     Returns self.
    #     """
    #     for ts in self.values():
    #         ts.update(*args, **kwargs)
    #     return self

    def shift(self, *args, **kwargs) -> "TimeSeriesDict":
        """
        Shift each TimeSeries in the dict.
        Returns a new TimeSeriesDict.
        """
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.shift(*args, **kwargs)
        return new_dict

    def gate(self, *args, **kwargs) -> "TimeSeriesDict":
        """
        Gate each TimeSeries in the dict.
        Returns a new TimeSeriesDict.
        """
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.gate(*args, **kwargs)
        return new_dict

    def mask(self, *args, **kwargs) -> "TimeSeriesDict":
        """
        Mask each TimeSeries in the dict.
        Returns a new TimeSeriesDict.
        """
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.mask(*args, **kwargs)
        return new_dict

    # --- Signal Processing ---

    def decimate(self, *args, **kwargs) -> "TimeSeriesDict":
        """
        Decimate each TimeSeries in the dict.
        Returns a new TimeSeriesDict.
        """
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.decimate(*args, **kwargs)
        return new_dict

    def filter(self, *args, **kwargs) -> "TimeSeriesDict":
        """
        Filter each TimeSeries in the dict.
        Returns a new TimeSeriesDict.
        """
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.filter(*args, **kwargs)
        return new_dict

    def whiten(self, *args, **kwargs) -> "TimeSeriesDict":
        """
        Whiten each TimeSeries in the dict.
        Returns a new TimeSeriesDict.
        """
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.whiten(*args, **kwargs)
        return new_dict

    def notch(self, *args, **kwargs) -> "TimeSeriesDict":
        """
        Notch filter each TimeSeries in the dict.
        Returns a new TimeSeriesDict.
        """
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.notch(*args, **kwargs)
        return new_dict

    def zpk(self, *args, **kwargs) -> "TimeSeriesDict":
        """
        Apply ZPK filter to each TimeSeries in the dict.
        Returns a new TimeSeriesDict.
        """
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.zpk(*args, **kwargs)
        return new_dict

    def detrend(self, *args, **kwargs) -> "TimeSeriesDict":
        """
        Detrend each TimeSeries in the dict.
        Returns a new TimeSeriesDict.
        """
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.detrend(*args, **kwargs)
        return new_dict

    def taper(self, *args, **kwargs) -> "TimeSeriesDict":
        """
        Taper each TimeSeries in the dict.
        Returns a new TimeSeriesDict.
        """
        new_dict = self.__class__()
        for key, ts in self.items():
            new_dict[key] = ts.taper(*args, **kwargs)
        return new_dict

    # --- Spectral Conversion ---

    def fft(self, *args, **kwargs):
        """
        Apply FFT to each TimeSeries in the dict.
        Returns a FrequencySeriesDict.
        """
        from gwexpy.frequencyseries import FrequencySeriesDict
        new_dict = FrequencySeriesDict()
        for key, ts in self.items():
            new_dict[key] = ts.fft(*args, **kwargs)
        return new_dict

    def average_fft(self, *args, **kwargs):
        """
        Apply averge_fft to each TimeSeries in the dict.
        Returns a FrequencySeriesDict.
        """
        from gwexpy.frequencyseries import FrequencySeriesDict
        new_dict = FrequencySeriesDict()
        for key, ts in self.items():
            new_dict[key] = ts.average_fft(*args, **kwargs)
        return new_dict

    def psd(self, *args, **kwargs):
        """
        Compute PSD for each TimeSeries in the dict.
        Returns a FrequencySeriesDict.
        """
        from gwexpy.frequencyseries import FrequencySeriesDict
        new_dict = FrequencySeriesDict()
        for key, ts in self.items():
            new_dict[key] = ts.psd(*args, **kwargs)
        return new_dict

    # --- Statistics & Measurements ---

    def _apply_scalar_or_map(self, method_name, *args, **kwargs):
        """
        Internal: apply a method that can return TimeSeries or scalar.
        If TimeSeries -> return TimeSeriesDict.
        If scalar -> return pandas.Series.
        """
        import pandas as pd
        results = {}
        is_ts = False
        first = True

        for key, ts in self.items():
            method = getattr(ts, method_name)
            res = method(*args, **kwargs)

            if first:
                first = False
                # Check for TimeSeries-like structure
                if hasattr(res, "value") and hasattr(res, "dt"):
                    is_ts = True
                    results = self.__class__()

            if is_ts:
                # Ensure consistency
                if not (hasattr(res, "value") and hasattr(res, "dt")):
                     # Mixed types not supported cleanly here, defaulting to dict of objects
                     pass

            results[key] = res

        if is_ts:
            return results
        else:
            return pd.Series(results)

    def rms(self, *args, **kwargs):
        return self._apply_scalar_or_map("rms", *args, **kwargs)

    def min(self, *args, **kwargs):
        return self._apply_scalar_or_map("min", *args, **kwargs)

    def max(self, *args, **kwargs):
        return self._apply_scalar_or_map("max", *args, **kwargs)

    def mean(self, *args, **kwargs):
        return self._apply_scalar_or_map("mean", *args, **kwargs)

    def std(self, *args, **kwargs):
        return self._apply_scalar_or_map("std", *args, **kwargs)

    def value_at(self, *args, **kwargs):
        return self._apply_scalar_or_map("value_at", *args, **kwargs)

    def is_contiguous(self, *args, **kwargs):
        return self._apply_scalar_or_map("is_contiguous", *args, **kwargs)

    # --- State Analysis ---

    def state_segments(self, *args, **kwargs):
        """Run state_segments on each item (returns Series of SegmentLists)."""
        return self._apply_scalar_or_map("state_segments", *args, **kwargs)

    # --- Multivariate ---

    def pca(self, *args, **kwargs):
        """Perform PCA decomposition across channels."""
        return self.to_matrix().pca(*args, **kwargs)

    def ica(self, *args, **kwargs):
        """Perform ICA decomposition across channels."""
        return self.to_matrix().ica(*args, **kwargs)


class TimeSeriesList(BaseTimeSeriesList):
    """List of TimeSeries objects."""

    def csd_matrix(self, other=None, *, fftlength=None, overlap=None, window='hann', hermitian=True, include_diagonal=True, **kwargs):
        """
        Compute Cross Spectral Density Matrix.

        Parameters
        ----------
        other : TimeSeriesDict or TimeSeriesList, optional
            Other collection for cross-CSD.
        fftlength, overlap, window :
            See TimeSeries.csd() arguments.
        hermitian : bool, default=True
            If True and other is None, compute only upper triangle and conjugate fill lower.
        include_diagonal : bool, default=True
            Whether to compute diagonal elements.

        Returns
        -------
        FrequencySeriesMatrix
        """
        return csd_matrix_from_collection(
            self, other,
            fftlength=fftlength, overlap=overlap, window=window,
            hermitian=hermitian, include_diagonal=include_diagonal,
            **kwargs
        )

    def coherence_matrix(self, other=None, *, fftlength=None, overlap=None, window='hann', symmetric=True, include_diagonal=True, diagonal_value=1.0, **kwargs):
        """
        Compute Coherence Matrix.

        Parameters
        ----------
        other : TimeSeriesDict or TimeSeriesList, optional
            Other collection.
        fftlength, overlap, window :
            See TimeSeries.coherence().
        symmetric : bool, default=True
            If True and other is None, compute only upper triangle and copy to lower.
        include_diagonal : bool, default=True
            Include diagonal.
        diagonal_value : float or None, default=1.0
            Value to fill diagonal if include_diagonal is True. If None, compute diagonal coherence.

        Returns
        -------
        FrequencySeriesMatrix
        """
        return coherence_matrix_from_collection(
            self, other,
            fftlength=fftlength, overlap=overlap, window=window,
            symmetric=symmetric, include_diagonal=include_diagonal,
            diagonal_value=diagonal_value,
            **kwargs
        )

    def csd(self, other=None, *, fftlength=None, overlap=None, window="hann", hermitian=True, include_diagonal=True, **kwargs):
        """
        Compute CSD for each element or as a matrix depending on `other`.
        """
        if other is self:
            other = None
        if other is None or (isinstance(other, str) and other.lower() == "self"):
            return self.csd_matrix(
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                hermitian=hermitian,
                include_diagonal=include_diagonal,
                **kwargs,
            )

        if isinstance(other, BaseTimeSeries):
            from gwexpy.frequencyseries import FrequencySeriesList
            new_list = FrequencySeriesList()
            for ts in self:
                list.append(new_list, ts.csd(
                    other, fftlength=fftlength, overlap=overlap, window=window, **kwargs
                ))
            return new_list

        if isinstance(other, (BaseTimeSeriesList, BaseTimeSeriesDict, list, dict)):
            return csd_matrix_from_collection(
                self,
                other,
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                hermitian=hermitian,
                include_diagonal=include_diagonal,
                **kwargs,
            )

        raise TypeError(
            "other must be TimeSeries, TimeSeriesList/Dict, or None/'self'"
        )

    def coherence(self, other=None, *, fftlength=None, overlap=None, window="hann", symmetric=True, include_diagonal=True, diagonal_value=1.0, **kwargs):
        """
        Compute coherence for each element or as a matrix depending on `other`.
        """
        if other is self:
            other = None
        if other is None or (isinstance(other, str) and other.lower() == "self"):
            return self.coherence_matrix(
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                symmetric=symmetric,
                include_diagonal=include_diagonal,
                diagonal_value=diagonal_value,
                **kwargs,
            )

        if isinstance(other, BaseTimeSeries):
            from gwexpy.frequencyseries import FrequencySeriesList
            new_list = FrequencySeriesList()
            for ts in self:
                list.append(new_list, ts.coherence(
                    other, fftlength=fftlength, overlap=overlap, window=window, **kwargs
                ))
            return new_list

        if isinstance(other, (BaseTimeSeriesList, BaseTimeSeriesDict, list, dict)):
            return coherence_matrix_from_collection(
                self,
                other,
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                symmetric=symmetric,
                include_diagonal=include_diagonal,
                diagonal_value=diagonal_value,
                **kwargs,
            )

        raise TypeError(
            "other must be TimeSeries, TimeSeriesList/Dict, or None/'self'"
        )

    def impute(self, *, method="interpolate", limit=None, axis="time", max_gap=None, **kwargs):
         new_list = self.__class__()
         for ts in self:
             list.append(new_list, ts.impute(method=method, limit=limit, axis=axis, max_gap=max_gap, **kwargs))
         return new_list

    def rolling_mean(self, window, *, center=False, min_count=1, nan_policy="omit", backend="auto"):
        """Apply rolling mean to each element."""
        from gwexpy.timeseries.rolling import rolling_mean
        return rolling_mean(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)

    def rolling_std(self, window, *, center=False, min_count=1, nan_policy="omit", backend="auto", ddof=0):
        """Apply rolling std to each element."""
        from gwexpy.timeseries.rolling import rolling_std
        return rolling_std(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend, ddof=ddof)

    def rolling_median(self, window, *, center=False, min_count=1, nan_policy="omit", backend="auto"):
        """Apply rolling median to each element."""
        from gwexpy.timeseries.rolling import rolling_median
        return rolling_median(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)

    def rolling_min(self, window, *, center=False, min_count=1, nan_policy="omit", backend="auto"):
        """Apply rolling min to each element."""
        from gwexpy.timeseries.rolling import rolling_min
        return rolling_min(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)

    def rolling_max(self, window, *, center=False, min_count=1, nan_policy="omit", backend="auto"):
        """Apply rolling max to each element."""
        from gwexpy.timeseries.rolling import rolling_max
        return rolling_max(self, window, center=center, min_count=min_count, nan_policy=nan_policy, backend=backend)

    def to_matrix(self, *, align="intersection", **kwargs):
        from gwexpy.timeseries.preprocess import align_timeseries_collection
        vals, times, meta = align_timeseries_collection(list(self), how=align, **kwargs)
        # Use names from metadata (from TS objects)
        names = meta.get("channel_names")

        data = vals.T[:, None, :]

        matrix = TimeSeriesMatrix(
            data,
            times=times,
        )
        if names:
             matrix.channel_names = names
        return matrix

    # ===============================
    # Batch Processing Methods (P1)
    # ===============================

    # --- Waveform Operations ---

    def crop(self, start=None, end=None, copy=False) -> "TimeSeriesList":
        """
        Crop each TimeSeries in the list.
        Accepts any time format supported by gwexpy.time.to_gps (str, datetime, pandas, obspy, etc).
        Returns a new TimeSeriesList.
        """
        from gwexpy.time import to_gps
        # Convert inputs to GPS if provided
        if start is not None:
            start = float(to_gps(start))
        if end is not None:
            end = float(to_gps(end))

        new_list = self.__class__()
        for ts in self:
            list.append(new_list, ts.crop(start=start, end=end, copy=copy))
        return new_list

    # def append(self, *args, **kwargs) -> "TimeSeriesList":
    #     """
    #     Append to each TimeSeries in the list (in-place).
    #     Returns self.
    #     """
    #     for ts in self:
    #         ts.append(*args, **kwargs)
    #     return self

    # def prepend(self, *args, **kwargs) -> "TimeSeriesList":
    #     """
    #     Prepend to each TimeSeries in the list (in-place).
    #     Returns self.
    #     """
    #     for ts in self:
    #         ts.prepend(*args, **kwargs)
    #     return self

    # def update(self, *args, **kwargs) -> "TimeSeriesList":
    #     """
    #     Update each TimeSeries in the list (in-place).
    #     Returns self.
    #     """
    #     for ts in self:
    #         ts.update(*args, **kwargs)
    #     return self

    def shift(self, *args, **kwargs) -> "TimeSeriesList":
        """
        Shift each TimeSeries in the list.
        Returns a new TimeSeriesList.
        """
        new_list = self.__class__()
        for ts in self:
            list.append(new_list, ts.shift(*args, **kwargs))
        return new_list

    def gate(self, *args, **kwargs) -> "TimeSeriesList":
        """
        Gate each TimeSeries in the list.
        Returns a new TimeSeriesList.
        """
        new_list = self.__class__()
        for ts in self:
            list.append(new_list, ts.gate(*args, **kwargs))
        return new_list

    def mask(self, *args, **kwargs) -> "TimeSeriesList":
        """
        Mask each TimeSeries in the list.
        Returns a new TimeSeriesList.
        """
        new_list = self.__class__()
        for ts in self:
            list.append(new_list, ts.mask(*args, **kwargs))
        return new_list

    # --- Signal Processing ---

    def resample(self, *args, **kwargs) -> "TimeSeriesList":
        """
        Resample each TimeSeries in the list.
        Returns a new TimeSeriesList.
        """
        new_list = self.__class__()
        for ts in self:
            list.append(new_list, ts.resample(*args, **kwargs))
        return new_list

    def decimate(self, *args, **kwargs) -> "TimeSeriesList":
        """
        Decimate each TimeSeries in the list.
        Returns a new TimeSeriesList.
        """
        new_list = self.__class__()
        for ts in self:
            list.append(new_list, ts.decimate(*args, **kwargs))
        return new_list

    def filter(self, *args, **kwargs) -> "TimeSeriesList":
        """
        Filter each TimeSeries in the list.
        Returns a new TimeSeriesList.
        """
        new_list = self.__class__()
        for ts in self:
            list.append(new_list, ts.filter(*args, **kwargs))
        return new_list

    def whiten(self, *args, **kwargs) -> "TimeSeriesList":
        """
        Whiten each TimeSeries in the list.
        Returns a new TimeSeriesList.
        """
        new_list = self.__class__()
        for ts in self:
            list.append(new_list, ts.whiten(*args, **kwargs))
        return new_list

    def notch(self, *args, **kwargs) -> "TimeSeriesList":
        """
        Notch filter each TimeSeries in the list.
        Returns a new TimeSeriesList.
        """
        new_list = self.__class__()
        for ts in self:
            list.append(new_list, ts.notch(*args, **kwargs))
        return new_list

    def zpk(self, *args, **kwargs) -> "TimeSeriesList":
        """
        ZPK filter each TimeSeries in the list.
        Returns a new TimeSeriesList.
        """
        new_list = self.__class__()
        for ts in self:
            list.append(new_list, ts.zpk(*args, **kwargs))
        return new_list

    def detrend(self, *args, **kwargs) -> "TimeSeriesList":
        """
        Detrend each TimeSeries in the list.
        Returns a new TimeSeriesList.
        """
        new_list = self.__class__()
        for ts in self:
            list.append(new_list, ts.detrend(*args, **kwargs))
        return new_list

    def taper(self, *args, **kwargs) -> "TimeSeriesList":
        """
        Taper each TimeSeries in the list.
        Returns a new TimeSeriesList.
        """
        new_list = self.__class__()
        for ts in self:
            list.append(new_list, ts.taper(*args, **kwargs))
        return new_list

    # --- Spectral Conversion ---

    def fft(self, *args, **kwargs):
        """
        Apply FFT to each TimeSeries in the list.
        Returns a FrequencySeriesList.
        """
        from gwexpy.frequencyseries import FrequencySeriesList
        new_list = FrequencySeriesList()
        for ts in self:
            list.append(new_list, ts.fft(*args, **kwargs))
        return new_list

    def average_fft(self, *args, **kwargs):
        """
        Apply average_fft to each TimeSeries in the list.
        Returns a FrequencySeriesList.
        """
        from gwexpy.frequencyseries import FrequencySeriesList
        new_list = FrequencySeriesList()
        for ts in self:
            list.append(new_list, ts.average_fft(*args, **kwargs))
        return new_list

    def psd(self, *args, **kwargs):
        """
        Compute PSD for each TimeSeries in the list.
        Returns a FrequencySeriesList.
        """
        from gwexpy.frequencyseries import FrequencySeriesList
        new_list = FrequencySeriesList()
        for ts in self:
            list.append(new_list, ts.psd(*args, **kwargs))
        return new_list

    def asd(self, *args, **kwargs):
        """
        Compute ASD for each TimeSeries in the list.
        Returns a FrequencySeriesList.
        """
        from gwexpy.frequencyseries import FrequencySeriesList
        new_list = FrequencySeriesList()
        for ts in self:
            list.append(new_list, ts.asd(*args, **kwargs))
        return new_list

    def spectrogram(self, *args, **kwargs):
        """
        Compute spectrogram for each TimeSeries in the list.
        Returns a SpectrogramList.
        """
        from gwexpy.spectrogram import SpectrogramList
        new_list = SpectrogramList()
        for ts in self:
            new_list.append(ts.spectrogram(*args, **kwargs))
        return new_list

    def spectrogram2(self, *args, **kwargs):
        """
        Compute spectrogram2 for each TimeSeries in the list.
        Returns a SpectrogramList.
        """
        from gwexpy.spectrogram import SpectrogramList
        new_list = SpectrogramList()
        for ts in self:
            new_list.append(ts.spectrogram2(*args, **kwargs))
        return new_list

    def q_transform(self, *args, **kwargs):
        """
        Compute Q-transform for each TimeSeries in the list.
        Returns a SpectrogramList.
        """
        from gwexpy.spectrogram import SpectrogramList
        new_list = SpectrogramList()
        for ts in self:
            new_list.append(ts.q_transform(*args, **kwargs))
        return new_list

    # --- Statistics & Measurements ---

    def _apply_scalar_or_map(self, method_name, *args, **kwargs):
        """
        Internal: apply a method that can return TimeSeries or scalar.
        If TimeSeries -> return TimeSeriesList.
        If scalar -> return list.
        """
        results = []
        is_ts = False
        first = True

        for ts in self:
            method = getattr(ts, method_name)
            res = method(*args, **kwargs)

            if first:
                first = False
                if hasattr(res, "value") and hasattr(res, "dt"):
                    is_ts = True
                    results = self.__class__()

            if is_ts:
                # Type check?
                pass

            if isinstance(results, self.__class__):
                list.append(results, res)
            else:
                list.append(results, res)

        return results

    def rms(self, *args, **kwargs):
        return self._apply_scalar_or_map("rms", *args, **kwargs)

    def min(self, *args, **kwargs):
        return self._apply_scalar_or_map("min", *args, **kwargs)

    def max(self, *args, **kwargs):
        return self._apply_scalar_or_map("max", *args, **kwargs)

    def mean(self, *args, **kwargs):
        return self._apply_scalar_or_map("mean", *args, **kwargs)

    def std(self, *args, **kwargs):
        return self._apply_scalar_or_map("std", *args, **kwargs)

    def value_at(self, *args, **kwargs):
        return self._apply_scalar_or_map("value_at", *args, **kwargs)

    def is_contiguous(self, *args, **kwargs):
        return self._apply_scalar_or_map("is_contiguous", *args, **kwargs)

    # --- State Analysis ---

    def state_segments(self, *args, **kwargs):
        return self._apply_scalar_or_map("state_segments", *args, **kwargs)

    # --- Multivariate ---

    def to_pandas(self, **kwargs):
        """
        Convert TimeSeriesList to pandas DataFrame.
        Each element becomes a column.
        ASSUMES common time axis.
        """
        import pandas as pd

        data = {}
        for i, ts in enumerate(self):
            name = ts.name or f"series_{i}"
            if hasattr(ts, "to_pandas"):
                 s = ts.to_pandas()
            else:
                 s = pd.Series(ts.value, index=ts.times.value)

            data[name] = s

        return pd.DataFrame(data)


    def to_tmultigraph(self, name: Optional[str] = None) -> Any:
        """Convert to ROOT TMultiGraph."""
        from gwexpy.interop import to_tmultigraph
        return to_tmultigraph(self, name=name)

    def write(self, target: str, *args: Any, **kwargs: Any) -> Any:
        fmt = kwargs.get("format")
        if fmt == "root" or (isinstance(target, str) and target.endswith(".root")):
             from gwexpy.interop.root_ import write_root_file
             return write_root_file(self, target, **kwargs)
        from astropy.io import registry
        return registry.write(self, target, *args, **kwargs)

    def pca(self, *args, **kwargs):
        """Perform PCA decomposition across channels."""
        return self.to_matrix().pca(*args, **kwargs)

    def ica(self, *args, **kwargs):
        """Perform ICA decomposition across channels."""
        return self.to_matrix().ica(*args, **kwargs)

