from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsIndex

import numpy as np
from astropy import units as u
from gwpy.spectrogram import Spectrogram as BaseSpectrogram

from gwexpy.types.mixin import InteropMixin, PhaseMethodsMixin
from gwexpy.types.mixin._plot_mixin import PlotMixin

if TYPE_CHECKING:
    from astropy.units import Quantity

    from gwexpy.frequencyseries import FrequencySeriesList
    from gwexpy.timeseries import TimeSeriesList


class Spectrogram(PlotMixin, PhaseMethodsMixin, InteropMixin, BaseSpectrogram):
    """A 2D time-frequency spectrogram.

    `Spectrogram` represents a 2-dimensional array of spectral data,
    where the first dimension corresponds to time and the second
    dimension corresponds to frequency. It extends the core GWpy
    `~gwpy.spectrogram.Spectrogram` with additional signal processing
    (e.g., bootstrap ASD estimation, cleaning) and interoperability
    methods.

    Parameters
    ----------
    data : array-like
        2D array of spectral data.

    times : array-like, optional
        Time values corresponding to each row. If provided, `dt` and
        `t0` are ignored.

    dt : `float`, `~astropy.units.Quantity`, optional
        Time step between rows.

    t0 : `float`, `~astropy.units.Quantity`, optional
        Start time of the data.

    frequencies : array-like, optional
        Frequency values corresponding to each column. If provided,
        `df` and `f0` are ignored.

    df : `float`, `~astropy.units.Quantity`, optional
        Frequency step between columns.

    f0 : `float`, `~astropy.units.Quantity`, optional
        Start frequency of the data.

    **kwargs
        Additional keyword arguments passed to the
        `~gwpy.spectrogram.Spectrogram` constructor.

    Notes
    -----
    Key methods:

    .. autosummary::

       ~Spectrogram.plot
       ~Spectrogram.imshow
       ~Spectrogram.pcolormesh
       ~Spectrogram.bootstrap
       ~Spectrogram.normalize
       ~Spectrogram.clean
       ~Spectrogram.rebin

    Examples
    --------
    >>> from gwexpy.spectrogram import Spectrogram
    >>> import numpy as np
    >>> data = np.ones((2, 2))
    >>> spec = Spectrogram(data, dt=1, f0=0, df=1)
    >>> spec
    <Spectrogram([[1., 1.],
                  [1., 1.]],
                 unit=Unit(dimensionless),
                 name=None,
                 epoch=<Time object: scale='utc' format='gps' value=0.0>,
                 channel=None,
                 x0=<Quantity 0. s>,
                 dx=<Quantity 1. s>,
                 xindex=<Index [0., 1.] s>,
                 y0=<Quantity 0. Hz>,
                 dy=<Quantity 1. Hz>,
                 yindex=<Index [0., 1.] Hz>)>
    """

    def __getitem__(self, item: Any) -> Any:
        """
        Custom getitem to handle 1D views safely.

        In some environments (e.g. newer matplotlib/astropy), iterating over a
        2D Spectrogram yields 1D Spectrogram views. GWpy's Array2D.__getitem__
        implementation expects 2D data and fails with a ValueError when
        unpacking slices for ndim=1. This override detects ndim=1 and falls
        back to the parent (Series) implementation.
        """
        if self.ndim == 1:
            from gwpy.types.series import Series

            return Series.__getitem__(self, item)
        return super().__getitem__(item)

    def __reduce_ex__(self, protocol: SupportsIndex):
        from gwexpy.io.pickle_compat import spectrogram_reduce_args

        return spectrogram_reduce_args(self)

    def bootstrap(
        self,
        n_boot=1000,
        method="median",
        average=None,
        ci=0.68,
        window="hann",
        fftlength=None,
        overlap=None,
        nfft=None,
        noverlap=None,
        block_size=None,
        rebin_width=None,
        return_map=False,
        ignore_nan=True,
        **kwargs,
    ):
        """
        Estimate robust ASD from this spectrogram using bootstrap resampling.

        This is a convenience wrapper around `gwexpy.spectral.bootstrap_spectrogram`.

        Parameters
        ----------
        fftlength : float or Quantity, optional
            FFT segment length in seconds (e.g. ``1.0`` or ``1.0 * u.s``).
            Used for VIF overlap-correlation correction. If None, the
            correction is estimated from spectrogram metadata.
            Cannot be used with `nfft`.
        overlap : float or Quantity, optional
            Overlap between FFT segments in seconds. If None, defaults to
            the recommended overlap for *window* (50 % for Hann).
            Cannot be used with `noverlap`.
        nfft : int, optional
            FFT segment length in samples. Alternative to `fftlength`.
            Cannot be used with `fftlength`.
        noverlap : int, optional
            Overlap length in samples. Must be used with `nfft`.
            Cannot be used with `overlap`.
        block_size : float, Quantity, or 'auto', optional
            Duration of blocks for block bootstrap in seconds. Can be
            specified as float (seconds), Quantity with time units, or 'auto'.
            If 'auto', estimates size based on overlap ratio. If None,
            perform standard bootstrap with analytical VIF correction.
        rebin_width : float, optional
            Frequency rebinning width in Hz before bootstrapping.
        **kwargs
            Additional keyword arguments. Passing the removed ``nperseg``
            or ``noverlap`` parameters will raise :class:`TypeError`.

        Examples
        --------
        >>> from gwexpy.spectrogram import Spectrogram
        >>> import numpy as np
        >>> from astropy import units as u
        >>>
        >>> # Create synthetic spectrogram
        >>> np.random.seed(42)
        >>> spec_data = np.random.random((100, 50))
        >>> spec = Spectrogram(spec_data, dt=1.0*u.s, f0=10*u.Hz, df=1*u.Hz)
        >>>
        >>> # Bootstrap estimation with time-based parameters
        >>> result = spec.bootstrap(
        ...     n_boot=100,
        ...     fftlength=4.0,    # 4 seconds
        ...     overlap=2.0,      # 2 seconds
        ...     block_size=2.0,   # 2 seconds block
        ...     window='hann',
        ...     method='median'
        ... )
        >>> print(result.value.shape)
        (50,)
        """
        from gwexpy.spectral import bootstrap_spectrogram
        from gwexpy.utils.fft_args import check_deprecated_kwargs

        check_deprecated_kwargs(**kwargs)

        if average is not None:
            method = average

        return bootstrap_spectrogram(
            self,
            n_boot=n_boot,
            method=method,
            average=None,
            ci=ci,
            window=window,
            fftlength=fftlength,
            overlap=overlap,
            nfft=nfft,
            noverlap=noverlap,
            block_size=block_size,
            rebin_width=rebin_width,
            return_map=return_map,
            ignore_nan=ignore_nan,
        )

    def bootstrap_asd(
        self,
        n_boot=1000,
        average="median",
        ci=0.68,
        window="hann",
        fftlength=None,
        overlap=None,
        nfft=None,
        noverlap=None,
        block_size=None,
        rebin_width=None,
        return_map=False,
        ignore_nan=True,
        **kwargs,
    ):
        """
        Convenience wrapper for bootstrap ASD estimation.

        Parameters
        ----------
        fftlength : float or Quantity, optional
            FFT segment length in seconds (e.g. ``1.0`` or ``1.0 * u.s``).
            Used for VIF overlap-correlation correction. Cannot be used with `nfft`.
        overlap : float or Quantity, optional
            Overlap between FFT segments in seconds. Cannot be used with `noverlap`.
        nfft : int, optional
            FFT segment length in samples. Alternative to `fftlength`.
        noverlap : int, optional
            Overlap length in samples. Must be used with `nfft`.
        block_size : float, Quantity, or 'auto', optional
            Duration of blocks for block bootstrap in seconds. Can be
            specified as float (seconds), Quantity with time units, or 'auto'.
        rebin_width : float, optional
            Frequency rebinning width in Hz before bootstrapping.
        **kwargs
            Additional keyword arguments. Passing the removed ``nperseg``
            or ``noverlap`` parameters will raise :class:`TypeError`.

        Examples
        --------
        >>> from gwexpy.spectrogram import Spectrogram
        >>> import numpy as np
        >>> from astropy import units as u
        >>>
        >>> # Create synthetic spectrogram
        >>> np.random.seed(42)
        >>> spec_data = np.random.random((100, 50))
        >>> spec = Spectrogram(spec_data, dt=1.0*u.s, f0=10*u.Hz, df=1*u.Hz)
        >>>
        >>> # Bootstrap ASD estimation
        >>> result = spec.bootstrap_asd(
        ...     n_boot=100,
        ...     fftlength=4.0,    # 4 seconds
        ...     overlap=2.0,      # 2 seconds
        ...     block_size=2.0,   # 2 seconds block
        ...     window='hann',
        ...     average='median'
        ... )
        >>> print(result.value.shape)
        (50,)
        """
        return self.bootstrap(
            n_boot=n_boot,
            method=average,
            ci=ci,
            window=window,
            fftlength=fftlength,
            overlap=overlap,
            nfft=nfft,
            noverlap=noverlap,
            block_size=block_size,
            rebin_width=rebin_width,
            return_map=return_map,
            ignore_nan=ignore_nan,
            **kwargs,
        )

    def to_th2d(self, error=None):
        """
        Convert to ROOT TH2D.
        """
        from gwexpy.interop import to_th2d

        return to_th2d(self, error=error)

    def to_quantities(self, units=None):
        """
        Convert to quantities.Quantity (Elephant/Neo compatible).
        """
        from gwexpy.interop import to_quantity

        return to_quantity(self, units=units)

    @classmethod
    def from_quantities(cls, q, times, frequencies):
        """
        Create Spectrogram from quantities.Quantity.

        Parameters
        ----------
        q : quantities.Quantity
            Input data (Time x Frequency matrix).
        times : array-like
            Time axis.
        frequencies : array-like
            Frequency axis.
        """
        from gwexpy.interop import from_quantity

        return from_quantity(cls, q, times=times, frequencies=frequencies)

    @classmethod
    def from_root(cls, obj, return_error=False):
        """
        Create Spectrogram from ROOT TH2D.
        """
        from gwexpy.interop import from_root

        return from_root(cls, obj, return_error=return_error)

    def to_mne(self, info: Any | None = None) -> Any:
        """
        Convert to MNE-Python object.

        Parameters
        ----------
        info : mne.Info, optional
            MNE Info object.

        Returns
        -------
        mne.time_frequency.EpochsTFRArray
        """
        from gwexpy.interop import to_mne

        return to_mne(self, info=info)

    @classmethod
    def from_mne(cls, tfr: Any, **kwargs: Any) -> Any:
        """
        Create Spectrogram from MNE-Python TFR object.

        Parameters
        ----------
        tfr : mne.time_frequency.EpochsTFR or AverageTFR
            Input TFR data.
        **kwargs
            Additional arguments passed to constructor.

        Returns
        -------
        Spectrogram or SpectrogramDict
        """
        from gwexpy.interop import from_mne

        return from_mne(cls, tfr, **kwargs)

    def to_obspy(self, **kwargs: Any) -> Any:
        """
        Convert to Obspy Stream.

        Returns
        -------
        obspy.Stream
        """
        from gwexpy.interop import to_obspy

        return to_obspy(self, **kwargs)

    @classmethod
    def from_obspy(cls, stream: Any, **kwargs: Any) -> Any:
        """
        Create Spectrogram from Obspy Stream.

        Parameters
        ----------
        stream : obspy.Stream
            Input stream.
        **kwargs
            Additional arguments.

        Returns
        -------
        Spectrogram
        """
        from gwexpy.interop import from_obspy

        return from_obspy(cls, stream, **kwargs)

    # ===============================
    # pyroomacoustics
    # ===============================

    @classmethod
    def from_pyroomacoustics_stft(
        cls,
        stft_obj: Any,
        *,
        channel: int | None = None,
        fs: float | None = None,
        unit: Any | None = None,
    ) -> Any:
        """
        Create from a pyroomacoustics STFT object.

        Parameters
        ----------
        stft_obj : pyroomacoustics.stft.STFT
            STFT object with ``.X``, ``.hop``, and ``.N`` attributes.
        channel : int, optional
            Channel index. If *None*, all channels are returned as a
            :class:`SpectrogramDict` for multi-channel data.
        fs : float, optional
            Sample rate in Hz. Required if ``stft_obj`` has no ``fs`` attribute.
        unit : str or astropy.units.Unit, optional
            Unit to assign to the result.

        Returns
        -------
        Spectrogram or SpectrogramDict
        """
        from gwexpy.interop import from_pyroomacoustics_stft

        return from_pyroomacoustics_stft(
            cls, stft_obj, channel=channel, fs=fs, unit=unit
        )

    def to_pyroomacoustics_stft(
        self,
        *,
        hop: int | None = None,
        analysis_window: Any | None = None,
    ) -> Any:
        """
        Export as a pyroomacoustics STFT object.

        Parameters
        ----------
        hop : int, optional
            Hop size in samples. If *None*, estimated from the spectrogram
            metadata.
        analysis_window : numpy.ndarray, optional
            Analysis window for the STFT object.

        Returns
        -------
        pyroomacoustics.stft.STFT
        """
        from gwexpy.interop import to_pyroomacoustics_stft

        return to_pyroomacoustics_stft(
            self, hop=hop, analysis_window=analysis_window
        )

    def rebin(
        self, dt: float | u.Quantity | None = None, df: float | u.Quantity | None = None
    ) -> Spectrogram:
        """
        Rebin the spectrogram in time and/or frequency.

        Parameters
        ----------
        dt : float or Quantity, optional
            New time bin width in seconds.
        df : float or Quantity, optional
            New frequency bin width in Hz.

        Returns
        -------
        Spectrogram
            The rebinned spectrogram.
        """
        data = self.value
        times = self.times
        freqs = self.frequencies

        # Frequency rebinning
        if df is not None:
            if hasattr(df, "to"):
                df = df.to("Hz").value
            orig_df = self.df.to("Hz").value if hasattr(self.df, "to") else self.df
            bin_size = int(round(df / orig_df))
            if bin_size > 1:
                nt, nf = data.shape
                nf_new = nf // bin_size
                data = (
                    data[:, : nf_new * bin_size]
                    .reshape(nt, nf_new, bin_size)
                    .mean(axis=2)
                )
                freqs = (
                    freqs[: nf_new * bin_size].reshape(nf_new, bin_size).mean(axis=1)
                )

        # Time rebinning
        if dt is not None:
            if hasattr(dt, "to"):
                dt = dt.to("s").value
            orig_dt = self.dt.to("s").value if hasattr(self.dt, "to") else self.dt
            bin_size = int(round(dt / orig_dt))
            if bin_size > 1:
                nt, nf = data.shape
                nt_new = nt // bin_size
                data = (
                    data[: nt_new * bin_size, :]
                    .reshape(nt_new, bin_size, nf)
                    .mean(axis=1)
                )
                times = (
                    times[: nt_new * bin_size].reshape(nt_new, bin_size).mean(axis=1)
                )

        return self.__class__(
            data,
            times=times,
            frequencies=freqs,
            unit=self.unit,
            name=self.name,
            channel=self.channel,
            epoch=self.epoch,
        )

    def imshow(self, **kwargs):
        """Plot using Matplotlib ``imshow`` (GWpy-compatible).

        This method is provided for convenience and forwards arguments to
        :meth:`gwpy.spectrogram.Spectrogram.imshow`.

        Common keyword arguments include ``ax``, ``cmap``, ``norm`` (or
        ``log=True`` in GWpy), and color scaling controls like ``vmin``/``vmax``.
        For the full set of supported keywords, see the GWpy documentation.
        """
        return super().imshow(**kwargs)

    def pcolormesh(self, **kwargs):
        """Plot using Matplotlib ``pcolormesh`` (GWpy-compatible).

        This method is provided for convenience and forwards arguments to
        :meth:`gwpy.spectrogram.Spectrogram.pcolormesh`.

        Common keyword arguments include ``ax``, ``cmap``, ``norm`` and
        ``vmin``/``vmax``. For the full set of supported keywords, see the GWpy
        documentation.
        """
        return super().pcolormesh(**kwargs)

    def radian(self, unwrap: bool = False) -> Spectrogram:
        """
        Calculate the phase of this Spectrogram in radians.

        Parameters
        ----------
        unwrap : bool, optional
            If True, unwrap the phase to remove discontinuities along the time axis.
            Default is False.

        Returns
        -------
        Spectrogram
            A new Spectrogram containing the phase in radians.
            All other metadata (times, frequencies, channel, epoch, etc.) are preserved.
        """
        # Copy to preserve all metadata (times, frequencies, channel, epoch, metadata dict, etc.)
        new = self.copy()

        # Calculate phase values
        val = np.angle(self.view(np.ndarray))
        if unwrap:
            # Unwrap along time axis (axis 0 of (Time, Freq) Spectrogram)
            val = np.unwrap(val, axis=0)

        # If original was complex, Ensure new is real-valued to hold phase
        if np.iscomplexobj(new):
            new = new.real.copy()

        # Update data and metadata
        # Use raw ndarray view for assignment to bypass Astropy's unit check
        new.view(np.ndarray)[:] = val
        new.override_unit("rad")
        if self.name:
            new.name = self.name + "_phase"
        else:
            new.name = "phase"

        return new

    def degree(self, unwrap: bool = False) -> Spectrogram:
        """
        Calculate the phase of this Spectrogram in degrees.

        Parameters
        ----------
        unwrap : bool, optional
            If True, unwrap the phase to remove discontinuities along the time axis.
            Default is False.

        Returns
        -------
        Spectrogram
            A new Spectrogram containing the phase in degrees.
            All other metadata (times, frequencies, channel, epoch, etc.) are preserved.
        """
        # Re-use radian() implementation which handles unwrap and metadata preservation
        p = self.radian(unwrap=unwrap)

        # Convert values to degrees
        val = np.rad2deg(p.view(np.ndarray))

        # Create final object (p already has correct metadata and is real-valued)
        new = p
        # Use raw ndarray view for assignment to bypass Astropy's unit check
        new.view(np.ndarray)[:] = val
        new.override_unit("deg")
        if self.name:
            new.name = self.name + "_phase_deg"
        else:
            new.name = "phase_deg"

        return new

    def normalize(
        self,
        method: str = "snr",
        reference: Any | None = None,
        *,
        percentile: float = 50.0,
    ) -> Spectrogram:
        """
        Normalize the spectrogram along the time axis.

        Parameters
        ----------
        method : {'snr', 'median', 'mean', 'percentile', 'reference'}
            Normalization method.

            - ``'snr'``: Divide each time slice by the median PSD along the
              time axis (equivalent to ``'median'``). If *reference* is given,
              use it as the denominator instead.
            - ``'median'``: Divide by the median along the time axis per
              frequency bin.
            - ``'mean'``: Divide by the mean along the time axis.
            - ``'percentile'``: Divide by the given *percentile* along the
              time axis.
            - ``'reference'``: Divide by a user-provided reference spectrum.
              *reference* must be given.

        reference : FrequencySeries or array-like, optional
            Reference spectrum used as the denominator for ``'snr'`` (if
            provided) or ``'reference'`` mode.

        percentile : float, optional
            Percentile value for ``'percentile'`` mode. Default is 50.0
            (equivalent to median).

        Returns
        -------
        Spectrogram
            Normalized spectrogram. Unit is set to dimensionless for ratio
            methods.
        """
        data = self.value.copy()

        n_freqs = data.shape[1]

        def _validate_reference(arr: Any) -> np.ndarray:
            r = np.asarray(arr, dtype=float).ravel()
            if r.shape[0] != n_freqs:
                raise ValueError(
                    f"reference length ({r.shape[0]}) must equal the number of "
                    f"frequency bins ({n_freqs})"
                )
            return r

        if method in ("snr", "median"):
            if reference is not None:
                ref = _validate_reference(reference)
            else:
                ref = np.median(data, axis=0)
        elif method == "mean":
            ref = np.mean(data, axis=0)
        elif method == "percentile":
            ref = np.percentile(data, percentile, axis=0)
        elif method == "reference":
            if reference is None:
                raise ValueError("reference must be provided for method='reference'")
            ref = _validate_reference(reference)
        else:
            raise ValueError(
                f"Unknown method: {method!r}. "
                "Choose from 'snr', 'median', 'mean', 'percentile', 'reference'."
            )

        # Safe division — replace zeros with NaN to avoid inf
        ref = np.asarray(ref, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = data / ref[np.newaxis, :]
            result[~np.isfinite(result)] = np.nan

        return self.__class__(
            result,
            times=self.times,
            frequencies=self.frequencies,
            unit=u.dimensionless_unscaled,
            name=self.name,
            channel=self.channel,
            epoch=self.epoch,
        )

    def clean(
        self,
        method: str = "threshold",
        *,
        threshold: float = 5.0,
        window_size: int | None = None,
        fill: str = "median",
        persistence_threshold: float = 0.8,
        amplitude_threshold: float = 3.0,
        return_mask: bool = False,
    ) -> Spectrogram | tuple[Spectrogram, np.ndarray]:
        """
        Clean the spectrogram by removing artifacts.

        Parameters
        ----------
        method : {'threshold', 'rolling_median', 'line_removal', 'combined'}
            Cleaning method.

            - ``'threshold'``: Remove outlier pixels using MAD-based
              detection.
            - ``'rolling_median'``: Normalize slow trends with a rolling
              median filter along the time axis.
            - ``'line_removal'``: Remove persistent narrowband lines.
            - ``'combined'``: Apply threshold, then rolling_median, then
              line_removal sequentially.

        threshold : float, optional
            MAD sigma threshold for outlier detection. Default 5.0.
        window_size : int, optional
            Rolling window size in time bins for ``'rolling_median'`` and
            ``'combined'`` modes. If *None*, defaults to ``shape[0] // 4``
            (clamped to at least 3).
        fill : {'median', 'nan', 'zero', 'interpolate'}
            How to fill masked/outlier values (for threshold method).
        persistence_threshold : float, optional
            Fraction threshold for line detection (0.0-1.0). Default 0.8.
        amplitude_threshold : float, optional
            Factor above global median for line detection. Default 3.0.
        return_mask : bool, optional
            If True, also return a boolean mask of cleaned pixels.

        Returns
        -------
        Spectrogram
            Cleaned spectrogram.
        mask : ndarray, optional
            Boolean mask where True = pixel was cleaned. Only returned when
            *return_mask* is True.
        """
        from .cleaning import (
            line_removal_clean,
            rolling_median_clean,
            threshold_clean,
        )

        data = self.value.copy()

        if window_size is None:
            window_size = max(3, data.shape[0] // 4)

        if method == "threshold":
            cleaned, mask = threshold_clean(data, threshold=threshold, fill=fill)
        elif method == "rolling_median":
            cleaned = rolling_median_clean(data, window_size=window_size)
            mask = np.zeros(data.shape, dtype=bool)
        elif method == "line_removal":
            cleaned, line_indices = line_removal_clean(
                data,
                persistence_threshold=persistence_threshold,
                amplitude_threshold=amplitude_threshold,
            )
            mask = np.zeros(data.shape, dtype=bool)
            for idx in line_indices:
                mask[:, idx] = True
        elif method == "combined":
            cleaned, mask = threshold_clean(data, threshold=threshold, fill=fill)
            cleaned = rolling_median_clean(cleaned, window_size=window_size)
            cleaned_lines, line_indices = line_removal_clean(
                cleaned,
                persistence_threshold=persistence_threshold,
                amplitude_threshold=amplitude_threshold,
            )
            cleaned = cleaned_lines
            for idx in line_indices:
                mask[:, idx] = True
        else:
            raise ValueError(
                f"Unknown method: {method!r}. "
                "Choose from 'threshold', 'rolling_median', 'line_removal', 'combined'."
            )

        result = self.__class__(
            cleaned,
            times=self.times,
            frequencies=self.frequencies,
            unit=self.unit,
            name=self.name,
            channel=self.channel,
            epoch=self.epoch,
        )

        if return_mask:
            return result, mask
        return result

    def to_timeseries_list(self) -> tuple[TimeSeriesList, Quantity]:
        """
        Convert this Spectrogram to a list of TimeSeries, one per frequency bin.

        For a Spectrogram with shape ``(ntimes, nfreqs)``, this extracts each
        column (frequency bin) as a TimeSeries with the same time axis.

        Returns
        -------
        ts_list : TimeSeriesList
            A list of TimeSeries, one for each frequency bin.
            Each TimeSeries has length ``ntimes``.
        frequencies : Quantity
            The frequency axis of this Spectrogram (length ``nfreqs``).

        Notes
        -----
        - Each TimeSeries inherits ``unit``, ``epoch``, ``channel`` from this
          Spectrogram.
        - ``name`` is set to ``"{original_name}_f{freq}"`` or ``"f{freq}"`` if
          the Spectrogram has no name, where ``freq`` is the frequency value.

        Examples
        --------
        >>> import numpy as np
        >>> data = np.ones((10, 5))
        >>> spec = Spectrogram(data, t0=0, dt=0.1, f0=10, df=5, name="test")
        >>> ts_list, freqs = spec.to_timeseries_list()
        >>> len(ts_list)  # equals nfreqs
        5
        >>> ts_list[0].name
        'test_f10.0 Hz'
        """
        from gwexpy.interop._registry import ConverterRegistry

        TimeSeries = ConverterRegistry.get_constructor("TimeSeries")
        TimeSeriesList = ConverterRegistry.get_constructor("TimeSeriesList")

        ntimes, nfreqs = self.shape
        # Extract raw ndarray to avoid unit doubling
        data_raw = self.view(np.ndarray)
        unit = self.unit
        times = self.times
        frequencies = self.frequencies
        # Convert epoch to GPS float to avoid astropy.Time interpretation issues
        epoch = (
            float(self.epoch.gps) if hasattr(self.epoch, "gps") else float(self.epoch)
        )
        channel = self.channel
        base_name = self.name

        ts_list = TimeSeriesList()
        for i in range(nfreqs):
            freq_val = frequencies[i]
            # Create descriptive name
            if base_name:
                name = f"{base_name}_f{freq_val}"
            else:
                name = f"f{freq_val}"

            # Extract column i (all times, single frequency)
            ts_data = data_raw[:, i].copy()

            # Create TimeSeries with explicit unit
            ts = TimeSeries(
                ts_data,
                times=times,
                unit=unit,
                name=name,
                channel=channel,
                epoch=epoch,
            )
            # Bypass validation by using list.append
            list.append(ts_list, ts)

        return ts_list, frequencies

    def to_frequencyseries_list(self) -> tuple[FrequencySeriesList, Quantity]:
        """
        Convert this Spectrogram to a list of FrequencySeries, one per time bin.

        For a Spectrogram with shape ``(ntimes, nfreqs)``, this extracts each
        row (time bin) as a FrequencySeries with the same frequency axis.

        Returns
        -------
        fs_list : FrequencySeriesList
            A list of FrequencySeries, one for each time bin.
            Each FrequencySeries has length ``nfreqs``.
        times : Quantity
            The time axis of this Spectrogram (length ``ntimes``).

        Notes
        -----
        - Each FrequencySeries inherits ``unit``, ``epoch``, ``channel`` from
          this Spectrogram.
        - ``name`` is set to ``"{original_name}_t{time}"`` or ``"t{time}"`` if
          the Spectrogram has no name, where ``time`` is the time value.

        Examples
        --------
        >>> import numpy as np
        >>> data = np.ones((10, 5))
        >>> spec = Spectrogram(data, t0=0, dt=0.1, f0=10, df=5, name="test")
        >>> fs_list, times = spec.to_frequencyseries_list()
        >>> len(fs_list)  # equals ntimes
        10
        >>> fs_list[0].name
        'test_t0.0 s'
        """
        from gwexpy.interop._registry import ConverterRegistry

        FrequencySeries = ConverterRegistry.get_constructor("FrequencySeries")
        FrequencySeriesList = ConverterRegistry.get_constructor("FrequencySeriesList")

        ntimes, nfreqs = self.shape
        # Extract raw ndarray to avoid unit doubling
        data_raw = self.view(np.ndarray)
        unit = self.unit
        times = self.times
        frequencies = self.frequencies
        # Convert epoch to GPS float to avoid astropy.Time interpretation issues
        epoch = (
            float(self.epoch.gps) if hasattr(self.epoch, "gps") else float(self.epoch)
        )
        channel = self.channel
        base_name = self.name

        fs_list = FrequencySeriesList()
        for j in range(ntimes):
            time_val = times[j]
            # Create descriptive name
            if base_name:
                name = f"{base_name}_t{time_val}"
            else:
                name = f"t{time_val}"

            # Extract row j (single time, all frequencies)
            fs_data = data_raw[j, :].copy()

            # Create FrequencySeries with explicit unit
            fs = FrequencySeries(
                fs_data,
                frequencies=frequencies,
                unit=unit,
                name=name,
                channel=channel,
                epoch=epoch,
            )
            # Bypass validation by using list.append
            list.append(fs_list, fs)

        return fs_list, times


# Import Matrix, List, and Dict to maintain backward compatibility if this file is imported
