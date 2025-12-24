"""
Spectral transform methods for TimeSeries.

This module provides spectral analysis functionality as a mixin class:
- FFT, RFFT, PSD, ASD, CSD, coherence
- DCT, Laplace transform
- CWT (Continuous Wavelet Transform)
- Cepstrum
- EMD (Empirical Mode Decomposition)
- HHT (Hilbert-Huang Transform)
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from typing import Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class TimeSeriesSpectralMixin:
    """
    Mixin class providing spectral transform methods for TimeSeries.

    This mixin is designed to be combined with TimeSeriesCore to create
    the full TimeSeries class. It provides frequency-domain transformations
    and time-frequency analysis methods.
    """

    # ===============================
    # Transform Data Preparation
    # ===============================

    def _prepare_data_for_transform(self, data=None, window=None, detrend=False):
        """Helper to copy data and apply detrending and windowing."""
        try:
            import scipy.signal
        except ImportError:
            raise ImportError("scipy is required for this transform")

        if data is None:
            data = self.value.copy()
        else:
            data = data.copy()  # Ensure we're working on a copy if data was passed

        if detrend:
            data = scipy.signal.detrend(data, type='linear')

        if window is not None:
            if isinstance(window, (str, tuple)):
                win = scipy.signal.get_window(window, len(data))
            else:
                win = np.asarray(window)
                if len(win) != len(data):
                    raise ValueError("Window length must match data length")
            data *= win
        return data

    # ===============================
    # FFT and Related Transforms
    # ===============================

    def fft(
        self,
        nfft: Optional[int] = None,
        *,
        mode: str = "gwpy",
        pad_mode: str = "zero",
        pad_left: int = 0,
        pad_right: int = 0,
        nfft_mode: Optional[str] = None,
        other_length: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Compute the one-dimensional discrete Fourier Transform.

        Parameters
        ----------
        nfft : `int`, optional
            Length of the FFT. If `None`, the length of the TimeSeries is used.
        mode : `str`, optional, default: "gwpy"
            "gwpy": Standard behavior (circular convolution assumption).
            "transient": Transient-friendly mode with padding options.
        pad_mode : `str`, optional, default: "zero"
            Padding mode for "transient" mode. "zero" or "reflect".
        pad_left : `int`, optional, default: 0
            Number of samples to pad on the left (for "transient" mode).
        pad_right : `int`, optional, default: 0
            Number of samples to pad on the right (for "transient" mode).
        nfft_mode : `str`, optional, default: None
            "next_fast_len": Use scipy.fft.next_fast_len to optimize FFT size.
            None: Use exact calculated length.
        other_length : `int`, optional, default: None
            If provided in "transient" mode, the target length is calculated as
            len(self) + other_length - 1 (linear convolution size).
        **kwargs
            Keyword arguments passed to the `FrequencySeries` constructor.

        Returns
        -------
        out : `FrequencySeries`
            The DFT of the TimeSeries.
        """
        self._check_regular("fft")

        # 1. GWpy compatible mode (default)
        if mode == "gwpy":
            if (
                pad_mode == "zero"
                and pad_left == 0
                and pad_right == 0
                and nfft_mode is None
                and other_length is None
            ):
                base_fs = super().fft(nfft=nfft, **kwargs)
                try:
                    from gwexpy.frequencyseries import FrequencySeries as GWEXFrequencySeries
                except ImportError:
                    return base_fs
                if isinstance(base_fs, GWEXFrequencySeries):
                    return base_fs
                return GWEXFrequencySeries(
                    base_fs.value,
                    frequencies=base_fs.frequencies,
                    unit=base_fs.unit,
                    name=base_fs.name,
                    channel=base_fs.channel,
                    epoch=base_fs.epoch,
                )

            # Fallback to super for gwpy mode even if explicit args (ignore extras)
            base_fs = super().fft(nfft=nfft, **kwargs)
            try:
                from gwexpy.frequencyseries import FrequencySeries as GWEXFrequencySeries
            except ImportError:
                return base_fs
            if isinstance(base_fs, GWEXFrequencySeries):
                return base_fs
            return GWEXFrequencySeries(
                base_fs.value,
                frequencies=base_fs.frequencies,
                unit=base_fs.unit,
                name=base_fs.name,
                channel=base_fs.channel,
                epoch=base_fs.epoch,
            )

        if mode != "transient":
            raise ValueError(f"Unknown mode: {mode}")

        # 2. Transient mode
        x = self.value

        # Padding
        if pad_left > 0 or pad_right > 0:
            if hasattr(pad_left, "to"):
                pad_left = pad_left.to("s").value * self.sample_rate.value
            elif isinstance(pad_left, float):
                pad_left = pad_left * self.sample_rate.value

            if hasattr(pad_right, "to"):
                pad_right = pad_right.to("s").value * self.sample_rate.value
            elif isinstance(pad_right, float):
                pad_right = pad_right * self.sample_rate.value

            pad_left = int(round(float(pad_left)))
            pad_right = int(round(float(pad_right)))

            if pad_left > 0 or pad_right > 0:
                if pad_mode == "zero":
                    x = np.pad(x, (pad_left, pad_right), mode="constant", constant_values=0)
                elif pad_mode == "reflect":
                    x = np.pad(x, (pad_left, pad_right), mode="reflect")
                else:
                    raise ValueError(f"Unknown pad_mode: {pad_mode}")

        len_x = x.shape[0]

        # Determine target_nfft
        if nfft is not None:
            if nfft < len_x:
                raise ValueError(f"nfft={nfft} must be >= padded length {len_x}")
            target_nfft = int(nfft)
        else:
            if other_length is not None:
                target_len = len_x + int(other_length) - 1
            else:
                target_len = len_x

            if nfft_mode == "next_fast_len":
                try:
                    import scipy.fft
                    def next_fast_len(n):
                        return scipy.fft.next_fast_len(n)
                except ImportError:
                    try:
                        from scipy.fftpack import next_fast_len
                    except ImportError:
                        def next_fast_len(n):
                            return n

                target_nfft = next_fast_len(target_len)
            else:
                target_nfft = target_len

        # Compute FFT
        dft = np.fft.rfft(x, n=target_nfft) / target_nfft

        if dft.shape[0] > 1:
            dft[1:] *= 2.0

        from gwexpy.frequencyseries import FrequencySeries

        fs = FrequencySeries(
            dft,
            epoch=self.epoch,
            unit=self.unit,
            name=self.name,
            channel=self.channel,
            **kwargs,
        )

        # Set frequencies with units
        fs.frequencies = np.fft.rfftfreq(target_nfft, d=self.dt.value) * u.Hz
        # Store transient metadata for round-trip ifft
        fs._gwex_fft_mode = "transient"
        fs._gwex_pad_left = pad_left
        fs._gwex_pad_right = pad_right
        fs._gwex_pad_mode = pad_mode
        fs._gwex_target_nfft = target_nfft
        fs._gwex_original_n = len(self)
        fs._gwex_other_length = other_length

        return fs

    def rfft(self, *args: Any, **kwargs: Any) -> Any:
        """Compute RFFT of the TimeSeries."""
        self._check_regular("rfft")
        return super().rfft(*args, **kwargs)

    def psd(self, *args: Any, **kwargs: Any) -> Any:
        """Compute PSD of the TimeSeries."""
        self._check_regular("psd")
        return super().psd(*args, **kwargs)

    def asd(self, *args: Any, **kwargs: Any) -> Any:
        """Compute ASD of the TimeSeries."""
        self._check_regular("asd")
        return super().asd(*args, **kwargs)

    def csd(self, *args: Any, **kwargs: Any) -> Any:
        """Compute CSD of the TimeSeries."""
        self._check_regular("csd")
        return super().csd(*args, **kwargs)

    def coherence(self, *args: Any, **kwargs: Any) -> Any:
        """Compute coherence of the TimeSeries."""
        self._check_regular("coherence")
        return super().coherence(*args, **kwargs)

    # ===============================
    # DCT (Discrete Cosine Transform)
    # ===============================

    def dct(
        self, type: int = 2, norm: str = "ortho", *, window: Any = None, detrend: bool = False
    ) -> Any:
        """
        Compute the Discrete Cosine Transform (DCT) of the TimeSeries.

        Parameters
        ----------
        type : `int`, optional
            Type of the DCT (1, 2, 3, 4). Default is 2.
        norm : `str`, optional
            Normalization mode. Default is "ortho".
        window : `str`, `numpy.ndarray`, or `None`, optional
            Window function to apply before DCT.
        detrend : `bool`, optional
            If `True`, remove the mean (or detrend) from the data before DCT.

        Returns
        -------
        out : `FrequencySeries`
             The DCT of the TimeSeries.
        """
        self._check_regular("dct")
        try:
            import scipy.fft
        except ImportError:
            raise ImportError(
                "scipy is required for dct. "
                "Please install it via `pip install scipy`."
            )

        data = self._prepare_data_for_transform(window=window, detrend=detrend)

        # DCT
        out_dct = scipy.fft.dct(data, type=type, norm=norm)

        # Frequencies
        N = len(data)
        if self.dt is None:
             raise ValueError("TimeSeries must have a valid dt for DCT frequency calculation")

        dt = self.dt.to("s").value
        k = np.arange(N)
        freqs_val = k / (2 * N * dt)

        frequencies = u.Quantity(freqs_val, "Hz")

        from gwexpy.frequencyseries import FrequencySeries

        fs = FrequencySeries(
            out_dct,
            frequencies=frequencies,
            unit=self.unit,
            name=self.name + "_dct" if self.name else "dct",
            channel=self.channel,
            epoch=self.epoch
        )

        # Metadata
        fs.transform = "dct"
        fs.dct_type = type
        fs.dct_norm = norm
        fs.original_n = N
        fs.dt = self.dt

        return fs

    # ===============================
    # Laplace Transform
    # ===============================

    def laplace(
        self,
        *,
        sigma: Any = 0.0,
        frequencies: Any = None,
        t_start: Any = None,
        t_stop: Any = None,
        window: Any = None,
        detrend: bool = False,
        normalize: str = "integral",
        dtype: Any = None,
        chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        One-sided finite-interval Laplace transform.

        Define s = sigma + i*2*pi*f (f in Hz).
        Compute on a cropped segment [t_start, t_stop] but shift time origin to t_start:
            L(s) = integral_0^T x(tau) exp(-(sigma + i 2pi f) tau) dtau
        Discrete approximation uses uniform dt.

        Parameters
        ----------
        sigma : `float` or `astropy.units.Quantity`, optional
            Real part of the Laplace variable s (1/s). Default 0.0.
        frequencies : `numpy.ndarray` or `astropy.units.Quantity`, optional
            Frequencies for the imaginary part of s (Hz).
            If None, defaults to `np.fft.rfftfreq(n, d=dt)`.
        t_start : `float` or `astropy.units.Quantity`, optional
            Start time relative to the beginning of the TimeSeries (seconds).
        t_stop : `float` or `astropy.units.Quantity`, optional
            Stop time relative to the beginning of the TimeSeries (seconds).
        window : `str`, `numpy.ndarray`, or `None`, optional
            Window function to apply.
        detrend : `bool`, optional
            If True, remove the mean before transformation.
        normalize : `str`, optional
            "integral": Sum * dt (standard Laplace integral approximation).
            "mean": Sum / n.
        dtype : `numpy.dtype`, optional
            Output data type.
        chunk_size : `int`, optional
            If provided, compute the transform in chunks along the frequency axis
            to save memory.

        Returns
        -------
        out : `FrequencySeries`
            The Laplace transform result (complex).
        """
        self._check_regular("laplace")

        n_total = len(self)
        if self.dt is None:
             raise ValueError("TimeSeries must have a valid dt for Laplace transform")

        dt_val = self.dt.to("s").value
        fs_rate = 1.0 / dt_val

        def resolve_time_arg(arg, default_idx):
             if arg is None:
                  return default_idx
             if isinstance(arg, u.Quantity):
                  t_s = arg.to("s").value
             else:
                  t_s = float(arg)
             idx = int(round(t_s * fs_rate))
             return np.clip(idx, 0, n_total)

        idx_start = resolve_time_arg(t_start, 0)
        idx_stop = resolve_time_arg(t_stop, n_total)

        if idx_stop <= idx_start:
             raise ValueError(f"Invalid time range: t_start index {idx_start} >= t_stop index {idx_stop}")

        # Extract data segment
        data = self.value[idx_start:idx_stop]

        # Preprocessing
        data = self._prepare_data_for_transform(data=data, window=window, detrend=detrend)
        n = len(data)

        # Frequency Axis
        if frequencies is None:
             freqs_val = np.fft.rfftfreq(n, d=dt_val)
             freqs_quant = u.Quantity(freqs_val, "Hz")
        else:
             if isinstance(frequencies, u.Quantity):
                  freqs_quant = frequencies.to("Hz")
                  freqs_val = freqs_quant.value
             else:
                  freqs_val = np.asarray(frequencies)
                  freqs_quant = u.Quantity(freqs_val, "Hz")

        # Sigma
        if isinstance(sigma, u.Quantity):
             sigma_val = sigma.to("1/s").value
        else:
             sigma_val = float(sigma)

        # Computation
        tau = np.arange(n) * dt_val

        # Determine output dtype
        if dtype is None:
             out_dtype = np.result_type(data.dtype, np.complex128)
        else:
             out_dtype = dtype

        n_freqs = len(freqs_val)
        out_data = np.zeros(n_freqs, dtype=out_dtype)

        # Factor for normalization
        if normalize == "integral":
             norm_factor = dt_val
        elif normalize == "mean":
             norm_factor = 1.0 / n
        else:
             raise ValueError(f"Unknown normalize mode: {normalize}")

        if chunk_size is None:
             # Heuristic: limit the size of intermediate complex matrices
             max_elements = 10_000_000
             if n_freqs * n > max_elements:
                 chunk_size = max(1, max_elements // n)
             else:
                 chunk_size = n_freqs

        # Precompute real weighting
        real_exp = np.exp(-sigma_val * tau)
        data_weighted = data * real_exp

        # Chunked computation
        for i in range(0, n_freqs, chunk_size):
             end = min(i + chunk_size, n_freqs)
             f_chunk = freqs_val[i:end]

             phase_chunk = 2 * np.pi * f_chunk[:, None] * tau[None, :]
             complex_exp_chunk = np.exp(-1j * phase_chunk)

             out_data[i:end] = np.dot(complex_exp_chunk, data_weighted) * norm_factor

        # Create Output
        from gwexpy.frequencyseries import FrequencySeries

        # Propagate units
        if normalize == "integral":
             out_unit = self.unit * u.s
        else:
             out_unit = self.unit

        fs = FrequencySeries(
            out_data,
            epoch=self.epoch,
            unit=out_unit,
            name=f"{self.name}_laplace" if self.name else "laplace",
            channel=self.channel,
            **kwargs,
        )

        fs.frequencies = freqs_quant
        fs.laplace_sigma = sigma_val

        return fs

    # ===============================
    # Cepstrum
    # ===============================

    def cepstrum(
        self,
        kind: str = "real",
        *,
        window: Any = None,
        detrend: bool = False,
        eps: Optional[float] = None,
        fft_mode: str = "gwpy",
    ) -> Any:
        """
        Compute the Cepstrum of the TimeSeries.

        Parameters
        ----------
        kind : {"real", "power", "complex"}, optional
            Type of cepstrum. Default is "real".
        window : `str`, `numpy.ndarray`, or `None`, optional
            Window function to apply.
        detrend : `bool`, optional
            If `True`, detrend (linear).
        eps : `float`, optional
            Small value to avoid log(0).
        fft_mode : `str`, optional
            Mode for the underlying FFT.

        Returns
        -------
        out : `FrequencySeries`
            The cepstrum, with frequencies interpreted as quefrency (seconds).
        """
        try:
             import scipy.fft
        except ImportError:
            raise ImportError(
                "scipy is required for cepstrum. "
                "Please install it via `pip install scipy`."
            )

        data = self._prepare_data_for_transform(window=window, detrend=detrend)

        if kind == "complex":
             spec = scipy.fft.rfft(data)
             if eps is not None:
                  spec += eps
             with np.errstate(divide='ignore', invalid='ignore'):
                  log_spec = np.log(spec)
             ceps = scipy.fft.irfft(log_spec, n=len(data))

        elif kind == "real":
             spec = scipy.fft.rfft(data)
             mag = np.abs(spec)
             if eps is not None:
                  mag += eps
             with np.errstate(divide='ignore'):
                  log_mag = np.log(mag)
             ceps = scipy.fft.irfft(log_mag, n=len(data))

        elif kind == "power":
             spec = scipy.fft.rfft(data)
             pwr = np.abs(spec)**2
             if eps is not None:
                  pwr += eps
             with np.errstate(divide='ignore'):
                  log_pwr = np.log(pwr)
             ceps = scipy.fft.irfft(log_pwr, n=len(data))

        else:
             raise ValueError(f"Unknown cepstrum kind: {kind}")

        # Quefrency axis
        if self.dt is None:
             dt = 1.0
        else:
             dt = self.dt.to("s").value

        n = len(ceps)
        quefrencies = np.arange(n) * dt
        quefrencies = u.Quantity(quefrencies, "s")

        from gwexpy.frequencyseries import FrequencySeries

        fs = FrequencySeries(
            ceps,
            frequencies=quefrencies,
            unit=u.dimensionless_unscaled,
            name=self.name + "_cepstrum" if self.name else "cepstrum",
            channel=self.channel,
            epoch=self.epoch
        )

        fs.axis_type = "quefrency"
        fs.transform = "cepstrum"
        fs.cepstrum_kind = kind
        fs.original_n = len(data)
        fs.dt = self.dt

        return fs

    # ===============================
    # CWT (Continuous Wavelet Transform)
    # ===============================

    def cwt(
        self,
        wavelet: str = "cmor1.5-1.0",
        widths: Any = None,
        frequencies: Any = None,
        *,
        window: Any = None,
        detrend: bool = False,
        output: str = "spectrogram",
        chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Compute the Continuous Wavelet Transform (CWT) using PyWavelets.

        Parameters
        ----------
        wavelet : `str`, optional
            Wavelet name (default "cmor1.5-1.0" for complex morlet).
        widths : `array-like`, optional
            Scales to use for CWT.
        frequencies : `array-like`, optional
             Frequencies to use (Hz). If provided, converts to scales.
        window : `str`, `numpy.ndarray`, or `None`
             Window function to apply before CWT.
        detrend : `bool`
             If True, detrend the time series.
        output : {"spectrogram", "ndarray"}
             Output format.
        chunk_size : `int`, optional
             If provided, compute in chunks to save memory.
        **kwargs :
             Additional arguments passed to `pywt.cwt`.

        Returns
        -------
        out : `gwpy.spectrogram.Spectrogram` or `(ndarray, freqs)`
        """
        self._check_regular("cwt")

        try:
             import pywt
             import scipy.signal
        except ImportError as e:
             raise ImportError(
                 f"pywt (PyWavelets) and scipy are required for cwt. "
                 f"Please install them via `pip install PyWavelets scipy`. Error: {e}"
             )

        data = self._prepare_data_for_transform(window=window, detrend=detrend)

        # Resolve widths(scales)/frequencies
        dt = self.dt.to("s").value

        if frequencies is not None:
             if widths is not None:
                  raise ValueError("Cannot specify both widths(scales) and frequencies")

             freqs_quant = u.Quantity(frequencies, "Hz")
             freqs_arr = freqs_quant.value

             center_freq = pywt.scale2frequency(wavelet, 1)

             with np.errstate(divide='ignore'):
                 scales = center_freq / (freqs_arr * dt)

        elif widths is None:
             raise ValueError("Must specify either widths(scales) or frequencies")
        else:
             scales = np.asarray(widths)
             center_freq = pywt.scale2frequency(wavelet, 1)
             freqs_arr = center_freq / (scales * dt)

        # Compute CWT
        if chunk_size is None:
             max_elements = 10_000_000
             n_times = len(data)
             n_scales = len(scales)
             if n_scales * n_times > max_elements:
                 chunk_size = max(1, max_elements // n_times)
             else:
                 chunk_size = n_scales

        coefs_list = []
        for i in range(0, len(scales), chunk_size):
             c_scales = scales[i : i + chunk_size]
             c_coefs, _ = pywt.cwt(data, c_scales, wavelet, sampling_period=dt, **kwargs)
             coefs_list.append(c_coefs)

        coefs = np.vstack(coefs_list) if len(coefs_list) > 1 else coefs_list[0]

        freqs_quant = u.Quantity(freqs_arr, "Hz")

        if output == "ndarray":
             return coefs, freqs_quant

        elif output == "spectrogram":
             try:
                  from gwpy.spectrogram import Spectrogram
             except ImportError:
                  return coefs, freqs_quant

             # Transpose to (time, freq) for GWpy Spectrogram
             out_spec = coefs.T

             # Sort frequencies
             idx_sorted = np.argsort(freqs_arr)
             freqs_sorted = freqs_arr[idx_sorted]
             out_spec_sorted = out_spec[:, idx_sorted]

             return Spectrogram(
                 out_spec_sorted,
                 times=self.times,
                 frequencies=u.Quantity(freqs_sorted, "Hz"),
                 unit=self.unit,
                 name=self.name,
                 channel=self.channel,
                 epoch=self.epoch
             )
        else:
             raise ValueError(f"Unknown output format: {output}")

    # ===============================
    # EMD (Empirical Mode Decomposition)
    # ===============================

    def emd(
        self,
        *,
        method: str = "eemd",
        max_imf: Optional[int] = None,
        sift_max_iter: int = 1000,
        stopping_criterion: Any = "default",
        eemd_noise_std: float = 0.2,
        eemd_trials: int = 100,
        random_state: Optional[int] = None,
        return_residual: bool = True,
    ) -> Any:
        """
        Perform Empirical Mode Decomposition (EMD) or EEMD.

        Parameters
        ----------
        method : {"emd", "eemd"}, optional
            Decomposition method. Default is "eemd".
        max_imf : `int`, optional
            Maximum number of IMFs to extract.
        sift_max_iter : `int`, optional
            Maximum number of sifting iterations.
        stopping_criterion : `str` or `dict`, optional
            Stopping criterion configuration.
        eemd_noise_std : `float`, optional
            Noise standard deviation for EEMD. Default 0.2.
        eemd_trials : `int`, optional
            Number of trials for EEMD. Default 100.
        random_state : `int` or `np.random.RandomState`, optional
            Seed for reproducibility.
        return_residual : `bool`, optional
            If True, include residual in the output.

        Returns
        -------
        imfs : `TimeSeriesDict`
            A dictionary of extracted IMFs (keys: "IMF1", "IMF2", ..., "residual").
        """
        try:
            import PyEMD
        except ImportError:
            raise ImportError(
                "PyEMD is required for EMD/EEMD. "
                "Please install it via `pip install EMD-signal`."
            )

        # Config RNG
        if random_state is not None:
             if isinstance(random_state, int):
                  np.random.seed(random_state)

        data = self.value

        # Initialize EMD/EEMD
        if method.lower() == "eemd":
             decomposer = PyEMD.EEMD(trials=eemd_trials, noise_width=eemd_noise_std)
             if max_imf is not None:
                  pass  # Handled in eemd call

             try:
                  imfs_array = decomposer.eemd(
                      data,
                      T=None,
                      max_imf=max_imf if max_imf is not None else -1,
                  )
             except (PermissionError, OSError) as exc:
                  if isinstance(exc, OSError) and exc.errno not in (None, 13):
                       raise
                  # Fallback to serial execution
                  decomposer.parallel = False
                  decomposer.processes = 1
                  imfs_array = decomposer.eemd(
                      data,
                      T=None,
                      max_imf=max_imf if max_imf is not None else -1,
                  )

        elif method.lower() == "emd":
             decomposer = PyEMD.EMD()
             imfs_array = decomposer.emd(data, T=None, max_imf=max_imf if max_imf is not None else -1)
        else:
             raise ValueError(f"Unknown EMD method: {method}")

        n_rows = imfs_array.shape[0]

        from .collections import TimeSeriesDict
        out_dict = TimeSeriesDict()

        n_imfs = n_rows - 1

        for i in range(n_imfs):
             key = f"IMF{i+1}"
             out_dict[key] = self.__class__(
                  imfs_array[i],
                  t0=self.t0,
                  dt=self.dt,
                  unit=self.unit,
                  name=f"{self.name}_{key}" if self.name else key,
                  channel=self.channel
             )

        if return_residual:
             key = "residual"
             out_dict[key] = self.__class__(
                  imfs_array[-1],
                  t0=self.t0,
                  dt=self.dt,
                  unit=self.unit,
                  name=f"{self.name}_{key}" if self.name else key,
                  channel=self.channel
             )

        return out_dict

    # ===============================
    # Hilbert Analysis
    # ===============================

    def hilbert_analysis(self, *, unwrap_phase: bool = True, frequency_unit: str = "Hz") -> dict[str, Any]:
        """
        Perform Hilbert Spectral Analysis (HSA) to get instantaneous properties.

        Returns
        -------
        results : `dict`
            {
                "analytic": TimeSeries (complex),
                "amplitude": TimeSeries (real),
                "phase": TimeSeries (rad),
                "frequency": TimeSeries (Hz)
            }
        """
        # 1. Analytic Signal
        analytic = self.analytic_signal()

        # 2. Amplitude
        amp = np.abs(analytic.value)
        amplitude = self.__class__(
             amp,
             t0=self.t0,
             dt=self.dt,
             unit=self.unit,
             name=f"{self.name}_IA" if self.name else "IA",
             channel=self.channel,
        )

        # 3. Phase
        pha = np.angle(analytic.value)
        if unwrap_phase:
             pha = np.unwrap(pha)

        phase = self.__class__(
             pha,
             t0=self.t0,
             dt=self.dt,
             unit="rad",
             name=f"{self.name}_Phase" if self.name else "Phase",
             channel=self.channel
        )

        # 4. Instantaneous Frequency
        if self.dt is None:
             raise ValueError("dt is required for Instantaneous Frequency")
        dt_val = self.dt.to("s").value

        dphi = np.gradient(pha, dt_val)
        freq_val = dphi / (2 * np.pi)

        frequency = self.__class__(
             freq_val,
             t0=self.t0,
             dt=self.dt,
             unit="Hz",
             name=f"{self.name}_IF" if self.name else "IF",
             channel=self.channel
        )

        if frequency_unit != "Hz":
             frequency = frequency.to(frequency_unit)

        return {
             "analytic": analytic,
             "amplitude": amplitude,
             "phase": phase,
             "frequency": frequency
        }

    # ===============================
    # HHT (Hilbert-Huang Transform)
    # ===============================

    def hht(
        self,
        *,
        emd_method: str = "eemd",
        emd_kwargs: Optional[dict[str, Any]] = None,
        hilbert_kwargs: Optional[dict[str, Any]] = None,
        output: str = "dict",
    ) -> Any:
        """
        Perform Hilbert-Huang Transform (HHT): EMD + HSA.

        Parameters
        ----------
        emd_method : `str`
            "emd" or "eemd".
        emd_kwargs : `dict`, optional
            Arguments for `emd()`.
        hilbert_kwargs : `dict`, optional
            Arguments for `hilbert_analysis()`.
        output : `str`
            "dict" (returns IMFs and properties) or "spectrogram" (returns Spectrogram).

        Returns
        -------
        result : `dict` or `Spectrogram`
        """
        if emd_kwargs is None:
             emd_kwargs = {}
        if hilbert_kwargs is None:
             hilbert_kwargs = {}

        # 1. EMD
        imfs = self.emd(method=emd_method, **emd_kwargs)

        # 2. HSA for each IMF
        from .collections import TimeSeriesDict
        ia_dict = TimeSeriesDict()
        if_dict = TimeSeriesDict()

        residual = None
        if "residual" in imfs:
             residual = imfs.pop("residual")

        for key, imf in imfs.items():
             res = imf.hilbert_analysis(**hilbert_kwargs)
             ia_dict[key] = res["amplitude"]
             if_dict[key] = res["frequency"]

        if output == "dict":
             return {
                  "imfs": imfs,
                  "ia": ia_dict,
                  "if": if_dict,
                  "residual": residual
             }

        elif output == "spectrogram":
             # 3. Construct Hilbert Spectrum (Spectrogram)
             fs_rate = self.sample_rate.to("Hz").value
             nyquist = fs_rate / 2.0

             n_bins = 100
             freq_bins = np.linspace(0, nyquist, n_bins + 1)

             n_time = len(self)

             keys = list(imfs.keys())
             if not keys:
                  return None

             if_stack = np.stack([if_dict[k].value for k in keys])
             ia_stack = np.stack([ia_dict[k].value for k in keys])

             # Digitize frequencies
             inds = np.digitize(if_stack, freq_bins) - 1

             # Valid indices
             mask = (inds >= 0) & (inds < n_bins)

             # Target grid: (N_Time, N_Bins)
             grid = np.zeros((n_time, n_bins))

             for k in range(len(keys)):
                  valid = mask[k]
                  t_inds = np.arange(n_time)[valid]
                  f_inds = inds[k][valid]
                  energies = ia_stack[k][valid] ** 2

                  np.add.at(grid, (t_inds, f_inds), energies)

             from gwpy.spectrogram import Spectrogram

             freq_centers = (freq_bins[:-1] + freq_bins[1:]) / 2.0

             return Spectrogram(
                  grid,
                  times=self.times,
                  frequencies=u.Quantity(freq_centers, "Hz"),
                  unit=self.unit**2,
                  name=self.name + " Hilbert Spectrum",
                  channel=self.channel,
                  epoch=self.epoch
             )

        else:
             raise ValueError(f"Unknown output format: {output}")
