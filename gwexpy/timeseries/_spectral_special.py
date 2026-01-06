"""
Special spectral transform methods for TimeSeries (HHT, EMD, Laplace, CWT).
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from typing import Optional, Any, TYPE_CHECKING
from gwexpy.interop._optional import require_optional

if TYPE_CHECKING:
    pass

class TimeSeriesSpectralSpecialMixin:
    """
    Mixin class providing special spectral transform methods.
    """

    def laplace(
        self,
        *,
        sigma: float | u.Quantity = 0.0,
        frequencies: Optional[np.ndarray | u.Quantity] = None,
        t_start: Optional[float | u.Quantity] = None,
        t_stop: Optional[float | u.Quantity] = None,
        window: Optional[str | tuple | np.ndarray] = None,
        detrend: bool = False,
        normalize: str = "integral",
        dtype: Optional[np.dtype] = None,
        chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Compute the Laplace Transform.

        Parameters
        ----------
        sigma : `float` or `Quantity`, optional
            The real part of the complex frequency s = sigma + j*omega.
        frequencies : `array_like` or `Quantity`, optional
            The frequencies (omega / 2pi) at which to evaluate.
        t_start, t_stop : `float` or `Quantity`, optional
            Time range for the transform.
        window : `str`, `tuple`, or `array_like`, optional
            Window function to apply.
        detrend : `bool`, optional
            If `True`, detrend the data before transforming.
        normalize : {"integral", "mean"}, optional
            Normalization mode.
        dtype : `dtype`, optional
            Output data type.
        chunk_size : `int`, optional
            Number of frequencies to process at once for memory efficiency.

        Returns
        -------
        `FrequencySeries`
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

        data = self.value[idx_start:idx_stop]
        data = self._prepare_data_for_transform(data=data, window=window, detrend=detrend)
        n = len(data)

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

        if isinstance(sigma, u.Quantity):
             sigma_val = sigma.to("1/s").value
        else:
             sigma_val = float(sigma)

        tau = np.arange(n) * dt_val
        if dtype is None:
             out_dtype = np.result_type(data.dtype, np.complex128)
        else:
             out_dtype = dtype

        n_freqs = len(freqs_val)
        out_data = np.zeros(n_freqs, dtype=out_dtype)

        if normalize == "integral":
             norm_factor = dt_val
        elif normalize == "mean":
             norm_factor = 1.0 / n
        else:
             raise ValueError(f"Unknown normalize mode: {normalize}")

        if chunk_size is None:
             max_elements = 10_000_000
             if n_freqs * n > max_elements:
                 chunk_size = max(1, max_elements // n)
             else:
                 chunk_size = n_freqs

        real_exp = np.exp(-sigma_val * tau)
        data_weighted = data * real_exp

        for i in range(0, n_freqs, chunk_size):
             end = min(i + chunk_size, n_freqs)
             f_chunk = freqs_val[i:end]
             # Optimized: Using broadcasting with complex exponential
             phase_chunk = (-2j * np.pi) * (f_chunk[:, None] * tau[None, :])
             complex_exp_chunk = np.exp(phase_chunk)
             out_data[i:end] = np.dot(complex_exp_chunk, data_weighted) * norm_factor

        from gwexpy.frequencyseries import FrequencySeries
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

    def stlt(
        self,
        stride: Any = None,
        window: Any = None,
        fftlength: Any = None,
        overlap: Any = None,
        *,
        sigmas: Any = 0.0,
        frequencies: Any = None,
        scaling: str = "dt",
        time_ref: str = "start",
        onesided: Optional[bool] = None,
        legacy: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Compute Short-Time Laplace Transform (STLT).

        Chunk the time series, apply window w[n] * exp(-sigma * t_rel[n]), and compute FFT.

        Parameters
        ----------
        stride : Quantity or str, optional
            Step size in seconds between chunks.
        window : Quantity, str, or array-like, optional
            If Quantity/str with units: window duration (legacy style or alias for fftlength).
            If str (no units) or array: window function (passed to scipy.signal.get_window).
            Default 'hann'.
        fftlength : Quantity or str, optional
            Window duration. Preferred over 'window' for specifying duration.
        overlap : Quantity or str, optional
            Overlap duration.
        sigmas : float or array-like or Quantity, optional
            Real part of Laplace frequency s = sigma + j*omega. Default 0.
        frequencies : array-like, optional
            Output frequencies. Currently ignored (always uses FFT grid).
        scaling : str, optional
            'dt' (multiply by dt, discrete integral), 'none' (raw FFT).
            Default 'dt'.
        time_ref : str, optional
            Time reference for the exponential term within each window.
            'start': t_rel in [0, T_win].
            'center': t_rel in [-T_win/2, T_win/2]. Recommended for large sigmas to avoid overflow.
            Default 'start'.
        onesided : bool, optional
            If True, return one-sided FFT (real input only).
            If False, return two-sided FFT.
            If None, defaults to True for real data, False for complex data.
        legacy : bool, optional
            If True, use the old magnitude-outer-product implementation (deprecated).

        Returns
        -------
        LaplaceGram
            3D transform result with shape (time, sigma, frequency).

        Notes
        -----
        This method uses a fully vectorized implementation for performance, broadcasting over sigmas and time chunks.
        """
        if legacy:
             return self._stlt_legacy(stride, window, **kwargs)

        try:
             import scipy.signal  # noqa: F401 - availability check
        except ImportError:
             raise ImportError("scipy is required for stlt.")

        # --- 1. Resolve Time Parameters ---
        if self.dt is None:
             raise ValueError("TimeSeries must have a valid dt.")
        dt_s = self.dt.to("s").value
        1.0 / dt_s

        # Helper to ensure quantity is in seconds
        def _to_sec(val, name):
             if val is None:
                 return None
             if isinstance(val, (int, float, np.number)):
                 return float(val)
             q = u.Quantity(val)
             if q.unit == u.dimensionless_unscaled:
                 # Assume seconds if dimensionless (e.g. from float) -> wait, Quantity(0.5) is dimless
                 # Quantity(0.5, 's') is time.
                 # If user passed Number, we handled above.
                 # If user passed Quantity(0.5), treating as seconds is ambiguous but likely intended if not specifed.
                 # But safer to assign unit "s" if original was just number.
                 return q.value
             try:
                 return q.to("s").value
             except u.UnitConversionError:
                 raise ValueError(f"{name} must be a time quantity (e.g. seconds).")

        # Resolve fftlength (duration) / window (function vs duration)
        window_func = 'hann'
        win_dur = None

        if fftlength is not None:
             win_dur = _to_sec(fftlength, "fftlength")
             if window is not None and not isinstance(window, (u.Quantity, float, int)):
                  window_func = window
        elif window is not None:
             is_dur = False
             if isinstance(window, (int, float, np.number)) or isinstance(window, u.Quantity):
                  is_dur = True
             elif isinstance(window, str) and any(c.isdigit() for c in window):
                  # Heuristic: if string contains digits, treat as quantity string "4s"
                  is_dur = True

             if is_dur:
                  win_dur = _to_sec(window, "window")
             else:
                  window_func = window

        if win_dur is None:
             raise ValueError("Must specify fftlength (or window duration).")

        nperseg = int(np.round(win_dur / dt_s))
        if nperseg < 1:
             raise ValueError("Window duration too short.")

        # Resolve Overlap/Stride
        if overlap is not None:
             ov_dur = _to_sec(overlap, "overlap")
             noverlap = int(np.round(ov_dur / dt_s))
             step = nperseg - noverlap
             str_dur = step * dt_s
        elif stride is not None:
             str_dur = _to_sec(stride, "stride")
             step = int(np.round(str_dur / dt_s))

             # Re-calculate str_dur from integer step to match actual execution?
             # Or keep user's request? Standard is usually to snap to samples.
             # step = max(1, step) # ensure at least 1 sample? Validated below.

             noverlap = nperseg - step
             ov_dur = noverlap * dt_s
        else:
             step = nperseg // 2
             noverlap = nperseg - step
             str_dur = step * dt_s
             ov_dur = noverlap * dt_s

        if step < 1:
             raise ValueError("Stride must be positive (overlap < window).")

        # --- 2. Data Chunking & Pre-checks ---
        from numpy.lib.stride_tricks import sliding_window_view

        data_arr = self.value
        if len(data_arr) < nperseg:
             raise ValueError("Data shorter than window length.")

        chunks = sliding_window_view(data_arr, window_shape=nperseg, axis=0)[::step]
        n_chunks = chunks.shape[0]

        t0_val = self.t0.to("s").value
        t_starts = t0_val + (np.arange(n_chunks) * step * dt_s)
        t_centers = t_starts + (win_dur / 2.0)
        times_q = u.Quantity(t_centers, "s")

        # --- 3. Window & Sigma Prep ---
        if isinstance(window_func, (str, tuple)):
             win_base = scipy.signal.get_window(window_func, nperseg)
        else:
             win_base = np.asarray(window_func)
             if len(win_base) != nperseg:
                  raise ValueError("Window function length mismatch.")

        # Handle Sigmas
        if np.isscalar(sigmas) or (isinstance(sigmas, u.Quantity) and sigmas.isscalar):
             sigmas_arr = [sigmas]
        else:
             sigmas_arr = sigmas

        sigmas_vals = []
        for s in sigmas_arr:
             if isinstance(s, u.Quantity):
                  sigmas_vals.append(s.to("1/s").value)
             else:
                  sigmas_vals.append(float(s))
        sigmas_vals = np.array(sigmas_vals)

        # Handle One-sided
        is_complex_input = np.iscomplexobj(data_arr)
        if onesided is None:
             onesided = not is_complex_input

        if onesided and is_complex_input:
             raise ValueError("Cannot perform one-sided FFT on complex data.")

        # Precompute t_rel
        if time_ref == "start":
             t_rel = np.arange(nperseg) * dt_s
        elif time_ref == "center":
             # centered around 0
             t_rel = (np.arange(nperseg) - (nperseg - 1) / 2.0) * dt_s
        else:
             raise ValueError(f"Unknown time_ref: {time_ref}. Use 'start' or 'center'.")

        # Stability Guardrails
        # Max exponent argument
        min_sigma = sigmas_vals.min()
        max_sigma = sigmas_vals.max()
        max_t = t_rel.max()
        min_t = t_rel.min() # negative if centered

        # We care about -sigma * t
        # Case 1: sigma positive, t negative (if centered) -> -sigma*t positive (growth)
        # Case 2: sigma negative, t positive -> -sigma*t positive (growth)

        max_exponent = max(
             -min_sigma * min_t,
             -min_sigma * max_t,
             -max_sigma * min_t,
             -max_sigma * max_t
        )

        if max_exponent > 700: # approx exp(709) is max double
             raise ValueError(
                 f"Configuration leads to overflow: max exponent ~{max_exponent:.1f} > 700. "
                 f"Try reducing sigma magnitude or using time_ref='center'."
             )

        # Prepare outputs
        if frequencies is None:
             if onesided:
                  freqs_val = np.fft.rfftfreq(nperseg, d=dt_s)
             else:
                  freqs_val = np.fft.fftfreq(nperseg, d=dt_s)
             freqs_q = u.Quantity(freqs_val, "Hz")
        else:
             # Just matching axes, we still compute full FFT (optimization for later phase)
             # If user supplied frequencies, we should warn if they don't match FFT grid?
             import warnings
             warnings.warn(
                 "Custom `frequencies` argument is currently ignored in `stlt`. "
                 "Output will always use the FFT frequency grid.",
                 UserWarning
             )
             if onesided:
                  freqs_val = np.fft.rfftfreq(nperseg, d=dt_s)
             else:
                  freqs_val = np.fft.fftfreq(nperseg, d=dt_s)
             freqs_q = u.Quantity(freqs_val, "Hz")

        n_freqs = len(freqs_val)
        n_sigmas = len(sigmas_vals)

        dtype = np.result_type(chunks.dtype, np.complex64)
        out_cube = np.zeros((n_chunks, n_sigmas, n_freqs), dtype=dtype)

        # --- 4. Computation Loop with Batching ---
        # Batch over chunks to control memory usage
        # weighted_chunks for 1 sigma = n_chunks * nperseg * 16 bytes (complex128)
        # We want to keep weighted_chunks < ~100MB?
        # 100MB / 16 bytes ~ 6e6 elements.
        # If nperseg=1000, batch_size=6000 chunks.

        BATCH_ELEMENTS = 5_000_000
        # Adjust batch size to account for expansion by n_sigmas
        # Each element in the batch loop will become (batch_size, n_sigmas, nperseg) complex128
        # So we divide BATCH_ELEMENTS by (nperseg * n_sigmas)
        elements_per_chunk = nperseg * n_sigmas
        chunk_batch_size = max(1, BATCH_ELEMENTS // elements_per_chunk)

        fft_func = np.fft.rfft if onesided else np.fft.fft

        # Precompute effective windows for all sigmas: (n_sigmas, nperseg)
        # decay = exp(-sigma * t)
        # shape: (S, N)
        decay_matrix = np.exp(-sigmas_vals[:, None] * t_rel[None, :])
        effective_windows = win_base[None, :] * decay_matrix

        for i_chunk in range(0, n_chunks, chunk_batch_size):
             end_chunk = min(i_chunk + chunk_batch_size, n_chunks)
             # Get batch of chunks: (batch_size, nperseg)
             batch_chunks = chunks[i_chunk:end_chunk] # View

             # Vectorized Window Application
             # batch_chunks: (B, N)
             # effective_windows: (S, N)
             # Broadcasting: (B, 1, N) * (1, S, N) -> (B, S, N)
             weighted_batch = batch_chunks[:, None, :] * effective_windows[None, :, :]

             # FFT along the last axis (time/window axis)
             # Result shape: (B, S, n_freqs)
             spec = fft_func(weighted_batch, axis=-1)

             # Store
             out_cube[i_chunk:end_chunk, :, :] = spec

        # --- 5. Scaling ---
        if scaling == "dt":
             out_cube *= dt_s
        elif scaling == "none":
             pass
        else:
             pass


        # --- 6. Container ---
        from gwexpy.types.time_plane_transform import LaplaceGram
        from gwexpy.types.array3d import Array3D

        # Axis 1 is Sigma
        axis_sigma = u.Quantity(sigmas_vals, "1/s")
        # Axis 2 is Frequency
        axis_freq = freqs_q

        # Unit
        res_unit = self.unit
        if scaling == "dt":
             res_unit = res_unit * u.s

        # Construct Array3D explicitly
        arr3d = Array3D(
            out_cube,
            unit=res_unit,
            axis0=times_q,
            axis1=axis_sigma,
            axis2=axis_freq,
            axis_names=["time", "sigma", "frequency"]
        )

        return LaplaceGram(
             arr3d,
             kind="stlt",
             meta={
                 "window": win_dur,
                 "stride": str_dur if stride is not None else None,
                 "overlap": ov_dur if overlap is not None else None,
                 "source": self.name
             }
        )

    def _stlt_legacy(self, stride: Any, window: Any, **kwargs: Any) -> Any:
        # Copied from original implementation
        from gwexpy.types.time_plane_transform import TimePlaneTransform
        try:
            import scipy.signal  # noqa: F401 - availability check
        except ImportError:
            raise ImportError(
                "scipy is required for stlt. "
                "Please install it via `pip install scipy`."
            )

        # Normalize inputs
        stride_q = u.Quantity(stride) if isinstance(stride, str) else stride
        window_q = u.Quantity(window) if isinstance(window, str) else window

        if not stride_q.unit.is_equivalent(u.s):
            raise ValueError("stride must be a time quantity")

        dt_s = self.dt.to(u.s).value
        fs_val = 1.0 / dt_s

        nperseg = int(np.round((window_q.to(u.s).value / dt_s)))
        stride_s = stride_q.to(u.s).value
        noverlap_samples = int(np.round((window_q.to(u.s).value - stride_s) / dt_s))

        if nperseg <= 0:
             raise ValueError("Window size too small")

        f, t_segs, Zxx = scipy.signal.stft(
             self.value,
             fs=fs_val,
             window='hann',
             nperseg=nperseg,
             noverlap=noverlap_samples,
             boundary=None,
             padded=False
        )

        Z = Zxx.T
        mag = np.abs(Z)
        data = mag[:, :, None] * mag[:, None, :]

        t0_val = self.t0.to(u.s).value
        times_q = (t0_val + t_segs) * u.s
        axis_f = f * u.Hz

        return TimePlaneTransform(
            (data, times_q, axis_f, axis_f, self.unit**2),
            kind="stlt_mag_outer",
            meta={"stride": stride, "window": window, "source": self.name}
        )

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
        self._check_regular("cwt")
        pywt = require_optional("pywt")
        require_optional("scipy") # Ensure scipy is available for prepare_data

        data = self._prepare_data_for_transform(window=window, detrend=detrend)
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
             out_spec = coefs.T
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
        PyEMD = require_optional("PyEMD")

        if random_state is not None:
             if isinstance(random_state, int):
                  np.random.seed(random_state)

        data = self.value
        if method.lower() == "eemd":
             decomposer = PyEMD.EEMD(trials=eemd_trials, noise_width=eemd_noise_std)
             try:
                  imfs_array = decomposer.eemd(
                      data,
                      T=None,
                      max_imf=max_imf if max_imf is not None else -1,
                  )
             except (PermissionError, OSError) as exc:
                  if isinstance(exc, OSError) and exc.errno not in (None, 13):
                       raise
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

    def hilbert_analysis(self, *, unwrap_phase: bool = True, frequency_unit: str = "Hz") -> dict[str, Any]:
        analytic = self.hilbert()
        amp = np.abs(analytic.value)

        amplitude = self.__class__(
             amp,
             t0=self.t0,
             dt=self.dt,
             unit=self.unit,
             name=f"{self.name}_IA" if self.name else "IA",
             channel=self.channel,
        )
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

    def hht(
        self,
        *,
        emd_method: str = "eemd",
        emd_kwargs: Optional[dict[str, Any]] = None,
        hilbert_kwargs: Optional[dict[str, Any]] = None,
        output: str = "dict",
    ) -> Any:
        if emd_kwargs is None:
             emd_kwargs = {}
        if hilbert_kwargs is None:
             hilbert_kwargs = {}

        imfs = self.emd(method=emd_method, **emd_kwargs)
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
             inds = np.digitize(if_stack, freq_bins) - 1
             mask = (inds >= 0) & (inds < n_bins)
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
