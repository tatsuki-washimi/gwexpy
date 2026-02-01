"""
Special spectral transform methods for TimeSeries (HHT, EMD, Laplace, CWT).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u

from gwexpy.interop._optional import require_optional

from ._typing import TimeSeriesAttrs

if TYPE_CHECKING:
    pass


class TimeSeriesSpectralSpecialMixin(TimeSeriesAttrs):
    """
    Mixin class providing special spectral transform methods.
    """

    def laplace(
        self,
        *,
        sigma: float | u.Quantity = 0.0,
        frequencies: np.ndarray | u.Quantity | None = None,
        t_start: float | u.Quantity | None = None,
        t_stop: float | u.Quantity | None = None,
        window: str | tuple | np.ndarray | None = None,
        detrend: bool = False,
        normalize: str = "integral",
        dtype: np.dtype | None = None,
        chunk_size: int | None = None,
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
            raise ValueError(
                f"Invalid time range: t_start index {idx_start} >= t_stop index {idx_stop}"
            )

        data = self.value[idx_start:idx_stop]
        data = self._prepare_data_for_transform(
            data=data, window=window, detrend=detrend
        )
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

        # Overflow guardrail for sigma < 0
        # exp(-sigma * tau) = exp(|sigma| * tau) for sigma < 0
        # This can overflow float64 when -sigma * tau_max > ~709
        max_exponent = -sigma_val * tau.max()
        if max_exponent > 700:
            raise ValueError(
                f"Configuration leads to overflow: exp({max_exponent:.1f}) exceeds float64 range. "
                f"sigma={sigma_val:.3g}/s, max_tau={tau.max():.3g}s. "
                f"Consider using smaller |sigma| or shorter data duration."
            )

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
        onesided: bool | None = None,
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
        window_func = "hann"
        win_dur = None

        if fftlength is not None:
            win_dur = _to_sec(fftlength, "fftlength")
            if window is not None and not isinstance(window, (u.Quantity, float, int)):
                window_func = window
        elif window is not None:
            is_dur = False
            if isinstance(window, (int, float, np.number)) or isinstance(
                window, u.Quantity
            ):
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
            sigmas_arr = list(sigmas)

        sigmas_vals_list: list[float] = []
        for s in sigmas_arr:
            if isinstance(s, u.Quantity):
                sigmas_vals_list.append(float(s.to("1/s").value))
            else:
                sigmas_vals_list.append(float(s))  # type: ignore[arg-type]
        sigmas_vals = np.asarray(sigmas_vals_list, dtype=float)

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
        min_t = t_rel.min()  # negative if centered

        # We care about -sigma * t
        # Case 1: sigma positive, t negative (if centered) -> -sigma*t positive (growth)
        # Case 2: sigma negative, t positive -> -sigma*t positive (growth)

        max_exponent = max(
            -min_sigma * min_t,
            -min_sigma * max_t,
            -max_sigma * min_t,
            -max_sigma * max_t,
        )

        if max_exponent > 700:  # approx exp(709) is max double
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
                UserWarning,
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
            batch_chunks = chunks[i_chunk:end_chunk]  # View

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
        from gwexpy.types.array3d import Array3D
        from gwexpy.types.time_plane_transform import LaplaceGram

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
            axis_names=["time", "sigma", "frequency"],
        )

        return LaplaceGram(
            arr3d,
            kind="stlt",
            meta={
                "window": win_dur,
                "stride": str_dur if stride is not None else None,
                "overlap": ov_dur if overlap is not None else None,
                "source": self.name,
            },
        )

    def _stlt_legacy(self, stride: Any, window: Any, **kwargs: Any) -> Any:
        # Copied from original implementation
        from gwexpy.types.time_plane_transform import TimePlaneTransform

        try:
            import scipy.signal  # noqa: F401 - availability check
        except ImportError:
            raise ImportError(
                "scipy is required for stlt. Please install it via `pip install scipy`."
            )

        # Normalize inputs
        stride_q = u.Quantity(stride) if isinstance(stride, str) else stride
        window_q = u.Quantity(window) if isinstance(window, str) else window

        if not stride_q.unit.is_equivalent(u.s):
            raise ValueError("stride must be a time quantity")

        if self.dt is None:
            raise ValueError("stlt legacy requires defined dt.")
        dt_s = self.dt.to(u.s).value
        fs_val = 1.0 / dt_s

        nperseg = int(np.round(window_q.to(u.s).value / dt_s))
        stride_s = stride_q.to(u.s).value
        noverlap_samples = int(np.round((window_q.to(u.s).value - stride_s) / dt_s))

        if nperseg <= 0:
            raise ValueError("Window size too small")

        f, t_segs, Zxx = scipy.signal.stft(
            self.value,
            fs=fs_val,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap_samples,
            boundary=None,
            padded=False,
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
            meta={"stride": stride, "window": window, "source": self.name},
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
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Compute the Continuous Wavelet Transform (CWT).

        The CWT provides a time-frequency representation of the signal
        using a wavelet basis. Unlike STFT, CWT uses wavelets of varying
        width to achieve better time resolution at high frequencies and
        better frequency resolution at low frequencies.

        Parameters
        ----------
        wavelet : str, optional
            Wavelet to use. Default is 'cmor1.5-1.0' (Complex Morlet).
            See pywt.wavelist() for available wavelets.
        widths : array-like, optional
            Wavelet scales to use. Mutually exclusive with `frequencies`.
        frequencies : array-like or Quantity, optional
            Target frequencies in Hz. Scales are computed from these.
            Mutually exclusive with `widths`.
        window : str, tuple, or array-like, optional
            Window function to apply before the transform.
        detrend : bool, optional
            If True, remove linear trend before the transform.
        output : {'spectrogram', 'ndarray'}, optional
            Output format. 'spectrogram' returns a gwpy Spectrogram.
            'ndarray' returns (coefficients, frequencies) tuple.
        chunk_size : int, optional
            Number of scales to process at once for memory efficiency.
        **kwargs
            Additional arguments passed to pywt.cwt().

        Returns
        -------
        Spectrogram or tuple
            If output='spectrogram': a Spectrogram object.
            If output='ndarray': (coefficients, frequencies) tuple.

        Notes
        -----
        Requires the `pywt` (PyWavelets) package.

        **Complex Coefficients**

        For complex wavelets (e.g., 'cmor', 'morl'), the CWT coefficients are
        **complex-valued**, containing both magnitude and phase information:

        - ``np.abs(coefs)`` : Envelope/magnitude (instantaneous amplitude)
        - ``np.angle(coefs)`` : Instantaneous phase (radians)
        - ``np.real(coefs)`` : Real part (in-phase component)
        - ``np.imag(coefs)`` : Imaginary part (quadrature component)

        For time-frequency power analysis, use ``np.abs(coefs)**2``.

        The returned Spectrogram object preserves complex dtype. To visualize,
        apply ``np.abs()`` before plotting.

        Examples
        --------
        >>> ts = TimeSeries(data, sample_rate=1024)
        >>> spec = ts.cwt(frequencies=np.logspace(0, 2, 50))
        >>> power = np.abs(spec.value)**2  # Time-frequency power
        """
        self._check_regular("cwt")

        pywt = require_optional("pywt")
        require_optional("scipy")  # Ensure scipy is available for prepare_data

        data = self._prepare_data_for_transform(window=window, detrend=detrend)
        if self.dt is None:
            raise ValueError("cwt requires defined dt.")
        dt = self.dt.to("s").value

        if frequencies is not None:
            if widths is not None:
                raise ValueError("Cannot specify both widths(scales) and frequencies")
            freqs_quant = u.Quantity(frequencies, "Hz")
            freqs_arr = freqs_quant.value
            center_freq = pywt.scale2frequency(wavelet, 1)
            with np.errstate(divide="ignore"):
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
                epoch=self.epoch,
            )
        else:
            raise ValueError(f"Unknown output format: {output}")

    def emd(
        self,
        *,
        method: str = "eemd",
        max_imf: int | None = None,
        sift_max_iter: int = 1000,
        stopping_criterion: Any = "default",
        eemd_noise_std: float = 0.2,
        eemd_trials: int = 100,
        random_state: int | None = None,
        return_residual: bool = True,
        eemd_parallel: bool | None = None,
        eemd_processes: int | None = None,
        eemd_noise_kind: str | None = None,
    ) -> Any:
        """
        Decompose the TimeSeries using Empirical Mode Decomposition (EMD).

        This method applies EMD or Ensemble EMD (EEMD) to decompose the signal
        into Intrinsic Mode Functions (IMFs) and a residual.

        Parameters
        ----------
        method : str, default='eemd'
            Decomposition method. Either 'emd' or 'eemd'.
        max_imf : int or None, default=None
            Maximum number of IMFs to extract. If None, extracts all.
        sift_max_iter : int, default=1000
            Maximum iterations per sifting process.
        stopping_criterion : Any, default='default'
            Stopping criterion for sifting.
        eemd_noise_std : float, default=0.2
            Standard deviation of added noise for EEMD (ratio of signal std).
        eemd_trials : int, default=100
            Number of ensemble trials for EEMD.
        random_state : int or None, default=None
            Random seed for reproducibility. If provided and the decomposer
            supports ``noise_seed()``, it will be used. Otherwise, NumPy's
            random state is temporarily set and restored.
        return_residual : bool, default=True
            If True, include the residual in the output.
        eemd_parallel : bool or None, default=None
            Enable parallel processing for EEMD. If None, uses PyEMD default.
        eemd_processes : int or None, default=None
            Number of processes for parallel EEMD. If None, uses PyEMD default.
        eemd_noise_kind : str or None, default=None
            Type of noise for EEMD ('normal', 'uniform'). If None, uses default.

        Returns
        -------
        TimeSeriesDict
            Dictionary containing IMFs (keys: 'IMF1', 'IMF2', ...) and
            optionally 'residual'.

        Raises
        ------
        ImportError
            If PyEMD is not installed.
        ValueError
            If an unknown method is specified or no IMFs are extracted.

        Notes
        -----
        **Optional Dependency**: Requires the ``PyEMD`` package.

        **EEMD Stochasticity**: EEMD adds noise to the signal and performs
        multiple decompositions. Results may vary between runs unless
        ``random_state`` is specified.

        **Endpoint Artifacts**: EMD envelope extrapolation can cause artifacts
        at signal boundaries. Consider padding or cropping edges in downstream
        analysis.

        **Residual Handling**: The residual is extracted using PyEMD's
        ``get_imfs_and_residue()`` method if available, otherwise via the
        ``residue`` attribute. This ensures correct IMF count.

        Examples
        --------
        >>> ts = TimeSeries(data, dt=0.01, unit='V')
        >>> imfs = ts.emd(method='eemd', eemd_trials=50, random_state=42)
        >>> for key, imf in imfs.items():
        ...     print(f"{key}: {imf.shape}")
        """
        PyEMD = require_optional("PyEMD")

        # Save and restore RNG state if needed
        saved_rng_state = None
        if random_state is not None:
            saved_rng_state = np.random.get_state()

        data = self.value
        residual = None

        try:
            if method.lower() == "eemd":
                decomposer = PyEMD.EEMD(trials=eemd_trials, noise_width=eemd_noise_std)

                # Apply optional EEMD controls
                if eemd_parallel is not None:
                    decomposer.parallel = eemd_parallel
                if eemd_processes is not None:
                    decomposer.processes = eemd_processes
                if eemd_noise_kind is not None:
                    decomposer.noise_kind = eemd_noise_kind

                # Set random seed using decomposer's method if available
                if random_state is not None:
                    if hasattr(decomposer, "noise_seed"):
                        decomposer.noise_seed(random_state)
                    else:
                        np.random.seed(random_state)

                try:
                    imfs_array = decomposer.eemd(
                        data,
                        T=None,
                        max_imf=max_imf if max_imf is not None else -1,
                    )
                except (PermissionError, OSError) as exc:
                    if isinstance(exc, OSError) and exc.errno not in (None, 13):
                        raise
                    # Fallback to non-parallel execution
                    decomposer.parallel = False
                    decomposer.processes = 1
                    imfs_array = decomposer.eemd(
                        data,
                        T=None,
                        max_imf=max_imf if max_imf is not None else -1,
                    )

                # Extract residual properly for EEMD
                if hasattr(decomposer, "get_imfs_and_residue"):
                    imfs_array, residual = decomposer.get_imfs_and_residue()
                elif hasattr(decomposer, "residue") and decomposer.residue is not None:
                    residual = decomposer.residue
                    # imfs_array already contains only IMFs in this case
                else:
                    # Fallback: no separate residual available
                    residual = None

            elif method.lower() == "emd":
                decomposer = PyEMD.EMD()
                # EMD is deterministic; random_state is not applicable here
                # (random_state only affects EEMD via noise_seed())

                imfs_array = decomposer.emd(
                    data, T=None, max_imf=max_imf if max_imf is not None else -1
                )

                # Extract residual properly for EMD
                if hasattr(decomposer, "get_imfs_and_residue"):
                    imfs_array, residual = decomposer.get_imfs_and_residue()
                elif hasattr(decomposer, "residue") and decomposer.residue is not None:
                    residual = decomposer.residue
                else:
                    residual = None

            else:
                raise ValueError(f"Unknown EMD method: {method}")

        finally:
            # Restore RNG state if we saved it
            if saved_rng_state is not None:
                np.random.set_state(saved_rng_state)

        # Validate results
        if imfs_array is None or imfs_array.shape[0] == 0:
            raise ValueError(
                "EMD decomposition returned no IMFs. Check input signal quality."
            )

        n_imfs = imfs_array.shape[0]
        from .collections import TimeSeriesDict

        out_dict = TimeSeriesDict()

        # Build IMF TimeSeries objects
        for i in range(n_imfs):
            key = f"IMF{i + 1}"
            out_dict[key] = self.__class__(
                imfs_array[i],
                t0=self.t0,
                dt=self.dt,
                unit=self.unit,
                name=f"{self.name}_{key}" if self.name else key,
                channel=self.channel,
            )

        # Add residual if requested and available
        if return_residual:
            if residual is not None:
                key = "residual"
                out_dict[key] = self.__class__(
                    residual,
                    t0=self.t0,
                    dt=self.dt,
                    unit=self.unit,
                    name=f"{self.name}_{key}" if self.name else key,
                    channel=self.channel,
                )
            else:
                import warnings

                warnings.warn(
                    "Residual requested but not available from decomposer. "
                    "The 'residual' key will not be included in output.",
                    UserWarning,
                )

        return out_dict

    def hilbert_analysis(
        self,
        *,
        unwrap_phase: bool = True,
        frequency_unit: str = "Hz",
        if_smooth: int | u.Quantity | None = None,
        **hilbert_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Perform Hilbert analysis to extract instantaneous amplitude, phase, and frequency.

        This method computes the analytic signal via Hilbert transform and extracts
        the instantaneous amplitude (IA), phase, and frequency (IF).

        Parameters
        ----------
        unwrap_phase : bool, default=True
            If True, unwrap the phase to remove discontinuities.
        frequency_unit : str, default='Hz'
            Unit for the instantaneous frequency output.
        if_smooth : int, Quantity, or None, default=None
            Optional smoothing window for instantaneous frequency.
            If int, number of samples. If Quantity, time duration.
            Applies a simple moving average filter. **Odd window lengths
            are recommended**; even values are automatically incremented.
        **hilbert_kwargs
            Additional keyword arguments passed to :meth:`hilbert`.
            Common options include:
            - ``pad``: Samples or duration to pad (reduces edge effects)
            - ``pad_mode``: Padding mode ('reflect', 'constant', etc.)
            - ``nan_policy``: How to handle NaN values

        Returns
        -------
        dict
            Dictionary containing:
            - 'analytic': Complex analytic signal
            - 'amplitude': Instantaneous amplitude (envelope)
            - 'phase': Instantaneous phase (radians, optionally unwrapped)
            - 'frequency': Instantaneous frequency

        Raises
        ------
        ValueError
            If dt is not defined.

        Notes
        -----
        **Edge Effects**: Both the Hilbert transform and numerical differentiation
        introduce artifacts at signal boundaries. Use the ``pad`` parameter to
        mitigate, or crop the output edges in downstream analysis.

        **IF Calculation**: Instantaneous frequency is computed as::

            IF = (1 / 2π) * d(phase) / dt

        This can be noisy for complex signals. Use ``if_smooth`` to apply
        post-hoc smoothing.

        **Padding**: To reduce edge effects, pass ``pad=N`` where N is the
        number of samples (or a duration Quantity) to pad on each side
        before the Hilbert transform.

        Examples
        --------
        >>> ts = TimeSeries(np.sin(2 * np.pi * 10 * t), dt=0.001, unit='V')
        >>> result = ts.hilbert_analysis(pad=100, if_smooth=10)
        >>> amplitude = result['amplitude']
        >>> frequency = result['frequency']
        """
        analytic = self.hilbert(**hilbert_kwargs)
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
            channel=self.channel,
        )
        if self.dt is None:
            raise ValueError("dt is required for Instantaneous Frequency")
        dt_val = self.dt.to("s").value
        dphi = np.gradient(pha, dt_val)
        freq_val = dphi / (2 * np.pi)

        # Apply IF smoothing if requested
        if if_smooth is not None:
            if isinstance(if_smooth, str):
                if_smooth = u.Quantity(if_smooth)

            if isinstance(if_smooth, u.Quantity):
                w_s = if_smooth.to("s").value
                w_samples = int(round(w_s / dt_val))
            else:
                w_samples = int(if_smooth)

            # Enforce odd window length for symmetric filtering
            if w_samples > 0 and w_samples % 2 == 0:
                w_samples += 1

            if w_samples > 1:
                window = np.ones(w_samples) / w_samples
                freq_pad = np.pad(freq_val, w_samples // 2, mode="edge")
                freq_smooth = np.convolve(freq_pad, window, mode="valid")

                if len(freq_smooth) > len(freq_val):
                    freq_smooth = freq_smooth[: len(freq_val)]
                freq_val = freq_smooth

        frequency = self.__class__(
            freq_val,
            t0=self.t0,
            dt=self.dt,
            unit="Hz",
            name=f"{self.name}_IF" if self.name else "IF",
            channel=self.channel,
        )
        if frequency_unit != "Hz":
            frequency = frequency.to(frequency_unit)
        return {
            "analytic": analytic,
            "amplitude": amplitude,
            "phase": phase,
            "frequency": frequency,
        }

    def hht(
        self,
        *,
        emd_method: str = "eemd",
        emd_kwargs: dict[str, Any] | None = None,
        hilbert_kwargs: dict[str, Any] | None = None,
        output: str = "dict",
        # Spectrogram output controls (only used when output="spectrogram")
        n_bins: int = 100,
        freq_bins: Any = None,
        fmin: float | u.Quantity | None = None,
        fmax: float | u.Quantity | None = None,
        weight: str = "ia2",
        if_policy: str = "drop",
        finite_only: bool = True,
    ) -> Any:
        """
        Perform Hilbert-Huang Transform (HHT) on the TimeSeries.

        HHT combines Empirical Mode Decomposition (EMD) with Hilbert Spectral
        Analysis to create a time-frequency representation of non-stationary
        signals.

        Parameters
        ----------
        emd_method : str, default='eemd'
            EMD method to use ('emd' or 'eemd').
        emd_kwargs : dict or None, default=None
            Additional keyword arguments for :meth:`emd`.
        hilbert_kwargs : dict or None, default=None
            Additional keyword arguments for :meth:`hilbert_analysis`.
            Common options include ``pad``, ``if_smooth``.
        output : str, default='dict'
            Output format: 'dict' or 'spectrogram'.

        Spectrogram Options (only used when ``output='spectrogram'``)
        -------------------------------------------------------------
        n_bins : int, default=100
            Number of frequency bins (used if ``freq_bins`` is None).
        freq_bins : array-like or Quantity, optional
            Custom frequency bin edges. If provided, overrides ``n_bins``
            and ``fmin``/``fmax``.
        fmin : float or Quantity, optional
            Minimum frequency for binning (default: 0).
        fmax : float or Quantity, optional
            Maximum frequency for binning (default: Nyquist frequency).
        weight : {'ia2', 'ia'}, default='ia2'
            Weighting for the spectrogram:
            - 'ia2': Squared instantaneous amplitude (power-like)
            - 'ia': Instantaneous amplitude (magnitude)
        if_policy : {'drop', 'clip'}, default='drop'
            Policy for IF values outside frequency bins:
            - 'drop': Ignore out-of-range values
            - 'clip': Clip to nearest bin edge
        finite_only : bool, default=True
            If True, exclude NaN/Inf values in IF/IA during spectrogram
            binning. **Note**: This does not allow NaN in the original
            signal passed to EMD; it only affects Hilbert analysis output.

        Returns
        -------
        dict or Spectrogram
            If ``output='dict'``:
                Dictionary with keys 'imfs', 'ia', 'if', 'residual'.
            If ``output='spectrogram'``:
                GWpy Spectrogram representing the Hilbert spectrum.

        Raises
        ------
        ImportError
            If PyEMD is not installed.
        ValueError
            If EMD returns no IMFs or unknown output format specified.

        Notes
        -----
        **Optional Dependency**: Requires PyEMD for EMD decomposition.

        **What is HHT?**: Unlike STFT or wavelet transforms, HHT provides
        instantaneous frequency estimates that can capture rapid frequency
        variations. The Hilbert spectrum (spectrogram output) is a binned
        representation of these IF curves, not a power spectral density.

        **Default Weighting**: The default ``weight='ia2'`` produces a
        power-like representation where energy is proportional to amplitude
        squared. Use ``weight='ia'`` for magnitude representation.

        **Edge Artifacts**: Both EMD envelope extrapolation and Hilbert
        transform can produce artifacts at boundaries. Consider:
        1. Pre-padding the signal
        2. Using ``hilbert_kwargs={'pad': N}``
        3. Cropping edges from the result

        Examples
        --------
        >>> ts = TimeSeries(data, dt=0.01, unit='V')
        >>> # Dictionary output
        >>> result = ts.hht(emd_method='eemd', output='dict')
        >>> imfs = result['imfs']
        >>> inst_freq = result['if']

        >>> # Spectrogram output with custom settings
        >>> spec = ts.hht(
        ...     output='spectrogram',
        ...     n_bins=200,
        ...     fmin=10,
        ...     fmax=100,
        ...     weight='ia',
        ...     hilbert_kwargs={'pad': 100, 'if_smooth': 10}
        ... )
        """
        if emd_kwargs is None:
            emd_kwargs = {}
        if hilbert_kwargs is None:
            hilbert_kwargs = {}

        # Validate if_policy
        if if_policy not in ("drop", "clip"):
            raise ValueError(f"Unknown if_policy: {if_policy!r}. Use 'drop' or 'clip'.")

        imfs = self.emd(method=emd_method, **emd_kwargs)
        from .collections import TimeSeriesDict

        ia_dict = TimeSeriesDict()
        if_dict = TimeSeriesDict()
        residual = None
        if "residual" in imfs:
            residual = imfs.pop("residual")

        # Prepare keywords for Hilbert analysis
        hk = hilbert_kwargs.copy()
        if finite_only and "nan_policy" not in hk:
            hk["nan_policy"] = "propagate"

        for key, imf in imfs.items():
            res = imf.hilbert_analysis(**hk)
            ia_dict[key] = res["amplitude"]
            if_dict[key] = res["frequency"]

        if output == "dict":
            return {"imfs": imfs, "ia": ia_dict, "if": if_dict, "residual": residual}

        elif output == "spectrogram":
            # Validate that we have IMFs
            keys = list(imfs.keys())
            if not keys:
                raise ValueError(
                    "EMD decomposition returned no IMFs. Cannot create spectrogram. "
                    "Check input signal quality or EMD parameters."
                )

            fs_rate = self.sample_rate.to("Hz").value
            nyquist = fs_rate / 2.0
            n_time = len(self)

            # Build frequency bin edges
            if freq_bins is not None:
                # Use custom frequency bins
                if isinstance(freq_bins, u.Quantity):
                    freq_bins_arr = freq_bins.to("Hz").value
                else:
                    freq_bins_arr = np.asarray(freq_bins)

                # Validate monotonicity
                if not np.all(np.diff(freq_bins_arr) > 0):
                    raise ValueError(
                        "freq_bins must be strictly monotonically increasing."
                    )
                actual_n_bins = len(freq_bins_arr) - 1
            else:
                # Determine fmin/fmax
                if fmin is None:
                    fmin_val = 0.0
                elif isinstance(fmin, u.Quantity):
                    fmin_val = fmin.to("Hz").value
                else:
                    fmin_val = float(fmin)

                if fmax is None:
                    fmax_val = nyquist
                elif isinstance(fmax, u.Quantity):
                    fmax_val = fmax.to("Hz").value
                else:
                    fmax_val = float(fmax)

                freq_bins_arr = np.linspace(fmin_val, fmax_val, n_bins + 1)
                actual_n_bins = n_bins

            # Stack IF and IA from all IMFs
            # Convert IF to Hz if needed (in case hilbert_kwargs changed unit)
            if_stack_list = []
            ia_stack_list = []
            for k in keys:
                if_ts = if_dict[k]
                ia_ts = ia_dict[k]

                # Ensure IF is in Hz for binning
                if hasattr(if_ts, "unit") and if_ts.unit != u.Hz:
                    if_val = if_ts.to("Hz").value
                else:
                    if_val = if_ts.value

                if_stack_list.append(if_val)
                ia_stack_list.append(ia_ts.value)

            if_stack = np.stack(if_stack_list)
            ia_stack = np.stack(ia_stack_list)

            # Handle non-finite values if requested
            if finite_only:
                finite_mask = np.isfinite(if_stack) & np.isfinite(ia_stack)
            else:
                finite_mask = np.ones_like(if_stack, dtype=bool)

            # Digitize IF values into frequency bins
            if if_policy == "clip":
                # Clip IF to valid range before digitizing
                if_clipped = np.clip(if_stack, freq_bins_arr[0], freq_bins_arr[-1])
                inds = np.digitize(if_clipped, freq_bins_arr) - 1
                # digitize returns n_bins for values == freq_bins_arr[-1]
                inds = np.clip(inds, 0, actual_n_bins - 1)
                mask = finite_mask  # All valid after clipping
            else:  # policy == "drop"
                inds = np.digitize(if_stack, freq_bins_arr) - 1
                mask = finite_mask & (inds >= 0) & (inds < actual_n_bins)

            # Compute weights
            if weight == "ia2":
                weights_stack = ia_stack**2
                out_unit = self.unit**2
                weight_label = "power"
            elif weight == "ia":
                weights_stack = ia_stack
                out_unit = self.unit
                weight_label = "amplitude"
            else:
                raise ValueError(f"Unknown weight: {weight}. Use 'ia2' or 'ia'.")

            # Build the spectrogram grid
            grid = np.zeros((n_time, actual_n_bins))
            for k in range(len(keys)):
                valid = mask[k]
                t_inds = np.arange(n_time)[valid]
                f_inds = inds[k][valid]
                energies = weights_stack[k][valid]
                np.add.at(grid, (t_inds, f_inds), energies)

            from gwexpy.types.hht_spectrogram import HHTSpectrogram

            freq_centers = (freq_bins_arr[:-1] + freq_bins_arr[1:]) / 2.0
            name_suffix = "Hilbert Spectrum"
            if weight_label == "amplitude":
                name_suffix += " (IA)"

            return HHTSpectrogram(
                grid,
                times=self.times,
                frequencies=u.Quantity(freq_centers, "Hz"),
                unit=out_unit,
                name=(self.name + " " + name_suffix) if self.name else name_suffix,
                channel=self.channel,
                epoch=self.epoch,
            )
        else:
            raise ValueError(f"Unknown output format: {output}")
