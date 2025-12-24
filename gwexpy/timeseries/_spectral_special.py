"""
Special spectral transform methods for TimeSeries (HHT, EMD, Laplace, CWT).
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from typing import Optional, Any, TYPE_CHECKING
from gwexpy.interop._optional import require_optional

if TYPE_CHECKING:
    from .timeseries import TimeSeries

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
        analytic = self.analytic_signal()
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
