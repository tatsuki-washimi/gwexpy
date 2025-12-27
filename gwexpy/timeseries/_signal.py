"""
Signal processing methods for TimeSeries.

This module provides signal processing functionality as a mixin class:
- Hilbert transform: analytic_signal, hilbert, envelope
- Phase/frequency: instantaneous_phase, unwrap_phase, instantaneous_frequency
- Demodulation: _build_phase_series, mix_down, baseband, lock_in
- Cross-correlation: xcorr, transfer_function
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from typing import Optional, Any, TYPE_CHECKING

try:
    import scipy.signal
except ImportError:
    scipy = None

if TYPE_CHECKING:
    pass


def _extract_axis_info(ts):
    """Extract axis information from a TimeSeries."""
    try:
        dt = ts.dt
        regular = True
    except (AttributeError, ValueError):
        dt = None
        regular = False
    return {'dt': dt, 'regular': regular}


class TimeSeriesSignalMixin:
    """
    Mixin class providing signal processing methods for TimeSeries.

    This mixin is designed to be combined with TimeSeriesCore to create
    the full TimeSeries class.
    """

    # ===============================
    # Hilbert Transform Methods
    # ===============================

    def analytic_signal(
        self,
        pad: Any = None,
        pad_mode: str = "reflect",
        pad_value: float = 0.0,
        nan_policy: str = "raise",
        copy: bool = True,
    ) -> "TimeSeriesSignalMixin":
        """
        Compute the analytic signal (Hilbert transform) of the TimeSeries.

        If input is real, returns complex analytic signal z(t) = x(t) + i H[x(t)].
        If input is complex, returns a copy (casting to complex if needed).
        """
        # 1. Complex check
        if np.iscomplexobj(self.value):
            return self.astype(complex, copy=copy)

        # 2. Check regular
        info = _extract_axis_info(self)
        if not info['regular']:
             raise ValueError("analytic_signal requires regular sampling (dt). Use asfreq/resample first.")

        # 3. Handle NaN
        if nan_policy == 'raise':
            if not np.isfinite(self.value).all():
                raise ValueError("Input contains NaNs or infinite values.")

        # 4. Padding
        data = self.value
        n_pad = 0
        if pad is not None:
             if isinstance(pad, u.Quantity):
                  n_pad = int(round(pad.to(self.times.unit).value / self.dt.to(self.times.unit).value))
             else:
                  n_pad = int(pad)

             if n_pad > 0:
                  if pad_mode == 'constant':
                       kwargs = {'constant_values': pad_value}
                  else:
                       kwargs = {}
                  data = np.pad(data, n_pad, mode=pad_mode, **kwargs)

        # 5. Hilbert
        analytic = scipy.signal.hilbert(data)

        # 6. Crop if padded
        if n_pad > 0:
             analytic = analytic[n_pad:-n_pad]

        # 7. Wrap
        return self.__class__(
             analytic,
             t0=self.t0,
             dt=self.dt,
             unit=self.unit,
             channel=self.channel,
             name=self.name
        )

    def hilbert(self, *args: Any, **kwargs: Any) -> "TimeSeriesSignalMixin":
        """Alias for analytic_signal."""
        return self.analytic_signal(*args, **kwargs)

    def envelope(self, *args: Any, **kwargs: Any) -> "TimeSeriesSignalMixin":
        """Compute the envelope of the TimeSeries."""
        analytic = self.analytic_signal(*args, **kwargs)
        return abs(analytic)

    # ===============================
    # Phase and Frequency Methods
    # ===============================

    def instantaneous_phase(self, deg: bool = False, unwrap: bool = False, **kwargs: Any) -> "TimeSeriesSignalMixin":
        """
        Compute the instantaneous phase of the TimeSeries.
        """
        analytic = self.analytic_signal(**kwargs)
        phi = np.angle(analytic.value, deg=deg)

        if unwrap:
             period = 360.0 if deg else 2 * np.pi
             phi = np.unwrap(phi, period=period)

        out = self.__class__(
             phi,
             t0=self.t0,
             dt=self.dt,
             channel=self.channel,
             name=self.name
        )

        # Override unit
        out.override_unit('deg' if deg else 'rad')
        return out

    def unwrap_phase(self, deg: bool = False, **kwargs: Any) -> "TimeSeriesSignalMixin":
        """Alias for instantaneous_phase(unwrap=True)."""
        return self.instantaneous_phase(deg=deg, unwrap=True, **kwargs)

    def instantaneous_frequency(self, unwrap: bool = True, smooth: Any = None, **kwargs: Any) -> "TimeSeriesSignalMixin":
        """
        Compute the instantaneous frequency of the TimeSeries.
        Returns unit 'Hz'.
        """
        # Force radians for calculation
        phi_ts = self.instantaneous_phase(deg=False, unwrap=unwrap, **kwargs)
        phi = phi_ts.value

        dt_s = self.dt.to('s').value

        # gradient
        dphi_dt = np.gradient(phi, dt_s)

        # f = (dphi/dt) / 2pi
        f_inst = dphi_dt / (2 * np.pi)

        # Smoothing
        if smooth is not None:
             if isinstance(smooth, str):
                  smooth = u.Quantity(smooth)

             if isinstance(smooth, u.Quantity):
                  w_s = smooth.to('s').value
                  w_samples = int(round(w_s / dt_s))
             else:
                  w_samples = int(smooth)

             if w_samples > 1:
                  window = np.ones(w_samples) / w_samples
                  f_pad = np.pad(f_inst, w_samples//2, mode='edge')
                  f_smooth = np.convolve(f_pad, window, mode='valid')

                  if len(f_smooth) > len(f_inst):
                       f_smooth = f_smooth[:len(f_inst)]
                  f_inst = f_smooth

        out = self.__class__(
             f_inst,
             t0=self.t0,
             dt=self.dt,
             channel=self.channel,
             name=self.name
        )
        out.override_unit('Hz')
        return out

    # ===============================
    # Demodulation Methods
    # ===============================

    def _build_phase_series(
        self,
        *,
        phase: Any = None,
        f0: Any = None,
        fdot: Any = 0.0,
        fddot: Any = 0.0,
        phase_epoch: Any = None,
        phase0: float = 0.0,
        prefer_dt: bool = True,
    ) -> np.ndarray:
        """Internal helper to build phase series in radians."""
        if (f0 is None and phase is None) or (f0 is not None and phase is not None):
             raise ValueError("Exactly one of 'f0' or 'phase' must be provided.")

        if phase is not None:
             if len(phase) != self.size:
                  raise ValueError(f"Length of phase ({len(phase)}) does not match TimeSeries ({self.size})")
             return np.asarray(phase, dtype=float) + phase0

        # Build from model
        if isinstance(f0, u.Quantity):
             f0 = f0.to('Hz').value
        else:
             f0 = float(f0)

        if isinstance(fdot, u.Quantity):
             fdot = fdot.to('Hz/s').value
        else:
             fdot = float(fdot)

        if isinstance(fddot, u.Quantity):
             fddot = float(fddot.to('Hz/s^2').value)
        else:
             fddot = float(fddot)

        # Determine t_rel
        has_dt = False
        dt_val = None
        try:
             if self.dt is not None:
                  dt_val = self.dt.to('s').value
                  has_dt = True
        except (AttributeError, ValueError):
             pass

        if has_dt and prefer_dt and dt_val is not None:
             if phase_epoch is None:
                  t_rel = dt_val * np.arange(self.size)
             else:
                  t0_abs = float(self.times.value[0])
                  t0_rel = t0_abs - float(phase_epoch)
                  t_rel = t0_rel + dt_val * np.arange(self.size)
        else:
             if self.times is None:
                  raise ValueError("TimeSeries requires times or dt to build phase model.")
             times_val = self.times.value if hasattr(self.times, 'value') else self.times
             times_val = np.asarray(times_val, dtype=float)

             if phase_epoch is None:
                  t_rel = times_val - times_val[0]
             else:
                  t_rel = times_val - float(phase_epoch)

        # Calculate phase
        cycles = f0 * t_rel
        if fdot != 0.0:
             cycles += 0.5 * fdot * t_rel**2
        if fddot != 0.0:
             cycles += (1.0/6.0) * fddot * t_rel**3

        return 2 * np.pi * cycles + phase0

    def mix_down(
        self,
        *,
        phase: Any = None,
        f0: Any = None,
        fdot: Any = 0.0,
        fddot: Any = 0.0,
        phase_epoch: Any = None,
        phase0: float = 0.0,
        singlesided: bool = False,
        copy: bool = True,
    ) -> "TimeSeriesSignalMixin":
        """
        Mix the TimeSeries with a complex oscillator.
        """
        phase_series = self._build_phase_series(
            phase=phase,
            f0=f0,
            fdot=fdot,
            fddot=fddot,
            phase_epoch=phase_epoch,
            phase0=phase0,
            prefer_dt=True
        )

        # Mix
        y = self.value * np.exp(-1j * phase_series)

        if singlesided:
             y *= 2.0

        # Prepare constructor args
        kwargs = {
             'unit': self.unit,
             'channel': self.channel,
             'name': self.name
        }

        # Check regularity for constructor
        try:
             if self.dt is not None:
                  kwargs['t0'] = self.t0
                  kwargs['dt'] = self.dt
             else:
                  kwargs['t0'] = self.t0
                  kwargs['sample_rate'] = self.sample_rate
        except (AttributeError, ValueError):
             kwargs['times'] = self.times

        out = self.__class__(y, **kwargs)
        return out

    def heterodyne(self, phase: Any, stride: Any = 1.0, singlesided: bool = True) -> "TimeSeriesSignalMixin":
        """
        Mix with phase and average over strides.

        Parameters
        ----------
        phase : `array_like` or `Series`
            Phase to mix with (radians).
        stride : `float` or `Quantity`, optional
            Time step for averaging (default 1.0s).
        singlesided : `bool`, optional
            If True, double the amplitude (useful for real signals).

        Returns
        -------
        TimeSeries
            Average (complex) demodulated signal.
        """
        # 1. Mix down
        z = self.mix_down(phase=phase, singlesided=singlesided)
        
        # 2. Resample (average) to stride-based rate
        if isinstance(stride, (float, int)):
             stride_dt = stride * u.s
        else:
             stride_dt = u.Quantity(stride)
        
        # Use our bin-based resample to get the average
        return z.resample(stride_dt, agg='mean')

    def baseband(
        self,
        *,
        phase: Any = None,
        f0: Any = None,
        fdot: Any = 0.0,
        fddot: Any = 0.0,
        phase_epoch: Any = None,
        phase0: float = 0.0,
        lowpass: Optional[float] = None,
        lowpass_kwargs: Optional[dict[str, Any]] = None,
        output_rate: Optional[float] = None,
        singlesided: bool = False,
    ) -> "TimeSeriesSignalMixin":
        """
        Demodulate the TimeSeries to baseband, optionally applying lowpass filter and resampling.
        """
        z = self.mix_down(
            phase=phase,
            f0=f0,
            fdot=fdot,
            fddot=fddot,
            phase_epoch=phase_epoch,
            phase0=phase0,
            singlesided=singlesided
        )

        if lowpass is not None:
             if z.sample_rate is None:
                  raise ValueError("lowpass requires defined sample rate.")
             lp_kwargs = lowpass_kwargs or {}
             z = z.lowpass(lowpass, **lp_kwargs)

        if output_rate is not None:
             z = z.resample(output_rate)

        return z

    def lock_in(
        self,
        *,
        phase: Any = None,
        f0: Any = None,
        fdot: Any = 0.0,
        fddot: Any = 0.0,
        phase_epoch: Any = None,
        phase0: float = 0.0,
        stride: float = 1.0,
        singlesided: bool = True,
        output: str = "amp_phase",
        deg: bool = True,
    ) -> Any:
        """
        Perform lock-in amplification (demodulation + averaging).
        """
        self._check_regular("lock_in")

        phase_series = self._build_phase_series(
            phase=phase,
            f0=f0,
            fdot=fdot,
            fddot=fddot,
            phase_epoch=phase_epoch,
            phase0=phase0,
            prefer_dt=True
        )

        outc = self.heterodyne(phase_series, stride=stride, singlesided=singlesided)

        if output == 'complex':
             return outc
        elif output == 'amp_phase':
             mag = outc.abs()
             ph = self.__class__(
                  np.angle(outc.value, deg=deg),
                  t0=outc.t0,
                  dt=outc.dt,
                  channel=self.channel,
                  name=self.name
             )
             ph.override_unit('deg' if deg else 'rad')
             return mag, ph
        elif output == 'iq':
             i = self.__class__(outc.value.real, t0=outc.t0, dt=outc.dt, channel=self.channel, name=self.name, unit=self.unit)
             q = self.__class__(outc.value.imag, t0=outc.t0, dt=outc.dt, channel=self.channel, name=self.name, unit=self.unit)
             return i, q
        else:
             raise ValueError(f"Unknown output format: {output}")

    # ===============================
    # Cross-correlation Methods
    # ===============================

    def transfer_function(
        self,
        other: Any,
        fftlength: Optional[float] = None,
        overlap: Optional[float] = None,
        window: Any = "hann",
        average: str = "mean",
        *,
        method: str = "gwpy",
        fft_kwargs: Optional[dict[str, Any]] = None,
        downsample: Optional[float] = None,
        align: str = "intersection",
        **kwargs: Any,
    ) -> Any:
        """
        Compute the transfer function between this TimeSeries and another.

        Parameters
        ----------
        other : `TimeSeries`
            The input TimeSeries.
        fftlength : `int`, optional
            Length of the FFT, in seconds (default) or samples.
        overlap : `int`, optional
            Overlap between segments, in seconds (default) or samples.
        window : `str`, `numpy.ndarray`, optional
            Window function to apply.
        average : `str`, optional
            Method to average viewing periods.
        method : `str`, optional
            "gwpy" or "csd_psd": Use GWpy CSD/PSD estimator.
            "fft": Use direct FFT ratio (other.fft() / self.fft()).
            "auto": Use "fft" if fftlength is None, else "gwpy".

        Returns
        -------
        out : `FrequencySeries`
            Transfer function.
        """
        import warnings

        use_fft = False
        if method in ("gwpy", "csd_psd"):
            use_fft = False
        elif method == "fft":
            use_fft = True
        elif method == "auto":
            use_fft = fftlength is None
        else:
            raise ValueError(f"Unknown method: {method}")

        if not use_fft:
            # CSD / PSD estimator
            csd = self.csd(
                other,
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                average=average,
                **kwargs,
            )
            psd = self.psd(
                fftlength=fftlength,
                overlap=overlap,
                window=window,
                average=average,
                **kwargs,
            )
            size = min(csd.size, psd.size)
            return csd[:size] / psd[:size]

        else:
            # FFT Ratio
            kw = dict(fft_kwargs) if fft_kwargs is not None else {}

            a = self
            b = other

            # 1. Sample Rate
            if a.sample_rate != b.sample_rate:
                if downsample is False:
                    raise ValueError("Sample rates differ and downsample=False")
                if downsample is None:
                    warnings.warn(
                        "Sample rates differ, downsampling to match.", UserWarning
                    )

                rate_a = a.sample_rate.value
                rate_b = b.sample_rate.value

                if rate_a > rate_b:
                    a = a.resample(b.sample_rate)
                elif rate_b > rate_a:
                    b = b.resample(a.sample_rate)

            # 2. Align
            if align == "intersection":
                start = max(a.span[0], b.span[0])
                end = min(a.span[1], b.span[1])

                if end <= start:
                    raise ValueError("No comparison overlap between TimeSeries")

                a = a.crop(start, end)
                b = b.crop(start, end)

            elif align == "none":
                pass
            else:
                raise ValueError("align must be 'intersection' or 'none'")

            # 3. Ensure equal length
            size = min(a.size, b.size)
            if a.size != size:
                a = a[:size]
            if b.size != size:
                b = b[:size]

            # 4. FFTs
            fx = a.fft(**kw)
            fy = b.fft(**kw)

            fsize = min(fx.size, fy.size)

            tf = fy[:fsize] / fx[:fsize]

            if b.name and a.name:
                tf.name = f"{b.name} / {a.name}"

            return tf

    def xcorr(
        self,
        other: Any,
        *,
        maxlag: Optional[float] = None,
        normalize: Optional[str] = None,
        mode: str = "full",
        demean: bool = True,
    ) -> "TimeSeriesSignalMixin":
        """
        Compute time-domain cross-correlation between two TimeSeries.
        """
        from scipy import signal

        dt_self = self.dt if isinstance(self.dt, u.Quantity) else u.Quantity(self.dt, "s")
        dt_other = other.dt if isinstance(other.dt, u.Quantity) else u.Quantity(other.dt, "s")
        if dt_self != dt_other:
            raise ValueError("TimeSeries must share the same dt/sample_rate for xcorr")
        dt = dt_self

        x = self.value.astype(float)
        y = other.value.astype(float)
        if demean:
            x = x - np.nanmean(x)
            y = y - np.nanmean(y)

        corr = signal.correlate(x, y, mode=mode, method="auto")
        lags = signal.correlation_lags(len(x), len(y), mode=mode)

        # Normalization
        if normalize is None:
            pass
        elif normalize == "biased":
            corr = corr / len(x)
        elif normalize == "unbiased":
            corr = corr / (len(x) - np.abs(lags))
        elif normalize == "coeff":
            denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
            if denom != 0:
                corr = corr / denom
        else:
            raise ValueError("normalize must be None|'biased'|'unbiased'|'coeff'")

        # Max lag trimming
        if maxlag is not None:
            if isinstance(maxlag, u.Quantity):
                lag_samples = int(np.floor(np.abs(maxlag.to(dt.unit).value / dt.value)))
            else:
                lag_samples = int(maxlag)
            mask = np.abs(lags) <= lag_samples
            corr = corr[mask]
            lags = lags[mask]

        lag_times = lags * dt
        name = f"xcorr({self.name},{getattr(other, 'name', '')})"
        return self.__class__(
            corr,
            times=lag_times,
            unit=self.unit * getattr(other, "unit", 1),
            name=name,
        )
