"""
Signal processing methods for TimeSeries.

This module provides signal processing functionality as a mixin class:
- Hilbert transform: hilbert, envelope
- Phase/frequency: instantaneous_phase, unwrap_phase, instantaneous_frequency
- Demodulation: _build_phase_series, mix_down, baseband, lock_in
- Cross-correlation: xcorr, transfer_function
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import numpy as np
from astropy import units as u
from numpy.typing import ArrayLike, NDArray

from ._typing import TimeSeriesAttrs

NumberLike: TypeAlias = int | float | np.number
QuantityLike: TypeAlias = ArrayLike | u.Quantity

try:
    import scipy.signal  # noqa: F401 - availability check
except ImportError:
    scipy = None

if TYPE_CHECKING:
    from gwexpy.timeseries.timeseries import TimeSeries


def _extract_axis_info(ts: Any) -> dict[str, Any]:
    """Extract axis information from a TimeSeries."""
    try:
        dt = ts.dt
        regular = True
    except (AttributeError, ValueError):
        dt = None
        regular = False
    return {"dt": dt, "regular": regular}


class TimeSeriesSignalMixin(TimeSeriesAttrs):
    """
    Mixin class providing signal processing methods for TimeSeries.

    This mixin is designed to be combined with TimeSeriesCore to create
    the full TimeSeries class.
    """

    # ===============================
    # Hilbert Transform Methods
    # ===============================

    def hilbert(
        self,
        pad: NumberLike | u.Quantity = 0,
        pad_mode: str = "reflect",
        pad_value: float = 0.0,
        nan_policy: Literal["raise", "propagate"] = "raise",
        copy: bool = True,
    ) -> TimeSeriesSignalMixin:
        """
        Compute the analytic signal via Hilbert transform.

        For a real input x(t), returns the complex analytic signal::

            z(t) = x(t) + i * H[x(t)]

        where H[x] is the Hilbert transform of x, computed via SciPy.

        If input is already complex, returns a copy.

        Parameters
        ----------
        pad : int or Quantity, default=0
            Number of samples (or time duration) to pad on each side before
            applying the Hilbert transform. Padding can help reduce endpoint
            artifacts. Default is 0 (no padding).
        pad_mode : str, default='reflect'
            Padding mode ('reflect', 'constant', 'edge', etc.).
        pad_value : float, default=0.0
            Value for 'constant' padding mode.
        nan_policy : {'raise', 'propagate'}, default='raise'
            How to handle NaNs/Infs: 'raise' raises ValueError (default),
            'propagate' allows NaNs to propagate through.
        copy : bool, default=True
            If input is complex, whether to return a copy.

        Returns
        -------
        TimeSeries
            Complex analytic signal with the same length as input.

        Raises
        ------
        ValueError
            If input contains NaN or infinite values and nan_policy='raise'.
        ValueError
            If the TimeSeries has irregular sampling.

        Notes
        -----
        **Preprocessing**: This method does NOT apply any automatic preprocessing.
        Users should apply demean, detrend, filtering, or windowing as needed
        before calling this method.

        **Endpoint artifacts**: The Hilbert transform can exhibit artifacts at
        the edges due to spectral leakage. Use padding or window the data
        appropriately if edge effects are a concern.

        **Mathematical definition**: The Hilbert transform H[x] is defined as
        the convolution of x(t) with 1/(πt). The analytic signal z(t) has the
        property that its Fourier transform is zero for negative frequencies.

        Examples
        --------
        >>> ts = TimeSeries(np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000)),
        ...                 dt=0.001, unit='V')
        >>> analytic = ts.hilbert()
        >>> envelope = np.abs(analytic.value)
        """
        # 1. Complex check
        if np.iscomplexobj(self.value):
            return self.astype(complex, copy=copy)

        # 2. Check regular
        info = _extract_axis_info(self)
        if not info["regular"]:
            raise ValueError(
                "hilbert requires regular sampling (dt). Use asfreq/resample first."
            )

        # 3. Handle NaN
        if nan_policy == "raise":
            if not np.isfinite(self.value).all():
                raise ValueError("Input contains NaNs or infinite values.")

        # 4. Padding
        data = self.value
        n_pad = 0
        if isinstance(pad, u.Quantity):
            if self.dt is None or getattr(self.times, "unit", None) is None:
                raise ValueError(
                    "hilbert requires defined dt and times when padding."
                )
            n_pad = int(
                round(
                    pad.to(self.times.unit).value
                    / self.dt.to(self.times.unit).value
                )
            )
        else:
            n_pad = int(pad)

        if n_pad > 0:
            if pad_mode == "constant":
                pad_kwargs: dict[str, Any] = {"constant_values": pad_value}
            else:
                pad_kwargs = {}
            data = np.pad(data, n_pad, mode=pad_mode, **pad_kwargs)  # type: ignore[call-overload]

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
            name=self.name,
        )

    def envelope(self, *args: Any, **kwargs: Any) -> TimeSeriesSignalMixin:
        """Compute the envelope (amplitude) of the TimeSeries via Hilbert transform."""
        analytic = self.hilbert(*args, **kwargs)
        return analytic.abs()

    # ===============================
    # Phase and Frequency Methods
    # ===============================

    def instantaneous_phase(
        self, deg: bool = False, unwrap: bool = False, **kwargs: Any
    ) -> TimeSeriesSignalMixin:
        """
        Compute the instantaneous phase of the TimeSeries via Hilbert transform.

        This method first computes the analytic signal using Hilbert transform,
        then extracts the phase using np.angle(). Use this for real-valued
        signals when you need the instantaneous phase.

        For complex-valued signals, consider using :meth:`radian` or :meth:`degree`
        which directly compute np.angle() without Hilbert transform.

        Parameters
        ----------
        deg : bool, default=False
            If True, return phase in degrees. Default is False (radians).
        unwrap : bool, default=False
            If True, unwrap the phase to remove discontinuities.
            Uses period=2π for radians, period=360 for degrees.
        **kwargs
            Passed to :meth:`hilbert`. Common options include `pad` for
            reducing endpoint artifacts.

        Returns
        -------
        TimeSeries
            Instantaneous phase with unit 'rad' or 'deg'.
            The output has the same length as the input (endpoints are not trimmed).

        Notes
        -----
        **Definition**: The instantaneous phase is computed as::

            analytic = hilbert(x)
            phase = np.angle(analytic)  # in radians
            if unwrap:
                phase = np.unwrap(phase, period=2*np.pi)  # or 360 for degrees

        **Preprocessing**: No automatic preprocessing is applied. Users should
        apply demean, detrend, filtering, or windowing as needed before calling.

        **Endpoint artifacts**: The underlying Hilbert transform may exhibit
        artifacts at the edges. Consider padding or windowing if edge effects
        are a concern.

        Examples
        --------
        >>> ts = TimeSeries(np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000)),
        ...                 dt=0.001, unit='V')
        >>> phase = ts.instantaneous_phase(unwrap=True)  # rad, unwrapped
        >>> phase_deg = ts.instantaneous_phase(deg=True, unwrap=True)  # degrees
        """
        analytic = self.hilbert(**kwargs)
        phi = np.asarray(np.angle(analytic.value, deg=deg))

        if unwrap:
            period = 360.0 if deg else 2 * np.pi
            phi = np.unwrap(phi, period=period)

        out = self.__class__(
            phi, t0=self.t0, dt=self.dt, channel=self.channel, name=self.name
        )

        # Override unit
        out.override_unit("deg" if deg else "rad")
        return out

    def radian(self, unwrap: bool = False) -> TimeSeriesSignalMixin:
        """
        Calculate the phase angle of the TimeSeries in radians.

        Computes np.angle(self.value) directly. Works for both real and complex
        time series. For real signals, this will return 0 or π depending on sign.

        For instantaneous phase of a real signal via Hilbert transform,
        use :meth:`instantaneous_phase` instead.

        Parameters
        ----------
        unwrap : bool, optional
            If True, unwrap the phase. Default is False.

        Returns
        -------
        TimeSeries
            Phase angle in radians.
        """
        phi = np.asarray(np.angle(self.value))

        if unwrap:
            phi = np.unwrap(phi)

        out = self.__class__(
            phi, t0=self.t0, dt=self.dt, channel=self.channel, name=self.name
        )
        out.override_unit("rad")
        return out

    def degree(self, unwrap: bool = False) -> TimeSeriesSignalMixin:
        """
        Calculate the phase angle of the TimeSeries in degrees.

        Computes np.angle(self.value) directly. Works for both real and complex
        time series. For real signals, this will return 0 or 180 depending on sign.

        For instantaneous phase of a real signal via Hilbert transform,
        use :meth:`instantaneous_phase` instead.

        Parameters
        ----------
        unwrap : bool, optional
            If True, unwrap the phase. Default is False.

        Returns
        -------
        TimeSeries
            Phase angle in degrees.
        """
        phi = np.asarray(np.angle(self.value, deg=True))

        if unwrap:
            phi = np.unwrap(phi, period=360.0)

        out = self.__class__(
            phi, t0=self.t0, dt=self.dt, channel=self.channel, name=self.name
        )
        out.override_unit("deg")
        return out

    def unwrap_phase(self, deg: bool = False, **kwargs: Any) -> TimeSeriesSignalMixin:
        """Alias for instantaneous_phase(unwrap=True)."""
        return self.instantaneous_phase(deg=deg, unwrap=True, **kwargs)

    def instantaneous_frequency(
        self,
        unwrap: bool = True,
        smooth: NumberLike | u.Quantity | None = None,
        **kwargs: Any,
    ) -> TimeSeriesSignalMixin:
        """
        Compute the instantaneous frequency of the TimeSeries.

        The instantaneous frequency is derived from the time derivative of
        the unwrapped instantaneous phase obtained via Hilbert transform.

        Parameters
        ----------
        unwrap : bool, default=True
            If True, unwrap the phase before differentiation (recommended).
            Setting to False may cause issues at phase wrap points.
        smooth : int, Quantity, or None, default=None
            Optional smoothing window. If int, number of samples. If Quantity,
            time duration. Default is None (no smoothing).
        **kwargs
            Passed to :meth:`hilbert` via :meth:`instantaneous_phase`.
            Common options include `pad` for reducing endpoint artifacts.

        Returns
        -------
        TimeSeries
            Instantaneous frequency with unit 'Hz'.
            The output has the same length as the input (endpoints are not trimmed).

        Raises
        ------
        ValueError
            If the TimeSeries has no defined dt (sample rate).

        Notes
        -----
        **Definition**: The instantaneous frequency is computed as::

            phase = instantaneous_phase(unwrap=True, deg=False)  # radians
            dphi_dt = np.gradient(phase, dt)  # time derivative
            f_inst = dphi_dt / (2 * π)  # convert to Hz

        **Preprocessing**: No automatic preprocessing is applied. Users should
        apply demean, detrend, filtering, or windowing as needed before calling.

        **Endpoint artifacts**: The underlying Hilbert transform may exhibit
        artifacts at the edges. The endpoints of the instantaneous frequency
        may also be less accurate due to numerical differentiation.
        When evaluating accuracy, consider using only the central portion
        of the output.

        **Smoothing**: The `smooth` parameter provides optional post-hoc
        smoothing via moving average. By default, no smoothing is applied.

        Examples
        --------
        >>> t = np.linspace(0, 1, 1000)
        >>> ts = TimeSeries(np.cos(2 * np.pi * 50 * t), dt=0.001, unit='V')
        >>> f_inst = ts.instantaneous_frequency()
        >>> # Central region should be close to 50 Hz
        >>> np.median(f_inst.value[100:-100])  # doctest: +SKIP
        50.0
        """
        # Force radians for calculation
        phi_ts = self.instantaneous_phase(deg=False, unwrap=unwrap, **kwargs)
        phi = phi_ts.value

        if self.dt is None:
            raise ValueError("instantaneous_frequency requires defined dt.")
        dt_s = self.dt.to("s").value

        # gradient
        dphi_dt = np.gradient(phi, dt_s)

        # f = (dphi/dt) / 2pi
        f_inst = dphi_dt / (2 * np.pi)

        # Smoothing
        if smooth is not None:
            if isinstance(smooth, str):
                smooth = u.Quantity(smooth)

            if isinstance(smooth, u.Quantity):
                w_s = smooth.to("s").value
                w_samples = int(round(w_s / dt_s))
            else:
                w_samples = int(smooth)

            if w_samples > 1:
                window = np.ones(w_samples) / w_samples
                f_pad = np.pad(f_inst, w_samples // 2, mode="edge")
                f_smooth = np.convolve(f_pad, window, mode="valid")

                if len(f_smooth) > len(f_inst):
                    f_smooth = f_smooth[: len(f_inst)]
                f_inst = f_smooth

        out = self.__class__(
            f_inst, t0=self.t0, dt=self.dt, channel=self.channel, name=self.name
        )
        out.override_unit("Hz")
        return out

    # ===============================
    # Demodulation Methods
    # ===============================

    def _build_phase_series(
        self,
        *,
        phase: (
            Sequence[NumberLike]
            | NDArray[np.floating]
            | NDArray[np.complexfloating]
            | None
        ) = None,
        f0: NumberLike | u.Quantity | None = None,
        fdot: NumberLike | u.Quantity = 0.0,
        fddot: NumberLike | u.Quantity = 0.0,
        phase_epoch: NumberLike | None = None,
        phase0: float = 0.0,
        prefer_dt: bool = True,
    ) -> NDArray[np.float_]:
        """Internal helper to build phase series in radians."""
        if (f0 is None and phase is None) or (f0 is not None and phase is not None):
            raise ValueError("Exactly one of 'f0' or 'phase' must be provided.")

        if phase is not None:
            if len(phase) != self.size:
                raise ValueError(
                    f"Length of phase ({len(phase)}) does not match TimeSeries ({self.size})"
                )
            return np.asarray(phase, dtype=float) + phase0

        # Build from model
        if isinstance(f0, u.Quantity):
            f0 = f0.to("Hz").value
        else:
            assert f0 is not None
            f0 = float(f0)

        if isinstance(fdot, u.Quantity):
            fdot = fdot.to("Hz/s").value
        else:
            fdot = float(fdot)

        if isinstance(fddot, u.Quantity):
            fddot = float(fddot.to("Hz/s^2").value)
        else:
            fddot = float(fddot)

        # Determine t_rel
        has_dt = False
        dt_val = None
        try:
            if self.dt is not None:
                dt_val = self.dt.to("s").value
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
                raise ValueError(
                    "TimeSeries requires times or dt to build phase model."
                )
            times_val = self.times.value if hasattr(self.times, "value") else self.times
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
            cycles += (1.0 / 6.0) * fddot * t_rel**3

        return 2 * np.pi * cycles + phase0

    def mix_down(
        self,
        *,
        phase: Sequence[NumberLike]
        | NDArray[np.floating]
        | NDArray[np.complexfloating]
        | None = None,
        f0: NumberLike | u.Quantity | None = None,
        fdot: NumberLike | u.Quantity = 0.0,
        fddot: NumberLike | u.Quantity = 0.0,
        phase_epoch: NumberLike | None = None,
        phase0: float = 0.0,
        singlesided: bool = False,
        copy: bool = True,
    ) -> TimeSeriesSignalMixin:
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
            prefer_dt=True,
        )

        # Mix
        y = self.value * np.exp(-1j * phase_series)

        if singlesided:
            y *= 2.0

        # Prepare constructor args
        kwargs = {"unit": self.unit, "channel": self.channel, "name": self.name}

        # Check regularity for constructor
        try:
            if self.dt is not None:
                kwargs["t0"] = self.t0
                kwargs["dt"] = self.dt
            else:
                kwargs["t0"] = self.t0
                kwargs["sample_rate"] = self.sample_rate
        except (AttributeError, ValueError):
            kwargs["times"] = self.times

        out = self.__class__(y, **kwargs)
        return out

    def heterodyne(
        self,
        phase: Sequence[NumberLike]
        | NDArray[np.floating]
        | NDArray[np.complexfloating],
        stride: NumberLike | u.Quantity = 1.0,
        singlesided: bool = True,
    ) -> TimeSeriesSignalMixin:
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
        return z.resample(stride_dt, agg="mean")

    def baseband(
        self,
        *,
        phase: Sequence[NumberLike]
        | NDArray[np.floating]
        | NDArray[np.complexfloating]
        | None = None,
        f0: NumberLike | u.Quantity | None = None,
        fdot: NumberLike | u.Quantity = 0.0,
        fddot: NumberLike | u.Quantity = 0.0,
        phase_epoch: NumberLike | None = None,
        phase0: float = 0.0,
        lowpass: float | u.Quantity | None = None,
        lowpass_kwargs: dict[str, Any] | None = None,
        output_rate: NumberLike | u.Quantity | None = None,
        singlesided: bool = False,
    ) -> TimeSeriesSignalMixin:
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
            singlesided=singlesided,
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
        f0: NumberLike | u.Quantity | None = None,
        *,
        phase: Sequence[NumberLike]
        | NDArray[np.floating]
        | NDArray[np.complexfloating]
        | None = None,
        fdot: NumberLike | u.Quantity = 0.0,
        fddot: NumberLike | u.Quantity = 0.0,
        phase_epoch: NumberLike | None = None,
        phase0: float = 0.0,
        stride: NumberLike | u.Quantity | None = None,
        bandwidth: NumberLike | u.Quantity | None = None,
        singlesided: bool = True,
        output: Literal["complex", "amp_phase", "iq"] = "amp_phase",
        deg: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Perform lock-in amplification (demodulation + averaging).
        """
        self._check_regular("lock_in")

        if bandwidth is not None:
            # LPF based
            outc = self.baseband(
                phase=phase,
                f0=f0,
                fdot=fdot,
                fddot=fddot,
                phase_epoch=phase_epoch,
                phase0=phase0,
                lowpass=bandwidth,
                singlesided=singlesided,
                **kwargs,
            )
        else:
            # Averaging based
            stride = stride or 1.0
            phase_series = self._build_phase_series(
                phase=phase,
                f0=f0,
                fdot=fdot,
                fddot=fddot,
                phase_epoch=phase_epoch,
                phase0=phase0,
                prefer_dt=True,
            )
            outc = self.heterodyne(phase_series, stride=stride, singlesided=singlesided)

        if output == "complex":
            return outc
        elif output == "amp_phase":
            mag = outc.abs()
            ph = self.__class__(
                np.angle(outc.value, deg=deg),
                t0=outc.t0,
                dt=outc.dt,
                channel=self.channel,
                name=self.name,
            )
            ph.override_unit("deg" if deg else "rad")
            return mag, ph
        elif output == "iq":
            i = self.__class__(
                outc.value.real,
                t0=outc.t0,
                dt=outc.dt,
                channel=self.channel,
                name=self.name,
                unit=self.unit,
            )
            q = self.__class__(
                outc.value.imag,
                t0=outc.t0,
                dt=outc.dt,
                channel=self.channel,
                name=self.name,
                unit=self.unit,
            )
            return i, q
        else:
            raise ValueError(f"Unknown output format: {output}")

    # ===============================
    # Cross-correlation Methods
    # ===============================

    def transfer_function(
        self,
        other: TimeSeries,
        mode: Literal["steady", "transient"] = "steady",
        fftlength: NumberLike | None = None,
        overlap: NumberLike | None = None,
        window: str | ArrayLike | None = "hann",
        average: str = "mean",
        *,
        method: str | None = None,  # Deprecated, for backward compatibility
        fft_kwargs: dict[str, Any] | None = None,
        downsample: NumberLike | None = None,
        align: Literal["intersection", "none"] = "intersection",
        **kwargs: Any,
    ) -> Any:
        """
        Compute the transfer function between this TimeSeries and another.

        This TimeSeries (`self`) is the 'A-channel' (reference, denominator),
        while `other` is the 'B-channel' (test, numerator).

        Parameters
        ----------
        other : `TimeSeries`
            The test TimeSeries (numerator).
        mode : `str`, optional
            "steady" (default): GWpy-compatible averaged estimator using
                H(f) = CSD_{A,B}(f) / PSD_A(f)
                Use for steady-state system identification with noise averaging.
            "transient": Instantaneous FFT ratio
                H(f) = FFT_B(f) / FFT_A(f)
                Use for single-shot transient response analysis.
        fftlength : `float`, optional
            Length of the FFT, in seconds. Only used for mode="steady".
        overlap : `float`, optional
            Overlap between segments, in seconds. Only used for mode="steady".
        window : `str`, `numpy.ndarray`, optional
            Window function to apply (mode="steady" only).
        average : `str`, optional
            Method to average segments (mode="steady" only).
        method : `str`, optional
            **Deprecated**: use `mode` instead.
            For backward compatibility: "gwpy"/"csd_psd" -> mode="steady",
            "fft" -> mode="transient", "auto" -> auto-select.
        fft_kwargs : `dict`, optional
            Additional keyword arguments for FFT (mode="transient" only).
        downsample : `float`, optional
            Whether to downsample if sample rates differ (mode="transient" only).
        align : `str`, optional
            Alignment method: "intersection" or "none" (mode="transient" only).

        Returns
        -------
        out : `FrequencySeries`
            Transfer function.

        Notes
        -----
        **steady mode (GWpy-compatible)**:
            Uses cross-spectral density and power spectral density:

            .. math::

                H(f) = \\frac{\\mathrm{CSD}_{A,B}(f)}{\\mathrm{PSD}_A(f)}

            This is the standard estimator for steady-state system identification,
            providing noise averaging through overlapped segmented FFTs.
            Results are numerically identical to GWpy's `transfer_function`.

        **transient mode (gwexpy extension)**:
            Uses direct FFT ratio without averaging:

            .. math::

                H_{\\mathrm{transient}}(f) = \\frac{\\mathrm{FFT}_B(f)}{\\mathrm{FFT}_A(f)}

            Use this for single-shot transient response analysis where
            averaging would obscure the instantaneous transfer characteristics.
            Employs the corrected transient FFT with proper DC/Nyquist handling.

        **division semantics (steady/transient)**:
            - den == 0, num == 0 -> NaN (real: np.nan, complex: np.nan + 1j*np.nan)
            - den == 0, num > 0 -> +inf (complex dtype: np.inf + 0j)
            - den == 0, num < 0 -> -inf (complex dtype: -np.inf + 0j)
            - den == 0, complex num (imag != 0) -> inf * exp(1j * angle(num))

        Examples
        --------
        Steady-state transfer function (GWpy-compatible)::

            >>> tf = reference.transfer_function(test, mode="steady", fftlength=1.0)

        Transient transfer function::

            >>> tf = input_signal.transfer_function(output_signal, mode="transient")
        """
        import warnings

        # Handle deprecated 'method' parameter
        if method is not None:
            warnings.warn(
                "The 'method' parameter is deprecated. Use 'mode' instead: "
                "'gwpy'/'csd_psd' -> mode='steady', 'fft' -> mode='transient'.",
                DeprecationWarning,
                stacklevel=2,
            )
            if method in ("gwpy", "csd_psd"):
                mode = "steady"
            elif method == "fft":
                mode = "transient"
            elif method == "auto":
                # Auto-select: if fftlength is None, use transient; else steady
                mode = "transient" if fftlength is None else "steady"
            else:
                raise ValueError(f"Unknown method: {method}")

        if mode not in ("steady", "transient"):
            raise ValueError(f"mode must be 'steady' or 'transient', got: {mode}")

        def _divide_with_special_rules(num_series: Any, den_series: Any) -> Any:
            size = min(num_series.size, den_series.size)
            num = num_series[:size]
            den = den_series[:size]

            num_vals = np.asarray(num.value)
            den_vals = np.asarray(den.value)
            out_dtype = np.result_type(num_vals, den_vals)
            out_vals = np.empty_like(num_vals, dtype=out_dtype)

            den_zero = den_vals == 0
            np.divide(num_vals, den_vals, out=out_vals, where=~den_zero)

            if np.any(den_zero):
                num_zero = num_vals == 0
                zero_zero = den_zero & num_zero
                if np.any(zero_zero):
                    if np.iscomplexobj(out_vals):
                        out_vals[zero_zero] = np.nan + 1j * np.nan
                    else:
                        out_vals[zero_zero] = np.nan

                num_nonzero = den_zero & ~num_zero
                if np.any(num_nonzero):
                    num_real = np.isreal(num_vals)
                    real_mask = num_nonzero & num_real
                    if np.any(real_mask):
                        signs = np.sign(np.real(num_vals[real_mask]))
                        out_vals[real_mask] = signs * np.inf

                    complex_mask = num_nonzero & ~num_real
                    if np.any(complex_mask):
                        out_vals[complex_mask] = (np.inf + 0j) * np.exp(
                            1j * np.angle(num_vals[complex_mask])
                        )

            unit = None
            num_unit = getattr(num, "unit", None)
            den_unit = getattr(den, "unit", None)
            try:
                if num_unit is not None and den_unit is not None:
                    unit = num_unit / den_unit
                elif num_unit is not None:
                    unit = num_unit
                elif den_unit is not None:
                    unit = 1 / den_unit
            except Exception:
                unit = None

            return num.__class__(
                out_vals,
                frequencies=num.frequencies,
                unit=unit,
                name=num.name,
                channel=num.channel,
                epoch=num.epoch,
            )

        if mode == "steady":
            # GWpy-compatible: CSD / PSD estimator
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
            tf = _divide_with_special_rules(csd, psd)
            if other.name and self.name:
                tf.name = f"{other.name} / {self.name}"
            return tf

        else:  # mode == "transient"
            # FFT Ratio using transient FFT
            kw = dict(fft_kwargs) if fft_kwargs is not None else {}
            kw.setdefault("mode", "transient")

            a: TimeSeriesSignalMixin = self
            b: TimeSeriesSignalMixin = other

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
                    a = cast(TimeSeriesSignalMixin, a.resample(b.sample_rate))
                elif rate_b > rate_a:
                    b = cast(TimeSeriesSignalMixin, b.resample(a.sample_rate))

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
                a = cast(Any, a)[:size]
            if b.size != size:
                b = cast(Any, b)[:size]

            # 4. FFTs (using transient mode for proper amplitude preservation)
            fx = a.fft(**kw)
            fy = b.fft(**kw)

            tf = _divide_with_special_rules(fy, fx)

            if b.name and a.name:
                tf.name = f"{b.name} / {a.name}"

            return tf

    def xcorr(
        self,
        other: Any,
        *,
        maxlag: float | None = None,
        normalize: str | None = None,
        mode: str = "full",
        demean: bool = True,
    ) -> TimeSeriesSignalMixin:
        """
        Compute time-domain cross-correlation between two TimeSeries.
        """
        from scipy import signal

        dt_self = (
            self.dt if isinstance(self.dt, u.Quantity) else u.Quantity(self.dt, "s")
        )
        dt_other = (
            other.dt if isinstance(other.dt, u.Quantity) else u.Quantity(other.dt, "s")
        )
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
