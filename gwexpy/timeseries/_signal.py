"""
Signal processing methods for TimeSeries.

This module provides signal processing functionality as a mixin class:
- Hilbert transform: hilbert, envelope
- Phase/frequency: instantaneous_phase, unwrap_phase, instantaneous_frequency
- Demodulation: _build_phase_series, mix_down, baseband, lock_in
- Cross-correlation: xcorr, transfer_function
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Optional, Union, cast

try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import numpy as np
from astropy import units as u
from numpy.typing import ArrayLike, NDArray

logger = logging.getLogger(__name__)

from ._typing import TimeSeriesAttrs

NumberLike: TypeAlias = Union[int, float, np.number]
QuantityLike: TypeAlias = Union[ArrayLike, u.Quantity]
PhaseLike: TypeAlias = Union[
    Sequence[NumberLike], NDArray[np.floating], NDArray[np.complex128], None
]

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
        pad: Union[NumberLike, u.Quantity] = 0,
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
        if isinstance(pad, str):
            pad = u.Quantity(pad)
        if isinstance(pad, u.Quantity):
            if self.dt is None or getattr(self.times, "unit", None) is None:
                raise ValueError("hilbert requires defined dt and times when padding.")
            n_pad = int(
                round(pad.to(self.times.unit).value / self.dt.to(self.times.unit).value)
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
        smooth: Optional[Union[NumberLike, u.Quantity]] = None,
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
        phase: PhaseLike = None,
        f0: Optional[Union[NumberLike, u.Quantity]] = None,
        fdot: Union[NumberLike, u.Quantity] = 0.0,
        fddot: Union[NumberLike, u.Quantity] = 0.0,
        phase_epoch: Optional[NumberLike] = None,
        phase0: float = 0.0,
        prefer_dt: bool = True,
    ) -> NDArray[np.floating]:
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
        phase: PhaseLike = None,
        f0: Optional[Union[NumberLike, u.Quantity]] = None,
        fdot: Union[NumberLike, u.Quantity] = 0.0,
        fddot: Union[NumberLike, u.Quantity] = 0.0,
        phase_epoch: Optional[NumberLike] = None,
        phase0: float = 0.0,
        singlesided: bool = False,
        copy: bool = True,
    ) -> TimeSeriesSignalMixin:
        """
        Mix the TimeSeries with a complex oscillator.
        """
        phase_series = self._build_phase_series(
            phase=phase,  # type: ignore[arg-type]
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
        phase: Union[
            Sequence[NumberLike], NDArray[np.floating], NDArray[np.complex128]
        ],
        stride: Union[NumberLike, u.Quantity] = 1.0,
        *,
        singlesided: bool = False,
    ) -> TimeSeriesSignalMixin:
        """
        Compute the average magnitude and phase of the TimeSeries after
        heterodyning with a given phase series.

        This method replicates the GWpy ``TimeSeries.heterodyne()`` algorithm
        exactly. The input TimeSeries is heterodyned against a phase series
        and averaged over fixed strides.

        Parameters
        ----------
        phase : array_like or TimeSeries
            Phase to mix with (radians). Must have ``len(phase) == len(self)``.
        stride : float or Quantity, default: 1.0
            Time step for averaging in seconds. Strides are rounded to
            the nearest number of samples (``int(stride * sample_rate)``).
            Trailing samples that do not form a full stride are discarded.
        singlesided : bool, default: False
            If True, double the amplitude of the output (conventional for real
            signals). Default is False, aligning with GWpy.

        Returns
        -------
        TimeSeries
            Complex demodulated signal with ``dt = stride``.
            The output value represents ``mag * exp(1j * phase_out)`` where
            mag/phase are the average magnitude and phase over each stride.

        Raises
        ------
        TypeError
            If ``phase`` is not array_like (i.e. ``len(phase)`` fails).
        ValueError
            If ``len(phase) != len(self)``.

        Notes
        -----
        **Algorithm (GWpy-identical)**

        1. ``stridesamp = int(stride * self.sample_rate.value)`` (floor truncation)
        2. ``nsteps = int(self.size // stridesamp)`` (trailing samples discarded)
        3. For each step ``step`` in ``range(nsteps)``:

           - ``istart = stridesamp * step``
           - ``iend = istart + stridesamp`` (exclusive end)
           - ``mixed = exp(-1j * phase[istart:iend]) * self.value[istart:iend]``
           - ``out[step] = 2 * mixed.mean() if singlesided else mixed.mean()``

        4. Output ``sample_rate = 1 / stride``

        See Also
        --------
        TimeSeries.demodulate
            for heterodyning at a fixed frequency (GWpy)
        TimeSeries.lock_in
            for a higher-level lock-in amplifier interface
        """
        # --- Phase validation (GWpy-compatible) ---
        try:
            phase = np.asarray(phase)  # make sure phase is a numpy array
            if phase.ndim != 1:
                raise TypeError(f"Phase is not array_like: ndim={phase.ndim}")
            _ = len(phase)  # ensure len() works
        except TypeError as e:
            raise TypeError(f"Phase is not array_like: {e}") from e

        if len(phase) != len(self):
            raise ValueError("Phase array must be the same length as the TimeSeries")

        # --- Stride calculation ---
        if isinstance(stride, u.Quantity):
            stride_s = stride.to("s").value
        else:
            stride_s = float(stride)

        stridesamp = int(stride_s * self.sample_rate.value)
        nsteps = int(self.size // stridesamp)

        # --- Heterodyne loop (GWpy-identical) ---
        out_data = np.zeros(nsteps, dtype=complex)
        for step in range(nsteps):
            istart = stridesamp * step
            iend = istart + stridesamp
            mixed = np.exp(-1j * phase[istart:iend]) * self.value[istart:iend]
            out_data[step] = 2 * mixed.mean() if singlesided else mixed.mean()

        # --- Build output TimeSeries (GWpy-compatible) ---
        # Use constructor with explicit sample_rate to ensure proper dt/sample_rate
        out = self.__class__(
            out_data,
            t0=self.t0,
            sample_rate=1 / stride_s,
            unit=self.unit,
            name=self.name,
            channel=self.channel,
        )

        return out

    def demodulate(
        self,
        f: Union[NumberLike, u.Quantity],
        stride: Union[NumberLike, u.Quantity] = 1.0,
        phase: float = 0.0,
        deg: bool = True,
        exp: bool = False,
    ) -> Any:
        r"""
        Compute the average magnitude and phase of the TimeSeries at a
        given frequency (GWpy-compatible).

        This method replicates the GWpy ``TimeSeries.demodulate()`` algorithm.
        It heterodynes the signal at a fixed frequency, averages over
        strides, and returns either the complex demodulated signal or
        (magnitude, phase) trends.

        Parameters
        ----------
        f : float or Quantity
            Frequency (Hz) at which to demodulate.
        stride : float or Quantity, default: 1.0
            Time step for averaging in seconds.
        phase : float, default: 0.0
            Initial phase offset in radians.
        deg : bool, default: True
            If True, return phase in degrees; else radians.
        exp : bool, default: False
            If True, return a single complex TimeSeries ($mag \cdot e^{i\phi}$).
            If False (default), return a tuple of (magnitude, phase) TimeSeries.

        Returns
        -------
        out : TimeSeries or tuple
            Demodulated result. If ``exp=True``, a complex TimeSeries.
            If ``exp=False``, a tuple of (magnitude, phase) TimeSeries.

        Notes
        -----
        **Phase Convention (GWpy-identical)**
        The phase model is built as $\phi(t) = 2\pi f t + \text{phase}$.
        The mixing operation is $x(t) \cdot e^{-i\phi(t)}$.
        The result is always multiplied by 2 (``singlesided=True``) to
        recover the full amplitude of real signals, matching GWpy.

        See Also
        --------
        TimeSeries.heterodyne
            for the underlying heterodyne method with arbitrary phase.
        TimeSeries.lock_in
            for a more flexible lock-in amplifier interface.
        """
        # Build phase for a fixed frequency
        from astropy.units import Quantity

        if isinstance(f, Quantity):
            f_val = f.to("Hz").value
        else:
            f_val = float(f)

        if self.dt is None:
            raise ValueError("demodulate requires defined dt")
        dt_val = self.dt.to("s").value if hasattr(self.dt, "to") else float(self.dt)

        phase_arr = 2 * np.pi * f_val * dt_val * np.arange(self.size)
        if phase != 0.0:
            phase_arr += float(phase)

        # Call heterodyne with singlesided=True (standard for demodulate)
        out = self.heterodyne(phase_arr, stride=stride, singlesided=True)

        if exp:
            return out

        mag = out.abs()
        ph_val = np.angle(out.value, deg=deg)
        ph = self.__class__(
            ph_val,
            t0=out.t0,
            dt=out.dt,
            channel=self.channel,
            name=self.name,
        )
        ph.override_unit("deg" if deg else "rad")
        return mag, ph

    def baseband(
        self,
        *,
        phase: PhaseLike = None,
        f0: Optional[Union[NumberLike, u.Quantity]] = None,
        fdot: Union[NumberLike, u.Quantity] = 0.0,
        fddot: Union[NumberLike, u.Quantity] = 0.0,
        phase_epoch: Optional[NumberLike] = None,
        phase0: float = 0.0,
        lowpass: Optional[Union[float, u.Quantity]] = None,
        lowpass_kwargs: Optional[dict[str, Any]] = None,
        output_rate: Optional[Union[NumberLike, u.Quantity]] = None,
        resample_kwargs: Optional[dict[str, Any]] = None,
        singlesided: bool = False,
    ) -> TimeSeriesSignalMixin:
        r"""
        Demodulate the TimeSeries to baseband with optional lowpass and resampling.

        This method performs frequency mixing (heterodyning) to shift a carrier
        frequency to baseband (DC), optionally followed by lowpass filtering
        and/or resampling. The processing chain is:

            mix_down(f0) → [lowpass(cutoff)] → [resample(output_rate)]

        Two primary modes are supported:

        **Mode A (Analysis bandwidth explicit)**:
            ``baseband(f0=fc, lowpass=cutoff, output_rate=None|...)``
            - Applies lowpass filter after mixing to define analysis bandwidth
            - Optionally resamples to reduce data rate

        **Mode B (Downsample priority)**:
            ``baseband(f0=fc, lowpass=None, output_rate=rate)``
            - Skips explicit lowpass; relies on resample's anti-aliasing
            - Useful when avoiding double-filtering

        Parameters
        ----------
        phase : array_like or None, optional
            Explicit phase array (radians) for mixing. Mutually exclusive
            with f0/fdot/fddot.
        f0 : float or Quantity, optional
            Center frequency (Hz) for mixing. The signal at f0 is shifted to DC.
            Must satisfy 0 < f0 < Nyquist for regular series.
        fdot : float or Quantity, default=0.0
            Frequency derivative (Hz/s) for chirp signals.
        fddot : float or Quantity, default=0.0
            Second frequency derivative (Hz/s²) for accelerating chirps.
        phase_epoch : float or None, optional
            Reference epoch for phase model.
        phase0 : float, default=0.0
            Initial phase offset (radians).
        lowpass : float or Quantity or None, optional
            Lowpass filter corner frequency (Hz). Defines the analysis bandwidth
            (half-bandwidth) around baseband. Must satisfy 0 < lowpass < Nyquist.
            If both lowpass and output_rate are specified, lowpass must be less
            than output_rate/2 (the new Nyquist).
        lowpass_kwargs : dict or None, optional
            Additional arguments passed to :meth:`lowpass`. GWpy-compatible
            options include ``type``, ``gpass``, ``gstop``, ``fstop``, ``filtfilt``.
        output_rate : float or Quantity or None, optional
            Output sample rate (Hz). If specified, resamples the output.
            Must be > 0. Uses GWpy's :meth:`resample` internally.
        resample_kwargs : dict or None, optional
            Additional arguments passed to :meth:`resample`. GWpy-compatible
            options include ``window``, ``ftype``, ``n``.
        singlesided : bool, default=False
            If True, double the amplitude (for real input signals).

        Returns
        -------
        TimeSeries
            Complex baseband signal.

        Raises
        ------
        ValueError
            If f0 <= 0.
        ValueError
            If f0 >= Nyquist (for regular series).
        ValueError
            If lowpass <= 0 or lowpass >= Nyquist.
        ValueError
            If output_rate <= 0.
        ValueError
            If both lowpass and output_rate are None.
        ValueError
            If lowpass >= output_rate/2 (exceeds new Nyquist).

        Notes
        -----
        **Preprocessing**: No automatic preprocessing is applied. Users should
        apply demean, detrend, or filtering as needed before calling. DC offset
        and low-frequency trends can affect the baseband result.

        **Lowpass vs f0**: It is generally recommended to set lowpass < f0 to
        capture only the modulation around the carrier. However, this is not
        enforced to allow flexibility in edge cases.

        **GWpy alignment**: The lowpass and resample operations delegate to
        GWpy's methods with their default parameters. Customization is available
        via lowpass_kwargs and resample_kwargs.

        Examples
        --------
        Mode A (with lowpass):

        >>> ts = TimeSeries(np.cos(2 * np.pi * 100 * t), dt=0.001, unit='V')
        >>> z = ts.baseband(f0=100, lowpass=10)  # 10 Hz analysis bandwidth

        Mode B (resample only):

        >>> z = ts.baseband(f0=100, lowpass=None, output_rate=50)

        With both:

        >>> z = ts.baseband(f0=100, lowpass=10, output_rate=50)
        """
        # === Input validation ===

        # Extract f0 value for validation
        f0_val = None
        if f0 is not None:
            if isinstance(f0, u.Quantity):
                f0_val = f0.to("Hz").value
            else:
                f0_val = float(f0)

            if f0_val <= 0:
                raise ValueError(f"f0 must be positive, got {f0_val}")

        # Get sample rate for Nyquist checks
        sample_rate_val = None
        nyquist = None
        if self.sample_rate is not None:
            try:
                sample_rate_val = self.sample_rate.to("Hz").value
                nyquist = sample_rate_val / 2.0
            except (AttributeError, u.UnitConversionError):
                sample_rate_val = float(self.sample_rate)
                nyquist = sample_rate_val / 2.0

        # Validate f0 < Nyquist
        if f0_val is not None and nyquist is not None:
            if f0_val >= nyquist:
                raise ValueError(
                    f"f0 ({f0_val} Hz) must be less than Nyquist ({nyquist} Hz)"
                )

        # Validate lowpass
        lowpass_val = None
        if lowpass is not None:
            if isinstance(lowpass, u.Quantity):
                lowpass_val = lowpass.to("Hz").value
            else:
                lowpass_val = float(lowpass)

            if lowpass_val <= 0:
                raise ValueError(f"lowpass must be positive, got {lowpass_val}")

            if nyquist is not None and lowpass_val >= nyquist:
                raise ValueError(
                    f"lowpass ({lowpass_val} Hz) must be less than Nyquist ({nyquist} Hz)"
                )

        # Validate output_rate
        output_rate_val = None
        if output_rate is not None:
            if isinstance(output_rate, u.Quantity):
                output_rate_val = output_rate.to("Hz").value
            else:
                output_rate_val = float(output_rate)

            if output_rate_val <= 0:
                raise ValueError(f"output_rate must be positive, got {output_rate_val}")

        # At least one of lowpass or output_rate must be specified
        if lowpass is None and output_rate is None:
            raise ValueError(
                "At least one of 'lowpass' or 'output_rate' must be specified. "
                "Use lowpass to define analysis bandwidth, or output_rate to resample."
            )

        # Validate lowpass < new Nyquist when both are specified
        if lowpass_val is not None and output_rate_val is not None:
            new_nyquist = output_rate_val / 2.0
            if lowpass_val >= new_nyquist:
                raise ValueError(
                    f"lowpass ({lowpass_val} Hz) must be less than output_rate/2 "
                    f"({new_nyquist} Hz) to avoid aliasing"
                )

        # === Processing ===

        # Step 1: Mix down
        z = self.mix_down(
            phase=phase,
            f0=f0,
            fdot=fdot,
            fddot=fddot,
            phase_epoch=phase_epoch,
            phase0=phase0,
            singlesided=singlesided,
        )

        # Step 2: Lowpass (optional)
        if lowpass is not None:
            if z.sample_rate is None:
                raise ValueError("lowpass requires defined sample rate.")
            lp_kwargs = lowpass_kwargs or {}
            z = z.lowpass(lowpass, **lp_kwargs)

        # Step 3: Resample (optional)
        if output_rate is not None:
            rs_kwargs = resample_kwargs or {}
            z = z.resample(output_rate, **rs_kwargs)

        return z

    def lock_in(
        self,
        f0: Optional[Union[NumberLike, u.Quantity]] = None,
        *,
        phase: PhaseLike = None,
        fdot: Union[NumberLike, u.Quantity] = 0.0,
        fddot: Union[NumberLike, u.Quantity] = 0.0,
        phase_epoch: Optional[NumberLike] = None,
        phase0: float = 0.0,
        stride: Optional[Union[NumberLike, u.Quantity]] = None,
        bandwidth: Optional[Union[NumberLike, u.Quantity]] = None,
        singlesided: bool = True,
        output: Literal["complex", "amp_phase", "iq"] = "amp_phase",
        deg: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Perform lock-in amplification (demodulation + filtering/averaging).

        This method extracts the complex amplitude (or magnitude and phase) of
        a specific frequency component from the TimeSeries. It supports two
        independent operational modes based on how the post-mixing signal is
        smoothed:

        **Mode 1: Stride-Average (bandwidth is None)**
            The signal is mixed with the reference and averaged over
            non-overlapping contiguous time intervals of length ``stride``.
            The output sample rate becomes ``1/stride``. This mode is
            numerically identical to GWpy's ``demodulate`` (with ``exp=True``).

        **Mode 2: LPF-Filtering (bandwidth is not None)**
            The signal is mixed with the reference and passed through a
            low-pass filter with a cutoff frequency of ``bandwidth``.
            By default, a zero-phase IIR filter is used (via ``filtfilt``).
            The output sample rate remains the same as the input unless
            ``output_rate`` is specified in ``**kwargs``.

        Parameters
        ----------
        f0 : float or Quantity, optional
            Center frequency (Hz) for the reference phase model.
            Mutually exclusive with ``phase``.
        phase : array_like, optional
            Explicit reference phase array in radians. Must have the same
            length as the input TimeSeries. If provided, f0-based parameters
            must NOT be specified.
        fdot : float or Quantity, default: 0.0
            First derivative of frequency (Hz/s) for chirped signals.
        fddot : float or Quantity, default: 0.0
            Second derivative of frequency (Hz/s²) for accelerating signals.
        phase_epoch : float or Quantity, optional
            Reference time (GPS) for the phase model. If None (default), use
            the start time of the TimeSeries.
        phase0 : float, default: 0.0
            Initial phase offset in radians at ``phase_epoch``.
        stride : float or Quantity, optional
            Averaging time step in seconds. Required for Stride-Average mode.
            Must NOT be specified if ``bandwidth`` is provided.
        bandwidth : float or Quantity, optional
            Low-pass filter cutoff frequency in Hz. Required for LPF mode.
            Defines the analysis bandwidth (half-bandwidth) around DC.
        singlesided : bool, default: True
            If True (default), multiply the output by 2. This is the convention
            for recovering the full amplitude of a real-valued input signal
            $A \\cos(\\omega t + \\phi)$.
        output : {'amp_phase', 'complex', 'iq'}, default: 'amp_phase'
            Formatting of the returned data:
            - ``'amp_phase'``: Returns a tuple of (amplitude, phase).
            - ``'complex'``: Returns a complex TimeSeries ($mag \\cdot e^{i\\phi}$).
            - ``'iq'``: Returns a tuple of (In-phase, Quadrature) TimeSeries.
        deg : bool, default: True
            If True and ``output='amp_phase'``, the phase is returned in
            degrees. Otherwise, radians.
        **kwargs : Any
            Additional keyword arguments passed to :meth:`baseband` in LPF mode.
            Common options: ``type`` (filter type), ``filtfilt`` (bool).

        Returns
        -------
        out : TimeSeries or tuple
            The demodulated signal in the requested format. Returns a tuple
            for ``'amp_phase'`` and ``'iq'``, and a single TimeSeries for
            ``'complex'``. Metadata like ``t0``, ``name``, and ``channel``
            are preserved. The ``unit`` of the amplitude/complex/IQ result
            is the same as the input signal.

        Notes
        -----
        **Phase Convention**
        The mixing uses the negative exponential convention:
        $Z(t) = 2 \\cdot \\text{LPF}\\{ x(t) \\cdot e^{-i\\phi(t)} \\}$ (if singlesided=True).
        For a real input $x(t) = A \\cos(\\omega t + \\phi_0)$, this operation
        correctly recovers the complex amplitude $A e^{i\\phi_0}$.

        **Edge Handling**
        - **Stride mode**: Discards any remainder samples at the end that do
          not form a full stride. The result is anchored at the start of
          each stride.
        - **LPF mode**: Filter transients occur at the boundaries. Using
          ``filtfilt=True`` (default) minimizes phase distortion but
          transients still exist. Padding the input is recommended for
          short series.

        **Numerical Precision**
        The phase is calculated relative to ``phase_epoch`` to maintain
        precision for high frequencies or long durations.

        Examples
        --------
        Recover amplitude and phase of a 100 Hz signal:

        >>> import numpy as np
        >>> t = np.linspace(0, 10, 163840)
        >>> data = np.cos(2 * np.pi * 100 * t + np.pi/4)
        >>> ts = TimeSeries(data, sample_rate=16384, t0=0)
        >>> amp, phase = ts.lock_in(f0=100, stride=1.0)
        >>> print(f"Amp: {amp.value.mean():.2f}, Phase: {phase.value.mean():.2f}")
        Amp: 1.00, Phase: 45.00

        Using LPF mode for higher time resolution:

        >>> complex_ts = ts.lock_in(f0=100, bandwidth=5.0, output='complex')
        """
        self._check_regular("lock_in")

        # === Validation: phase vs f0 mutual exclusivity ===
        has_phase = phase is not None
        has_f0_params = (
            f0 is not None
            or fdot != 0.0
            or fddot != 0.0
            or phase_epoch is not None
            or phase0 != 0.0
        )

        if has_phase and has_f0_params:
            raise ValueError(
                "Cannot specify both 'phase' and any of 'f0/fdot/fddot/phase_epoch/phase0'. "
                "When 'phase' is provided, it takes precedence and the f0-based parameters "
                "must not be specified."
            )

        if not has_phase and f0 is None:
            raise ValueError(
                "Either 'phase' or 'f0' must be specified. "
                "Use 'phase' for explicit phase array, or 'f0' to build phase from frequency model."
            )

        # === Validation: bandwidth vs stride mutual exclusivity ===
        if bandwidth is not None and stride is not None:
            raise ValueError(
                "Cannot specify both 'bandwidth' and 'stride'. "
                "Use 'bandwidth' for LPF mode (with baseband), or 'stride' for stride-average mode."
            )

        if bandwidth is None and stride is None:
            raise ValueError(
                "Either 'bandwidth' or 'stride' must be specified. "
                "Use 'stride' for stride-average mode (heterodyne), or 'bandwidth' for LPF mode."
            )

        # === Output validation ===
        if output not in ("complex", "amp_phase", "iq"):
            raise ValueError(
                f"Unknown output format: {output}. Must be one of 'complex', 'amp_phase', 'iq'."
            )

        # === Mode dispatch ===
        if bandwidth is not None:
            # LPF mode: use baseband
            outc = self.baseband(
                phase=phase,  # type: ignore[arg-type]
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
            # Stride-average mode: build phase series and use heterodyne
            assert stride is not None  # guaranteed by validation above
            phase_series = self._build_phase_series(
                phase=phase,  # type: ignore[arg-type]
                f0=f0,
                fdot=fdot,
                fddot=fddot,
                phase_epoch=phase_epoch,
                phase0=phase0,
                prefer_dt=True,
            )
            outc = self.heterodyne(phase_series, stride=stride, singlesided=singlesided)

        # === Output formatting ===
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
        else:  # output == "iq"
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

    # ===============================
    # Cross-correlation Methods
    # ===============================

    def transfer_function(
        self,
        other: TimeSeries,
        mode: Literal["steady", "transient"] = "steady",
        fftlength: Optional[NumberLike] = None,
        overlap: Optional[NumberLike] = None,
        window: Optional[Union[str, ArrayLike]] = "hann",
        average: str = "mean",
        *,
        method: Optional[
            Literal["gwpy", "csd_psd", "fft", "auto"]
        ] = None,  # Deprecated
        fft_kwargs: Optional[dict[str, Any]] = None,
        downsample: Optional[NumberLike] = None,
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
            except (TypeError, ValueError):
                logger.debug(
                    "Failed to determine unit for transfer function.", exc_info=True
                )
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
        maxlag: Optional[float] = None,
        normalize: Optional[str] = None,
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
