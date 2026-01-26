"""gwexpy.noise.wave - Generate time-series waveforms.

This module provides functions for generating common time-series waveforms
including noise, periodic signals, and transient signals.

All functions return gwexpy.TimeSeries objects.

Submodule Structure
-------------------
- `gwexpy.noise.asd`: Functions that return ASD (FrequencySeries)
- `gwexpy.noise.wave`: Functions that return time-series waveforms (TimeSeries)

Examples
--------
>>> from gwexpy.noise.wave import sine, gaussian, chirp
>>> sine_wave = sine(duration=1.0, sample_rate=1024, frequency=10.0)
>>> noise = gaussian(duration=1.0, sample_rate=1024, std=0.1)
>>> sweep = chirp(duration=1.0, sample_rate=1024, f0=10, f1=100)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u
from scipy import signal as scipy_signal

if TYPE_CHECKING:
    from numpy.random import Generator

    from ..frequencyseries import FrequencySeries
    from ..timeseries import TimeSeries


def _make_timeseries(
    data: np.ndarray,
    sample_rate: float,
    t0: float = 0.0,
    unit: Any = None,
    name: str | None = None,
    channel: str | None = None,
) -> TimeSeries:
    """Helper to create a TimeSeries from data."""
    from ..timeseries import TimeSeries

    return TimeSeries(
        data,
        sample_rate=sample_rate,
        t0=t0,
        unit=unit,
        name=name,
        channel=channel,
    )


def _get_times(duration: float, sample_rate: float) -> np.ndarray:
    """Generate time array for given duration and sample rate."""
    n_samples = int(duration * sample_rate)
    return np.arange(n_samples) / sample_rate


# =============================================================================
# Noise Generators
# =============================================================================


def gaussian(
    duration: float,
    sample_rate: float,
    std: float = 1.0,
    mean: float = 0.0,
    t0: float = 0.0,
    rng: Generator | None = None,
    seed: int | None = None,
    unit: Any = None,
    name: str | None = None,
    channel: str | None = None,
) -> TimeSeries:
    """
    Generate Gaussian (normal) white noise.

    Parameters
    ----------
    duration : float
        Duration of the output in seconds.
    sample_rate : float
        Sample rate in Hz.
    std : float, optional
        Standard deviation of the noise. Default is 1.0.
    mean : float, optional
        Mean of the noise. Default is 0.0.
    t0 : float, optional
        Start time. Default is 0.0.
    rng : numpy.random.Generator, optional
        Random number generator. If None, creates a new one.
    unit : astropy.units.Unit, optional
        Unit of the output.
    name : str, optional
        Name for the TimeSeries.
    channel : str, optional
        Channel name.

    Returns
    -------
    TimeSeries
        Gaussian noise time-series.

    Examples
    --------
    >>> from gwexpy.noise.wave import gaussian
    >>> noise = gaussian(duration=1.0, sample_rate=1024, std=0.1)
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    n_samples = int(duration * sample_rate)
    data = rng.normal(loc=mean, scale=std, size=n_samples)

    return _make_timeseries(data, sample_rate, t0, unit, name, channel)


def uniform(
    duration: float,
    sample_rate: float,
    low: float = -1.0,
    high: float = 1.0,
    t0: float = 0.0,
    rng: Generator | None = None,
    seed: int | None = None,
    unit: Any = None,
    name: str | None = None,
    channel: str | None = None,
) -> TimeSeries:
    """
    Generate uniform white noise.

    Parameters
    ----------
    duration : float
        Duration of the output in seconds.
    sample_rate : float
        Sample rate in Hz.
    low : float, optional
        Lower bound of the uniform distribution. Default is -1.0.
    high : float, optional
        Upper bound of the uniform distribution. Default is 1.0.
    t0 : float, optional
        Start time. Default is 0.0.
    rng : numpy.random.Generator, optional
        Random number generator.
    unit : astropy.units.Unit, optional
        Unit of the output.
    name : str, optional
        Name for the TimeSeries.
    channel : str, optional
        Channel name.

    Returns
    -------
    TimeSeries
        Uniform noise time-series.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    n_samples = int(duration * sample_rate)
    data = rng.uniform(low=low, high=high, size=n_samples)

    return _make_timeseries(data, sample_rate, t0, unit, name, channel)


def colored(
    duration: float,
    sample_rate: float,
    exponent: float,
    amplitude: float = 1.0,
    f_ref: float = 1.0,
    t0: float = 0.0,
    rng: Generator | None = None,
    seed: int | None = None,
    unit: Any = None,
    name: str | None = None,
    channel: str | None = None,
) -> TimeSeries:
    """
    Generate colored (power-law) noise.

    The noise follows a power-law spectrum: ASD(f) ~ f^(-exponent).

    Parameters
    ----------
    duration : float
        Duration of the output in seconds.
    sample_rate : float
        Sample rate in Hz.
    exponent : float
        Power-law exponent in ASD. Common values:
        - 0: white noise
        - 0.5: pink noise (1/f^0.5)
        - 1: red/Brownian noise (1/f)
    amplitude : float, optional
        Amplitude at the reference frequency. Default is 1.0.
    f_ref : float, optional
        Reference frequency in Hz. Default is 1.0.
    t0 : float, optional
        Start time. Default is 0.0.
    rng : numpy.random.Generator, optional
        Random number generator.
    unit : astropy.units.Unit, optional
        Unit of the output.
    name : str, optional
        Name for the TimeSeries.
    channel : str, optional
        Channel name.

    Returns
    -------
    TimeSeries
        Colored noise time-series.

    Examples
    --------
    >>> from gwexpy.noise.wave import colored
    >>> pink = colored(duration=1.0, sample_rate=1024, exponent=0.5)
    """
    from .asd import power_law

    if rng is None:
        rng = np.random.default_rng(seed)

    n_samples = int(duration * sample_rate)
    fft_freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

    # Generate power-law ASD
    asd = power_law(
        exponent=exponent,
        amplitude=amplitude,
        f_ref=f_ref,
        frequencies=fft_freqs,
    )

    # Handle DC component
    asd_vals = asd.value.copy()
    if fft_freqs[0] == 0:
        asd_vals[0] = asd_vals[1] if len(asd_vals) > 1 else amplitude

    # Convert ASD to FFT amplitudes
    psd = asd_vals**2
    amplitudes = np.sqrt(psd * n_samples * sample_rate / 2)

    # Generate random phases
    phases = rng.uniform(0, 2 * np.pi, size=len(fft_freqs))
    fft_coeffs = amplitudes * np.exp(1j * phases)

    # DC and Nyquist must be real
    fft_coeffs[0] = np.real(fft_coeffs[0])
    if n_samples % 2 == 0:
        fft_coeffs[-1] = np.real(fft_coeffs[-1])

    # Inverse FFT
    data = np.fft.irfft(fft_coeffs, n=n_samples)

    return _make_timeseries(data, sample_rate, t0, unit, name, channel)


def white_noise(
    duration: float,
    sample_rate: float,
    amplitude: float = 1.0,
    **kwargs: Any,
) -> TimeSeries:
    """
    Generate white noise (flat spectrum).

    Shortcut for colored(exponent=0).

    Parameters
    ----------
    duration : float
        Duration in seconds.
    sample_rate : float
        Sample rate in Hz.
    amplitude : float, optional
        Noise amplitude. Default is 1.0.
    **kwargs
        Additional arguments passed to colored().

    Returns
    -------
    TimeSeries
        White noise time-series.
    """
    return colored(duration, sample_rate, exponent=0.0, amplitude=amplitude, **kwargs)


def pink_noise(
    duration: float,
    sample_rate: float,
    amplitude: float = 1.0,
    **kwargs: Any,
) -> TimeSeries:
    """
    Generate pink noise (1/f^0.5 spectrum).

    Shortcut for colored(exponent=0.5).

    Parameters
    ----------
    duration : float
        Duration in seconds.
    sample_rate : float
        Sample rate in Hz.
    amplitude : float, optional
        Noise amplitude at reference frequency. Default is 1.0.
    **kwargs
        Additional arguments passed to colored().

    Returns
    -------
    TimeSeries
        Pink noise time-series.
    """
    return colored(duration, sample_rate, exponent=0.5, amplitude=amplitude, **kwargs)


def red_noise(
    duration: float,
    sample_rate: float,
    amplitude: float = 1.0,
    **kwargs: Any,
) -> TimeSeries:
    """
    Generate red/Brownian noise (1/f spectrum).

    Shortcut for colored(exponent=1.0).

    Parameters
    ----------
    duration : float
        Duration in seconds.
    sample_rate : float
        Sample rate in Hz.
    amplitude : float, optional
        Noise amplitude at reference frequency. Default is 1.0.
    **kwargs
        Additional arguments passed to colored().

    Returns
    -------
    TimeSeries
        Red noise time-series.
    """
    return colored(duration, sample_rate, exponent=1.0, amplitude=amplitude, **kwargs)


# =============================================================================
# Periodic Waveforms
# =============================================================================


def sine(
    duration: float,
    sample_rate: float,
    frequency: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
    t0: float = 0.0,
    unit: Any = None,
    name: str | None = None,
    channel: str | None = None,
) -> TimeSeries:
    """
    Generate a sine wave.

    Parameters
    ----------
    duration : float
        Duration in seconds.
    sample_rate : float
        Sample rate in Hz.
    frequency : float
        Frequency in Hz.
    amplitude : float, optional
        Amplitude. Default is 1.0.
    phase : float, optional
        Initial phase in radians. Default is 0.0.
    t0 : float, optional
        Start time. Default is 0.0.
    unit : astropy.units.Unit, optional
        Unit of the output.
    name : str, optional
        Name for the TimeSeries.
    channel : str, optional
        Channel name.

    Returns
    -------
    TimeSeries
        Sine wave time-series.

    Examples
    --------
    >>> from gwexpy.noise.wave import sine
    >>> wave = sine(duration=1.0, sample_rate=1024, frequency=10.0)
    """
    t = _get_times(duration, sample_rate)
    data = amplitude * np.sin(2 * np.pi * frequency * t + phase)

    return _make_timeseries(data, sample_rate, t0, unit, name, channel)


def square(
    duration: float,
    sample_rate: float,
    frequency: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
    duty: float = 0.5,
    t0: float = 0.0,
    unit: Any = None,
    name: str | None = None,
    channel: str | None = None,
) -> TimeSeries:
    """
    Generate a square wave.

    Parameters
    ----------
    duration : float
        Duration in seconds.
    sample_rate : float
        Sample rate in Hz.
    frequency : float
        Frequency in Hz.
    amplitude : float, optional
        Amplitude. Default is 1.0.
    phase : float, optional
        Initial phase in radians. Default is 0.0.
    duty : float, optional
        Duty cycle (0 to 1). Default is 0.5.
    t0 : float, optional
        Start time. Default is 0.0.
    unit : astropy.units.Unit, optional
        Unit of the output.
    name : str, optional
        Name for the TimeSeries.
    channel : str, optional
        Channel name.

    Returns
    -------
    TimeSeries
        Square wave time-series.
    """
    t = _get_times(duration, sample_rate)
    data = amplitude * scipy_signal.square(2 * np.pi * frequency * t + phase, duty=duty)

    return _make_timeseries(data, sample_rate, t0, unit, name, channel)


def sawtooth(
    duration: float,
    sample_rate: float,
    frequency: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
    width: float = 1.0,
    t0: float = 0.0,
    unit: Any = None,
    name: str | None = None,
    channel: str | None = None,
) -> TimeSeries:
    """
    Generate a sawtooth wave.

    Parameters
    ----------
    duration : float
        Duration in seconds.
    sample_rate : float
        Sample rate in Hz.
    frequency : float
        Frequency in Hz.
    amplitude : float, optional
        Amplitude. Default is 1.0.
    phase : float, optional
        Initial phase in radians. Default is 0.0.
    width : float, optional
        Width parameter (0 to 1). 1 gives rising sawtooth, 0 gives falling.
        Default is 1.0.
    t0 : float, optional
        Start time. Default is 0.0.
    unit : astropy.units.Unit, optional
        Unit of the output.
    name : str, optional
        Name for the TimeSeries.
    channel : str, optional
        Channel name.

    Returns
    -------
    TimeSeries
        Sawtooth wave time-series.
    """
    t = _get_times(duration, sample_rate)
    data = amplitude * scipy_signal.sawtooth(
        2 * np.pi * frequency * t + phase, width=width
    )

    return _make_timeseries(data, sample_rate, t0, unit, name, channel)


def triangle(
    duration: float,
    sample_rate: float,
    frequency: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
    t0: float = 0.0,
    unit: Any = None,
    name: str | None = None,
    channel: str | None = None,
) -> TimeSeries:
    """
    Generate a triangle wave.

    This is a sawtooth with width=0.5.

    Parameters
    ----------
    duration : float
        Duration in seconds.
    sample_rate : float
        Sample rate in Hz.
    frequency : float
        Frequency in Hz.
    amplitude : float, optional
        Amplitude. Default is 1.0.
    phase : float, optional
        Initial phase in radians. Default is 0.0.
    t0 : float, optional
        Start time. Default is 0.0.
    unit : astropy.units.Unit, optional
        Unit of the output.
    name : str, optional
        Name for the TimeSeries.
    channel : str, optional
        Channel name.

    Returns
    -------
    TimeSeries
        Triangle wave time-series.
    """
    return sawtooth(
        duration,
        sample_rate,
        frequency,
        amplitude=amplitude,
        phase=phase,
        width=0.5,
        t0=t0,
        unit=unit,
        name=name,
        channel=channel,
    )


def chirp(
    duration: float,
    sample_rate: float,
    f0: float,
    f1: float,
    t1: float | None = None,
    method: str = "linear",
    amplitude: float = 1.0,
    phase: float = 0.0,
    t0: float = 0.0,
    unit: Any = None,
    name: str | None = None,
    channel: str | None = None,
) -> TimeSeries:
    """
    Generate a swept-frequency cosine (chirp) signal.

    Parameters
    ----------
    duration : float
        Duration in seconds.
    sample_rate : float
        Sample rate in Hz.
    f0 : float
        Initial frequency in Hz.
    f1 : float
        Final frequency in Hz (at time t1).
    t1 : float, optional
        Time at which f1 is reached. Default is duration.
    method : str, optional
        Frequency sweep method: 'linear', 'quadratic', 'logarithmic', 'hyperbolic'.
        Default is 'linear'.
    amplitude : float, optional
        Amplitude. Default is 1.0.
    phase : float, optional
        Initial phase in radians. Default is 0.0.
    t0 : float, optional
        Start time. Default is 0.0.
    unit : astropy.units.Unit, optional
        Unit of the output.
    name : str, optional
        Name for the TimeSeries.
    channel : str, optional
        Channel name.

    Returns
    -------
    TimeSeries
        Chirp signal time-series.

    Examples
    --------
    >>> from gwexpy.noise.wave import chirp
    >>> sweep = chirp(duration=1.0, sample_rate=1024, f0=10, f1=100)
    """
    if t1 is None:
        t1 = duration

    t = _get_times(duration, sample_rate)
    data = amplitude * scipy_signal.chirp(
        t, f0, t1, f1, method=method, phi=np.degrees(phase)
    )

    return _make_timeseries(data, sample_rate, t0, unit, name, channel)


# =============================================================================
# Transient Signals
# =============================================================================


def step(
    duration: float,
    sample_rate: float,
    t_step: float = 0.0,
    amplitude: float = 1.0,
    t0: float = 0.0,
    unit: Any = None,
    name: str | None = None,
    channel: str | None = None,
) -> TimeSeries:
    """
    Generate a step (Heaviside) function.

    Parameters
    ----------
    duration : float
        Duration in seconds.
    sample_rate : float
        Sample rate in Hz.
    t_step : float, optional
        Time of the step (relative to t0). Default is 0.0.
    amplitude : float, optional
        Amplitude of the step. Default is 1.0.
    t0 : float, optional
        Start time. Default is 0.0.
    unit : astropy.units.Unit, optional
        Unit of the output.
    name : str, optional
        Name for the TimeSeries.
    channel : str, optional
        Channel name.

    Returns
    -------
    TimeSeries
        Step function time-series.

    Examples
    --------
    >>> from gwexpy.noise.wave import step
    >>> s = step(duration=1.0, sample_rate=1024, t_step=0.5)
    """
    t = _get_times(duration, sample_rate)
    data = np.where(t >= t_step, amplitude, 0.0)

    return _make_timeseries(data, sample_rate, t0, unit, name, channel)


def impulse(
    duration: float,
    sample_rate: float,
    t_impulse: float = 0.0,
    amplitude: float = 1.0,
    t0: float = 0.0,
    unit: Any = None,
    name: str | None = None,
    channel: str | None = None,
) -> TimeSeries:
    """
    Generate an impulse (delta-like) signal.

    The impulse is placed at the sample closest to t_impulse.

    Parameters
    ----------
    duration : float
        Duration in seconds.
    sample_rate : float
        Sample rate in Hz.
    t_impulse : float, optional
        Time of the impulse (relative to t0). Default is 0.0.
    amplitude : float, optional
        Amplitude of the impulse. Default is 1.0.
    t0 : float, optional
        Start time. Default is 0.0.
    unit : astropy.units.Unit, optional
        Unit of the output.
    name : str, optional
        Name for the TimeSeries.
    channel : str, optional
        Channel name.

    Returns
    -------
    TimeSeries
        Impulse signal time-series.

    Examples
    --------
    >>> from gwexpy.noise.wave import impulse
    >>> i = impulse(duration=1.0, sample_rate=1024, t_impulse=0.5)
    """
    n_samples = int(duration * sample_rate)
    data = np.zeros(n_samples)

    # Find nearest sample
    sample_idx = int(round(t_impulse * sample_rate))
    if 0 <= sample_idx < n_samples:
        data[sample_idx] = amplitude

    return _make_timeseries(data, sample_rate, t0, unit, name, channel)


def exponential(
    duration: float,
    sample_rate: float,
    tau: float,
    amplitude: float = 1.0,
    t_start: float = 0.0,
    decay: bool = True,
    t0: float = 0.0,
    unit: Any = None,
    name: str | None = None,
    channel: str | None = None,
) -> TimeSeries:
    """
    Generate an exponential signal.

    Parameters
    ----------
    duration : float
        Duration in seconds.
    sample_rate : float
        Sample rate in Hz.
    tau : float
        Time constant in seconds.
    amplitude : float, optional
        Initial amplitude. Default is 1.0.
    t_start : float, optional
        Start time of the exponential (relative to t0). Default is 0.0.
    decay : bool, optional
        If True, decay (exp(-t/tau)). If False, growth (exp(+t/tau)).
        Default is True.
    t0 : float, optional
        Start time. Default is 0.0.
    unit : astropy.units.Unit, optional
        Unit of the output.
    name : str, optional
        Name for the TimeSeries.
    channel : str, optional
        Channel name.

    Returns
    -------
    TimeSeries
        Exponential signal time-series.

    Examples
    --------
    >>> from gwexpy.noise.wave import exponential
    >>> decay_signal = exponential(duration=1.0, sample_rate=1024, tau=0.2)
    """
    t = _get_times(duration, sample_rate)
    t_rel = t - t_start

    if decay:
        data = np.where(t_rel >= 0, amplitude * np.exp(-t_rel / tau), 0.0)
    else:
        data = np.where(t_rel >= 0, amplitude * np.exp(t_rel / tau), 0.0)

    return _make_timeseries(data, sample_rate, t0, unit, name, channel)


# =============================================================================
# ASD-based waveform generation (original function)
# =============================================================================


def from_asd(
    asd: FrequencySeries,
    duration: float,
    sample_rate: float,
    t0: float = 0.0,
    rng: Generator | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> TimeSeries:
    """
    Generate colored noise TimeSeries from an ASD (Amplitude Spectral Density).

    Uses FFT-based colored noise synthesis to produce a time-series with the
    spectral characteristics defined by the input ASD.

    Parameters
    ----------
    asd : FrequencySeries
        The amplitude spectral density defining the noise spectrum.
    duration : float
        Duration of the output time-series in seconds.
    sample_rate : float
        Sample rate of the output time-series in Hz.
    t0 : float, optional
        Start time of the output TimeSeries.
    rng : numpy.random.Generator, optional
        Random number generator instance. If None, a new default generator
        is created.
    **kwargs
        Additional arguments passed to TimeSeries constructor (e.g., name, channel, unit).
        If provided, these override values derived from the ASD.

    Returns
    -------
    TimeSeries
        The generated noise time-series as a gwexpy TimeSeries.

    Examples
    --------
    >>> from gwexpy.noise.asd import from_pygwinc
    >>> from gwexpy.noise.wave import from_asd
    >>> asd = from_pygwinc('aLIGO', fmin=4.0, fmax=1024.0, df=0.01)
    >>> noise = from_asd(asd, duration=128, sample_rate=2048, t0=0)
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    n_samples = int(duration * sample_rate)
    fft_freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)

    # Interpolate ASD to required frequency bins
    asd_interp = np.interp(
        fft_freqs,
        asd.frequencies.value,
        asd.value,
        left=asd.value[0],
        right=asd.value[-1],
    )

    # Convert ASD to PSD and scale to FFT amplitudes
    # P(f) = ASD^2, FFT amplitude = sqrt(P(f) * N * sample_rate / 2)
    psd = asd_interp**2
    amplitudes = np.sqrt(psd * n_samples * sample_rate / 2)

    # Generate complex FFT coefficients with random phases
    phases = rng.uniform(0, 2 * np.pi, size=len(fft_freqs))
    fft_coeffs = amplitudes * np.exp(1j * phases)

    # DC and Nyquist components must be real
    fft_coeffs[0] = np.real(fft_coeffs[0])
    if n_samples % 2 == 0:
        fft_coeffs[-1] = np.real(fft_coeffs[-1])

    # Inverse FFT to get time-series
    noise = np.fft.irfft(fft_coeffs, n=n_samples)

    # Determine metadata (kwargs override derived values)
    if "unit" not in kwargs:
        unit = getattr(asd, "unit", None)
        ts_unit = None
        if unit is not None:
            try:
                unit = u.Unit(unit)
                ts_unit = unit * (u.Hz**0.5)
            except (TypeError, ValueError):
                ts_unit = unit
        kwargs["unit"] = ts_unit

    if "name" not in kwargs:
        kwargs["name"] = getattr(asd, "name", None)

    if "channel" not in kwargs:
        kwargs["channel"] = getattr(asd, "channel", None)

    return _make_timeseries(noise, sample_rate, t0, **kwargs)


__all__ = [
    # ASD-based
    "from_asd",
    # Noise generators
    "gaussian",
    "uniform",
    "colored",
    "white_noise",
    "pink_noise",
    "red_noise",
    # Periodic waveforms
    "sine",
    "square",
    "sawtooth",
    "triangle",
    "chirp",
    # Transient signals
    "step",
    "impulse",
    "exponential",
]
