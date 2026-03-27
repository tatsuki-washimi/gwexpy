"""gwexpy.noise.non_gaussian - Non-Gaussian noise simulators."""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u
from scipy import signal

from ..timeseries import TimeSeries


def transient_gaussian_noise(
    duration: float | u.Quantity,
    sample_rate: float | u.Quantity,
    A1: float,
    psd: Any | None = None,
    **kwargs: Any,
) -> TimeSeries:
    """
    Generate Model I non-Gaussian noise: Superposition of transient Gaussian noise.
    x(t) = n0(t) + A1 * B(t) * n1(t)

    Parameters
    ----------
    duration : float, u.Quantity
        Duration of the noise in seconds.
    sample_rate : float, u.Quantity
        Sample rate in Hz.
    A1 : float
        Amplitude scaling factor for the transient part.
    psd : FrequencySeries, optional
        PSD to color the noise. If None, white noise is used.
    **kwargs
        Additional arguments for TimeSeries.

    Returns
    -------
    TimeSeries
    """
    if isinstance(duration, u.Quantity):
        duration = duration.to("s").value
    if isinstance(sample_rate, u.Quantity):
        sample_rate = sample_rate.to("Hz").value

    n_samples = int(duration * sample_rate)
    dt = 1.0 / sample_rate

    # Generate n0(t) and n1(t)
    # For now, we use white noise if psd is not provided.
    # In the future, we can use colored noise.
    n0 = np.random.randn(n_samples)
    n1 = np.random.randn(n_samples)

    if psd is not None:
        # TODO: Implement coloring if PSD is provided
        # For now, just a placeholder as the simple model often uses white noise
        pass

    # B(t): Tukey window (alpha=0.5), random start, duration T/6
    transient_len = n_samples // 6
    if transient_len > 0:
        win = signal.windows.tukey(transient_len, alpha=0.5)
        start_idx = np.random.randint(0, n_samples - transient_len)
        B = np.zeros(n_samples)
        B[start_idx : start_idx + transient_len] = win
    else:
        B = np.zeros(n_samples)

    data = n0 + A1 * B * n1
    return TimeSeries(data, sample_rate=sample_rate, **kwargs)


def scatter_light_noise(
    duration: float | u.Quantity,
    sample_rate: float | u.Quantity,
    A2: float,
    f_sc: float = 0.2,
    G: float = 3e-22,
    lambda_val: float = 1064e-9,
    x0: float = 1.0,
    f_amp: float = 0.1,
    **kwargs: Any,
) -> TimeSeries:
    """
    Generate Model II non-Gaussian noise: Scattered light noise.
    x(t) = n0(t) + G * sin(4pi/lambda * (x0 + delta_x_sc(t)))
    delta_x_sc = A2 * (1 + 0.25 * sin(2pi * f_amp * t)) * cos(2pi * f_sc * t)

    Parameters
    ----------
    duration : float, u.Quantity
    sample_rate : float, u.Quantity
    A2 : float
        Amplitude of the scattering motion.
    f_sc : float, default=0.2
        Frequency of the scattering motion in Hz.
    G : float, default=3e-22
        Overall amplitude scaling.
    lambda_val : float, default=1064e-9
        Wavelength in meters.
    x0 : float, default=1.0
        Static offset in meters.
    f_amp : float, default=0.1
        Frequency of the amplitude modulation in Hz.
    **kwargs
        Additional arguments for TimeSeries.

    Returns
    -------
    TimeSeries
    """
    if isinstance(duration, u.Quantity):
        duration = duration.to("s").value
    if isinstance(sample_rate, u.Quantity):
        sample_rate = sample_rate.to("Hz").value

    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate

    n0 = np.random.randn(n_samples)

    delta_x_sc = A2 * (1 + 0.25 * np.sin(2 * np.pi * f_amp * t)) * np.cos(2 * np.pi * f_sc * t)
    phase = (4 * np.pi / lambda_val) * (x0 + delta_x_sc)
    glitch = G * np.sin(phase)

    data = n0 + glitch
    return TimeSeries(data, sample_rate=sample_rate, **kwargs)


def inject_noise(clean_ts: TimeSeries, noise_ts: TimeSeries) -> TimeSeries:
    """
    Inject noise into a clean time series.
    """
    return clean_ts + noise_ts
