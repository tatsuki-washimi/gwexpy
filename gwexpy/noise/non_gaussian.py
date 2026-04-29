"""gwexpy.noise.non_gaussian - Non-Gaussian noise simulators."""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u

try:
    from scipy import signal
except ImportError as _exc:
    raise ImportError(
        "scipy is required for gwexpy.noise. Install with: pip install scipy"
    ) from _exc

from ..timeseries import TimeSeries
from .wave import from_asd


def _psd_to_asd(psd: Any) -> Any:
    """Return an ASD FrequencySeries derived from a PSD FrequencySeries."""
    from ..frequencyseries import FrequencySeries

    psd_values = np.asarray(psd.value, dtype=float)
    psd_freqs = np.asarray(psd.frequencies.to_value(u.Hz), dtype=float)
    if psd_values.ndim != 1 or psd_freqs.ndim != 1:
        raise ValueError("PSD values and frequencies must be one-dimensional")
    if psd_values.size == 0 or psd_values.size != psd_freqs.size:
        raise ValueError("PSD values and frequencies must be non-empty and aligned")
    if np.any(~np.isfinite(psd_freqs)) or np.any(psd_freqs < 0.0):
        raise ValueError("PSD frequencies must be finite and non-negative")
    if np.any(np.diff(psd_freqs) <= 0.0):
        raise ValueError("PSD frequencies must be strictly increasing")
    if np.any(~np.isfinite(psd_values)) or np.any(psd_values < 0.0):
        raise ValueError("PSD values must be finite and non-negative")

    psd_unit = getattr(psd, "unit", None)
    asd_unit = None
    if psd_unit is not None:
        asd_unit = u.Unit(psd_unit) ** 0.5

    return FrequencySeries(
        np.sqrt(psd_values),
        frequencies=psd.frequencies,
        unit=asd_unit,
        name=getattr(psd, "name", None),
        channel=getattr(psd, "channel", None),
    )


def _normal_pair(
    n_samples: int,
    rng: Any | None,
) -> tuple[np.ndarray, np.ndarray]:
    if rng is not None:
        return rng.normal(size=n_samples), rng.normal(size=n_samples)
    return np.random.randn(n_samples), np.random.randn(n_samples)


def _window_start(n_samples: int, transient_len: int, rng: Any | None) -> int:
    high = n_samples - transient_len
    if rng is not None:
        return int(rng.integers(0, high))
    return int(np.random.randint(0, high))


def transient_gaussian_noise(
    duration: float | u.Quantity,
    sample_rate: float | u.Quantity,
    A1: float,
    psd: Any | None = None,
    rng: Any | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> TimeSeries:
    """Generate Model I non-Gaussian noise: Superposition of transient Gaussian noise.

    x(t) = n0(t) + A1 * B(t) * n1(t).

    Parameters
    ----------
    duration : float, u.Quantity
        Duration of the noise in seconds.
    sample_rate : float, u.Quantity
        Sample rate in Hz.
    A1 : float
        Amplitude scaling factor for the transient part.
    psd : FrequencySeries, optional
        One-sided PSD used to color both Gaussian components. If None, white
        noise is used.
    rng : numpy.random.Generator, optional
        Random number generator. If None and ``seed`` is not provided, the
        legacy global NumPy RNG is used for the white-noise path.
    seed : int, optional
        Seed used to initialize a default random generator.
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
    if rng is None and seed is not None:
        rng = np.random.default_rng(seed)
    if psd is not None:
        asd = _psd_to_asd(psd)
        n0 = from_asd(asd, float(duration), float(sample_rate), rng=rng, **kwargs).value
        n1 = from_asd(asd, float(duration), float(sample_rate), rng=rng, **kwargs).value
    else:
        n0, n1 = _normal_pair(n_samples, rng)

    # B(t): Tukey window (alpha=0.5), random start, duration T/6
    transient_len = n_samples // 6
    if transient_len > 0:
        win = signal.windows.tukey(transient_len, alpha=0.5)
        start_idx = _window_start(n_samples, transient_len, rng)
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
    """Generate Model II non-Gaussian noise: Scattered light noise.

    x(t) = n0(t) + G * sin(4pi/lambda * (x0 + delta_x_sc(t)))
    delta_x_sc = A2 * (1 + 0.25 * sin(2pi * f_amp * t)) * cos(2pi * f_sc * t).

    Parameters
    ----------
    duration : float, u.Quantity
        Duration of the noise in seconds.
    sample_rate : float, u.Quantity
        Sample rate in Hz.
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

    delta_x_sc = (
        A2 * (1 + 0.25 * np.sin(2 * np.pi * f_amp * t)) * np.cos(2 * np.pi * f_sc * t)
    )
    phase = (4 * np.pi / lambda_val) * (x0 + delta_x_sc)
    glitch = G * np.sin(phase)

    data = n0 + glitch
    return TimeSeries(data, sample_rate=sample_rate, **kwargs)


def inject_noise(clean_ts: TimeSeries, noise_ts: TimeSeries) -> TimeSeries:
    """Inject `noise_ts` into `clean_ts` and return the noisy TimeSeries.

    The function returns the pointwise sum of the clean time series and the
    noise time series. Both series must be aligned in time and have compatible
    sample rates. This convenience wrapper preserves metadata where possible.

    Parameters
    ----------
    clean_ts : TimeSeries
        Clean (signal-only) time series that will receive the noise. The series
        may have `name`, `unit`, and time span attributes.
    noise_ts : TimeSeries
        Noise time series to add to `clean_ts`. Should be aligned with
        `clean_ts` (same sample rate and overlapping span). If sample rates
        differ, users should resample externally prior to calling this function.

    Returns
    -------
    TimeSeries
        New TimeSeries equal to `clean_ts + noise_ts`. Metadata from `clean_ts`
        are preserved where possible; `unit` will follow arithmetic rules
        (e.g., addition requires compatible units).

    Raises
    ------
    ValueError
        If the time alignment or sample rates are incompatible (unless the
        underlying TimeSeries implementation handles resampling).

    Examples
    --------
    >>> from gwexpy import TimeSeries
    >>> import numpy as np
    >>> t = np.linspace(0, 1, 100)
    >>> clean_ts = TimeSeries(np.sin(2 * np.pi * 10 * t), t0=0, sample_rate=100)
    >>> noise_ts = TimeSeries(np.random.randn(100) * 0.1, t0=0, sample_rate=100)
    >>> noisy = inject_noise(clean_ts, noise_ts)
    >>> isinstance(noisy, TimeSeries)
    True

    """
    return clean_ts + noise_ts
