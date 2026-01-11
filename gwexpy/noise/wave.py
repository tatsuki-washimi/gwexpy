"""gwexpy.noise.wavegen - Generate time-series waveforms from spectral data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np

if TYPE_CHECKING:
    from numpy.random import Generator

    from ..frequencyseries import FrequencySeries


def from_asd(
    asd: "FrequencySeries",
    duration: float,
    sample_rate: float,
    rng: Union["Generator", None] = None,
) -> np.ndarray:
    """
    Generate colored noise time-series from an ASD (Amplitude Spectral Density).

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
    rng : numpy.random.Generator, optional
        Random number generator instance. If None, a new default generator
        is created.

    Returns
    -------
    np.ndarray
        The generated noise time-series as a numpy array.

    Examples
    --------
    >>> from gwexpy.noise import from_pygwinc, from_asd
    >>> asd = from_pygwinc('aLIGO', fmin=4.0, fmax=1024.0, df=0.01)
    >>> noise = from_asd(asd, duration=128, sample_rate=2048)
    """
    if rng is None:
        rng = np.random.default_rng()

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

    return noise
