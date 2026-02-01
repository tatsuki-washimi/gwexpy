"""gwexpy.noise.peaks - Generic peak generation functions."""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u
from scipy.special import wofz

from ..frequencyseries import FrequencySeries


def _get_freqs(frequencies: np.ndarray | None, kwargs: dict[str, Any]) -> np.ndarray:
    """Helper to determine frequency array from args or kwargs."""
    if frequencies is not None:
        if isinstance(frequencies, u.Quantity):
            return frequencies.to("Hz").value
        return np.asarray(frequencies)

    fmin = kwargs.get("fmin", 0.1)
    fmax = kwargs.get("fmax", 8192.0)
    df = kwargs.get("df", 1.0)
    return np.arange(fmin, fmax + df, df)


def lorentzian_line(
    f0: float,
    amplitude: float | u.Quantity,
    Q: float | None = None,
    gamma: float | None = None,
    frequencies: np.ndarray | None = None,
    **kwargs: Any,
) -> FrequencySeries:
    """
    Generate a Lorentzian peak (ASD form, Peak Normalization).

    Formula: ASD(f) = A * gamma / sqrt( (f-f0)^2 + gamma^2 )

    This implementation uses **Peak Normalization** where the profile peaks
    exactly at `amplitude` when f = f0. This differs from the physics
    convention of Area Normalization (where the integral is A).

    This form is suitable for modeling spectral lines in ASD (Amplitude
    Spectral Density) where the peak height is known.

    Parameters
    ----------
    f0 : float
        Center frequency.
    amplitude : float or Quantity
        Peak ASD value.
    Q : float, optional
        Quality factor. gamma = f0 / (2*Q). Either Q or gamma must be provided.
    gamma : float, optional
        HWHM (Half Width at Half Maximum).
    frequencies : array-like, optional
        Target frequency array.
    """
    f_vals = _get_freqs(frequencies, kwargs)

    if Q is not None:
        gamma_val = f0 / (2 * Q)
    elif gamma is not None:
        gamma_val = gamma
    else:
        raise ValueError("Either 'Q' or 'gamma' must be provided.")

    # Unit handling
    target_unit = kwargs.get("unit", None)
    if isinstance(amplitude, u.Quantity):
        amp_val = amplitude.value
        if target_unit is None:
            target_unit = amplitude.unit
    else:
        amp_val = float(amplitude)

    # Lorentzian calculation (ASD)
    # L(f) = gamma / sqrt((f-f0)^2 + gamma^2) -> Peak is 1 at f=f0
    denom = np.sqrt((f_vals - f0) ** 2 + gamma_val**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        shape = gamma_val / denom
        # Handle cases where peak might be undefined or calculation unstable (though standard form is fine)

    data = amp_val * shape

    # If kwargs contains generation params that FrequencySeries doesn't need if we pass data/frequencies
    # we filter them out or let FrequencySeries ignore? FrequencySeries works with data+frequencies.
    # Note: If frequencies was None, we generated f_vals. We must pass this generated array.

    # Remove 'unit' from kwargs if present to avoid multiple values error
    kwargs.pop("unit", None)

    return FrequencySeries(data, frequencies=f_vals, unit=target_unit, **kwargs)


def gaussian_line(
    f0: float,
    amplitude: float | u.Quantity,
    sigma: float,
    frequencies: np.ndarray | None = None,
    **kwargs: Any,
) -> FrequencySeries:
    """
    Generate a Gaussian peak (ASD).
    Formula: ASD(f) = A * exp( - (f-f0)^2 / (2*sigma^2) )
    """
    f_vals = _get_freqs(frequencies, kwargs)

    target_unit = kwargs.get("unit", None)
    if isinstance(amplitude, u.Quantity):
        amp_val = amplitude.value
        if target_unit is None:
            target_unit = amplitude.unit
    else:
        amp_val = float(amplitude)

    data = amp_val * np.exp(-((f_vals - f0) ** 2) / (2 * sigma**2))

    kwargs.pop("unit", None)
    return FrequencySeries(data, frequencies=f_vals, unit=target_unit, **kwargs)


def voigt_line(
    f0: float,
    amplitude: float | u.Quantity,
    sigma: float,
    gamma: float,
    frequencies: np.ndarray | None = None,
    **kwargs: Any,
) -> FrequencySeries:
    """
    Generate a Voigt peak (ASD) using Faddeeva function.
    Normalized to have peak value = 'amplitude'.
    """
    f_vals = _get_freqs(frequencies, kwargs)

    target_unit = kwargs.get("unit", None)
    if isinstance(amplitude, u.Quantity):
        amp_val = amplitude.value
        if target_unit is None:
            target_unit = amplitude.unit
    else:
        amp_val = float(amplitude)

    # Calculate Voigt profile
    z = ((f_vals - f0) + 1j * gamma) / (sigma * np.sqrt(2))
    v = wofz(z).real

    # We need to normalize so peak is 'amp_val'.
    # Peak of unnormalized wofz for centered Voigt is roughly wofz(1j * gamma / (sigma*sqrt(2))).real
    # Let's calculate the value at f0 explicitly to normalize
    z0 = (1j * gamma) / (sigma * np.sqrt(2))
    peak_factor = wofz(z0).real

    data = amp_val * (v / peak_factor)

    kwargs.pop("unit", None)
    return FrequencySeries(data, frequencies=f_vals, unit=target_unit, **kwargs)
