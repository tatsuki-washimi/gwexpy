"""gwexpy.noise.colored - Power-law noise generation."""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u

from ..frequencyseries import FrequencySeries


def power_law(
    exponent: float,
    amplitude: float | u.Quantity = 1.0,
    f_ref: float | u.Quantity = 1.0,
    frequencies: np.ndarray | None = None,
    **kwargs: Any,
) -> FrequencySeries:
    """Generate an ASD FrequencySeries following a power law.

    ASD(f) = amplitude * (f / f_ref) ** (-exponent).
    """
    if frequencies is None:
        if "df" in kwargs and (
            "N" in kwargs or "f0" in kwargs or ("fmin" in kwargs and "fmax" in kwargs)
        ):
            fmin = kwargs.pop("fmin", 0.0)
            fmax = kwargs.pop("fmax", 100.0)
            df = kwargs.pop("df", 1.0)
            if "frequencies" not in kwargs:
                frequencies = np.arange(fmin, fmax + df, df)
        else:
            raise ValueError(
                "The 'frequencies' argument is required to evaluate the power law."
            )

    if isinstance(frequencies, u.Quantity):
        f_vals = frequencies.to("Hz").value
        f_unit = u.Hz
    else:
        f_vals = np.asarray(frequencies)
        f_unit = u.Hz

    if isinstance(f_ref, u.Quantity):
        f_ref_val = f_ref.to(f_unit).value
    else:
        f_ref_val = float(f_ref)

    target_unit = kwargs.get("unit", None)
    if isinstance(amplitude, u.Quantity):
        amp_val = amplitude.value
        if target_unit is None:
            target_unit = amplitude.unit
    else:
        amp_val = float(amplitude)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = f_vals / f_ref_val
        data = amp_val * (ratio ** (-exponent))

    # Handle f=0 singularities
    zero_mask = f_vals == 0
    if np.any(zero_mask):
        if exponent > 0:
            data[zero_mask] = np.inf
        elif exponent < 0:
            data[zero_mask] = 0.0
        # exponent=0 -> const, data is fine

    # Remove 'unit' from kwargs to avoid double passing
    kwargs.pop("unit", None)

    return FrequencySeries(data, frequencies=frequencies, unit=target_unit, **kwargs)


def white_noise(amplitude: float | u.Quantity, **kwargs: Any) -> FrequencySeries:
    """Generate White noise ASD (~ f^0).

    Parameters
    ----------
    amplitude : float or astropy.units.Quantity
        The noise amplitude. If a float is provided, it is assumed to be in the
        same units as the target spectrum (e.g., [m/rtHz] or [V/rtHz]).
        If a Quantity is provided, its units are preserved in the output.
    **kwargs : Any
        Additional keyword arguments passed to `power_law`.
        Typically includes `df`, `fmin`, `fmax`, or `frequencies`.

    Returns
    -------
    FrequencySeries
        The generated white noise ASD.

    Examples
    --------
    >>> from gwexpy.noise import white_noise
    >>> asd = white_noise(1e-18, df=1.0, fmin=0, fmax=100)
    >>> asd.unit
    Unit(dimensionless)

    """
    return power_law(0.0, amplitude=amplitude, **kwargs)


def pink_noise(
    amplitude: float | u.Quantity, f_ref: float | u.Quantity = 1.0, **kwargs: Any
) -> FrequencySeries:
    """Generate Pink noise ASD (~ f^-0.5).

    The power spectral density (PSD) follows f^-1, meaning the amplitude
    spectral density (ASD) follows f^-0.5.

    Parameters
    ----------
    amplitude : float or astropy.units.Quantity
        The noise amplitude at the reference frequency.
    f_ref : float or astropy.units.Quantity, optional
        The reference frequency in Hz (default is 1.0).
    **kwargs : Any
        Additional keyword arguments passed to `power_law`.

    Returns
    -------
    FrequencySeries
        The generated pink noise ASD.

    Examples
    --------
    >>> from gwexpy.noise import pink_noise
    >>> from astropy import units as u
    >>> asd = pink_noise(10.0, f_ref=10.0, df=1.0, fmin=1, fmax=100)
    >>> # 10 Hz is at index 9 (if fmin=1, f_0=1, index 9 is f=10)
    >>> bool(abs(asd[9].value - 10.0) < 1e-10)
    True

    """
    return power_law(0.5, amplitude=amplitude, f_ref=f_ref, **kwargs)


def red_noise(
    amplitude: float | u.Quantity, f_ref: float | u.Quantity = 1.0, **kwargs: Any
) -> FrequencySeries:
    """Generate Red (Brownian) noise ASD (~ f^-1).

    The power spectral density (PSD) follows f^-2, meaning the amplitude
    spectral density (ASD) follows f^-1.

    Parameters
    ----------
    amplitude : float or astropy.units.Quantity
        The noise amplitude at the reference frequency.
    f_ref : float or astropy.units.Quantity, optional
        The reference frequency in Hz (default is 1.0).
    **kwargs : Any
        Additional keyword arguments passed to `power_law`.

    Returns
    -------
    FrequencySeries
        The generated red noise ASD.

    Examples
    --------
    >>> from gwexpy.noise import red_noise
    >>> # Generate red noise with 1e-10 unit/rtHz at 1 Hz
    >>> asd = red_noise(1e-10, f_ref=1.0, df=0.1, fmin=0.1, fmax=10)
    >>> bool(abs(asd.value[9] - 1e-10) < 1e-18)  # At index 9, f=1.0Hz
    True

    """
    return power_law(1.0, amplitude=amplitude, f_ref=f_ref, **kwargs)
