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
    """
    Generate an ASD FrequencySeries following a power law.
    ASD(f) = amplitude * (f / f_ref) ** (-exponent)
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

    if f_vals[0] == 0:
        if exponent > 0:
            data[0] = np.inf
        elif exponent < 0:
            data[0] = 0.0
        # exponent=0 -> const, data[0] is fine

    # Remove 'unit' from kwargs to avoid double passing
    kwargs.pop("unit", None)

    return FrequencySeries(data, frequencies=frequencies, unit=target_unit, **kwargs)


def white_noise(amplitude: float | u.Quantity, **kwargs: Any) -> FrequencySeries:
    """Generate White noise ASD (~ f^0)."""
    return power_law(0.0, amplitude=amplitude, **kwargs)


def pink_noise(
    amplitude: float | u.Quantity, f_ref: float | u.Quantity = 1.0, **kwargs: Any
) -> FrequencySeries:
    """Generate Pink noise ASD (~ f^-0.5)."""
    return power_law(0.5, amplitude=amplitude, f_ref=f_ref, **kwargs)


def red_noise(
    amplitude: float | u.Quantity, f_ref: float | u.Quantity = 1.0, **kwargs: Any
) -> FrequencySeries:
    """Generate Red (Brownian) noise ASD (~ f^-1)."""
    return power_law(1.0, amplitude=amplitude, f_ref=f_ref, **kwargs)
