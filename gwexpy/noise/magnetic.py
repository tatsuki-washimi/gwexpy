"""gwexpy.noise.magnetic - Geomagnetic noise models."""
from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u

from ..frequencyseries import FrequencySeries
from .colored import power_law
from .peaks import lorentzian_line


def schumann_resonance(
    frequencies: np.ndarray | None = None,
    modes: list[tuple[float, float, float]] | None = None,
    amplitude_scale: float = 1.0,
    **kwargs: Any,
) -> FrequencySeries:
    """
    Generate Schumann Resonance ASD by combining Lorentzians in PSD space.

    Parameters
    ----------
    frequencies : array-like, optional
        Target frequency array.
    modes : list of tuples, optional
        (f0, Q, amplitude_asd in pT/rtHz).
        If None, uses defaults from observation.
    amplitude_scale : float
        Scaling factor for amplitude.
    **kwargs
        Additional arguments for FrequencySeries.
    """
    if frequencies is None:
        # Need valid frequencies to sum over
        frequencies = kwargs.pop("frequencies", None)
        if frequencies is None:
            raise ValueError("frequencies argument is required for schumann_resonance.")

    if modes is None:
        # Default Schumann modes: f0 (Hz), Q, Amplitude (pT/rtHz)
        modes = [
            (7.83, 5.0, 1.5),
            (14.3, 6.0, 0.8),
            (20.8, 7.0, 0.6),
            (27.3, 8.0, 0.4),
            (33.8, 8.0, 0.3),
            (39.0, 9.0, 0.2),
            (45.0, 9.0, 0.1),
        ]

    # Handle unit
    if "unit" not in kwargs:
        kwargs["unit"] = u.Unit("pT / Hz^(1/2)")

    # We'll work with the raw values of the target unit
    # But lorentzian_line handles units.
    # To sum in PSD space: Total PSD = SUM( (ASD_i)^2 )

    # Check frequencies type
    if isinstance(frequencies, u.Quantity):
        f_arr = frequencies.to("Hz").value
    else:
        f_arr = np.asarray(frequencies)

    total_psd = np.zeros_like(f_arr, dtype=float)

    for f0, Q, A in modes:
        # Generate single peak ASD
        # We pass frequencies explicitly.
        # Amplitude is scaled.
        # We use the correct unit in kwargs so lorentzian_line returns correct Quantities if unit present

        # NOTE: logic requires adding SQUARES of values.
        # If lorentzian_line returns Quantity, we get value.

        peak_asd_series = lorentzian_line(
            f0, A * amplitude_scale, Q=Q, frequencies=f_arr, unit=kwargs["unit"]
        )

        # Add to total PSD (incoherently)
        total_psd += peak_asd_series.value**2

    # Final ASD
    total_asd = np.sqrt(total_psd)

    return FrequencySeries(total_asd, frequencies=f_arr, **kwargs)


def geomagnetic_background(
    frequencies: np.ndarray,
    amplitude_1hz: float = 10e-12,  # 10 pT
    exponent: float = 1.0,
    **kwargs: Any,
) -> FrequencySeries:
    """
    Generate 1/f^alpha magnetic background noise.
    Wrapper for colored.power_law.

    Parameters
    ----------
    amplitude_1hz : float
        Amplitude at 1Hz, typically in pT/rtHz (default 10e-12 = 10pT).
        Note: The user provided example says 10e-12 (10pT), assumed unit of result.
        However, if user passes unit='pT...', amplitude should be consistent.
    """

    if "unit" not in kwargs:
        # Default to pT/rtHz if not specified, assuming input implies T?
        # Actually usually inputs are in same unit base.
        # If user passes 10e-12, it implies Tesla.
        # If output is pT, 10e-12 T = 10 pT.
        # Let's default to a safe unit or respect input.
        kwargs["unit"] = u.Unit("pT / Hz^(1/2)")

        # If amplitude is float 10e-12 (which is 10pT in Tesla),
        # and we want pT/rtHz output...
        # If we pass 10e-12 directly to power_law with unit=pT, we get 1e-11 pT... very small.
        # Let's allow user to specify. If defaults:

        if amplitude_1hz < 1e-9:  # Likely Tesla
            # Convert to pT for the default unit
            amplitude_1hz *= 1e12

    return power_law(
        exponent, amplitude=amplitude_1hz, f_ref=1.0, frequencies=frequencies, **kwargs
    )
