"""gwexpy.noise.asd - Functions that generate ASD (Amplitude Spectral Density).

This module provides convenient access to ASD-generating functions.
All functions in this module return FrequencySeries objects.
"""

from __future__ import annotations

# Re-export ASD-generating functions
from .gwinc_ import from_pygwinc
from .obspy_ import from_obspy
from .colored import power_law, white_noise, pink_noise, red_noise
from .magnetic import schumann_resonance, geomagnetic_background
from .peaks import lorentzian_line, gaussian_line, voigt_line

__all__ = [
    "from_pygwinc",
    "from_obspy",
    "power_law",
    "white_noise",
    "pink_noise",
    "red_noise",
    "schumann_resonance",
    "geomagnetic_background",
    "lorentzian_line",
    "gaussian_line",
    "voigt_line",
]
