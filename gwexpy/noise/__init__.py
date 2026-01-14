"""gwexpy.noise - Noise generation and modeling.

This module provides tools for generating and modeling noise spectra.

Submodules
----------
asd : Functions that return FrequencySeries (ASD)
wave : Functions that return time-series waveforms (ndarray)

Examples
--------
>>> from gwexpy.noise.asd import from_pygwinc
>>> from gwexpy.noise.wave import from_asd

>>> asd = from_pygwinc('aLIGO', fmin=4.0, fmax=1024.0, df=0.01)
>>> noise = from_asd(asd, duration=128, sample_rate=2048, t0=0)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

# Import submodules for direct access
from . import asd, wave

if TYPE_CHECKING:
    from .colored import pink_noise, power_law, red_noise, white_noise
    from .gwinc_ import from_pygwinc
    from .magnetic import geomagnetic_background, schumann_resonance
    from .obspy_ import from_obspy
    from .peaks import gaussian_line, lorentzian_line, voigt_line
    from .wave import from_asd

__all__ = [
    # Submodules
    "asd",
    "wave",
    # ASD functions (for backward compatibility)
    "from_pygwinc",
    "from_obspy",
    "from_asd",
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


def __getattr__(name: str) -> Any:
    if name == "from_pygwinc":
        from .gwinc_ import from_pygwinc

        return from_pygwinc
    if name == "from_obspy":
        from .obspy_ import from_obspy

        return from_obspy
    if name in ("power_law", "white_noise", "pink_noise", "red_noise"):
        from .colored import pink_noise, power_law, red_noise, white_noise

        if name == "power_law":
            return power_law
        if name == "white_noise":
            return white_noise
        if name == "pink_noise":
            return pink_noise
        if name == "red_noise":
            return red_noise
    if name in ("schumann_resonance", "geomagnetic_background"):
        from .magnetic import geomagnetic_background, schumann_resonance

        if name == "schumann_resonance":
            return schumann_resonance
        if name == "geomagnetic_background":
            return geomagnetic_background
    if name in ("lorentzian_line", "gaussian_line", "voigt_line"):
        from .peaks import gaussian_line, lorentzian_line, voigt_line

        if name == "lorentzian_line":
            return lorentzian_line
        if name == "gaussian_line":
            return gaussian_line
        if name == "voigt_line":
            return voigt_line
    if name == "from_asd":
        from .wave import from_asd

        return from_asd

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
