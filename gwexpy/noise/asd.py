"""gwexpy.noise.asd - Functions that generate ASD (Amplitude Spectral Density).

This module provides convenient access to ASD-generating functions.
All functions in this module return FrequencySeries objects with appropriate
units based on the requested physical quantity.

ASD Functions
-------------
from_pygwinc : Generate ASD from pyGWINC detector noise models
    Supported quantities: "strain" [1/sqrt(Hz)], "darm" [m/sqrt(Hz)],
    "displacement" (alias of "darm")

from_obspy : Generate ASD from ObsPy seismic/infrasound noise models
    Supported quantities: "displacement" [m/sqrt(Hz)],
    "velocity" [m/(s·sqrt(Hz))], "acceleration" [m/(s²·sqrt(Hz))]
    Note: "strain" is NOT supported for seismic models.

power_law, white_noise, pink_noise, red_noise : Colored noise ASD generators
schumann_resonance, geomagnetic_background : Geomagnetic noise models
lorentzian_line, gaussian_line, voigt_line : Spectral line shapes

Examples
--------
>>> from gwexpy.noise.asd import from_pygwinc, from_obspy

# Get strain ASD from pyGWINC
>>> strain_asd = from_pygwinc("aLIGO", quantity="strain")

# Get DARM (differential arm length) ASD
>>> darm_asd = from_pygwinc("aLIGO", quantity="darm")

# Get displacement ASD from ObsPy NLNM
>>> disp_asd = from_obspy("NLNM", quantity="displacement")
"""
from __future__ import annotations

from .colored import pink_noise, power_law, red_noise, white_noise

# Re-export ASD-generating functions
from .gwinc_ import from_pygwinc
from .magnetic import geomagnetic_background, schumann_resonance
from .obspy_ import from_obspy
from .peaks import gaussian_line, lorentzian_line, voigt_line

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
