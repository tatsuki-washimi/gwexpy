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

# Import submodules for direct access
from . import asd, wave

# Re-export key functions for convenience
from .asd import from_obspy, from_pygwinc
from .wave import from_asd

__all__ = [
    # Submodules
    "asd",
    "wave",
    # ASD Functions
    "from_pygwinc",
    "from_obspy",
    # Waveform Functions
    "from_asd",
]
