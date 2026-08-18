"""GWexpy's public spectral API.

GWpy's registered ``median-mean`` PSD/ASD method is intentionally re-exported
without a duplicate registration. ``median_bias`` is the GWexpy-owned helper
for the independent-ordinate correction described in its docstring.
"""

from __future__ import annotations

from gwpy.signal.spectral import (
    average_spectrogram,
    bartlett,
    coherence,
    csd,
    get_default_fft_api,
    get_method,
    median,
    psd,
    rayleigh,
    register_method,
    spectrogram,
    welch,
)

from ._median_mean import median_bias

__all__ = [
    "average_spectrogram",
    "bartlett",
    "coherence",
    "csd",
    "get_default_fft_api",
    "get_method",
    "median",
    "median_bias",
    "psd",
    "rayleigh",
    "register_method",
    "spectrogram",
    "welch",
]
