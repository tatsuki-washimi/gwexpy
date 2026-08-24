"""GWpy-compatible spectral APIs plus GWexpy-owned extensions."""

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

from ._median_bias import median_bias
from ._median_mean import median_mean

__all__ = [
    "average_spectrogram",
    "bartlett",
    "coherence",
    "csd",
    "get_default_fft_api",
    "get_method",
    "median",
    "median_bias",
    "median_mean",
    "psd",
    "rayleigh",
    "register_method",
    "spectrogram",
    "welch",
]
