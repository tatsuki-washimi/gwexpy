"""
Spectral transform methods for TimeSeries.

This module integrates modular spectral analysis functionality:
- Fourier transforms (FFT, PSD, etc.)
- Special transforms (HHT, EMD, Laplace, CWT)
"""
from __future__ import annotations

from ._spectral_fourier import TimeSeriesSpectralFourierMixin
from ._spectral_special import TimeSeriesSpectralSpecialMixin


class TimeSeriesSpectralMixin(
    TimeSeriesSpectralFourierMixin, TimeSeriesSpectralSpecialMixin
):
    """
    Mixin class providing spectral transform methods for TimeSeries.

    Inherits from:
    - TimeSeriesSpectralFourierMixin: Standard Fourier transforms
    - TimeSeriesSpectralSpecialMixin: Special/Time-Frequency transforms
    """

    pass
