"""gwexpy.spectrogram - Spectrogram data containers and operations."""

from __future__ import annotations

from .collections import SpectrogramDict, SpectrogramList
from .matrix import SpectrogramMatrix
from .spectrogram import Spectrogram

__all__ = [
    "Spectrogram",
    "SpectrogramList",
    "SpectrogramDict",
    "SpectrogramMatrix",
]

# Register constructors for cross-module lookup (avoids circular imports)
from gwexpy.interop._registry import ConverterRegistry as _CR

_CR.register_constructor("Spectrogram", Spectrogram)
_CR.register_constructor("SpectrogramDict", SpectrogramDict)
_CR.register_constructor("SpectrogramList", SpectrogramList)
_CR.register_constructor("SpectrogramMatrix", SpectrogramMatrix)
del _CR

# Dynamic import from gwpy (PEP 562)
import gwpy.spectrogram as _gwpy_spectrogram


def __getattr__(name):
    return getattr(_gwpy_spectrogram, name)


def __dir__():
    return sorted(set(__all__) | set(dir(_gwpy_spectrogram)))
