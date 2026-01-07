"""gwexpy.spectrogram - Spectrogram data containers and operations."""

from .spectrogram import Spectrogram
from .collections import SpectrogramList, SpectrogramDict
from .matrix import SpectrogramMatrix

__all__ = [
    "Spectrogram",
    "SpectrogramList",
    "SpectrogramDict",
    "SpectrogramMatrix",
]

# Dynamic import from gwpy (PEP 562)
import gwpy.spectrogram as _gwpy_spectrogram

def __getattr__(name):
    return getattr(_gwpy_spectrogram, name)

def __dir__():
    return sorted(set(__all__) | set(dir(_gwpy_spectrogram)))
