"""gwexpy.frequencyseries - Frequency series data containers and operations."""

from .frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesList,
    FrequencySeriesMatrix,
)

__all__ = [
    "FrequencySeries",
    "FrequencySeriesDict",
    "FrequencySeriesList",
    "FrequencySeriesMatrix",
]

# Register I/O readers on import
from . import io as _io  # noqa: F401

# Dynamic import from gwpy (PEP 562)
import gwpy.frequencyseries as _gwpy_frequencyseries

def __getattr__(name):
    return getattr(_gwpy_frequencyseries, name)

def __dir__():
    return sorted(set(__all__) | set(dir(_gwpy_frequencyseries)))
