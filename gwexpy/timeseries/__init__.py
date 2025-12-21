"""gwexpy.timeseries - Time series data containers and operations."""

from .timeseries import TimeSeries
from .collections import TimeSeriesDict, TimeSeriesList
from .matrix import TimeSeriesMatrix
from .pipeline import (
    Transform,
    Pipeline,
    ImputeTransform,
    StandardizeTransform,
    WhitenTransform,
    PCATransform,
    ICATransform,
)

__all__ = [
    "TimeSeries",
    "TimeSeriesDict",
    "TimeSeriesList",
    "TimeSeriesMatrix",
    "Transform",
    "Pipeline",
    "ImputeTransform",
    "StandardizeTransform",
    "WhitenTransform",
    "PCATransform",
    "ICATransform",
]

# Register I/O readers on import
from . import io as _io  # noqa: F401

# Dynamic import from gwpy (PEP 562)
import gwpy.timeseries as _gwpy_timeseries

def __getattr__(name):
    return getattr(_gwpy_timeseries, name)

def __dir__():
    return sorted(set(__all__) | set(dir(_gwpy_timeseries)))
