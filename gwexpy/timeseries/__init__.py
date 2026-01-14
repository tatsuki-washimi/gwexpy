"""gwexpy.timeseries - Time series data containers and operations."""

from .collections import TimeSeriesDict, TimeSeriesList
from .matrix import TimeSeriesMatrix
from .pipeline import (
    ICATransform,
    ImputeTransform,
    PCATransform,
    Pipeline,
    StandardizeTransform,
    Transform,
    WhitenTransform,
)
from .timeseries import TimeSeries

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
# Dynamic import from gwpy (PEP 562)
import gwpy.timeseries as _gwpy_timeseries

from . import io as _io  # noqa: F401


def __getattr__(name):
    return getattr(_gwpy_timeseries, name)


def __dir__():
    return sorted(set(__all__) | set(dir(_gwpy_timeseries)))
