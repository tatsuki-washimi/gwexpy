from .timeseries import TimeSeries, TimeSeriesDict, TimeSeriesList, TimeSeriesMatrix
from .pipeline import (
    Transform,
    Pipeline,
    ImputeTransform,
    StandardizeTransform,
    WhitenTransform,
    PCATransform,
    ICATransform,
)

# Register I/O readers on import
from . import io as _io  # noqa: F401

# Dynamic import from gwpy
import sys
import gwpy.timeseries

_this_module = sys.modules[__name__]
_gwpy_module = gwpy.timeseries

for _name in dir(_gwpy_module):
    if not _name.startswith("_") and not hasattr(_this_module, _name):
        setattr(_this_module, _name, getattr(_gwpy_module, _name))

__all__ = [k for k in dir(_this_module) if not k.startswith("_") and k not in ["gwpy", "sys"]]
