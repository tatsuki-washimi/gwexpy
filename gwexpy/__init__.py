"""
gwexpy: GWpy Expansions for Experiments
========================================

This package extends GWpy with additional functionality for
gravitational wave and time-series data analysis.
"""

from ._version import __version__
from typing import TYPE_CHECKING, Any

# Core data types - explicitly imported for IDE support and clear API
from .timeseries import (
    TimeSeries,
    TimeSeriesDict,
    TimeSeriesList,
    TimeSeriesMatrix,
)
from .frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesList,
    FrequencySeriesMatrix,
)
from .spectrogram import (
    Spectrogram,
    SpectrogramList,
    SpectrogramDict,
    SpectrogramMatrix,
)

# Types
from .types import (
    SeriesMatrix,
    MetaData,
    MetaDataDict,
    MetaDataMatrix,
    as_series,
)

# Signal processing utilities
from .signal.preprocessing import (
    whiten,
    standardize,
    impute,
    WhiteningModel,
    StandardizationModel,
)

# Subpackages are available via namespace
from . import (
    timeseries,
    frequencyseries,
    spectrogram,
    spectral,
    astro,
    detector,
    plot,
    segments,
    signal,
    table,
    time,
    types,
    io,
    interop,
    noise,
)

__all__ = [
    # Version
    "__version__",
    # TimeSeries types
    "TimeSeries",
    "TimeSeriesDict",
    "TimeSeriesList",
    "TimeSeriesMatrix",
    # FrequencySeries types
    "FrequencySeries",
    "FrequencySeriesDict",
    "FrequencySeriesList",
    "FrequencySeriesMatrix",
    # Spectrogram types
    "Spectrogram",
    "SpectrogramList",
    "SpectrogramDict",
    "SpectrogramMatrix",
    # Types
    "SeriesMatrix",
    "MetaData",
    "MetaDataDict",
    "MetaDataMatrix",
    "as_series",
    # Signal preprocessing
    "whiten",
    "standardize",
    "impute",
    "WhiteningModel",
    "StandardizationModel",
    # Subpackages
    "timeseries",
    "frequencyseries",
    "spectrogram",
    "spectral",
    "astro",
    "detector",
    "plot",
    "segments",
    "signal",
    "table",
    "time",
    "types",
    "io",
    "interop",
    "noise",
    "fitting",
]

if TYPE_CHECKING:  # pragma: no cover
    import gwexpy.fitting as fitting


def __getattr__(name: str) -> Any:
    if name == "fitting":
        import importlib

        mod = importlib.import_module(".fitting", __name__)
        globals()["fitting"] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
