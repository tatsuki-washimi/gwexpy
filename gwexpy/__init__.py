"""
gwexpy: GWpy Expansions for Experiments
========================================

This package extends GWpy with additional functionality for 
gravitational wave and time-series data analysis.
"""

from ._version import __version__

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
]
