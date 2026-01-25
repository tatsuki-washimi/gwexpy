"""
gwexpy: GWpy Expansions for Experiments
========================================

This package extends GWpy with additional functionality for
gravitational wave and time-series data analysis.
"""

import warnings

warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
warnings.filterwarnings("ignore", category=UserWarning, module="gwpy")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
from astropy.utils.exceptions import AstropyWarning

warnings.filterwarnings("ignore", category=AstropyWarning)

from typing import TYPE_CHECKING, Any

# Subpackages are available via namespace
from . import (
    astro,
    detector,
    frequencyseries,
    interop,
    io,
    noise,
    plot,
    segments,
    signal,
    spectral,
    spectrogram,
    table,
    time,
    timeseries,
    types,
)
from ._version import __version__
from .fields import FieldDict, FieldList, ScalarField, TensorField, VectorField
from .frequencyseries import (
    FrequencySeries,
    FrequencySeriesDict,
    FrequencySeriesList,
    FrequencySeriesMatrix,
)

# Signal processing utilities
from .signal.preprocessing import (
    StandardizationModel,
    WhiteningModel,
    impute,
    standardize,
    whiten,
)
from .spectrogram import (
    Spectrogram,
    SpectrogramDict,
    SpectrogramList,
    SpectrogramMatrix,
)

# Core data types - explicitly imported for IDE support and clear API
from .timeseries import (
    TimeSeries,
    TimeSeriesDict,
    TimeSeriesList,
    TimeSeriesMatrix,
)

# Types
from .types import (
    MetaData,
    MetaDataDict,
    MetaDataMatrix,
    SeriesMatrix,
    as_series,
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
    "ScalarField",
    "VectorField",
    "TensorField",
    "FieldList",
    "FieldDict",
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


# Enable fitting monkeypatch by default for user convenience
try:
    from .fitting import enable_series_fit

    enable_series_fit()
except ImportError:
    pass
