"""gwexpy: GWpy Expansions for Experiments.

This package extends GWpy with additional functionality for
gravitational wave and time-series data analysis.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from astropy.utils.exceptions import AstropyWarning

from . import _warnings  # noqa: F401 – registers package-level warning filters

# Keep docs/tutorial output readable by suppressing known noisy warnings.
# This must run before importing GWpy/LAL (which can emit warnings at import time).
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

warnings.filterwarnings(
    "ignore",
    message=r"xindex was given to .*\(\), x0 will be ignored",
    category=UserWarning,
    module="gwpy",
)
warnings.filterwarnings(
    "ignore",
    message=r"xindex was given to .*\(\), dx will be ignored",
    category=UserWarning,
    module="gwpy",
)
warnings.filterwarnings(
    "ignore",
    message=r"yindex was given to .*\(\), dy will be ignored",
    category=UserWarning,
    module="gwpy",
)
warnings.filterwarnings("ignore", category=AstropyWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="control")
warnings.filterwarnings(
    "ignore", message="Protobuf gencode version", category=UserWarning
)
warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*joblib will operate in serial mode.*",
    category=UserWarning,
)

from ._bootstrap import register_all as _register_all
from ._version import __version__


def register_all(*, include_io: bool = True) -> None:
    """Ensure all constructors and optional I/O formats are registered."""
    _register_all(include_io=include_io)


__all__ = [
    # Bootstrap
    "register_all",
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
    # Histogram types
    "Histogram",
    "HistogramDict",
    "HistogramList",
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
    "histogram",
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
    import gwexpy.spectral as spectral


_LAZY_ROOT_MODULES: dict[str, str] = {
    name: f".{name}"
    for name in (
        "timeseries",
        "frequencyseries",
        "histogram",
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
    )
}


_LAZY_ROOT_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "TimeSeries": (".timeseries", "TimeSeries"),
    "TimeSeriesDict": (".timeseries", "TimeSeriesDict"),
    "TimeSeriesList": (".timeseries", "TimeSeriesList"),
    "TimeSeriesMatrix": (".timeseries", "TimeSeriesMatrix"),
    "FrequencySeries": (".frequencyseries", "FrequencySeries"),
    "FrequencySeriesDict": (".frequencyseries", "FrequencySeriesDict"),
    "FrequencySeriesList": (".frequencyseries", "FrequencySeriesList"),
    "FrequencySeriesMatrix": (".frequencyseries", "FrequencySeriesMatrix"),
    "Spectrogram": (".spectrogram", "Spectrogram"),
    "SpectrogramList": (".spectrogram", "SpectrogramList"),
    "SpectrogramDict": (".spectrogram", "SpectrogramDict"),
    "SpectrogramMatrix": (".spectrogram", "SpectrogramMatrix"),
    "Histogram": (".histogram", "Histogram"),
    "HistogramDict": (".histogram", "HistogramDict"),
    "HistogramList": (".histogram", "HistogramList"),
    "ScalarField": (".fields", "ScalarField"),
    "VectorField": (".fields", "VectorField"),
    "TensorField": (".fields", "TensorField"),
    "FieldList": (".fields", "FieldList"),
    "FieldDict": (".fields", "FieldDict"),
    "SeriesMatrix": (".types", "SeriesMatrix"),
    "MetaData": (".types", "MetaData"),
    "MetaDataDict": (".types", "MetaDataDict"),
    "MetaDataMatrix": (".types", "MetaDataMatrix"),
    "as_series": (".types", "as_series"),
    "whiten": (".signal.preprocessing", "whiten"),
    "standardize": (".signal.preprocessing", "standardize"),
    "impute": (".signal.preprocessing", "impute"),
    "WhiteningModel": (".signal.preprocessing", "WhiteningModel"),
    "StandardizationModel": (".signal.preprocessing", "StandardizationModel"),
}

_LAZY_IMPORT_PREREQUISITES: dict[str, tuple[str, ...]] = {
    ".fields": (".interop",),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ROOT_MODULES:
        import importlib

        mod = importlib.import_module(_LAZY_ROOT_MODULES[name], __name__)
        globals()[name] = mod
        return mod
    if name in _LAZY_ROOT_ATTRIBUTES:
        import importlib

        module_name, attribute_name = _LAZY_ROOT_ATTRIBUTES[name]
        for prerequisite in _LAZY_IMPORT_PREREQUISITES.get(module_name, ()):
            importlib.import_module(prerequisite, __name__)
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attribute_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals().keys()))
