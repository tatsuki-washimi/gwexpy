"""
gwexpy: GWpy Expansions for Experiments
========================================

This package extends GWpy with additional functionality for
gravitational wave and time-series data analysis.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from astropy.utils.exceptions import AstropyWarning

# Keep docs/tutorial output readable by suppressing known noisy warnings.
# This must run before importing GWpy/LAL (which can emit warnings at import time).
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")

# -----------------------------------------------------------------------------
# Compatibility: some minimal or newer gwpy builds used in docs/CI may lack
# `gwpy.io.registry.register_reader` / `register_identifier` / `register_writer`.
# Ensure the attributes exist so our IO modules' registration calls don't
# explode during import in stripped-down environments.
# -----------------------------------------------------------------------------
try:  # pragma: no cover - defensive
    import gwpy.io as _gwpy_io

    _io_reg = getattr(_gwpy_io, "registry", None)
    if _io_reg is not None:

        def _noop(*_args, **_kwargs):
            return None

        for _name in ("register_reader", "register_identifier", "register_writer"):
            if not hasattr(_io_reg, _name):
                setattr(_io_reg, _name, _noop)  # type: ignore[attr-defined]
except (ImportError, AttributeError):
    pass

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
