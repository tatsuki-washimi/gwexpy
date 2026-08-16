from __future__ import annotations

import importlib
import sys

# gwexpy.signal
# Extends gwpy.signal with additional preprocessing and analysis utilities.
# Local preprocessing module
from . import preprocessing
from .preprocessing import (
    StandardizationModel as StandardizationModel,
)
from .preprocessing import (
    WhiteningModel as WhiteningModel,
)
from .preprocessing import (
    impute as impute,
)
from .preprocessing import (
    standardize as standardize,
)
from .preprocessing import (
    whiten as whiten,
)

__all__ = [
    "spectral",
    "preprocessing",
    "WhiteningModel",
    "whiten",
    "StandardizationModel",
    "standardize",
    "impute",
]


def __getattr__(name):
    if name == "spectral":
        module = importlib.import_module(f"{__name__}.spectral")
        globals()[name] = module
        return module
    return getattr(importlib.import_module("gwpy.signal"), name)


def __dir__():
    local_names = {
        "spectral",
        "preprocessing",
        "WhiteningModel",
        "whiten",
        "StandardizationModel",
        "standardize",
        "impute",
    }
    gwpy_signal = sys.modules.get("gwpy.signal")
    gwpy_names = set(dir(gwpy_signal)) if gwpy_signal is not None else set()
    return sorted(local_names | gwpy_names)
