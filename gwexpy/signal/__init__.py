from __future__ import annotations

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
    "preprocessing",
    "WhiteningModel",
    "whiten",
    "StandardizationModel",
    "standardize",
    "impute",
]

# Dynamic import from gwpy (PEP 562)
import gwpy.signal


def __getattr__(name):
    return getattr(gwpy.signal, name)


def __dir__():
    local_names = {
        "preprocessing",
        "WhiteningModel",
        "whiten",
        "StandardizationModel",
        "standardize",
        "impute",
    }
    return sorted(local_names | set(dir(gwpy.signal)))
