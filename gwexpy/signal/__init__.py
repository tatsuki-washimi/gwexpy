# gwexpy.signal
# Extends gwpy.signal with additional preprocessing and analysis utilities.

# Local preprocessing module
from . import preprocessing
from .preprocessing import (
    WhiteningModel as WhiteningModel,
    whiten as whiten,
    StandardizationModel as StandardizationModel,
    standardize as standardize,
    impute as impute,
)

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

