# gwexpy.signal
# Extends gwpy.signal with additional preprocessing and analysis utilities.

# Local preprocessing module
from . import preprocessing
from .preprocessing import (
    WhiteningModel,
    whiten,
    StandardizationModel,
    standardize,
    impute,
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

