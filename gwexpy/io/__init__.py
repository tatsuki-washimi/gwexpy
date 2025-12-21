"""
gwexpy.io helpers and registration hooks.
"""

from . import utils  # noqa: F401

# Dynamic import from gwpy (PEP 562)
import gwpy.io as _gwpy_io

def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    return getattr(_gwpy_io, name)

def __dir__():
    local = {"utils"}
    return sorted(local | set(dir(_gwpy_io)))
