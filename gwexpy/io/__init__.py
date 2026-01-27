"""
gwexpy.io helpers and registration hooks.
"""

from __future__ import annotations

# Dynamic import from gwpy (PEP 562)
import gwpy.io as _gwpy_io

from . import utils  # noqa: F401


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    return getattr(_gwpy_io, name)


def __dir__():
    local = {"utils"}
    return sorted(local | set(dir(_gwpy_io)))
