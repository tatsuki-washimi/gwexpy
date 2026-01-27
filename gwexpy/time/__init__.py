from __future__ import annotations

# Dynamic import from gwpy (PEP 562)
import gwpy.time

from .core import from_gps as from_gps
from .core import tconvert as tconvert
from .core import to_gps as to_gps


def __getattr__(name):
    return getattr(gwpy.time, name)


def __dir__():
    return sorted(set(globals().keys()) | set(dir(gwpy.time)))
