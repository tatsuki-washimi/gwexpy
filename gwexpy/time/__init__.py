from .core import to_gps as to_gps, from_gps as from_gps, tconvert as tconvert
# Dynamic import from gwpy (PEP 562)
import gwpy.time

def __getattr__(name):
    return getattr(gwpy.time, name)

def __dir__():
    return sorted(set(globals().keys()) | set(dir(gwpy.time)))
