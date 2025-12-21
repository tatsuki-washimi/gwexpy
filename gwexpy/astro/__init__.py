# Dynamic import from gwpy (PEP 562)
import gwpy.astro

def __getattr__(name):
    return getattr(gwpy.astro, name)

def __dir__():
    return sorted(set(globals().keys()) | set(dir(gwpy.astro)))
