# Dynamic import from gwpy (PEP 562)
import gwpy.segments

def __getattr__(name):
    return getattr(gwpy.segments, name)

def __dir__():
    return sorted(set(globals().keys()) | set(dir(gwpy.segments)))
