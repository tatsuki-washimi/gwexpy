# Dynamic import from gwpy (PEP 562)
import gwpy.detector

def __getattr__(name):
    return getattr(gwpy.detector, name)

def __dir__():
    return sorted(set(globals().keys()) | set(dir(gwpy.detector)))
