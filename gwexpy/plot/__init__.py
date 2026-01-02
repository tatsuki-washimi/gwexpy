from .plot import Plot
from .skymap import SkyMap
from .geomap import GeoMap
# Dynamic import from gwpy (PEP 562)
import gwpy.plot

def __getattr__(name):
    return getattr(gwpy.plot, name)

def __dir__():
    return sorted(set(globals().keys()) | set(dir(gwpy.plot)))
