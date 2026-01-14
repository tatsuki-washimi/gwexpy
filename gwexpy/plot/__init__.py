from .pairplot import PairPlot
from .plot import Plot, plot_mmm
from .skymap import SkyMap

__all__ = ["Plot", "plot_mmm", "SkyMap", "GeoMap", "PairPlot"]

# Dynamic import from gwpy (PEP 562)
import importlib

import gwpy.plot


def __getattr__(name):
    if name == "GeoMap":
        geomap = importlib.import_module(".geomap", __name__)
        globals()["GeoMap"] = geomap.GeoMap
        return geomap.GeoMap
    return getattr(gwpy.plot, name)


def __dir__():
    return sorted(set(globals().keys()) | set(dir(gwpy.plot)) | {"GeoMap"})
