"""
Plotting helpers.

Note: This module intentionally avoids importing optional/heavy dependencies
(e.g. ligo.skymap) at import time because Sphinx autodoc imports `gwexpy`.
"""

from typing import TYPE_CHECKING

from .pairplot import PairPlot
from .plot import Plot, plot_mmm

__all__ = ["Plot", "plot_mmm", "SkyMap", "GeoMap", "PairPlot"]

# Dynamic import from gwpy (PEP 562)
import importlib

import gwpy.plot


def __getattr__(name):
    if name == "SkyMap":
        skymap = importlib.import_module(".skymap", __name__)
        globals()["SkyMap"] = skymap.SkyMap
        return skymap.SkyMap
    if name == "GeoMap":
        geomap = importlib.import_module(".geomap", __name__)
        globals()["GeoMap"] = geomap.GeoMap
        return geomap.GeoMap
    return getattr(gwpy.plot, name)


def __dir__():
    return sorted(set(globals().keys()) | set(dir(gwpy.plot)) | {"GeoMap", "SkyMap"})


if TYPE_CHECKING:  # pragma: no cover
    from .skymap import SkyMap as SkyMap
