from .frequencyseries import FrequencySeries, FrequencySeriesDict, FrequencySeriesList, FrequencySeriesMatrix

# Register I/O readers on import
from . import io as _io  # noqa: F401

# Dynamic import from gwpy
import gwpy.frequencyseries
for key in dir(gwpy.frequencyseries):
    if not key.startswith("_") and key not in locals():
        locals()[key] = getattr(gwpy.frequencyseries, key)

__all__ = [k for k in locals().keys() if not k.startswith("_") and k != "gwpy"]
