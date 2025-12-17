#from gwpy.core import FrequencySeriesBase
from gwpy.frequencyseries.hist import SpectralVariance
from .frequencyseries import (
    FrequencySeries,
    FrequencySeriesBaseDict,
    FrequencySeriesBaseList,
    FrequencySeriesDict,
    FrequencySeriesList,
    FrequencySeriesMatrix,
)

__all__ = ["SpectralVariance",
           "FrequencySeries",
           "FrequencySeriesBaseDict", "FrequencySeriesBaseList",
           "FrequencySeriesDict", "FrequencySeriesList", "FrequencySeriesMatrix"]
