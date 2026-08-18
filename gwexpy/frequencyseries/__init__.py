from __future__ import annotations

from .bifrequencymap import BifrequencyMap
from .collections import (
    FrequencySeriesBaseDict,
    FrequencySeriesBaseList,
    FrequencySeriesDict,
    FrequencySeriesList,
)
from .frequencyseries import FrequencySeries, SeriesType
from .matrix import FrequencySeriesMatrix

__all__ = [
    "BifrequencyMap",
    "FrequencySeries",
    "SeriesType",
    "FrequencySeriesMatrix",
    "FrequencySeriesBaseDict",
    "FrequencySeriesDict",
    "FrequencySeriesBaseList",
    "FrequencySeriesList",
]

# Register constructors for cross-module lookup (avoids circular imports)
from gwexpy.interop._registry import ConverterRegistry as _CR

_CR.register_constructor("FrequencySeries", FrequencySeries)
_CR.register_constructor("FrequencySeriesDict", FrequencySeriesDict)
_CR.register_constructor("FrequencySeriesList", FrequencySeriesList)
_CR.register_constructor("FrequencySeriesMatrix", FrequencySeriesMatrix)
_CR.register_constructor("BifrequencyMap", BifrequencyMap)
del _CR
