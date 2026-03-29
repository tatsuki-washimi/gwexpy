from __future__ import annotations

from .collections import HistogramDict, HistogramList
from .histogram import Histogram

__all__ = [
    "Histogram",
    "HistogramDict",
    "HistogramList",
]

# Register to ConverterRegistry
def _register() -> None:
    from gwexpy.interop._registry import ConverterRegistry

    ConverterRegistry.register_constructor("Histogram", Histogram)

_register()
