from __future__ import annotations

from ._protocols import (
    AxisApiHost,
    Copyable,
    HasPhaseMethods,
    HasSeriesData,
    HasSeriesMetadata,
    SupportsPhaseMethods,
    SupportsSignalAnalysis,
)
from .mixin_legacy import PhaseMethodsMixin, RegularityMixin
from .signal_interop import InteropMixin, SignalAnalysisMixin

__all__ = [
    "RegularityMixin",
    "PhaseMethodsMixin",
    "SignalAnalysisMixin",
    "InteropMixin",
    "HasSeriesData",
    "HasSeriesMetadata",
    "HasPhaseMethods",
    "Copyable",
    "AxisApiHost",
    "SupportsSignalAnalysis",
    "SupportsPhaseMethods",
]
