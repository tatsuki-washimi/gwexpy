"""
Legacy TimeSeries class - Minimal base for gwexpy.

This module provides a minimal TimeSeries class that serves as the base
for the modular gwexpy TimeSeries implementation. Most functionality has
been extracted into separate mixin modules:

- _core.py: Basic operations
- _spectral.py: Spectral transforms
- _signal.py: Signal processing
- _resampling.py: Resampling operations
- _analysis.py: Statistical analysis
- _interop.py: Interoperability

This file now only contains:
1. Base class imports and inheritance
2. Helper functions and utilities used internally
"""
from __future__ import annotations

try:
    import scipy.signal  # noqa: F401 - availability check
except ImportError:
    pass

from gwpy.timeseries import TimeSeries as BaseTimeSeries

# --- Imports for delegation ---


class TimeSeries(BaseTimeSeries):
    """
    Minimal TimeSeries base class.

    This class provides the basic GWpy TimeSeries functionality.
    All extended gwexpy functionality is provided by Mixin classes
    that are combined in gwexpy.timeseries.timeseries.TimeSeries.

    Note: This class should not be used directly. Use
    `from gwexpy.timeseries import TimeSeries` instead.
    """

    pass
