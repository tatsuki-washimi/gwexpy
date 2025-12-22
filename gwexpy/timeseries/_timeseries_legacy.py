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

import inspect
from enum import Enum
import numpy as np
from astropy import units as u
from typing import Optional, Union, Any, List, Iterable

try:
    import scipy.signal
except ImportError:
    pass

from gwpy.timeseries import TimeSeries as BaseTimeSeries
from gwpy.timeseries import TimeSeriesDict as BaseTimeSeriesDict
from gwpy.timeseries import TimeSeriesList as BaseTimeSeriesList

from gwexpy.types.seriesmatrix import SeriesMatrix
from gwexpy.types.metadata import MetaData, MetaDataMatrix

# --- Imports for delegation ---
from .preprocess import (
    impute_timeseries, standardize_timeseries, align_timeseries_collection, 
    standardize_matrix, whiten_matrix
)
from .arima import fit_arima
from .hurst import hurst, local_hurst
from .decomposition import (
    pca_fit, pca_transform, pca_inverse_transform, 
    ica_fit, ica_transform, ica_inverse_transform
)
from .spectral import csd_matrix_from_collection, coherence_matrix_from_collection

from .utils import *


def _extract_axis_info(ts):
    """Extract axis information from a TimeSeries.
    
    Returns a dict with:
    - 'dt': sample interval (Quantity or None)
    - 'regular': whether the series has regular sampling
    """
    try:
        dt = ts.dt
        regular = True
    except (AttributeError, ValueError):
        dt = None
        regular = False
    return {'dt': dt, 'regular': regular}


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
