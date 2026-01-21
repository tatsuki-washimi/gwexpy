"""gwexpy.types - Data type definitions and utilities."""

from .array import Array
from .array2d import Array2D
from .array3d import Array3D
from .array4d import Array4D
from .axis import AxisDescriptor, coerce_1d_quantity
from .axis_api import AxisApiMixin
from .field4d import Field4D
from .field4d_collections import Field4DDict, Field4DList
from .field4d_demo import (
    make_demo_field4d,
    make_propagating_gaussian,
    make_sinusoidal_wave,
    make_standing_wave,
)
from .field4d_signal import (
    coherence_map,
    compute_freq_space,
    compute_psd,
    compute_xcorr,
    freq_space_map,
    time_delay_map,
)
from .metadata import MetaData, MetaDataDict, MetaDataMatrix
from .plane2d import Plane2D
from .series_creator import as_series
from .seriesmatrix import SeriesMatrix
from .time_plane_transform import TimePlaneTransform

__all__ = [
    # Metadata
    "MetaData",
    "MetaDataDict",
    "MetaDataMatrix",
    # Series
    "SeriesMatrix",
    "as_series",
    # Axis
    "AxisDescriptor",
    "coerce_1d_quantity",
    "AxisApiMixin",
    # Array types
    "Array",
    "Array2D",
    "Plane2D",
    "Array3D",
    "Array4D",
    "Field4D",
    "Field4DList",
    "Field4DDict",
    "TimePlaneTransform",
    # Field4D demo
    "make_demo_field4d",
    "make_propagating_gaussian",
    "make_sinusoidal_wave",
    "make_standing_wave",
    # Field4D signal processing
    "compute_psd",
    "freq_space_map",
    "compute_freq_space",
    "compute_xcorr",
    "time_delay_map",
    "coherence_map",
]

# Dynamic import from gwpy (PEP 562)
import gwpy.types as _gwpy_types


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    return getattr(_gwpy_types, name)


def __dir__():
    return sorted(set(__all__) | set(dir(_gwpy_types)))
