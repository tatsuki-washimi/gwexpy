"""gwexpy.types - Data type definitions and utilities."""

from .array import Array
from .array2d import Array2D
from .array3d import Array3D
from .array4d import Array4D
from .axis import AxisDescriptor, coerce_1d_quantity
from .axis_api import AxisApiMixin
from .metadata import MetaData, MetaDataDict, MetaDataMatrix
from .plane2d import Plane2D
from .series_creator import as_series
from .seriesmatrix import SeriesMatrix
from .time_plane_transform import TimePlaneTransform
from .typing import (
    ArrayLike,
    IndexLike,
    MetaDataCollectionType,
    MetaDataDictLike,
    MetaDataLike,
    MetaDataMatrixLike,
    MetaDataType,
    UnitLike,
    XIndex,
)

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
    "TimePlaneTransform",
    # Type definitions (Protocols and TypeAliases)
    "XIndex",
    "MetaDataLike",
    "MetaDataDictLike",
    "MetaDataMatrixLike",
    "ArrayLike",
    "IndexLike",
    "UnitLike",
    "MetaDataType",
    "MetaDataCollectionType",
]

# Dynamic import from gwpy (PEP 562)
import gwpy.types as _gwpy_types


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(name)
    return getattr(_gwpy_types, name)


def __dir__():
    return sorted(set(__all__) | set(dir(_gwpy_types)))
