"""Precision type definitions for gwexpy.

This module provides Protocol definitions and TypeAliases to eliminate
broad 'Any' usage and enable stricter static type checking across the
codebase. These definitions serve as shared contracts for duck-typed
interfaces that are frequently used but not formally declared.

The protocols here are structural (duck-typed), meaning a class need
not explicitly inherit from them to be considered compatible—it just
needs to implement the required methods and attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, Union, runtime_checkable

import numpy as np
from astropy import units as u
from numpy.typing import ArrayLike as NumpyArrayLike
from numpy.typing import NDArray

if TYPE_CHECKING:
    from gwexpy.types.metadata import MetaData

# =============================
# Index-like Protocols
# =============================


@runtime_checkable
class XIndex(Protocol):
    """Protocol for index-like objects (times, frequencies, etc.).

    This represents array-like objects that serve as coordinate axes
    for SeriesMatrix and related classes. They must support:
    - Length queries via __len__
    - Item access via __getitem__
    - Physical units via .unit attribute
    - Numeric value extraction via .value attribute
    """

    @property
    def unit(self) -> u.UnitBase:
        """Physical unit of the index."""
        ...

    @property
    def value(self) -> NDArray[Any]:
        """Numeric values as a numpy array."""
        ...

    def __len__(self) -> int:
        """Number of elements in the index."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the index."""
        ...

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        ...

    def copy(self) -> Any:
        """Return a copy of the index."""
        ...

    def __getitem__(self, key: Any) -> Any:
        """Access element(s) by index or slice."""
        ...


# =============================
# Metadata-like Protocols
# =============================


@runtime_checkable
class MetaDataLike(Protocol):
    """Protocol for single-object metadata containers.

    This represents objects that carry name, channel, and unit
    information for a single data series or field component.
    """

    name: str
    channel: Any  # gwpy.detector.Channel
    unit: u.UnitBase


@runtime_checkable
class MetaDataDictLike(Protocol):
    """Protocol for ordered collections of metadata.

    This represents dict-like containers mapping keys to MetaData
    objects, used for row/column metadata in SeriesMatrix.
    """

    def keys(self) -> Any:
        """Return keys of the metadata collection."""
        ...

    def values(self) -> Any:
        """Return MetaData instances."""
        ...

    def items(self) -> Any:
        """Return (key, MetaData) pairs."""
        ...

    def __getitem__(self, key: Any) -> MetaData:
        """Access metadata by key."""
        ...

    def __len__(self) -> int:
        """Number of metadata entries."""
        ...


@runtime_checkable
class MetaDataMatrixLike(Protocol):
    """Protocol for 2D metadata matrices.

    This represents structured metadata for matrix-like data,
    with units, names, and channels per element.
    """

    shape: tuple[int, ...]

    @property
    def units(self) -> NDArray[Any]:
        """2D array of units for each matrix element."""
        ...

    @property
    def names(self) -> NDArray[Any]:
        """2D array of names for each matrix element."""
        ...

    @property
    def channels(self) -> NDArray[Any]:
        """2D array of channels for each matrix element."""
        ...

    def __getitem__(self, key: Any) -> Any:
        """Access metadata by index."""
        ...


# =============================
# Type Aliases
# =============================

# Generic array-like (numpy-compatible)
ArrayLike: TypeAlias = Union[
    np.ndarray,
    list,
    tuple,
    NumpyArrayLike,
]

# Index types (coordinate axes)
IndexLike: TypeAlias = Union[
    XIndex,
    u.Quantity,
    np.ndarray,
]

# Unit types
UnitLike: TypeAlias = Union[
    u.UnitBase,
    str,
    None,
]

# Metadata types
MetaDataType: TypeAlias = Union[
    "MetaData",
    MetaDataLike,
    dict[str, Any],
]

MetaDataCollectionType: TypeAlias = Union[
    MetaDataDictLike,
    dict[str, MetaDataType],
    list[MetaDataType],
    None,
]

__all__ = [
    # Protocols
    "XIndex",
    "MetaDataLike",
    "MetaDataDictLike",
    "MetaDataMatrixLike",
    # Type Aliases
    "ArrayLike",
    "IndexLike",
    "UnitLike",
    "MetaDataType",
    "MetaDataCollectionType",
]
