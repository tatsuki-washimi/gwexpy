"""Protocol definitions for type-safe mixin classes.

These protocols define the structural requirements that mixins expect
from their host classes, enabling MyPy to verify type safety.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from astropy import units as u


@runtime_checkable
class HasSeriesData(Protocol):
    """Protocol for classes that have value and unit attributes."""

    @property
    def value(self) -> Any:
        """The numerical data values."""
        ...

    @property
    def unit(self) -> u.Unit:
        """The unit of the data."""
        ...


@runtime_checkable
class HasSeriesMetadata(Protocol):
    """Protocol for classes that have series metadata."""

    @property
    def name(self) -> str | None:
        """Name of the series."""
        ...

    @property
    def channel(self) -> str | None:
        """Channel identifier."""
        ...


@runtime_checkable
class Copyable(Protocol):
    """Protocol for classes that support copy()."""

    def copy(self) -> Any:
        """Return a copy of the object."""
        ...


@runtime_checkable
class HasPhaseMethods(Protocol):
    """Protocol for classes that provide degree/radian phase methods."""

    def degree(self, unwrap: bool = False, **kwargs: Any) -> Any:
        """Return phase in degrees."""
        ...

    def radian(self, unwrap: bool = False, **kwargs: Any) -> Any:
        """Return phase in radians."""
        ...


class SupportsSignalAnalysis(HasSeriesData, HasSeriesMetadata, Copyable, Protocol):
    """Combined protocol for SignalAnalysisMixin host classes.

    Classes using SignalAnalysisMixin must provide:
    - value: numerical data
    - unit: astropy unit
    - name: series name
    - channel: channel identifier
    - copy(): method to create a copy
    """

    def _get_meta_for_constructor(self) -> dict[str, Any]:
        """Return constructor metadata for rebuilding the object."""
        ...

    def __getitem__(self, key: Any) -> Any:
        """Support index-based slicing."""
        ...


class SupportsPhaseMethods(HasPhaseMethods, Protocol):
    """Protocol for PhaseMethodsMixin host classes.

    Classes using PhaseMethodsMixin must provide:
    - degree(): method returning phase in degrees
    - radian(): method returning phase in radians
    """

    def phase(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any:
        """Return phase (radians by default, degrees if requested)."""
        ...


class AxisApiHost(Copyable, Protocol):
    """Protocol for objects implementing AxisApiMixin operations."""

    @property
    def axis_names(self) -> tuple[str, ...]: ...

    def _set_axis_name(self, index: int, name: str) -> None: ...

    def _get_axis_index(self, key: int | str) -> int: ...

    def _swapaxes_int(self, a: int, b: int) -> Any: ...
