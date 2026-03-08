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


# ----------------------------------------------------------------
# Structural protocols for cross-module type checking
# (used to break circular dependencies between modules)
# ----------------------------------------------------------------


@runtime_checkable
class SupportsPlot(HasSeriesData, HasSeriesMetadata, Protocol):
    """Protocol for objects that can be plotted.

    Matches TimeSeries, FrequencySeries, and other series types
    that expose value, xindex, unit, and name.
    """

    @property
    def xindex(self) -> Any:
        """The x-axis index (times, frequencies, etc.)."""
        ...


@runtime_checkable
class SupportsTimeSeries(HasSeriesData, HasSeriesMetadata, Protocol):
    """Structural protocol for TimeSeries-like objects.

    Used by plot, spectrogram, and interop modules to check for
    time-domain data without importing the concrete TimeSeries class.
    """

    @property
    def t0(self) -> Any:
        """Start time."""
        ...

    @property
    def dt(self) -> Any:
        """Time step."""
        ...

    @property
    def times(self) -> Any:
        """Time axis."""
        ...

    @property
    def sample_rate(self) -> Any:
        """Sample rate."""
        ...


@runtime_checkable
class SupportsFrequencySeries(HasSeriesData, HasSeriesMetadata, Protocol):
    """Structural protocol for FrequencySeries-like objects.

    Used by plot, spectrogram, and interop modules to check for
    frequency-domain data without importing the concrete class.
    """

    @property
    def frequencies(self) -> Any:
        """Frequency axis."""
        ...

    @property
    def df(self) -> Any:
        """Frequency step."""
        ...

    @property
    def f0(self) -> Any:
        """Starting frequency."""
        ...


@runtime_checkable
class SupportsSpectrogram(HasSeriesData, Protocol):
    """Structural protocol for Spectrogram-like objects.

    Used by plot and interop modules to check for time-frequency
    data without importing the concrete Spectrogram class.
    """

    @property
    def times(self) -> Any:
        """Time axis."""
        ...

    @property
    def frequencies(self) -> Any:
        """Frequency axis."""
        ...


@runtime_checkable
class SupportsSeriesCollection(Protocol):
    """Protocol for dict-like series collections (TimeSeriesDict, etc.)."""

    def keys(self) -> Any: ...
    def values(self) -> Any: ...
    def items(self) -> Any: ...


@runtime_checkable
class SupportsSeriesList(Protocol):
    """Protocol for list-like series collections (TimeSeriesList, etc.)."""

    def __iter__(self) -> Any: ...
    def __len__(self) -> int: ...
    def append(self, item: Any) -> None: ...


@runtime_checkable
class SupportsMatrix(Protocol):
    """Protocol for SeriesMatrix-like objects."""

    def to_series_1Dlist(self) -> Any: ...
    def row_keys(self) -> Any: ...
    def col_keys(self) -> Any: ...
