"""SegmentCell — lazy payload cell for SegmentTable.

Each cell holds either a concrete *value* or a *loader* callable (or both).
Accessing :meth:`get` resolves the cell on demand and optionally caches the result.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

__all__ = ["SegmentCell"]


@dataclass
class SegmentCell:
    """A single lazy-loadable payload cell.

    Parameters
    ----------
    value:
        The concrete payload object.  May be *None* when a *loader* is set.
    loader:
        A zero-argument callable that returns the payload.  Ignored when
        *value* is already set.
    cacheable:
        When *True* (default) the result of *loader* is stored in *value*
        after the first call, so subsequent calls return directly.

    Examples
    --------
    >>> cell = SegmentCell(loader=lambda: 42, cacheable=True)
    >>> cell.get()
    42
    >>> cell.is_loaded()
    True
    """

    value: Optional[Any] = field(default=None)
    loader: Optional[Callable[[], Any]] = field(default=None, repr=False)
    cacheable: bool = field(default=True)

    # Internal sentinel so we can distinguish "not yet set" from explicit None.
    _loaded: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # If a concrete value was provided at construction time, mark as loaded.
        if self.value is not None:
            self._loaded = True

    def get(self) -> Any:
        """Return the payload, loading it if necessary.

        Returns
        -------
        object
            The resolved payload.

        Raises
        ------
        ValueError
            If both *value* and *loader* are ``None``.
        """
        if self._loaded:
            return self.value

        if self.loader is not None:
            result = self.loader()
            if self.cacheable:
                self.value = result
                self._loaded = True
            return result

        raise ValueError(
            "SegmentCell has no value and no loader; cannot resolve payload."
        )

    def is_loaded(self) -> bool:
        """Return ``True`` if the value has been loaded (or was set directly)."""
        return self._loaded

    def clear(self) -> None:
        """Discard the cached value so the next :meth:`get` call re-invokes the loader."""
        if self.loader is not None:
            self.value = None
            self._loaded = False
        # If there is no loader, clearing is a no-op (value is the only source).

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _summary(self, kind: str = "object") -> str:
        """Short human-readable summary used by SegmentTable display."""
        if not self._loaded and self.loader is not None:
            return f"<lazy: {kind}>"
        val = self.value
        if val is None:
            return "<empty>"
        # Kind-specific summaries
        try:
            from gwpy.frequencyseries import FrequencySeries
            from gwpy.timeseries import TimeSeries, TimeSeriesDict

            if isinstance(val, TimeSeriesDict):
                return f"<timeseriesdict: {len(val)} ch>"
            if isinstance(val, TimeSeries):
                return f"<timeseries: {len(val)} samples>"
            if isinstance(val, FrequencySeries):
                return f"<frequencyseries: {len(val)} bins>"
        except ImportError:
            pass
        # Generic fallback
        summary = repr(val)
        return summary if len(summary) <= 30 else summary[:27] + "..."
