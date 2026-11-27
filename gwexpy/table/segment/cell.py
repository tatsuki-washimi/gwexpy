"""SegmentCell and CellStatus — explicit state model for SegmentTable cells.

Each cell tracks an explicit :class:`CellStatus` alongside either a concrete
*value*, a lazy *loader* callable, or a recorded *error*.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Optional

from gwexpy.table.segment.exceptions import CellLoadError

__all__ = ["CellStatus", "SegmentCell"]

_UNSET: Final = object()


class CellStatus(StrEnum):
    """Explicit lifecycle state of a :class:`SegmentCell`."""

    UNLOADED = "unloaded"
    VALID = "valid"
    MISSING = "missing"
    FAILED = "failed"
    MASKED = "masked"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class SegmentCell:
    """A single lazy-loadable payload cell with explicit state tracking.

    Parameters
    ----------
    value:
        The concrete payload object.  Passing ``value=None`` without an explicit
        ``status=CellStatus.VALID`` retains the legacy empty-cell meaning
        (``UNLOADED`` with no loader, raising ``ValueError`` on :meth:`get`).
    loader:
        A zero-argument callable that returns the payload.  Ignored when
        *status* is already :attr:`CellStatus.VALID`.
    cacheable:
        When *True* (default) the result of *loader* is cached on the cell
        after the first call, transitioning its status to :attr:`CellStatus.VALID`.
    status:
        Explicit :class:`CellStatus`.  Inferred automatically when a concrete
        non-``None`` *value* is supplied at construction time.
    error:
        Optional exception recorded when *status* is :attr:`CellStatus.FAILED`.

    """

    value: Any = field(default=_UNSET)
    loader: Optional[Callable[[], Any]] = field(default=None, repr=False)
    cacheable: bool = field(default=True)
    status: CellStatus = field(default=CellStatus.UNLOADED)
    error: Optional[BaseException] = field(default=None, repr=False)

    _loaded: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.status, CellStatus):
            self.status = CellStatus(self.status)

        if self.status == CellStatus.VALID:
            self._loaded = True
            return

        # Backward compatibility: SegmentCell(value=None) without explicit
        # status=CellStatus.VALID remains an empty UNLOADED cell.
        if (
            self.status == CellStatus.UNLOADED
            and self.value is not _UNSET
            and self.value is not None
        ):
            self.status = CellStatus.VALID
            self._loaded = True
        else:
            self._loaded = False

    # ------------------------------------------------------------------
    # Explicit state constructors
    # ------------------------------------------------------------------

    @classmethod
    def valid(cls, value: Any) -> SegmentCell:
        """Create a ``VALID`` cell holding *value* (including explicit ``None``)."""
        return cls(value=value, status=CellStatus.VALID)

    @classmethod
    def missing(cls) -> SegmentCell:
        """Create a ``MISSING`` cell whose :meth:`get` returns ``None``."""
        return cls(value=_UNSET, status=CellStatus.MISSING)

    @classmethod
    def failed(cls, error: BaseException | str) -> SegmentCell:
        """Create a ``FAILED`` cell recording *error*."""
        err = RuntimeError(error) if isinstance(error, str) else error
        return cls(value=_UNSET, status=CellStatus.FAILED, error=err)

    @classmethod
    def masked(cls) -> SegmentCell:
        """Create a ``MASKED`` cell (excluded by quality criteria)."""
        return cls(value=_UNSET, status=CellStatus.MASKED)

    @classmethod
    def not_applicable(cls) -> SegmentCell:
        """Create a ``NOT_APPLICABLE`` cell."""
        return cls(value=_UNSET, status=CellStatus.NOT_APPLICABLE)

    # ------------------------------------------------------------------
    # Resolution & cache control
    # ------------------------------------------------------------------

    def get(self, default: Any = _UNSET) -> Any:
        """Return the payload, loading it if necessary.

        Parameters
        ----------
        default:
            Optional fallback value returned instead of raising when the cell
            is ``FAILED`` or empty ``UNLOADED``, or instead of ``None`` when
            the cell is ``MISSING`` / ``MASKED`` / ``NOT_APPLICABLE``.

        Returns
        -------
        object
            The resolved payload, or ``None`` / *default* for non-valid states.

        Raises
        ------
        CellLoadError
            If the cell is already in ``FAILED`` state and *default* is not given.
        ValueError
            If the cell is ``UNLOADED`` with no *loader* and *default* is not given.

        """
        if self.status == CellStatus.VALID:
            return None if self.value is _UNSET else self.value

        if self.status in (
            CellStatus.MISSING,
            CellStatus.MASKED,
            CellStatus.NOT_APPLICABLE,
        ):
            return None if default is _UNSET else default

        if self.status == CellStatus.FAILED:
            if default is not _UNSET:
                return default
            if self.error is not None:
                raise CellLoadError(
                    f"SegmentCell failed to load ({type(self.error).__name__}: {self.error})"
                ) from self.error
            raise CellLoadError("SegmentCell is in FAILED state.")

        # UNLOADED
        if self.loader is not None:
            try:
                result = self.loader()
            except Exception as exc:
                self.status = CellStatus.FAILED
                self.error = exc
                self._loaded = False
                if default is not _UNSET:
                    return default
                raise
            if self.cacheable:
                self.value = result
                self.status = CellStatus.VALID
                self._loaded = True
            return result

        if default is not _UNSET:
            return default
        raise ValueError(
            "SegmentCell has no value and no loader; cannot resolve payload."
        )

    def get_or_none(self) -> Any:
        """Return the payload if available, otherwise ``None`` without raising."""
        return self.get(default=None)

    def is_loaded(self) -> bool:
        """Return ``True`` if the value has been loaded (or was set directly)."""
        return self._loaded

    def clear(self) -> None:
        """Discard the cached value so the next :meth:`get` call re-invokes the loader."""
        if self.loader is not None:
            self.value = _UNSET
            self.status = CellStatus.UNLOADED
            self.error = None
            self._loaded = False

    def clone(self, *, deep: bool = False) -> SegmentCell:
        """Return a new :class:`SegmentCell` with independent state tracking."""
        if self.value is _UNSET:
            val = _UNSET
        elif deep:
            val = copy.deepcopy(self.value)
        else:
            val = self.value

        err = self.error
        if deep and self.error is not None:
            try:
                err = copy.deepcopy(self.error)
            except Exception:
                err = self.error

        cell = SegmentCell(
            value=val,
            loader=self.loader,
            cacheable=self.cacheable,
            status=self.status,
            error=err,
        )
        cell._loaded = self._loaded
        return cell

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _summary(self, kind: str = "object") -> str:
        """Short human-readable summary used by SegmentTable display."""
        if self.status == CellStatus.UNLOADED:
            if self.loader is not None:
                return f"<lazy: {kind}>"
            return "<empty>"
        if self.status == CellStatus.MISSING:
            return "<missing>"
        if self.status == CellStatus.MASKED:
            return "<masked>"
        if self.status == CellStatus.NOT_APPLICABLE:
            return "<not_applicable>"
        if self.status == CellStatus.FAILED:
            if self.error is not None:
                return f"<failed: {type(self.error).__name__}>"
            return "<failed>"

        # VALID
        val = None if self.value is _UNSET else self.value
        if val is None:
            return "None"
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
        summary = repr(val)
        return summary if len(summary) <= 30 else summary[:27] + "..."
