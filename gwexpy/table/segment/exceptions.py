"""Exceptions for SegmentTable and SegmentCell."""

from __future__ import annotations

__all__ = ["CellLoadError"]


class CellLoadError(RuntimeError):
    """Raised when accessing a :class:`SegmentCell` in the ``FAILED`` state.

    When the failure originated from an exception in a loader or row-wise
    operation, the original exception is attached as ``__cause__``.
    """
