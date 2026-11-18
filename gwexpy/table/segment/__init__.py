"""SegmentTable subpackage — segment-keyed analysis container."""

from __future__ import annotations

from gwexpy.table.segment.cell import CellStatus, SegmentCell
from gwexpy.table.segment.core import RowProxy, SegmentTable
from gwexpy.table.segment.exceptions import CellLoadError

__all__ = [
    "CellLoadError",
    "CellStatus",
    "RowProxy",
    "SegmentCell",
    "SegmentTable",
]
