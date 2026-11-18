"""SegmentTable subpackage — segment-keyed analysis container."""

from __future__ import annotations

from gwexpy.table.segment.cell import SegmentCell
from gwexpy.table.segment.core import RowProxy, SegmentTable

__all__ = [
    "RowProxy",
    "SegmentCell",
    "SegmentTable",
]
