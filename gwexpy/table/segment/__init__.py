"""SegmentTable subpackage — segment-keyed analysis container."""

from __future__ import annotations

from gwexpy.table.segment.aggregate import Quantile, SegmentTableGroupBy, quantile
from gwexpy.table.segment.cell import CellStatus, SegmentCell
from gwexpy.table.segment.core import RowProxy, SegmentTable
from gwexpy.table.segment.exceptions import CellLoadError
from gwexpy.table.segment.expr import (
    BinOp,
    Call,
    ColumnRef,
    Expr,
    Literal,
    SegmentColumn,
    UnaryOp,
    col,
)
from gwexpy.table.segment.io import (
    get_payload_serializer,
    register_payload_serializer,
)
from gwexpy.table.segment.stats import Reducer
from gwexpy.table.segment.summary import SummaryTable

__all__ = [
    "BinOp",
    "Call",
    "CellLoadError",
    "CellStatus",
    "ColumnRef",
    "Expr",
    "Literal",
    "RowProxy",
    "Reducer",
    "Quantile",
    "SegmentTableGroupBy",
    "SummaryTable",
    "SegmentCell",
    "SegmentColumn",
    "SegmentTable",
    "UnaryOp",
    "col",
    "get_payload_serializer",
    "register_payload_serializer",
    "quantile",
]
