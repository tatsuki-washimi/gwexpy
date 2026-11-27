"""Backward-compatible re-export shim for :mod:`gwexpy.table.segment.core`."""

from __future__ import annotations

from gwexpy.table.segment.cell import SegmentCell as SegmentCell
from gwexpy.table.segment.core import (
    _ALL_KINDS as _ALL_KINDS,
)
from gwexpy.table.segment.core import (
    _META_KINDS as _META_KINDS,
)
from gwexpy.table.segment.core import (
    _PAYLOAD_KINDS as _PAYLOAD_KINDS,
)
from gwexpy.table.segment.core import (
    RowProxy as RowProxy,
)
from gwexpy.table.segment.core import (
    SegmentTable as SegmentTable,
)
from gwexpy.table.segment.core import (
    _coerce_span_value as _coerce_span_value,
)
from gwexpy.table.segment.core import (
    _infer_kind as _infer_kind,
)
from gwexpy.table.segment.core import (
    _is_sequence as _is_sequence,
)
from gwexpy.table.segment.core import (
    _parse_span_string as _parse_span_string,
)

__all__ = ["RowProxy", "SegmentTable"]
