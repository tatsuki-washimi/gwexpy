from __future__ import annotations

# Dynamic import from gwpy (PEP 562)
from importlib import import_module as _import_module

import gwpy.table

# gwexpy-native additions
from gwexpy.table.segment_cell import SegmentCell as SegmentCell
from gwexpy.table.segment_table import RowProxy as RowProxy
from gwexpy.table.segment_table import SegmentTable as SegmentTable

_gwpy_all = [name for name in dir(gwpy.table) if not name.startswith("_")]
_CURATED_PROXY_MODULES = {"filter", "table"}

__all__ = sorted(set(_gwpy_all) | {"SegmentCell", "SegmentTable", "RowProxy"})


def __getattr__(name: str):
    if name in _CURATED_PROXY_MODULES:
        return _import_module(f"{__name__}.{name}")
    # Fall back to gwpy.table for names not defined here
    if name in _gwpy_all:
        return getattr(gwpy.table, name)
    raise AttributeError(f"module 'gwexpy.table' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(dir(gwpy.table)))
