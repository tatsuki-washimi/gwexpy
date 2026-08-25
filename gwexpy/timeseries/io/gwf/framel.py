"""Lazy compatibility proxy for the optional GWpy FrameL backend."""

# ruff: noqa: F822

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_UPSTREAM = "gwpy.timeseries.io.gwf.framel"
__all__ = (
    "FRAME_LIBRARY",
    "Segment",
    "TimeSeries",
    "file_list",
    "file_path",
    "framel",
    "read",
    "warnings",
    "write",
)  # noqa: F822
_module: ModuleType | None = None
_DIRECTORY = tuple(sorted(__all__))


def _load() -> ModuleType:
    global _module
    if _module is None:
        _module = import_module(_UPSTREAM)
    return _module


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_load(), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return list(_DIRECTORY)
