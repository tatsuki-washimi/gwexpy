"""Lazy compatibility proxy for the optional GWpy FrameL backend."""

# ruff: noqa: F822

from __future__ import annotations

import importlib
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


def import_module(name: str, package: str | None = None) -> ModuleType:
    """Resolve an optional backend at call time.

    Keeping this small indirection (rather than binding
    ``importlib.import_module`` at module import) is important for two reasons:
    tests can still replace ``proxy.import_module`` to model a missing
    backend, and temporary import guards do not leak into this lazy proxy after
    their fixture has been torn down.
    """
    return importlib.import_module(name, package)


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
