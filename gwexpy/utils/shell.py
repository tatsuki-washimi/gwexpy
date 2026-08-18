from __future__ import annotations

_IMPORT_ERROR = ImportError("gwpy.utils.shell was removed in GWpy 4")
_UNAVAILABLE_NAMES = {
    "PIPE",
    "CalledProcessError",
    "Popen",
    "call",
    "deprecated_function",
    "which",
}

__all__: list[str] = []


def __getattr__(name: str):
    if name not in _UNAVAILABLE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    msg = "gwpy.utils.shell is unavailable in GWpy 4"
    raise ImportError(msg) from _IMPORT_ERROR
