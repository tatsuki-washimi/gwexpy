from __future__ import annotations

_IMPORT_ERROR = ImportError("gwpy.utils.sphinx.ex2rst was removed in GWpy 4")
_UNAVAILABLE_NAMES = {
    "METADATA",
    "Path",
    "create_parser",
    "ex2rst",
    "main",
    "postprocess_code",
}

__all__: list[str] = []


def __getattr__(name: str):
    if name not in _UNAVAILABLE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    msg = "gwpy.utils.sphinx.ex2rst is unavailable in GWpy 4"
    raise ImportError(msg) from _IMPORT_ERROR
