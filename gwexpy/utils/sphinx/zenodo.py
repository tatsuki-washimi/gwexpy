from __future__ import annotations

_IMPORT_ERROR = ImportError("gwpy.utils.sphinx.zenodo was removed in GWpy 4")
_UNAVAILABLE_NAMES = {
    "DEFAULT_HITS",
    "DEFAULT_ZENODO_URL",
    "create_parser",
    "format_citations",
    "main",
}

__all__: list[str] = []


def __getattr__(name: str):
    if name not in _UNAVAILABLE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    msg = "gwpy.utils.sphinx.zenodo is unavailable in GWpy 4"
    raise ImportError(msg) from _IMPORT_ERROR
