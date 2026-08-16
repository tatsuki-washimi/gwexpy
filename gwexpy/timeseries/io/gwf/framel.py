from __future__ import annotations

try:
    import gwpy.timeseries.io.gwf.framel as _gwpy_framel
except ModuleNotFoundError as exc:
    if exc.name != "framel":
        raise
    _gwpy_framel = None
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None

if _gwpy_framel is None:
    _PUBLIC: list[str] = []
    _UNAVAILABLE_NAMES = {
        "FRAMEL_COMPRESSION_GZIP",
        "FrameLVectDict",
        "framel",
        "read",
        "write",
    }
else:
    _PUBLIC = getattr(_gwpy_framel, "__all__", None) or [
        name for name in dir(_gwpy_framel) if not name.startswith("_")
    ]
    globals().update({name: getattr(_gwpy_framel, name) for name in _PUBLIC})

__all__ = list(_PUBLIC)


def __getattr__(name: str):
    if _IMPORT_ERROR is not None and name in _UNAVAILABLE_NAMES:
        msg = "gwpy FrameL support is unavailable; install python-framel"
        raise ImportError(msg) from _IMPORT_ERROR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
