from __future__ import annotations

try:
    import gwpy.timeseries.io.gwf.framecpp as _gwpy_framecpp
except ModuleNotFoundError as exc:
    if exc.name != "LDAStools":
        raise
    _gwpy_framecpp = None
    _IMPORT_ERROR: ImportError | None = exc
else:
    _IMPORT_ERROR = None

if _gwpy_framecpp is None:
    _PUBLIC: list[str] = []
    _UNAVAILABLE_NAMES = {
        "FRAME_LIBRARY",
        "FRERR_NO_CHANNEL_OF_TYPE",
        "FRERR_NO_FRAME_AT_NUM",
        "frameCPP",
        "io_framecpp",
        "io_gwf",
        "read",
        "read_frdata",
        "read_frvect",
        "read_gwf",
        "write",
    }
else:
    _PUBLIC = getattr(_gwpy_framecpp, "__all__", None) or [
        name for name in dir(_gwpy_framecpp) if not name.startswith("_")
    ]
    globals().update({name: getattr(_gwpy_framecpp, name) for name in _PUBLIC})

__all__ = list(_PUBLIC)


def __getattr__(name: str):
    if _IMPORT_ERROR is not None and name in _UNAVAILABLE_NAMES:
        msg = "gwpy frameCPP support is unavailable; install LDAStools"
        raise ImportError(msg) from _IMPORT_ERROR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
