from __future__ import annotations

import gwpy.timeseries.io.gwf.framecpp as _gwpy_framecpp

_PUBLIC = getattr(_gwpy_framecpp, "__all__", None) or [
    name for name in dir(_gwpy_framecpp) if not name.startswith("_")
]

globals().update({name: getattr(_gwpy_framecpp, name) for name in _PUBLIC})

__all__ = list(_PUBLIC)
