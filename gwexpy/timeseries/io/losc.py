from __future__ import annotations

import gwpy.timeseries.io.losc as _gwpy_losc

_PUBLIC = getattr(_gwpy_losc, "__all__", None) or [
    name for name in dir(_gwpy_losc) if not name.startswith("_")
]

globals().update({name: getattr(_gwpy_losc, name) for name in _PUBLIC})

__all__ = list(_PUBLIC)
