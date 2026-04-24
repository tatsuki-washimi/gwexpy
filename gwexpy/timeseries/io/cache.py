from __future__ import annotations

import gwpy.timeseries.io.cache as _gwpy_cache

_PUBLIC = getattr(_gwpy_cache, "__all__", None) or [
    name for name in dir(_gwpy_cache) if not name.startswith("_")
]

globals().update({name: getattr(_gwpy_cache, name) for name in _PUBLIC})

__all__ = list(_PUBLIC)
