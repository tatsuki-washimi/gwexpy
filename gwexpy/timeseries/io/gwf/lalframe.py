from __future__ import annotations

import gwpy.timeseries.io.gwf.lalframe as _gwpy_lalframe

_PUBLIC = getattr(_gwpy_lalframe, "__all__", None) or [
    name for name in dir(_gwpy_lalframe) if not name.startswith("_")
]

globals().update({name: getattr(_gwpy_lalframe, name) for name in _PUBLIC})

__all__ = list(_PUBLIC)
