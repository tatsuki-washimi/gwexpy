from __future__ import annotations

import gwpy.timeseries.statevector as _sv

# Dynamically re-export everything from gwpy.timeseries.statevector
globals().update({name: getattr(_sv, name) for name in getattr(_sv, "__all__", [])})

__all__ = list(getattr(_sv, "__all__", []))
