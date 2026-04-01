from __future__ import annotations

# This module is a proxy for gwpy.io._framecpp.
# Since it is an internal module, we only import if it exists.
try:
    from gwpy.io import _framecpp
    __all__ = [name for name in dir(_framecpp) if not name.startswith("_")]
    # Note: we are not using wildcard imports here to satisfy audit requirements.
    for name in __all__:
        globals()[name] = getattr(_framecpp, name)
except ImportError:
    __all__ = []
