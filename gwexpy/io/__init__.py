"""
gwexpy.io helpers and registration hooks.
"""

from . import utils  # noqa: F401


# Dynamic import from gwpy
import gwpy.io
for key in dir(gwpy.io):
    if not key.startswith("_") and key not in locals():
        locals()[key] = getattr(gwpy.io, key)
