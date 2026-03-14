from __future__ import annotations

from gwpy.segments.io import hdf5 as _gwpy_hdf5
from gwpy.segments.io.hdf5 import *  # noqa: F403

__all__ = getattr(_gwpy_hdf5, "__all__", [name for name in dir(_gwpy_hdf5) if not name.startswith("_")])
