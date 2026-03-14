from __future__ import annotations

from gwpy.segments import flag as _gwpy_flag
from gwpy.segments.flag import *  # noqa: F403

__all__ = getattr(_gwpy_flag, "__all__", [name for name in dir(_gwpy_flag) if not name.startswith("_")])
