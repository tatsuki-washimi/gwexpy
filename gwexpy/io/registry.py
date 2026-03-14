from __future__ import annotations

from gwpy.io import registry as _gwpy_registry
from gwpy.io.registry import *  # noqa: F403

__all__ = getattr(
    _gwpy_registry,
    "__all__",
    [name for name in dir(_gwpy_registry) if not name.startswith("_")],
)
