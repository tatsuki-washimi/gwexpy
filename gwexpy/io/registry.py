from __future__ import annotations

from gwpy.io import registry as _gwpy_registry
from gwpy.io.registry import (
    GetExceptionGroup,
    UnifiedFetch,
    UnifiedFetchRegistry,
    UnifiedGet,
    UnifiedGetRegistry,
    UnifiedIORegistry,
    UnifiedRead,
    UnifiedWrite,
    default_registry,
    identify_factory,
    inherit_unified_io,
)

__all__ = getattr(
    _gwpy_registry,
    "__all__",
    [name for name in dir(_gwpy_registry) if not name.startswith("_")],
)
