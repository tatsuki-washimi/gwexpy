from __future__ import annotations

from gwpy.io import gwf as _gwpy_gwf
from gwpy.io.gwf import (
    BACKENDS,
    backend,
    channel_exists,
    core,
    data_segments,
    get_backend,
    get_backend_function,
    get_channel_names,
    get_channel_type,
    identify_gwf,
    import_backend,
    iter_channel_names,
    num_channels,
)

__all__ = getattr(_gwpy_gwf, "__all__", [name for name in dir(_gwpy_gwf) if not name.startswith("_")])
