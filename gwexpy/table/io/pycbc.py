from __future__ import annotations

from ._proxy import bind_gwpy_proxy

__all__ = bind_gwpy_proxy(globals(), "gwpy.table.io.pycbc")
