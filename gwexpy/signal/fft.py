from __future__ import annotations

try:
    from gwpy.signal.fft import *  # type: ignore[import-not-found]  # noqa: F403
except ImportError:
    pass
