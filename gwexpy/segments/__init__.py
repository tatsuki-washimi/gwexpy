from __future__ import annotations

# Dynamic import from gwpy (PEP 562)
import gwpy.segments


__all__ = [name for name in dir(gwpy.segments) if not name.startswith("_")]


def __getattr__(name):
    return getattr(gwpy.segments, name)


def __dir__():
    return sorted(set(globals().keys()) | set(dir(gwpy.segments)))
