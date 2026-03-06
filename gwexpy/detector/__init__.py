from __future__ import annotations

# Dynamic import from gwpy (PEP 562)
import gwpy.detector


__all__ = [name for name in dir(gwpy.detector) if not name.startswith("_")]


def __getattr__(name):
    return getattr(gwpy.detector, name)


def __dir__():
    return sorted(set(globals().keys()) | set(dir(gwpy.detector)))
