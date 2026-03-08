from __future__ import annotations

# Dynamic import from gwpy (PEP 562)
import gwpy.astro


__all__ = [name for name in dir(gwpy.astro) if not name.startswith("_")]


def __getattr__(name):
    return getattr(gwpy.astro, name)


def __dir__():
    return sorted(set(globals().keys()) | set(dir(gwpy.astro)))
