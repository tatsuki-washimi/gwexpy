from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gwinc_ import from_pygwinc
    from .obspy_ import from_obspy

__all__ = ["from_pygwinc", "from_obspy"]


def __getattr__(name: str) -> Any:
    if name == "from_pygwinc":
        from .gwinc_ import from_pygwinc

        return from_pygwinc
    if name == "from_obspy":
        from .obspy_ import from_obspy

        return from_obspy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
