from __future__ import annotations

_REMOVED_MESSAGE = (
    "gwpy.table.io.fetch was removed from GWpy 4.x; use gwexpy.table.io.gwosc "
    "for GWOSC catalog access or EventTable fetch/read APIs where available."
)

__all__: list[str] = []


def __getattr__(name: str) -> object:
    raise AttributeError(f"{name!r} is unavailable: {_REMOVED_MESSAGE}")
