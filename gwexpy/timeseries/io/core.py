from __future__ import annotations

_REMOVED_MESSAGE = (
    "gwpy.timeseries.io.core was removed from GWpy 4.x; use concrete "
    "gwexpy.timeseries.io modules or the TimeSeries/TimeSeriesDict read "
    "methods instead."
)

__all__: list[str] = []


def __getattr__(name: str) -> object:
    raise AttributeError(f"{name!r} is unavailable: {_REMOVED_MESSAGE}")
