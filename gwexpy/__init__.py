from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("gwexpy")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

from . import (
    astro,
    cli,
    detector,
    frequencyseries,
    io,
    plot,
    segments,
    signal,
    spectrogram,
    table,
    testing,
    time,
    timeseries,
    types,
    utils,
)

__all__ = ["__version__"]

