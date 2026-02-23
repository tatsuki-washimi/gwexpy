"""
Placeholder readers for unsupported formats (WIN/WIN32, SDB, vendor loggers).
"""

from __future__ import annotations

from astropy.io import registry as io_registry

from .. import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

_P2_FORMATS = [
    "orf",
    "mem",
    "wvf",
    "wdf",
    "taffmat",
    "lsf",
    "li",
]


def _stub(fmt):
    def _reader(*args, **kwargs):
        from ...interop.errors import raise_unimplemented_io

        raise_unimplemented_io(fmt)

    return _reader


for _fmt in _P2_FORMATS:
    reader = _stub(_fmt)
    io_registry.register_reader(_fmt, TimeSeries, reader, force=True)
    io_registry.register_reader(_fmt, TimeSeriesDict, reader, force=True)
    io_registry.register_reader(_fmt, TimeSeriesMatrix, reader, force=True)
