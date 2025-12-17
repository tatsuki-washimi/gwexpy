"""
Placeholder readers for unsupported formats (WIN/WIN32, SDB, vendor loggers).
"""

from gwpy.io import registry as io_registry

from ..timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix

_P2_FORMATS = [
    "win",
    "win32",
    "sdb",
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
        raise NotImplementedError(
            f"Format '{fmt}' is not implemented yet. "
            "Please supply a specification / sample file / reference parser."
        )

    return _reader


for _fmt in _P2_FORMATS:
    reader = _stub(_fmt)
    io_registry.register_reader(_fmt, TimeSeries, reader)
    io_registry.register_reader(_fmt, TimeSeriesDict, reader)
    io_registry.register_reader(_fmt, TimeSeriesMatrix, reader)

