"""
gwexpy.timeseries.io
--------------------

Registrations for additional TimeSeries readers.
"""

from __future__ import annotations

# Readers are registered on import
from . import (
    ats,  # noqa: F401
    dttxml,  # noqa: F401
    gbd,  # noqa: F401
    sdb,  # noqa: F401
    seismic,  # noqa: F401
    stubs,  # noqa: F401
    tdms,  # noqa: F401
    wav,  # noqa: F401
    win,  # noqa: F401
)
