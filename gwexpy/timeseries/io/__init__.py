"""
gwexpy.timeseries.io
--------------------

Registrations for additional TimeSeries readers.
"""

# Readers are registered on import
from . import gbd  # noqa: F401
from . import dttxml  # noqa: F401
from . import seismic  # noqa: F401
from . import tdms  # noqa: F401
from . import win  # noqa: F401
from . import wav  # noqa: F401
from . import ats  # noqa: F401
from . import sdb  # noqa: F401
from . import stubs  # noqa: F401

