"""Field I/O subpackage."""

from __future__ import annotations

from . import _simulation as _simulation  # noqa: F401
from . import geotiff as _geotiff  # noqa: F401
from . import gsi_dem as _gsi_dem  # noqa: F401
from ._registration import (
    get_field_lister,
    register_field_format,
    registered_field_formats,
)
from .core import (
    list_fields,
    read_fielddict,
    read_scalarfield,
    read_tensorfield,
    read_vectorfield,
    write_scalarfield,
)
from .geotiff import (
    list_fields_geotiff,
    read_scalarfield_geotiff,
    write_scalarfield_geotiff,
)
from .gsi_dem import (
    identify_gsi_dem,
    list_fields_gsi_dem,
    read_scalarfield_gsi_dem,
)

__all__ = [
    "get_field_lister",
    "identify_gsi_dem",
    "list_fields",
    "list_fields_geotiff",
    "list_fields_gsi_dem",
    "read_fielddict",
    "read_scalarfield",
    "read_scalarfield_geotiff",
    "read_scalarfield_gsi_dem",
    "read_tensorfield",
    "read_vectorfield",
    "register_field_format",
    "registered_field_formats",
    "write_scalarfield",
    "write_scalarfield_geotiff",
]
