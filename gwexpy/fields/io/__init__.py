"""Field I/O subpackage."""

from __future__ import annotations

from . import _simulation as _simulation  # noqa: F401
from . import geotiff as _geotiff  # noqa: F401
from . import gsi_dem as _gsi_dem  # noqa: F401
from . import hdf5 as _hdf5  # noqa: F401
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
    write_fielddict,
    write_scalarfield,
    write_tensorfield,
    write_vectorfield,
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
from .hdf5 import (
    FIELD_DICT_SCHEMA_NAME,
    FIELD_SCHEMA_NAME,
    FIELD_SCHEMA_VERSION,
    identify_field_hdf5,
    list_fields_hdf5,
    read_fielddict_hdf5,
    read_scalarfield_hdf5,
    read_tensorfield_hdf5,
    read_vectorfield_hdf5,
    write_fielddict_hdf5,
    write_scalarfield_hdf5,
    write_tensorfield_hdf5,
    write_vectorfield_hdf5,
)

__all__ = [
    "FIELD_DICT_SCHEMA_NAME",
    "FIELD_SCHEMA_NAME",
    "FIELD_SCHEMA_VERSION",
    "get_field_lister",
    "identify_field_hdf5",
    "identify_gsi_dem",
    "list_fields",
    "list_fields_geotiff",
    "list_fields_gsi_dem",
    "list_fields_hdf5",
    "read_fielddict",
    "read_fielddict_hdf5",
    "read_scalarfield",
    "read_scalarfield_geotiff",
    "read_scalarfield_gsi_dem",
    "read_scalarfield_hdf5",
    "read_tensorfield",
    "read_tensorfield_hdf5",
    "read_vectorfield",
    "read_vectorfield_hdf5",
    "register_field_format",
    "registered_field_formats",
    "write_fielddict",
    "write_fielddict_hdf5",
    "write_scalarfield",
    "write_scalarfield_geotiff",
    "write_scalarfield_hdf5",
    "write_tensorfield",
    "write_tensorfield_hdf5",
    "write_vectorfield",
    "write_vectorfield_hdf5",
]
