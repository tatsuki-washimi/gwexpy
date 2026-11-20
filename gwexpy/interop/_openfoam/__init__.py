"""Internal implementation package for OpenFOAM simulation readers."""

from .backend import (
    LegacyVTKOpenFOAMBackend,
    OpenFOAMCase,
    PyVistaOpenFOAMBackend,
    load_openfoam_density,
    read_openfoam_vtk,
)
from .convert import (
    from_openfoam,
    list_fields_openfoam,
    read_fielddict_openfoam,
    read_scalarfield_openfoam,
    read_vectorfield_openfoam,
    register_openfoam_format,
    register_openfoam_formats,
)
from .dimensions import (
    OPENFOAM_BASE_UNITS,
    OpenFOAMDimensions,
    OpenFOAMHeader,
    classify_pressure_dimensions,
    kinematic_to_physical_pressure,
    parse_control_dict,
    parse_foam_field_metadata,
    parse_foam_header,
    parse_openfoam_dimensions,
)
from .grid import (
    RegularGridSpec,
    check_moving_mesh,
    infer_exact_cartesian_grid,
    sample_to_regular_grid,
)
from .model import (
    FoamFieldData,
    FoamSnapshot,
    FoamVolumeMesh,
    OpenFOAMDensitySnapshot,
    OpenFOAMProvenance,
)

__all__ = [
    "OPENFOAM_BASE_UNITS",
    "FoamFieldData",
    "FoamSnapshot",
    "FoamVolumeMesh",
    "LegacyVTKOpenFOAMBackend",
    "OpenFOAMCase",
    "OpenFOAMDensitySnapshot",
    "OpenFOAMDimensions",
    "OpenFOAMHeader",
    "OpenFOAMProvenance",
    "PyVistaOpenFOAMBackend",
    "RegularGridSpec",
    "check_moving_mesh",
    "classify_pressure_dimensions",
    "from_openfoam",
    "infer_exact_cartesian_grid",
    "kinematic_to_physical_pressure",
    "list_fields_openfoam",
    "load_openfoam_density",
    "parse_control_dict",
    "parse_foam_field_metadata",
    "parse_foam_header",
    "parse_openfoam_dimensions",
    "read_fielddict_openfoam",
    "read_openfoam_vtk",
    "read_scalarfield_openfoam",
    "read_vectorfield_openfoam",
    "register_openfoam_format",
    "register_openfoam_formats",
    "sample_to_regular_grid",
]
