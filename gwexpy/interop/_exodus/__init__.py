"""Internal implementation package for EXODUS II and FLOW-3D simulation readers."""

from .backend import ExodusCase, ExodusIIBackend
from .convert import (
    from_exodus,
    list_fields_exodus,
    read_fielddict_exodus,
    read_scalarfield_exodus,
    read_vectorfield_exodus,
    register_exodus_format,
    register_exodus_formats,
)
from .flow3d_profile import FLOW3D_PROFILE_SPECS, Flow3DExodusProfile
from .grid import RegularGridSpec, infer_exact_cartesian_grid, sample_exodus_to_grid
from .masks import Flow3DMaskSet, apply_flow3d_masks, compute_flow3d_masks
from .model import (
    ExodusBlock,
    ExodusProvenance,
    ExodusSnapshot,
    ExodusVariable,
    Flow3DProvenance,
)
from .profiles import ExodusProfile, get_exodus_profile, register_exodus_profile
from .units import (
    FLOW3D_CGS_UNITS,
    FLOW3D_SI_UNITS,
    parse_exodus_unit_str,
    resolve_coordinate_scale,
    resolve_variable_unit,
)

__all__ = [
    "FLOW3D_CGS_UNITS",
    "FLOW3D_PROFILE_SPECS",
    "FLOW3D_SI_UNITS",
    "ExodusBlock",
    "ExodusCase",
    "ExodusIIBackend",
    "ExodusProfile",
    "ExodusProvenance",
    "ExodusSnapshot",
    "ExodusVariable",
    "Flow3DExodusProfile",
    "Flow3DMaskSet",
    "Flow3DProvenance",
    "RegularGridSpec",
    "apply_flow3d_masks",
    "compute_flow3d_masks",
    "from_exodus",
    "get_exodus_profile",
    "infer_exact_cartesian_grid",
    "list_fields_exodus",
    "parse_exodus_unit_str",
    "read_fielddict_exodus",
    "read_scalarfield_exodus",
    "read_vectorfield_exodus",
    "register_exodus_format",
    "register_exodus_formats",
    "register_exodus_profile",
    "resolve_coordinate_scale",
    "resolve_variable_unit",
    "sample_exodus_to_grid",
]
