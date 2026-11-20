"""Explicit FLOW-3D EXODUS II profile (``format='exo.flow3d'``).

Documented FLOW-3D EXODUS II Profile Contract
--------------------------------------------
Since genuine proprietary FLOW-3D ``.exo`` outputs are not bundled in this
repository, this module defines and documents the explicit supported profile for
``format="exo.flow3d"``:

- **Blocks**: One or more EXODUS II element blocks (``HEX8``, ``QUAD4``,
  ``TETRA``, etc.), selectable via ``block=`` and combinable via
  ``block_mode="auto" | "single" | "separate" | "merge"``.
- **Variables & Canonical Aliases** (case-insensitive):
  - ``density``: ``density``, ``rho``, ``rhor``, ``fluid_density`` [SI: ``kg/m^3``]
  - ``pressure``: ``pressure``, ``p``, ``p_abs``, ``dynamic_pressure`` [SI: ``Pa``]
  - ``velocity``: vector ``velocity``/``vel``/``U`` or scalar component triples
    ``(u, v, w)``, ``(vx, vy, vz)``, ``(vel_x, vel_y, vel_z)``,
    ``(velocity_x, velocity_y, velocity_z)`` [SI: ``m/s``]
  - ``flow_depth``: ``flow_depth``, ``depth``, ``h``, ``water_depth``,
    ``fluid_depth`` -> 2D surface-like field of shape ``(nt, nx, ny, 1)`` [SI: ``m``]
  - ``terrain_elevation``: ``terrain_elevation``, ``bed_elevation``,
    ``bathymetry``, ``elevation``, ``zb``, ``bottom_elevation`` -> static 2D
    surface field of shape ``(1, nx, ny, 1)`` [SI: ``m``]
  - ``fluid_fraction``: ``f``, ``vof``, ``fluid_fraction``,
    ``fraction_of_fluid`` [dimensionless]
  - ``favor_open_volume``: ``vf``, ``favor``, ``open_volume_fraction``,
    ``vf_open``, ``favor_open_volume`` [dimensionless]
- **Association**: Preserves native EXODUS II ``cell`` (element) vs ``point``
  (nodal) association; never silently averages cell values onto nodes.
- **Masking**: FAVOR open-volume fraction distinguishes physically zero fluid
  quantities (dry bed ``flow_depth=0``, zero velocity, gauge ``p=0``) from
  FAVOR-blocked solid obstacle cells (``vf <= favor_threshold`` -> ``NaN``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

from astropy import units as u

from .profiles import CanonicalVariableSpec, ExodusProfile, register_exodus_profile
from .units import FLOW3D_SI_UNITS

__all__ = ["FLOW3D_PROFILE_SPECS", "Flow3DExodusProfile"]

FLOW3D_PROFILE_SPECS: tuple[CanonicalVariableSpec, ...] = (
    CanonicalVariableSpec(
        canonical_name="density",
        quantity_kind="fluid_density",
        default_si_unit=FLOW3D_SI_UNITS["density"],
        aliases=("density", "rho", "rhor", "fluid_density", "mass_density"),
    ),
    CanonicalVariableSpec(
        canonical_name="pressure",
        quantity_kind="pressure",
        default_si_unit=FLOW3D_SI_UNITS["pressure"],
        aliases=("pressure", "p", "p_abs", "dynamic_pressure", "static_pressure"),
    ),
    CanonicalVariableSpec(
        canonical_name="velocity",
        quantity_kind="velocity",
        default_si_unit=FLOW3D_SI_UNITS["velocity"],
        is_vector=True,
        aliases=("velocity", "vel", "u_vec", "velocity_vec"),
        component_triples=(
            ("u", "v", "w"),
            ("vx", "vy", "vz"),
            ("vel_x", "vel_y", "vel_z"),
            ("velocity_x", "velocity_y", "velocity_z"),
            ("u_x", "u_y", "u_z"),
        ),
    ),
    CanonicalVariableSpec(
        canonical_name="flow_depth",
        quantity_kind="flow_depth",
        default_si_unit=FLOW3D_SI_UNITS["flow_depth"],
        is_surface_2d=True,
        aliases=("flow_depth", "depth", "water_depth", "fluid_depth", "h"),
    ),
    CanonicalVariableSpec(
        canonical_name="terrain_elevation",
        quantity_kind="terrain_elevation",
        default_si_unit=FLOW3D_SI_UNITS["terrain_elevation"],
        is_surface_2d=True,
        is_static_time=True,
        aliases=(
            "terrain_elevation",
            "bed_elevation",
            "bathymetry",
            "elevation",
            "zb",
            "bottom_elevation",
            "topography",
        ),
    ),
    CanonicalVariableSpec(
        canonical_name="fluid_fraction",
        quantity_kind="vof_fraction",
        default_si_unit=FLOW3D_SI_UNITS["fluid_fraction"],
        is_mask=True,
        aliases=("f", "vof", "fluid_fraction", "fraction_of_fluid"),
    ),
    CanonicalVariableSpec(
        canonical_name="favor_open_volume",
        quantity_kind="favor_open_volume",
        default_si_unit=FLOW3D_SI_UNITS["favor_open_volume"],
        is_mask=True,
        aliases=(
            "vf",
            "favor",
            "open_volume_fraction",
            "vf_open",
            "favor_open_volume",
        ),
    ),
)

_ALIAS_LOOKUP: dict[str, CanonicalVariableSpec] = {
    alias.lower(): spec for spec in FLOW3D_PROFILE_SPECS for alias in spec.aliases
}

# Also register scalar velocity component names -> velocity_component unit
_VEL_COMP_NAMES: frozenset[str] = frozenset(
    name.lower()
    for spec in FLOW3D_PROFILE_SPECS
    if spec.canonical_name == "velocity"
    for triple in spec.component_triples
    for name in triple
)


class Flow3DExodusProfile(ExodusProfile):
    """Profile implementing FLOW-3D EXODUS II conventions (``exo.flow3d``)."""

    name: str = "flow3d"
    format_name: str = "exo.flow3d"
    profile_version: str = "flow3d-exodus-v1"
    default_unit_system: str = "si"

    def resolve_variable_spec(self, var_name: str) -> CanonicalVariableSpec | None:
        low = var_name.strip().lower()
        if low in _ALIAS_LOOKUP:
            return _ALIAS_LOOKUP[low]
        if low in _VEL_COMP_NAMES:
            return CanonicalVariableSpec(
                canonical_name="velocity",
                quantity_kind="velocity_component",
                default_si_unit=FLOW3D_SI_UNITS["velocity"],
                is_vector=False,
            )
        return None

    def profile_units_map(self) -> Mapping[str, u.UnitBase]:
        return FLOW3D_SI_UNITS

    def find_vector_components(
        self,
        available_names: Sequence[str],
        requested: str = "velocity",
    ) -> tuple[str, str, str] | None:
        if requested.lower() not in {"velocity", "vel", "u", "u_vec"}:
            return None
        by_lower = {name.lower(): name for name in available_names}
        vel_spec = next(
            s for s in FLOW3D_PROFILE_SPECS if s.canonical_name == "velocity"
        )
        for cx, cy, cz in vel_spec.component_triples:
            if cx in by_lower and cy in by_lower and cz in by_lower:
                return (by_lower[cx], by_lower[cy], by_lower[cz])
        return None

    def find_favor_variable(self, available_names: Sequence[str]) -> str | None:
        by_lower = {name.lower(): name for name in available_names}
        favor_spec = next(
            s for s in FLOW3D_PROFILE_SPECS if s.canonical_name == "favor_open_volume"
        )
        for alias in favor_spec.aliases:
            if alias in by_lower:
                return by_lower[alias]
        return None

    def find_vof_variable(self, available_names: Sequence[str]) -> str | None:
        by_lower = {name.lower(): name for name in available_names}
        vof_spec = next(
            s for s in FLOW3D_PROFILE_SPECS if s.canonical_name == "fluid_fraction"
        )
        for alias in vof_spec.aliases:
            if alias in by_lower:
                return by_lower[alias]
        return None

    def is_surface_variable(self, var_name: str) -> bool:
        spec = self.resolve_variable_spec(var_name)
        return bool(spec and spec.is_surface_2d)

    def is_static_variable(self, var_name: str) -> bool:
        spec = self.resolve_variable_spec(var_name)
        return bool(spec and spec.is_static_time)

    def is_mask_variable(self, var_name: str) -> bool:
        spec = self.resolve_variable_spec(var_name)
        return bool(spec and spec.is_mask)

    def default_association(self) -> Literal["cell", "point"] | None:
        return "cell"


_FLOW3D_PROFILE = Flow3DExodusProfile()
register_exodus_profile("exo.flow3d", _FLOW3D_PROFILE)
register_exodus_profile("flow3d", _FLOW3D_PROFILE)
