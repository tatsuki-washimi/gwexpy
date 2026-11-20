"""Profile abstraction and registry for EXODUS II datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from astropy import units as u

__all__ = [
    "ExodusProfile",
    "get_exodus_profile",
    "register_exodus_profile",
]


@dataclass(frozen=True)
class CanonicalVariableSpec:
    """Profile definition for a canonical physical variable."""

    canonical_name: str
    quantity_kind: str
    default_si_unit: u.UnitBase
    is_vector: bool = False
    is_surface_2d: bool = False
    is_static_time: bool = False
    is_mask: bool = False
    aliases: tuple[str, ...] = ()
    component_triples: tuple[tuple[str, str, str], ...] = ()


class ExodusProfile:
    """Base EXODUS II interpretation profile (generic ``format='exo'``)."""

    name: str = "generic"
    format_name: str = "exo"
    default_unit_system: str = "si"

    def resolve_variable_spec(self, var_name: str) -> CanonicalVariableSpec | None:
        return None

    def canonical_name(self, var_name: str) -> str:
        spec = self.resolve_variable_spec(var_name)
        return spec.canonical_name if spec is not None else var_name

    def quantity_kind(self, var_name: str) -> str:
        spec = self.resolve_variable_spec(var_name)
        return spec.quantity_kind if spec is not None else "custom"

    def profile_units_map(self) -> Mapping[str, u.UnitBase]:
        return {}

    def find_vector_components(
        self,
        available_names: Sequence[str],
        requested: str = "velocity",
    ) -> tuple[str, str, str] | None:
        return None

    def find_favor_variable(self, available_names: Sequence[str]) -> str | None:
        return None

    def find_vof_variable(self, available_names: Sequence[str]) -> str | None:
        return None

    def is_surface_variable(self, var_name: str) -> bool:
        return False

    def is_static_variable(self, var_name: str) -> bool:
        return False

    def is_mask_variable(self, var_name: str) -> bool:
        return False

    def default_association(self) -> Literal["cell", "point"] | None:
        return None


def find_vector_components(
    profile: ExodusProfile, available: Sequence[str], name: str
) -> tuple[str, str, str] | None:
    """Resolve a named vector using the profile or a component naming convention."""
    resolved = profile.find_vector_components(available, requested=name)
    if resolved is not None:
        return resolved
    by_lower = {item.lower(): item for item in available}
    for separator in ("_", ""):
        prefix = f"{name.lower()}{separator}"
        x, y, z = (by_lower.get(prefix + axis) for axis in "xyz")
        if x is not None and y is not None and z is not None:
            return x, y, z
    return None


_PROFILES: dict[str, ExodusProfile] = {
    "exo": ExodusProfile(),
    "generic": ExodusProfile(),
}


def register_exodus_profile(key: str, profile: ExodusProfile) -> None:
    """Register an EXODUS II interpretation profile under ``key``."""
    _PROFILES[key] = profile


def get_exodus_profile(
    format_or_profile: str | ExodusProfile | None = None,
) -> ExodusProfile:
    """Resolve an EXODUS II profile by format name (``'exo'``, ``'exo.flow3d'``)."""
    if isinstance(format_or_profile, ExodusProfile):
        return format_or_profile
    if format_or_profile is None:
        return _PROFILES["exo"]
    key = str(format_or_profile).lower()
    if key not in _PROFILES:
        raise ValueError(
            f"Unknown EXODUS II profile/format {format_or_profile!r}; "
            f"available: {sorted(_PROFILES)}"
        )
    return _PROFILES[key]
