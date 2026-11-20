"""Unit resolution and unit-system conversion helpers for EXODUS II / FLOW-3D."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from astropy import units as u

__all__ = [
    "FLOW3D_CGS_UNITS",
    "FLOW3D_SI_UNITS",
    "parse_exodus_unit_str",
    "resolve_coordinate_scale",
    "resolve_variable_unit",
]

FLOW3D_SI_UNITS: dict[str, u.UnitBase] = {
    "density": u.kg / u.m**3,
    "pressure": u.Pa,
    "velocity": u.m / u.s,
    "flow_depth": u.m,
    "terrain_elevation": u.m,
    "fluid_fraction": u.dimensionless_unscaled,
    "favor_open_volume": u.dimensionless_unscaled,
    "temperature": u.K,
    "length": u.m,
    "time": u.s,
}

FLOW3D_CGS_UNITS: dict[str, u.UnitBase] = {
    "density": u.g / u.cm**3,
    "pressure": u.barye,  # dyn / cm^2 = 0.1 Pa
    "velocity": u.cm / u.s,
    "flow_depth": u.cm,
    "terrain_elevation": u.cm,
    "fluid_fraction": u.dimensionless_unscaled,
    "favor_open_volume": u.dimensionless_unscaled,
    "temperature": u.K,
    "length": u.cm,
    "time": u.s,
}

_UNIT_ALIASES: dict[str, str] = {
    "": "1",
    "dimensionless": "1",
    "none": "1",
    "unitless": "1",
    "-": "1",
    "dyn/cm^2": "Ba",
    "dyn/cm2": "Ba",
    "dyne/cm^2": "Ba",
    "dyne/cm2": "Ba",
    "kg/m^3": "kg / m**3",
    "kg/m3": "kg / m**3",
    "g/cm^3": "g / cm**3",
    "g/cm3": "g / cm**3",
    "m/sec": "m / s",
    "cm/sec": "cm / s",
    "meters": "m",
    "meter": "m",
    "seconds": "s",
    "second": "s",
}


def parse_exodus_unit_str(raw: str | None) -> u.UnitBase | None:
    """Parse an EXODUS II variable ``units`` attribute string into an Astropy Unit."""
    if raw is None:
        return None
    cleaned = str(raw).strip()
    if not cleaned:
        return None
    alias = _UNIT_ALIASES.get(cleaned.lower())
    if alias is not None:
        return u.Unit(alias)
    try:
        return u.Unit(cleaned)
    except ValueError as exc:
        raise ValueError(f"Could not parse EXODUS unit string {raw!r}") from exc


def resolve_coordinate_scale(
    *,
    file_coord_unit: str | None = None,
    unit_system: str = "si",
    coordinate_unit: str | u.UnitBase | None = None,
) -> tuple[float, u.UnitBase]:
    """Return scale factor to convert file coordinates into metres."""
    if coordinate_unit is not None:
        src = u.Unit(coordinate_unit)
    elif file_coord_unit:
        parsed = parse_exodus_unit_str(file_coord_unit)
        src = parsed if parsed is not None else u.m
    elif unit_system.lower() == "cgs":
        src = u.cm
    elif unit_system.lower() == "si":
        src = u.m
    else:
        raise ValueError(
            f"Unsupported unit_system {unit_system!r}; expected 'si' or 'cgs'"
        )

    if not src.is_equivalent(u.m):
        raise u.UnitConversionError(
            f"Spatial coordinate unit {src} is not convertible to metres"
        )
    return float(src.to(u.m)), src


def resolve_variable_unit(
    *,
    canonical_name: str,
    file_unit_str: str | None = None,
    unit_system: str = "si",
    unit_override: Any | None = None,
    convert_to_si: bool = True,
    profile_units: Mapping[str, u.UnitBase] | None = None,
) -> tuple[u.UnitBase, float, str]:
    """Resolve source unit, numerical multiplier, and provenance source label.

    Returns ``(target_unit, scale_factor, unit_source)``.
    """
    sys_low = unit_system.lower()
    if sys_low not in {"si", "cgs"}:
        raise ValueError(
            f"Unsupported unit_system {unit_system!r}; expected 'si' or 'cgs'"
        )

    file_unit = parse_exodus_unit_str(file_unit_str)
    si_map = profile_units if profile_units is not None else FLOW3D_SI_UNITS
    cgs_map = FLOW3D_CGS_UNITS

    if file_unit is not None:
        src_unit = file_unit
        unit_source = "file_attribute"
    elif sys_low == "cgs" and canonical_name in cgs_map:
        src_unit = cgs_map[canonical_name]
        unit_source = "unit_system_cgs"
    elif canonical_name in si_map:
        src_unit = si_map[canonical_name]
        unit_source = f"profile_{sys_low}"
    else:
        src_unit = u.dimensionless_unscaled
        unit_source = "dimensionless"

    # Validate file unit against profile canonical dimension when known
    expected_si = si_map.get(canonical_name)
    if (
        file_unit is not None
        and expected_si is not None
        and unit_override is None
        and not file_unit.is_equivalent(expected_si)
    ):
        raise u.UnitConversionError(
            f"EXODUS variable {canonical_name!r} declares unit {file_unit}, "
            f"which is incompatible with expected physical dimension {expected_si}"
        )

    if unit_override is not None:
        target = u.Unit(unit_override)
        if src_unit == u.dimensionless_unscaled and file_unit is None:
            return target, 1.0, "override"
        if not src_unit.is_equivalent(target):
            raise u.UnitConversionError(
                f"Cannot convert {canonical_name!r} from {src_unit} to {target}"
            )
        return target, float(src_unit.to(target)), "override"

    if (
        convert_to_si
        and expected_si is not None
        and src_unit.is_equivalent(expected_si)
    ):
        return expected_si, float(src_unit.to(expected_si)), unit_source

    return src_unit, 1.0, unit_source
