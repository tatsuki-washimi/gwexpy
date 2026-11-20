"""Physical unit conventions for SPECFEM3D seismograms and volumetric fields."""

from __future__ import annotations

from astropy import units as u

__all__ = [
    "SPECFEM_FIELD_UNITS",
    "SPECFEM_SEISMOGRAM_UNITS",
    "resolve_field_unit",
    "resolve_seismogram_unit",
]

SPECFEM_SEISMOGRAM_UNITS: dict[str, tuple[u.UnitBase, str]] = {
    "semd": (u.m, "displacement"),
    "semv": (u.m / u.s, "velocity"),
    "sema": (u.m / u.s**2, "acceleration"),
    "semp": (u.Pa, "pressure"),
}

SPECFEM_FIELD_UNITS: dict[str, u.UnitBase] = {
    "displacement": u.m,
    "velocity": u.m / u.s,
    "acceleration": u.m / u.s**2,
    "rho": u.kg / u.m**3,
    "density": u.kg / u.m**3,
    "vp": u.m / u.s,
    "vs": u.m / u.s,
    "qkappa": u.dimensionless_unscaled,
    "qmu": u.dimensionless_unscaled,
    "pressure": u.Pa,
    "curl": 1 / u.s,
    "divergence": u.dimensionless_unscaled,
    "quadrature_volume": u.m**3,
    "coordinates": u.m,
    "material_id": u.dimensionless_unscaled,
    "domain_id": u.dimensionless_unscaled,
    "jacobian_min": u.dimensionless_unscaled,
}


def resolve_seismogram_unit(
    ext_or_quantity: str,
    *,
    unit_override: object | None = None,
) -> tuple[u.UnitBase, str]:
    """Resolve Astropy unit and physical quantity name for a SPECFEM3D seismogram."""
    key = ext_or_quantity.lower().lstrip(".")
    if key in SPECFEM_SEISMOGRAM_UNITS:
        default_unit, qty_name = SPECFEM_SEISMOGRAM_UNITS[key]
    elif key in {"displacement", "velocity", "acceleration", "pressure"}:
        qty_name = key
        default_unit = SPECFEM_FIELD_UNITS[key]
    else:
        default_unit = u.dimensionless_unscaled
        qty_name = "custom"

    if unit_override is not None:
        return u.Unit(unit_override), qty_name
    return default_unit, qty_name


def resolve_field_unit(
    field_name: str,
    *,
    unit_override: object | None = None,
) -> u.UnitBase:
    """Resolve Astropy unit for a SPECFEM3D volumetric or GLL field."""
    if unit_override is not None:
        return u.Unit(unit_override)
    low = field_name.strip().lower()
    if low in SPECFEM_FIELD_UNITS:
        return SPECFEM_FIELD_UNITS[low]
    if low.startswith(("displ", "u_")):
        return u.m
    if low.startswith(("vel", "v_")):
        return u.m / u.s
    if low.startswith(("accel", "a_")):
        return u.m / u.s**2
    if low.startswith("curl"):
        return 1 / u.s
    return u.dimensionless_unscaled
