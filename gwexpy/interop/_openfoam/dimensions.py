"""OpenFOAM dimension-set and FoamFile header parsers."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u

__all__ = [
    "OPENFOAM_BASE_UNITS",
    "OpenFOAMDimensions",
    "OpenFOAMHeader",
    "classify_pressure_dimensions",
    "kinematic_to_physical_pressure",
    "parse_control_dict",
    "parse_foam_field_metadata",
    "parse_foam_header",
    "parse_openfoam_dimensions",
]

# OpenFOAM 7-exponent order: [kg, m, s, K, mol, A, cd]
OPENFOAM_BASE_UNITS: tuple[u.UnitBase, ...] = (
    u.kg,
    u.m,
    u.s,
    u.K,
    u.mol,
    u.A,
    u.cd,
)

_DIMENSIONS_RE = re.compile(r"dimensions\s*\[([^\]]+)\]|^\s*\[([^\]]+)\]\s*$")
_FOAMFILE_RE = re.compile(r"FoamFile\s*\{(.*?)\}", re.DOTALL)
_KV_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s+([^;{}]+?)\s*;")
_INTERNAL_UNIFORM_RE = re.compile(
    r"\binternalField\s+uniform\s+([^;]+?)\s*;", re.DOTALL
)
_INTERNAL_NONUNIFORM_RE = re.compile(
    r"\binternalField\s+nonuniform\s+List<([^>]+)>", re.DOTALL
)

_CANONICAL_UNIT_MAP: dict[tuple[int | float, ...], tuple[u.UnitBase, str]] = {
    (0, 0, 0, 0, 0, 0, 0): (u.dimensionless_unscaled, "dimensionless"),
    (1, -1, -2, 0, 0, 0, 0): (u.Pa, "pressure_physical"),
    (0, 2, -2, 0, 0, 0, 0): (u.m**2 / u.s**2, "pressure_kinematic_or_specific_energy"),
    (1, -3, 0, 0, 0, 0, 0): (u.kg / u.m**3, "density"),
    (0, 1, -1, 0, 0, 0, 0): (u.m / u.s, "velocity"),
    (0, 1, -2, 0, 0, 0, 0): (u.m / u.s**2, "acceleration"),
    (0, 0, 0, 1, 0, 0, 0): (u.K, "temperature"),
    (0, 0, -1, 0, 0, 0, 0): (1 / u.s, "rate"),
    (0, 2, -1, 0, 0, 0, 0): (u.m**2 / u.s, "kinematic_viscosity"),
    (1, -1, -1, 0, 0, 0, 0): (u.Pa * u.s, "dynamic_viscosity"),
    (0, 2, -3, 0, 0, 0, 0): (u.m**2 / u.s**3, "dissipation_rate"),
    (0, 1, 0, 0, 0, 0, 0): (u.m, "length"),
    (0, 0, 1, 0, 0, 0, 0): (u.s, "time"),
    (1, 0, 0, 0, 0, 0, 0): (u.kg, "mass"),
}


def strip_foam_comments(text: str) -> str:
    """Remove C++ style ``/* ... */`` and ``// ...`` comments from OpenFOAM text."""
    no_block = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", no_block)


def _coerce_exponent(token: Any) -> int | float:
    if isinstance(token, bool):
        raise ValueError(
            f"Boolean is not a valid OpenFOAM dimension exponent: {token!r}"
        )
    if isinstance(token, (int, np.integer)):
        return int(token)
    if isinstance(token, (float, np.floating)):
        val = float(token)
        if not math.isfinite(val):
            raise ValueError(f"Non-finite OpenFOAM dimension exponent: {token!r}")
        return int(val) if val.is_integer() else val
    if isinstance(token, str):
        s = token.strip()
        try:
            ival = int(s)
            return ival
        except ValueError:
            pass
        try:
            fval = float(s)
        except ValueError as exc:
            raise ValueError(
                f"Invalid OpenFOAM dimension exponent token: {token!r}"
            ) from exc
        if not math.isfinite(fval):
            raise ValueError(f"Non-finite OpenFOAM dimension exponent: {token!r}")
        return int(fval) if fval.is_integer() else fval
    raise TypeError(
        f"Unsupported OpenFOAM dimension exponent type: {type(token).__name__}"
    )


@dataclass(frozen=True)
class OpenFOAMDimensions:
    """Parsed 7-exponent OpenFOAM ``dimensions`` entry."""

    exponents: tuple[int | float, ...]
    unit: u.UnitBase
    quantity_kind: str

    @property
    def is_physical_pressure(self) -> bool:
        return self.exponents == (1, -1, -2, 0, 0, 0, 0)

    @property
    def is_kinematic_pressure(self) -> bool:
        return self.exponents == (0, 2, -2, 0, 0, 0, 0)

    def to_foam_string(self) -> str:
        return "[" + " ".join(str(x) for x in self.exponents) + "]"


def _build_unit_from_exponents(
    exponents: tuple[int | float, ...],
) -> tuple[u.UnitBase, str]:
    if exponents in _CANONICAL_UNIT_MAP:
        return _CANONICAL_UNIT_MAP[exponents]

    unit_expr: u.UnitBase = u.dimensionless_unscaled
    for base, exp in zip(OPENFOAM_BASE_UNITS, exponents, strict=True):
        if exp != 0:
            unit_expr = unit_expr * (base**exp)
    return u.Unit(unit_expr), "custom"


def parse_openfoam_dimensions(
    spec: str | Sequence[int | float],
) -> OpenFOAMDimensions:
    """Parse an OpenFOAM 7-exponent ``dimensions`` specification into Astropy units.

    Parameters
    ----------
    spec : str or sequence of 7 numbers
        Either a bracketed OpenFOAM dimension string such as
        ``"[1 -1 -2 0 0 0 0]"`` or ``"dimensions [0 1 -1 0 0 0 0];"``,
        or a 7-element sequence of exponents in OpenFOAM SI order
        ``[kg, m, s, K, mol, A, cd]``.

    Returns
    -------
    OpenFOAMDimensions
        Parsed dimension exponents, resolved Astropy unit, and physical kind tag.

    """
    if isinstance(spec, str):
        cleaned = strip_foam_comments(spec).strip()
        m = _DIMENSIONS_RE.search(cleaned)
        if m is None:
            raise ValueError(f"Could not parse OpenFOAM dimensions from {spec!r}")
        inner = m.group(1) if m.group(1) is not None else m.group(2)
        tokens = inner.split()
    elif isinstance(spec, Sequence) and not isinstance(spec, (bytes, bytearray)):
        tokens = list(spec)
    else:
        raise TypeError(
            f"OpenFOAM dimensions spec must be str or sequence, got {type(spec).__name__}"
        )

    if len(tokens) != 7:
        raise ValueError(
            f"OpenFOAM dimensions must have exactly 7 exponents [kg m s K mol A cd], "
            f"got {len(tokens)}: {tokens!r}"
        )

    exponents = tuple(_coerce_exponent(tok) for tok in tokens)
    unit, kind = _build_unit_from_exponents(exponents)
    return OpenFOAMDimensions(exponents=exponents, unit=unit, quantity_kind=kind)


def classify_pressure_dimensions(
    dims: OpenFOAMDimensions | str | Sequence[int | float],
    *,
    field_name: str | None = None,
) -> str | None:
    """Classify pressure dimensions as ``'physical'``, ``'kinematic'``, or ``None``.

    Physical pressure has dimensions ``[1 -1 -2 0 0 0 0]`` (Pa = kg / (m s^2)).
    Kinematic pressure (incompressible OpenFOAM solvers such as ``simpleFoam``)
    has dimensions ``[0 2 -2 0 0 0 0]`` (m^2 / s^2 = p / rho).
    """
    parsed = (
        dims
        if isinstance(dims, OpenFOAMDimensions)
        else parse_openfoam_dimensions(dims)
    )
    if parsed.is_physical_pressure:
        return "physical"
    if parsed.is_kinematic_pressure:
        # [0 2 -2 0 0 0 0] is shared by kinematic pressure (p, p_rgh, pd) and
        # turbulent kinetic energy (k). Only label as kinematic pressure when
        # field_name is omitted or looks like a pressure field.
        if field_name is None:
            return "kinematic"
        low = field_name.lower()
        if low == "p" or low.startswith(("p_", "pd", "prgh", "p_rgh")):
            return "kinematic"
    return None


def kinematic_to_physical_pressure(
    values: Any,
    rho_ref: float | u.Quantity,
) -> Any:
    """Explicitly convert kinematic pressure ``p / rho`` (m^2/s^2) to physical pressure (Pa).

    This conversion is never applied implicitly during file loading. Callers
    must supply an explicit reference density ``rho_ref`` (positive real value
    in kg/m^3 or Quantity convertible to kg/m^3).
    """
    if isinstance(rho_ref, bool):
        raise TypeError("rho_ref must be a positive numeric density, not bool")
    if isinstance(rho_ref, u.Quantity):
        rho_val = float(rho_ref.to_value(u.kg / u.m**3))
        rho_qty = rho_ref.to(u.kg / u.m**3)
    else:
        rho_val = float(rho_ref)
        rho_qty = rho_val * (u.kg / u.m**3)

    if not math.isfinite(rho_val) or rho_val <= 0.0:
        raise ValueError(f"rho_ref must be finite and positive, got {rho_ref!r}")

    if hasattr(values, "unit") and hasattr(values, "copy"):
        # ScalarField or Quantity
        src_unit = values.unit if values.unit is not None else u.dimensionless_unscaled
        if not src_unit.is_equivalent(u.m**2 / u.s**2):
            raise u.UnitConversionError(
                f"Expected kinematic pressure unit equivalent to m^2/s^2, got {src_unit}"
            )
        return (values * rho_qty).to(u.Pa)

    arr = np.asarray(values, dtype=np.float64)
    return (arr * rho_val) * u.Pa


@dataclass(frozen=True)
class OpenFOAMHeader:
    """Parsed metadata from an OpenFOAM dictionary or field file."""

    version: str | None = None
    file_format: str | None = None
    foam_class: str | None = None
    location: str | None = None
    object_name: str | None = None
    dimensions: OpenFOAMDimensions | None = None
    internal_field_kind: str | None = None
    internal_field_type: str | None = None
    pressure_kind: str | None = None


def parse_foam_header(source: str | Path) -> OpenFOAMHeader:
    """Parse ``FoamFile { ... }`` header and optional ``dimensions`` from a file or string."""
    if isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).is_file()
    ):
        text = Path(source).read_text(encoding="utf-8", errors="replace")
    else:
        text = str(source)

    cleaned = strip_foam_comments(text)
    header_vals: dict[str, str] = {}
    m_block = _FOAMFILE_RE.search(cleaned)
    if m_block is not None:
        for k, v in _KV_RE.findall(m_block.group(1)):
            header_vals[k] = v.strip().strip('"')

    dims: OpenFOAMDimensions | None = None
    m_dim = re.search(r"\bdimensions\s*\[([^\]]+)\]\s*;", cleaned)
    if m_dim is not None:
        dims = parse_openfoam_dimensions(f"[{m_dim.group(1)}]")

    internal_kind: str | None = None
    internal_type: str | None = None
    if _INTERNAL_UNIFORM_RE.search(cleaned):
        internal_kind = "uniform"
    else:
        m_nonuni = _INTERNAL_NONUNIFORM_RE.search(cleaned)
        if m_nonuni is not None:
            internal_kind = "nonuniform"
            internal_type = m_nonuni.group(1).strip()

    obj_name = header_vals.get("object")
    p_kind = classify_pressure_dimensions(dims, field_name=obj_name) if dims else None

    return OpenFOAMHeader(
        version=header_vals.get("version"),
        file_format=header_vals.get("format"),
        foam_class=header_vals.get("class"),
        location=header_vals.get("location"),
        object_name=obj_name,
        dimensions=dims,
        internal_field_kind=internal_kind,
        internal_field_type=internal_type,
        pressure_kind=p_kind,
    )


def parse_foam_field_metadata(path: str | Path) -> OpenFOAMHeader:
    """Parse header and dimension metadata from an OpenFOAM field file."""
    return parse_foam_header(Path(path))


def parse_control_dict(path: str | Path) -> dict[str, Any]:
    """Parse common metadata entries from ``system/controlDict``."""
    p = Path(path)
    cleaned = strip_foam_comments(p.read_text(encoding="utf-8", errors="replace"))
    # Remove FoamFile {...} block before extracting top-level key-values
    without_header = _FOAMFILE_RE.sub("", cleaned, count=1)
    result: dict[str, Any] = {}
    for k, v in _KV_RE.findall(without_header):
        val_str = v.strip().strip('"')
        if k in {"startTime", "endTime", "deltaT", "writeInterval"}:
            try:
                result[k] = float(val_str)
                continue
            except ValueError:
                pass
        result[k] = val_str
    return result
