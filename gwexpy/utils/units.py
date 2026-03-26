"""
gwexpy.utils.units
------------------

Bidirectional conversion between Pint and astropy unit systems.

MetPy and several other scientific Python packages use Pint for unit handling,
while GWexpy uses ``astropy.units``.  This module provides thin wrappers that
convert between the two systems by parsing unit strings, with a lookup table
for known naming mismatches.

Only ``pint`` is imported lazily (optional dependency); ``astropy`` is always
available in GWexpy.
"""

from __future__ import annotations

import warnings
from typing import Any

from astropy import units as u

from gwexpy.interop._optional import require_optional

__all__ = [
    "pint_to_astropy",
    "astropy_to_pint",
    "pint_unit_to_astropy_unit",
    "astropy_unit_to_pint_unit",
]

# ---------------------------------------------------------------------------
# Known unit-string mismatches between Pint and astropy
# ---------------------------------------------------------------------------

# Pint name → astropy name
_PINT_TO_ASTROPY: dict[str, str] = {
    "degC": "deg_C",
    "degree_Celsius": "deg_C",
    "percent": "%",
    "dimensionless": "",
    "gal": "Gal",
    "mbar": "mbar",
    "knot": "kn",
    "knots": "kn",
    "kelvin": "K",
}

# astropy name → Pint name
_ASTROPY_TO_PINT: dict[str, str] = {
    "deg_C": "degC",
    "%": "percent",
    "Gal": "gal",
    "kn": "knot",
}


# ---------------------------------------------------------------------------
# Public API — unit objects
# ---------------------------------------------------------------------------


def pint_unit_to_astropy_unit(pint_unit: Any) -> u.UnitBase:
    """Convert a Pint unit to an astropy Unit.

    Parameters
    ----------
    pint_unit : pint.Unit
        A Pint unit object (e.g., ``ureg.meter / ureg.second``).

    Returns
    -------
    astropy.units.UnitBase
        Equivalent astropy unit.

    Raises
    ------
    ValueError
        If the unit string cannot be parsed by astropy.
    """
    raw = str(pint_unit)

    # Apply known overrides
    unit_str = _PINT_TO_ASTROPY.get(raw, raw)

    # Handle dimensionless
    if unit_str in ("", "dimensionless"):
        return u.dimensionless_unscaled

    try:
        return u.Unit(unit_str)
    except ValueError:
        # Attempt decomposition: replace ** with ^ notation differences etc.
        unit_str_cleaned = unit_str.replace("**", "^") if "**" in unit_str else unit_str
        try:
            return u.Unit(unit_str_cleaned)
        except ValueError as exc:
            raise ValueError(
                f"Cannot convert Pint unit {pint_unit!r} (string: {raw!r}) "
                f"to an astropy Unit."
            ) from exc


def astropy_unit_to_pint_unit(astropy_unit: u.UnitBase | str) -> Any:
    """Convert an astropy Unit to a Pint unit.

    Parameters
    ----------
    astropy_unit : astropy.units.UnitBase or str
        An astropy unit object or string representation.

    Returns
    -------
    pint.Unit
        Equivalent Pint unit.

    Raises
    ------
    ValueError
        If the unit string cannot be parsed by Pint.
    ImportError
        If ``pint`` is not installed.
    """
    pint = require_optional("pint")
    ureg = pint.UnitRegistry()

    raw = str(astropy_unit)
    unit_str = _ASTROPY_TO_PINT.get(raw, raw)

    if raw in ("", "dimensionless"):
        return ureg.dimensionless

    try:
        return ureg.Unit(unit_str)
    except Exception as exc:
        raise ValueError(
            f"Cannot convert astropy unit {astropy_unit!r} (string: {raw!r}) "
            f"to a Pint unit."
        ) from exc


# ---------------------------------------------------------------------------
# Public API — quantities (value + unit)
# ---------------------------------------------------------------------------


def pint_to_astropy(pint_quantity: Any) -> u.Quantity:
    """Convert a Pint Quantity to an astropy Quantity.

    Parameters
    ----------
    pint_quantity : pint.Quantity
        A Pint quantity (magnitude + unit).

    Returns
    -------
    astropy.units.Quantity
        Equivalent astropy quantity with the same magnitude and unit.
    """
    magnitude = pint_quantity.magnitude
    astropy_unit = pint_unit_to_astropy_unit(pint_quantity.units)
    return u.Quantity(magnitude, astropy_unit)


def astropy_to_pint(astropy_quantity: u.Quantity) -> Any:
    """Convert an astropy Quantity to a Pint Quantity.

    Parameters
    ----------
    astropy_quantity : astropy.units.Quantity
        An astropy quantity.

    Returns
    -------
    pint.Quantity
        Equivalent Pint quantity with the same magnitude and unit.

    Raises
    ------
    ImportError
        If ``pint`` is not installed.
    """
    pint = require_optional("pint")
    ureg = pint.UnitRegistry()

    magnitude = astropy_quantity.value
    pint_unit = astropy_unit_to_pint_unit(astropy_quantity.unit)

    # Use a fresh ureg to build the quantity, matching the unit's registry
    return ureg.Quantity(magnitude, str(pint_unit))
