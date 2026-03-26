"""
gwexpy.interop.metpy_
---------------------

Interoperability with MetPy xarray DataArrays.

MetPy uses Pint quantities via ``.metpy.quantify()`` / ``.metpy.dequantify()``
and CF Convention axis attributes (``_metpy_axis``).  This module strips the
Pint layer, converts units to astropy via :mod:`gwexpy.utils.units`, and
delegates to :func:`gwexpy.interop.xarray_.from_xarray_field`.

References
----------
https://unidata.github.io/MetPy/latest/
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np

from gwexpy.fields import ScalarField

from ._optional import require_optional
from .xarray_ import from_xarray_field

__all__ = ["from_metpy_dataarray"]


def from_metpy_dataarray(
    cls: type,
    da: Any,
    *,
    dequantify: bool = True,
    axis0_domain: Literal["time", "frequency"] = "time",
) -> ScalarField:
    """Convert a MetPy-enhanced ``xarray.DataArray`` to a ``ScalarField``.

    MetPy attaches Pint units to data arrays via ``.metpy.quantify()``.  This
    function strips the Pint layer (``dequantify``) and converts the unit to
    ``astropy.units`` before creating a ScalarField.

    Parameters
    ----------
    cls : type
        ``ScalarField`` class.
    da : xarray.DataArray
        MetPy-enhanced DataArray.  May have Pint-backed data or plain
        ``float64`` data with a ``"units"`` attribute.
    dequantify : bool, default True
        Call ``.metpy.dequantify()`` to strip Pint units and move them to
        ``attrs["units"]``.  Set to ``False`` if the array is already plain.
    axis0_domain : {"time", "frequency"}, default "time"
        Physical domain of axis0.

    Returns
    -------
    ScalarField

    Notes
    -----
    MetPy must be installed for full functionality, but the converter also
    works with plain xarray DataArrays that have ``"units"`` and
    ``"_metpy_axis"`` attributes.
    """
    xr = require_optional("xarray")

    # Dequantify: strip Pint wrapper if present
    if dequantify and hasattr(da, "metpy"):
        try:
            da = da.metpy.dequantify()
        except Exception:
            pass  # already plain

    # Try to convert Pint unit string → astropy
    unit_str = da.attrs.get("units")
    if unit_str is not None:
        try:
            from gwexpy.utils.units import pint_unit_to_astropy_unit  # noqa: PLC0415

            import pint  # noqa: PLC0415

            ureg = pint.UnitRegistry()
            astropy_unit = pint_unit_to_astropy_unit(ureg.Unit(unit_str))
            # Replace attrs so from_xarray_field can parse it
            da = da.assign_attrs(units=str(astropy_unit))
        except Exception:
            pass  # leave as-is, from_xarray_field will try to parse

    return from_xarray_field(cls, da, axis0_domain=axis0_domain)
