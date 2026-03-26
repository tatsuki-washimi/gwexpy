"""
gwexpy.interop.harmonica_
-------------------------

Interoperability with Harmonica gravity/magnetic grids.

Harmonica (from the Fatiando a Terra project) works primarily with
``xarray.DataArray`` and ``xarray.Dataset`` objects using
``easting``/``northing`` or ``longitude``/``spherical_latitude`` coordinates.
Units are implicitly SI (metres, mGal, nT, etc.).

This module maps Harmonica grids to GWexpy ``ScalarField`` or ``VectorField``
by delegating to :func:`gwexpy.interop.xarray_.from_xarray_field`.

References
----------
https://www.fatiando.org/harmonica/
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gwexpy.fields import ScalarField, VectorField

from ._optional import require_optional
from .xarray_ import from_xarray_field

__all__ = ["from_harmonica_grid"]


def from_harmonica_grid(
    cls: type,
    ds: Any,
    *,
    data_name: str | None = None,
) -> ScalarField | VectorField:
    """Convert a Harmonica xarray grid to a ``ScalarField`` or ``VectorField``.

    Harmonica grids typically have dimensions ``easting``/``northing`` (for
    projected coordinates) or ``longitude``/``latitude`` (for geographic
    coordinates), with an optional ``upward`` dimension for height.

    Parameters
    ----------
    cls : type
        ``ScalarField`` or ``VectorField``.
    ds : xarray.DataArray or xarray.Dataset
        Harmonica grid data.  A ``Dataset`` is treated as a ``VectorField``
        unless *data_name* is given, in which case a single variable is
        extracted.
    data_name : str, optional
        Variable name to extract from a ``Dataset``.  If not given and *ds*
        is a ``Dataset``, all variables become VectorField components.

    Returns
    -------
    ScalarField
        When *ds* is a DataArray or *data_name* is given.
    VectorField
        When *ds* is a Dataset and *data_name* is not given.
    """
    xr = require_optional("xarray")

    # Extract single variable from Dataset if requested
    if isinstance(ds, xr.Dataset) and data_name is not None:
        if data_name not in ds.data_vars:
            raise ValueError(
                f"Variable {data_name!r} not found in Dataset. "
                f"Available: {list(ds.data_vars)}"
            )
        ds = ds[data_name]

    # Ensure units attribute exists (Harmonica uses implicit SI)
    if isinstance(ds, xr.DataArray):
        if "units" not in ds.attrs:
            # Infer from common variable names
            ds = ds.assign_attrs(units=_guess_harmonica_unit(ds.name or ""))

    return from_xarray_field(cls, ds, space_domain="real")


def _guess_harmonica_unit(name: str) -> str:
    """Guess the physical unit from a Harmonica variable name."""
    low = name.lower()
    if "gravity" in low or "grav" in low or "bouguer" in low or "free_air" in low:
        return "mGal"
    if "magnetic" in low or "mag" in low:
        return "nT"
    if "topography" in low or "topo" in low or "height" in low:
        return "m"
    return ""
