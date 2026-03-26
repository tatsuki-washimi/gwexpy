"""
gwexpy.interop.wrf_
-------------------

Interoperability with wrf-python (``wrf.getvar()``) output.

WRF model output uses non-standard dimension names (``south_north``,
``west_east``, ``bottom_top``) and 2-D latitude/longitude arrays
(``XLAT``, ``XLONG``).  This module extracts 1-D axis coordinates where
possible and delegates to :func:`gwexpy.interop.xarray_.from_xarray_field`.

References
----------
https://wrf-python.readthedocs.io/
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from gwexpy.fields import ScalarField

from ._optional import require_optional
from .xarray_ import from_xarray_field

__all__ = ["from_wrf_variable"]

# WRF dimension → GWexpy axis role mapping
_WRF_DIM_ROLES = {
    "time": 0,
    "Time": 0,
    "bottom_top": 3,
    "bottom_top_stag": 3,
    "south_north": 2,
    "south_north_stag": 2,
    "west_east": 1,
    "west_east_stag": 1,
}


def _extract_1d_from_2d(da: Any, coord_name: str, axis: int) -> np.ndarray | None:
    """Try to extract a 1-D coordinate from a 2-D WRF coordinate array.

    For regular grids the latitude/longitude values are constant along one
    axis, so we can take a single row/column.  Returns ``None`` for
    irregularly spaced grids.
    """
    if coord_name not in da.coords:
        return None

    coord = np.asarray(da.coords[coord_name].values, dtype=np.float64)

    if coord.ndim == 1:
        return coord

    if coord.ndim != 2:
        return None

    # Check if constant along axis 0 (→ take row 0)
    if np.allclose(coord, coord[0:1, :], atol=1e-6):
        return coord[0, :]

    # Check if constant along axis 1 (→ take column 0)
    if np.allclose(coord, coord[:, 0:1], atol=1e-6):
        return coord[:, 0]

    return None


def from_wrf_variable(
    cls: type,
    da: Any,
    *,
    vertical_dim: str | None = None,
    axis0_domain: Literal["time", "frequency"] = "time",
) -> ScalarField:
    """Convert a ``wrf.getvar()`` xarray.DataArray to a ``ScalarField``.

    WRF dimension names are mapped to GWexpy axes:

    - ``Time`` / ``time`` → axis0
    - ``west_east`` → axis1 (x)
    - ``south_north`` → axis2 (y)
    - ``bottom_top`` → axis3 (z)

    2-D ``XLAT`` / ``XLONG`` coordinates are collapsed to 1-D when the grid
    is regular.

    Parameters
    ----------
    cls : type
        ``ScalarField`` class.
    da : xarray.DataArray
        Output of ``wrf.getvar()``.
    vertical_dim : str, optional
        Override the vertical dimension name.  Auto-detected from
        ``bottom_top`` / ``bottom_top_stag`` when ``None``.
    axis0_domain : {"time", "frequency"}, default "time"
        Physical domain of axis0.

    Returns
    -------
    ScalarField
    """
    xr = require_optional("xarray")

    # Try to extract 1D coords from 2D XLAT/XLONG
    lon_1d = _extract_1d_from_2d(da, "XLONG", axis=0)
    lat_1d = _extract_1d_from_2d(da, "XLAT", axis=1)

    # Build a cleaned DataArray with 1D coords
    new_coords = dict(da.coords)

    # Map WRF dims to our spatial dims
    dim_mapping: dict[str, str] = {}
    spatial_dims: list[str] = []

    for dim in da.dims:
        role = _WRF_DIM_ROLES.get(dim)
        if role == 1:  # west_east → x
            dim_mapping[dim] = dim
            if lon_1d is not None and len(lon_1d) == da.sizes[dim]:
                new_coords[dim] = lon_1d
            spatial_dims.append(dim)
        elif role == 2:  # south_north → y
            dim_mapping[dim] = dim
            if lat_1d is not None and len(lat_1d) == da.sizes[dim]:
                new_coords[dim] = lat_1d
            spatial_dims.append(dim)
        elif role == 3:  # bottom_top → z
            dim_mapping[dim] = dim
            spatial_dims.append(dim)

    # Re-wrap with updated coords for 1D lat/lon
    # Only update if we actually changed something
    da_clean = da.assign_coords(
        {k: v for k, v in new_coords.items() if k in da.dims}
    )

    # Parse unit from WRF attrs
    unit_str = da.attrs.get("units")
    if unit_str is not None:
        da_clean = da_clean.assign_attrs(units=unit_str)

    return from_xarray_field(cls, da_clean, axis0_domain=axis0_domain)
