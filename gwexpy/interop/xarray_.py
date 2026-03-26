from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
from gwpy.time import LIGOTimeGPS

from ._optional import require_optional


def to_xarray(ts, time_coord="datetime"):
    """
    TimeSeries -> xarray.DataArray
    """
    xr = require_optional("xarray")

    data = ts.value
    attrs = {
        "unit": str(ts.unit),
        "name": str(ts.name),
        "channel": str(ts.channel),
        "epoch": float(ts.t0.value if hasattr(ts.t0, "value") else ts.t0),
        "time_coord": time_coord,
    }

    times_gps = ts.times.value
    if time_coord == "datetime":
        from astropy.time import Time

        t_vals = Time(times_gps, format="gps").to_datetime()
    elif time_coord == "seconds":
        from astropy.time import Time

        t_vals = Time(times_gps, format="gps").unix
    elif time_coord == "gps":
        t_vals = times_gps
    else:
        raise ValueError("time_coord must be 'datetime'|'seconds'|'gps'")

    da = xr.DataArray(
        data,
        dims=("time",),
        coords={"time": t_vals},
        name=ts.name,
        attrs=attrs,
    )
    return da


def from_xarray(cls, da, unit=None):
    """DataArray -> TimeSeries"""
    require_optional("xarray")

    val = da.values
    t_coord = da.coords["time"].values

    time_coord = da.attrs.get("time_coord")
    if np.issubdtype(t_coord.dtype, np.datetime64):
        from astropy.time import Time

        t_obj = Time(t_coord, format="datetime64")
        t0_gps = float(t_obj[0].gps)
        dt = float(t_obj[1].gps - t0_gps) if len(t_coord) > 1 else 1.0
    elif time_coord == "seconds":
        from astropy.time import Time

        t_obj = Time(t_coord, format="unix")
        t0_gps = float(t_obj[0].gps)
        dt = float(t_obj[1].gps - t0_gps) if len(t_coord) > 1 else 1.0
    else:
        t0_gps = float(t_coord[0])
        dt = float(t_coord[1] - t0_gps) if len(t_coord) > 1 else 1.0

    _unit = unit or da.attrs.get("unit")
    name = da.name or da.attrs.get("name")

    return cls(val, t0=LIGOTimeGPS(t0_gps), dt=dt, unit=_unit, name=name)


# ---------------------------------------------------------------------------
# ScalarField / VectorField  ↔  xarray.DataArray / Dataset
# ---------------------------------------------------------------------------

# Dimension name → axis role detection (CF Convention > MetPy > heuristic)
_AXIS0_CANDIDATES = {"time", "t", "frequency", "freq", "f"}
_AXIS1_CANDIDATES = {"x", "easting", "west_east", "lon", "longitude"}
_AXIS2_CANDIDATES = {"y", "northing", "south_north", "lat", "latitude"}
_AXIS3_CANDIDATES = {"z", "upward", "height", "altitude", "bottom_top", "level"}

_CF_AXIS_TO_ROLE = {"T": 0, "X": 1, "Y": 2, "Z": 3}
_METPY_TO_ROLE = {"time": 0, "x": 1, "y": 2, "vertical": 3}

_ROLE_CANDIDATES = [
    _AXIS0_CANDIDATES,
    _AXIS1_CANDIDATES,
    _AXIS2_CANDIDATES,
    _AXIS3_CANDIDATES,
]


def _detect_dim_role(da: Any, dim: str) -> int | None:
    """Return the axis role (0-3) for *dim* in *da*, or ``None``."""
    coords = da.coords if hasattr(da, "coords") else {}
    # 1. CF Convention `axis` attribute on the coordinate variable
    if dim in coords:
        coord_attrs = coords[dim].attrs
        cf_axis = coord_attrs.get("axis", "").upper()
        if cf_axis in _CF_AXIS_TO_ROLE:
            return _CF_AXIS_TO_ROLE[cf_axis]
        # 2. MetPy `_metpy_axis`
        metpy_axis = coord_attrs.get("_metpy_axis", "")
        if metpy_axis in _METPY_TO_ROLE:
            return _METPY_TO_ROLE[metpy_axis]
    # 3. Heuristic name match
    low = dim.lower()
    for role, candidates in enumerate(_ROLE_CANDIDATES):
        if low in candidates:
            return role
    return None


def _detect_dim_roles(da: Any, dims: tuple[str, ...]) -> dict[int, str]:
    """Map axis roles → dimension names for a DataArray's dimensions.

    Returns ``{role: dim_name}`` for roles 0-3.  Unknown dims are ignored.
    When multiple dims map to the same role, the first one wins.
    """
    role_to_dim: dict[int, str] = {}
    for dim in dims:
        role = _detect_dim_role(da, dim)
        if role is not None and role not in role_to_dim:
            role_to_dim[role] = dim
    return role_to_dim


def _try_parse_unit(unit_str: str) -> Any:
    """Try to parse *unit_str* as an astropy Unit; return ``None`` on failure."""
    try:
        from astropy import units as u  # noqa: PLC0415

        return u.Unit(unit_str)
    except Exception:
        warnings.warn(
            f"Could not parse unit string {unit_str!r} as an astropy Unit. "
            "Unit will be set to None.",
            stacklevel=4,
        )
        return None


def from_xarray_field(
    cls: type,
    da: Any,
    *,
    axis0_dim: str | None = None,
    spatial_dims: tuple[str, ...] | None = None,
    axis0_domain: Literal["time", "frequency"] = "time",
    space_domain: Literal["real", "k"] = "real",
) -> Any:
    """Convert an ``xarray.DataArray`` or ``Dataset`` to a ScalarField / VectorField.

    Dimension auto-detection priority:

    1. CF Convention ``axis`` attribute (``T`` / ``X`` / ``Y`` / ``Z``)
    2. MetPy ``_metpy_axis`` attribute
    3. Heuristic name matching

    Parameters
    ----------
    cls : type
        ``ScalarField`` or ``VectorField``.
    da : xarray.DataArray or xarray.Dataset
        Input array.  A ``Dataset`` is treated as a VectorField where each
        data variable becomes one component.
    axis0_dim : str, optional
        Dimension name to use as axis0.  Auto-detected when ``None``.
    spatial_dims : tuple of str, optional
        Ordered tuple ``(dim_x, dim_y, dim_z)`` for spatial axes.
        Auto-detected when ``None``.  Missing axes are filled with singleton
        dimensions.
    axis0_domain : {"time", "frequency"}, default "time"
        Physical domain of axis0.
    space_domain : {"real", "k"}, default "real"
        Spatial domain label for the resulting ScalarField.

    Returns
    -------
    ScalarField
        When *da* is a DataArray or *cls* is ScalarField.
    VectorField
        When *da* is a Dataset or *cls* is VectorField with a DataArray.
    """
    xr = require_optional("xarray")
    from gwexpy.fields import ScalarField, VectorField  # noqa: PLC0415

    # Dataset → VectorField by converting each variable
    if isinstance(da, xr.Dataset):
        components: dict[str, Any] = {}
        for var_name in da.data_vars:
            sf = from_xarray_field(
                ScalarField,
                da[var_name],
                axis0_dim=axis0_dim,
                spatial_dims=spatial_dims,
                axis0_domain=axis0_domain,
                space_domain=space_domain,
            )
            components[var_name] = sf
        return VectorField(components)

    # ----- DataArray path -----
    dims = da.dims  # tuple of str

    # Detect roles
    role_to_dim = _detect_dim_roles(da, dims)

    # Resolve axis0
    if axis0_dim is not None:
        role_to_dim[0] = axis0_dim
    ax0_dim = role_to_dim.get(0)

    # Resolve spatial dims (1, 2, 3)
    if spatial_dims is not None:
        for i, spdim in enumerate(spatial_dims[:3], start=1):
            role_to_dim[i] = spdim

    # Collect coordinate arrays
    def _coord_arr(dim: str | None) -> np.ndarray | None:
        if dim is None:
            return None
        if da.coords and dim in da.coords:
            arr = np.asarray(da.coords[dim].values, dtype=np.float64)
            return arr
        return None

    ax0_vals = _coord_arr(ax0_dim)
    ax1_vals = _coord_arr(role_to_dim.get(1))
    ax2_vals = _coord_arr(role_to_dim.get(2))
    ax3_vals = _coord_arr(role_to_dim.get(3))

    # Build the 4-D data array
    # Order: move axis0_dim → position 0, then spatial dims → 1,2,3
    ordered_spatial = [role_to_dim.get(r) for r in (1, 2, 3)]
    spatial_present = [d for d in ordered_spatial if d is not None]
    remaining_dims = [d for d in dims if d != ax0_dim and d not in spatial_present]
    # remaining_dims should be empty in well-formed input; ignored if extra

    # Transpose: axis0 first, then spatial, then any extras
    new_order = []
    if ax0_dim and ax0_dim in dims:
        new_order.append(ax0_dim)
    for d in spatial_present:
        if d in dims:
            new_order.append(d)
    for d in dims:
        if d not in new_order:
            new_order.append(d)

    data = np.asarray(da.transpose(*new_order).values)

    # Pad to 4D: prepend singleton axis0 if missing, append singleton spatial axes
    # Resulting shape: (n_ax0, n_x, n_y, n_z)
    n_spatial_found = len([d for d in spatial_present if d in dims])
    ax0_present = ax0_dim and ax0_dim in dims

    if not ax0_present:
        data = data[np.newaxis, ...]

    # Pad to exactly 4D by appending singleton axes
    while data.ndim < 4:
        data = data[..., np.newaxis]

    # Parse unit
    unit = _try_parse_unit(da.attrs["units"]) if "units" in da.attrs else None

    return ScalarField(
        data,
        unit=unit,
        axis0=ax0_vals,
        axis0_domain=axis0_domain,
        axis1=ax1_vals,
        axis2=ax2_vals,
        axis3=ax3_vals,
        space_domain=space_domain,
    )


def to_xarray_field(
    field: Any,
    *,
    dim_names: tuple[str, str, str, str] | None = None,
) -> Any:
    """Convert a ScalarField or VectorField to an xarray DataArray / Dataset.

    Parameters
    ----------
    field : ScalarField or VectorField
        Source field.
    dim_names : tuple of 4 str, optional
        Dimension names ``(axis0, axis1, axis2, axis3)``.  Defaults to
        ``field.axis_names`` when available, then to ``("t", "x", "y", "z")``.

    Returns
    -------
    xarray.DataArray
        When *field* is a ScalarField.
    xarray.Dataset
        When *field* is a VectorField.
    """
    xr = require_optional("xarray")
    from gwexpy.fields import ScalarField, VectorField  # noqa: PLC0415

    if isinstance(field, VectorField):
        arrays = {}
        for comp_name, sf in field.items():
            arrays[comp_name] = to_xarray_field(sf, dim_names=dim_names)
        return xr.Dataset(arrays)

    # ScalarField path
    default_names = ("t", "x", "y", "z")
    if dim_names is not None:
        names = dim_names
    elif hasattr(field, "axis_names") and field.axis_names is not None:
        names = tuple(field.axis_names)
        if len(names) < 4:
            names = names + default_names[len(names):]
    else:
        names = default_names

    data = np.asarray(field.value)

    # Build coordinate dicts
    coords: dict[str, np.ndarray] = {}

    def _get_axis(attr: str) -> np.ndarray | None:
        ax = getattr(field, attr, None)
        if ax is None:
            return None
        arr = np.asarray(ax.value if hasattr(ax, "value") else ax)
        return arr if arr.size > 0 else None

    for i, (name, attr) in enumerate(
        zip(names, ("_axis0_index", "_axis1_index", "_axis2_index", "_axis3_index"))
    ):
        vals = _get_axis(attr)
        if vals is not None and vals.size == data.shape[i]:
            coords[name] = vals

    attrs: dict[str, Any] = {}
    if field.unit is not None:
        attrs["units"] = str(field.unit)

    return xr.DataArray(
        data,
        dims=names[:data.ndim],
        coords=coords,
        attrs=attrs,
    )
