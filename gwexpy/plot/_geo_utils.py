"""Geographic plotting context, consistency checks, and terrain math helpers."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from astropy import units as u
from matplotlib.colors import LightSource

# Default layer zorder conventions for FieldPlot (overridable via kwargs)
ZORDER_BASEMAP: float = 0.0
ZORDER_TERRAIN: float = 1.0
ZORDER_SCALAR: float = 2.0
ZORDER_VECTOR: float = 3.0
ZORDER_MARKERS: float = 4.0

EARTH_MEAN_RADIUS_M: float = 6371000.0

_ELLIPSOIDS: dict[str, tuple[float, float]] = {
    "GRS80": (6378137.0, 1.0 / 298.257222101),
    "WGS84": (6378137.0, 1.0 / 298.257223563),
}

__all__ = [
    "EARTH_MEAN_RADIUS_M",
    "ZORDER_BASEMAP",
    "ZORDER_MARKERS",
    "ZORDER_SCALAR",
    "ZORDER_TERRAIN",
    "ZORDER_VECTOR",
    "GeoPlotContext",
    "compute_hillshade",
    "compute_shaded_relief",
    "convert_axis_values",
    "get_axis_unit",
    "get_geo_context",
    "get_geo_metadata",
    "get_space_domains",
    "local_xy_to_lonlat",
    "lonlat_to_local_xy",
    "resolve_geo_context",
    "set_geo_context",
]


@dataclass
class GeoPlotContext:
    """Per-Axes coordinate consistency state for layered geographic plots."""

    kind: Literal["angle", "local", "plain"]
    crs: str | None = None
    local_origin: tuple[float, float] | None = None
    axis_units: tuple[u.UnitBase, u.UnitBase] = (
        u.dimensionless_unscaled,
        u.dimensionless_unscaled,
    )


def get_geo_metadata(field: Any) -> Any:
    """Return GeoMetadata attached to ``field``, or ``None`` if absent."""
    geo = getattr(field, "geo", None)
    if geo is not None:
        return geo
    return getattr(field, "_gwex_geo", None)


def get_space_domains(field: Any) -> dict[str, str]:
    """Return the spatial domain mapping for ``field`` via duck-typing."""
    domains = getattr(field, "space_domains", None)
    if isinstance(domains, dict):
        return dict(domains)
    domains = getattr(field, "_space_domains", None)
    if isinstance(domains, dict):
        return dict(domains)
    domain = getattr(field, "space_domain", None)
    if isinstance(domain, dict):
        return dict(domain)
    return {}


def _infer_axis_names(field: Any) -> list[str]:
    names = getattr(field, "axis_names", None)
    if names is not None:
        return list(names)
    inferred: list[str] = []
    for i in range(4):
        name = getattr(field, f"_axis{i}_name", None)
        if name is not None:
            inferred.append(str(name))
    return inferred


def get_axis_unit(field: Any, axis_name: str | None) -> u.UnitBase:
    """Return the ``astropy.units.UnitBase`` for ``axis_name`` on ``field``."""
    if axis_name is None:
        return u.dimensionless_unscaled
    names = _infer_axis_names(field)
    for i, name in enumerate(names):
        if name == axis_name:
            idx = getattr(field, f"_axis{i}_index", None)
            unit = getattr(idx, "unit", None)
            if isinstance(unit, u.UnitBase):
                return unit
            if unit is not None:
                return u.Unit(unit)
            return u.dimensionless_unscaled
    return u.dimensionless_unscaled


def _coerce_unit(unit: Any) -> u.UnitBase:
    if unit is None:
        return u.dimensionless_unscaled
    if isinstance(unit, u.UnitBase):
        return unit
    return u.Unit(unit)


def get_geo_context(ax: Any) -> GeoPlotContext | None:
    """Return the ``GeoPlotContext`` stored on ``ax``, or ``None``."""
    return getattr(ax, "_gwexpy_geo_context", None)


def set_geo_context(ax: Any, ctx: GeoPlotContext) -> GeoPlotContext:
    """Store ``ctx`` on ``ax`` and return it."""
    ax._gwexpy_geo_context = ctx
    return ctx


def convert_axis_values(index: Any, target_unit: u.UnitBase) -> np.ndarray:
    """Convert an axis index array/Quantity to ``target_unit`` values."""
    if target_unit == u.dimensionless_unscaled:
        return np.asarray(getattr(index, "value", index), dtype=float)
    if hasattr(index, "to_value"):
        idx_unit = getattr(index, "unit", u.dimensionless_unscaled)
        if idx_unit == u.dimensionless_unscaled:
            return np.asarray(index.value, dtype=float)
        if idx_unit == target_unit:
            return np.asarray(index.value, dtype=float)
        return np.asarray(index.to_value(target_unit), dtype=float)
    return np.asarray(index, dtype=float)


def _ellipsoid_radii(lat0_deg: float, ellipsoid: str = "GRS80") -> tuple[float, float]:
    """Return (N, M) prime-vertical and meridional radii of curvature in meters."""
    key = ellipsoid.upper()
    if key == "SPHERE":
        return (EARTH_MEAN_RADIUS_M, EARTH_MEAN_RADIUS_M)
    if key not in _ELLIPSOIDS:
        raise ValueError(
            f"Unsupported ellipsoid '{ellipsoid}'. Expected 'GRS80', 'WGS84', or 'sphere'."
        )
    a, f = _ELLIPSOIDS[key]
    e2 = 2.0 * f - f * f
    sin_lat = np.sin(np.deg2rad(float(lat0_deg)))
    w2 = 1.0 - e2 * sin_lat * sin_lat
    w = float(np.sqrt(w2))
    r_east = a / w
    r_north = a * (1.0 - e2) / (w2 * w)
    return (r_east, r_north)


def lonlat_to_local_xy(
    lon: Any,
    lat: Any,
    lon0: float,
    lat0: float,
    *,
    ellipsoid: str = "GRS80",
) -> tuple[Any, Any]:
    """Convert geographic (lon, lat) [deg] to local tangent-plane (x, y) [m].

    Uses the local tangent-plane approximation ``x = R_east * cos(lat0) * dlon``
    and ``y = R_north * dlat`` matching ``ScalarField.to_local_cartesian()``.
    """
    is_scalar = np.ndim(lon) == 0 and np.ndim(lat) == 0
    r_east, r_north = _ellipsoid_radii(lat0, ellipsoid=ellipsoid)
    dlon_rad = np.deg2rad(np.asarray(lon, dtype=float) - float(lon0))
    dlat_rad = np.deg2rad(np.asarray(lat, dtype=float) - float(lat0))
    x = r_east * np.cos(np.deg2rad(float(lat0))) * dlon_rad
    y = r_north * dlat_rad
    if is_scalar:
        return (float(x), float(y))
    return (x, y)


def local_xy_to_lonlat(
    x: Any,
    y: Any,
    lon0: float,
    lat0: float,
    *,
    ellipsoid: str = "GRS80",
) -> tuple[Any, Any]:
    """Convert local tangent-plane (x, y) [m] back to (lon, lat) [deg]."""
    is_scalar = np.ndim(x) == 0 and np.ndim(y) == 0
    r_east, r_north = _ellipsoid_radii(lat0, ellipsoid=ellipsoid)
    cos_lat0 = float(np.cos(np.deg2rad(float(lat0))))
    if abs(cos_lat0) < 1e-12:
        raise ValueError(
            "lat0 is too close to a pole for local tangent-plane conversion."
        )
    lon = float(lon0) + np.rad2deg(np.asarray(x, dtype=float) / (r_east * cos_lat0))
    lat = float(lat0) + np.rad2deg(np.asarray(y, dtype=float) / r_north)
    if is_scalar:
        return (float(lon), float(lat))
    return (lon, lat)


def resolve_geo_context(
    ax: Any,
    field: Any = None,
    x_name: str | None = None,
    y_name: str | None = None,
    *,
    x_unit: Any = None,
    y_unit: Any = None,
) -> GeoPlotContext:
    """Resolve and validate the ``GeoPlotContext`` for an Axes layer.

    The first geo-aware layer establishes the context on ``ax`` (stored in
    ``ax._gwexpy_geo_context``). Subsequent layers are checked for coordinate
    consistency:

    - Mixing ``angle``-domain axes with ``real``/``local`` axes raises ``ValueError``.
    - Active CRS string mismatches emit a ``UserWarning``.
    - Differing ``local_origin`` coordinates emit a ``UserWarning`` with the
      approximate offset distance in meters.
    - Compatible angle/length units (e.g. ``deg`` vs ``rad``) are accepted so
      callers can convert coordinates to the context's ``axis_units``.
    """
    existing = get_geo_context(ax)
    if field is None:
        return existing if existing is not None else GeoPlotContext(kind="plain")

    names = _infer_axis_names(field)
    if x_name is None and len(names) >= 2:
        x_name = names[1]
    if y_name is None and len(names) >= 3:
        y_name = names[2]

    domains = get_space_domains(field)
    x_dom = domains.get(x_name) if x_name is not None else None
    y_dom = domains.get(y_name) if y_name is not None else None

    x_unit_obj = (
        _coerce_unit(x_unit) if x_unit is not None else get_axis_unit(field, x_name)
    )
    y_unit_obj = (
        _coerce_unit(y_unit) if y_unit is not None else get_axis_unit(field, y_name)
    )

    geo = get_geo_metadata(field)
    geo_active = bool(getattr(geo, "active", False)) if geo is not None else False
    geo_crs = getattr(geo, "crs", None) if geo is not None else None
    geo_origin = getattr(geo, "local_origin", None) if geo is not None else None

    incoming_kind: Literal["angle", "local", "plain"]
    if x_dom == "angle" or y_dom == "angle":
        incoming_kind = "angle"
    elif geo_origin is not None:
        incoming_kind = "local"
    else:
        incoming_kind = "plain"

    active_crs = (
        str(geo_crs)
        if (geo is not None and geo_active and geo_crs is not None)
        else None
    )
    origin_tuple = (
        (float(geo_origin[0]), float(geo_origin[1])) if geo_origin is not None else None
    )

    if existing is None:
        ctx = GeoPlotContext(
            kind=incoming_kind,
            crs=active_crs,
            local_origin=origin_tuple,
            axis_units=(x_unit_obj, y_unit_obj),
        )
        return set_geo_context(ax, ctx)

    # Rule 1: angle vs real/local mixing is forbidden
    if (existing.kind == "angle") != (incoming_kind == "angle"):
        raise ValueError(
            "Cannot mix angle-domain axes with real/local-Cartesian axes. "
            "Use `to_local_cartesian()` to convert geographic fields to meters, "
            "or keep the DEM in degrees."
        )

    # Rule 2: active CRS mismatch emits UserWarning
    if (
        existing.crs is not None
        and active_crs is not None
        and existing.crs != active_crs
    ):
        warnings.warn(
            f"CRS mismatch between layers: '{existing.crs}' vs '{active_crs}'. "
            "Overlaying without reprojection.",
            UserWarning,
            stacklevel=3,
        )

    # Rule 3: local_origin mismatch emits UserWarning with approximate offset
    if (
        existing.kind == "local"
        and incoming_kind == "local"
        and existing.local_origin is not None
        and origin_tuple is not None
        and not np.allclose(existing.local_origin, origin_tuple, rtol=0.0, atol=1e-9)
    ):
        dx_m, dy_m = lonlat_to_local_xy(
            origin_tuple[0],
            origin_tuple[1],
            existing.local_origin[0],
            existing.local_origin[1],
        )
        offset_m = float(np.hypot(dx_m, dy_m))
        warnings.warn(
            f"local_origin mismatch between layers: {existing.local_origin} vs "
            f"{origin_tuple} (approx. {offset_m:.1f} m offset).",
            UserWarning,
            stacklevel=3,
        )

    # Rule 4: unit equivalence across layers (e.g. deg vs rad)
    for existing_u, incoming_u in zip(
        existing.axis_units, (x_unit_obj, y_unit_obj), strict=True
    ):
        if (
            existing_u != u.dimensionless_unscaled
            and incoming_u != u.dimensionless_unscaled
            and not existing_u.is_equivalent(incoming_u)
        ):
            raise ValueError(
                f"Incompatible axis units across layers: {existing_u} vs {incoming_u}."
            )

    # Enrich existing context if an earlier layer (e.g. basemap) left fields unset
    if existing.crs is None and active_crs is not None:
        existing.crs = active_crs
    if existing.local_origin is None and origin_tuple is not None:
        existing.local_origin = origin_tuple

    return existing


def _fill_nonfinite_nearest(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (filled_array, invalid_mask) filling non-finite cells by nearest neighbor."""
    z_arr = np.asarray(z, dtype=float)
    invalid = ~np.isfinite(z_arr)
    if not invalid.any() or invalid.all():
        return z_arr, invalid
    from scipy.ndimage import distance_transform_edt

    _, indices = distance_transform_edt(
        invalid, return_distances=True, return_indices=True
    )
    return z_arr[tuple(indices)], invalid


def compute_hillshade(
    z: Any,
    dx: float = 1.0,
    dy: float = 1.0,
    azdeg: float = 315.0,
    altdeg: float = 45.0,
    vert_exag: float = 1.0,
) -> np.ndarray:
    """Compute a hillshade array in ``[0, 1]`` preserving non-finite values as ``NaN``.

    Parameters
    ----------
    z : array-like
        2D elevation array.
    dx, dy : float, optional
        Grid spacing along the horizontal and vertical axes. Positive ``dy``
        corresponds to top-to-bottom (descending y / ``origin="upper"``) layout;
        negative ``dy`` corresponds to ascending y (``origin="lower"``).
    azdeg, altdeg : float, optional
        Illumination azimuth and altitude in degrees.
    vert_exag : float, optional
        Vertical exaggeration factor passed to :class:`matplotlib.colors.LightSource`.

    Returns
    -------
    numpy.ndarray
        2D float array of hillshade intensities in ``[0, 1]``, with ``NaN`` at
        non-finite input locations.

    """
    z_work, invalid = _fill_nonfinite_nearest(np.asarray(z, dtype=float))
    if invalid.all():
        return np.full(z_work.shape, np.nan, dtype=float)
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    hs = np.asarray(
        ls.hillshade(z_work, vert_exag=vert_exag, dx=dx, dy=dy), dtype=float
    )
    if invalid.any():
        hs[invalid] = np.nan
    return hs


def compute_shaded_relief(
    z: Any,
    cmap: Any,
    dx: float = 1.0,
    dy: float = 1.0,
    azdeg: float = 315.0,
    altdeg: float = 45.0,
    vert_exag: float = 1.0,
    vmin: float | None = None,
    vmax: float | None = None,
    blend_mode: Literal["hsv", "overlay", "soft"] | Callable[..., Any] = "overlay",
) -> np.ndarray:
    """Compute an RGBA shaded-relief array with transparent ``NaN`` pixels."""
    z_work, invalid = _fill_nonfinite_nearest(np.asarray(z, dtype=float))
    if invalid.all():
        return np.zeros((*z_work.shape, 4), dtype=float)
    ls = LightSource(azdeg=azdeg, altdeg=altdeg)
    rgba = np.asarray(
        ls.shade(
            z_work,
            cmap=cmap,
            blend_mode=blend_mode,
            vmin=vmin,
            vmax=vmax,
            vert_exag=vert_exag,
            dx=dx,
            dy=dy,
        ),
        dtype=np.float64,
    )
    if invalid.any():
        rgba[invalid] = (0.0, 0.0, 0.0, 0.0)
    return rgba
