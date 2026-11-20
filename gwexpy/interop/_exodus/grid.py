"""Exact Cartesian grid validation and explicit sampling for EXODUS II meshes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from astropy import units as u

__all__ = [
    "RegularGridSpec",
    "infer_exact_cartesian_grid",
    "sample_exodus_to_grid",
]


def _coerce_axis_metres(values: Any, name: str) -> np.ndarray:
    if isinstance(values, u.Quantity):
        arr = np.asarray(values.to_value(u.m), dtype=np.float64)
    else:
        arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D coordinate array")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} coordinates must be finite")
    if arr.size > 1 and np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} coordinates must be strictly increasing")
    out = arr.copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class RegularGridSpec:
    """Explicit target Cartesian grid specification in metres for ``grid_mode='sample'``."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _coerce_axis_metres(self.x, "x"))
        object.__setattr__(self, "y", _coerce_axis_metres(self.y, "y"))
        object.__setattr__(self, "z", _coerce_axis_metres(self.z, "z"))

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        return (int(self.x.size), int(self.y.size), int(self.z.size))

    @classmethod
    def from_bounds(
        cls,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        zlim: tuple[float, float],
        *,
        shape: tuple[int, int, int],
    ) -> RegularGridSpec:
        """Build a regular grid from bounds and ``(nx, ny, nz)`` shape."""
        if len(shape) != 3:
            raise ValueError("shape must have length 3 (nx, ny, nz)")
        axes = []
        for (lo, hi), n in zip((xlim, ylim, zlim), shape, strict=True):
            if n < 1:
                raise ValueError("grid shape dimensions must be >= 1")
            if n == 1:
                axes.append(np.array([0.5 * (float(lo) + float(hi))], dtype=np.float64))
            else:
                axes.append(np.linspace(float(lo), float(hi), int(n), dtype=np.float64))
        return cls(x=axes[0], y=axes[1], z=axes[2])


def _cluster_axis(
    coords_1d: np.ndarray, *, atol: float, rtol: float
) -> tuple[np.ndarray, np.ndarray]:
    sorted_vals = np.sort(coords_1d)
    scale = max(1.0, float(np.max(np.abs(sorted_vals))))
    tol = atol + rtol * scale
    clusters: list[list[float]] = [[float(sorted_vals[0])]]
    for val in sorted_vals[1:]:
        fval = float(val)
        if abs(fval - clusters[-1][-1]) <= tol:
            clusters[-1].append(fval)
        else:
            clusters.append([fval])
    centers = np.array([float(np.mean(c)) for c in clusters], dtype=np.float64)
    idx = np.searchsorted(centers, coords_1d)
    idx = np.clip(idx, 0, len(centers) - 1)
    left = np.clip(idx - 1, 0, len(centers) - 1)
    choose_left = np.abs(coords_1d - centers[left]) < np.abs(coords_1d - centers[idx])
    nearest_idx = np.where(choose_left, left, idx)
    max_err = float(np.max(np.abs(coords_1d - centers[nearest_idx])))
    if max_err > tol:
        raise ValueError(
            f"EXODUS coordinates deviate from regular Cartesian axis by {max_err:.3e} m"
        )
    return centers, nearest_idx


def infer_exact_cartesian_grid(
    coords_m: np.ndarray,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
    force_surface_2d: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate that ``coords_m`` forms an exact Cartesian tensor-product grid.

    Returns ``(x_1d, y_1d, z_1d, index_grid)``.
    """
    pts = np.asarray(coords_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"coords_m must have shape (N, 3), got {pts.shape}")
    n_pts = pts.shape[0]
    if n_pts == 0:
        raise ValueError("Cannot infer Cartesian grid from zero entities")
    if not np.isfinite(pts).all():
        raise ValueError("coords_m must contain finite coordinates")

    x_1d, ix = _cluster_axis(pts[:, 0], atol=atol, rtol=rtol)
    y_1d, iy = _cluster_axis(pts[:, 1], atol=atol, rtol=rtol)
    z_1d, iz = _cluster_axis(pts[:, 2], atol=atol, rtol=rtol)

    if force_surface_2d and len(z_1d) != 1:
        raise ValueError(
            f"Surface variable requires single vertical plane (nz=1), got nz={len(z_1d)}"
        )

    nx, ny, nz = len(x_1d), len(y_1d), len(z_1d)
    if nx * ny * nz != n_pts:
        raise ValueError(
            f"EXODUS mesh coordinates do not form an exact Cartesian tensor product: "
            f"{n_pts} != {nx}*{ny}*{nz}. Use grid_mode='sample' with RegularGridSpec."
        )

    index_grid = np.full((nx, ny, nz), -1, dtype=np.int64)
    for src_i in range(n_pts):
        gx, gy, gz = int(ix[src_i]), int(iy[src_i]), int(iz[src_i])
        if index_grid[gx, gy, gz] != -1:
            raise ValueError(
                f"Duplicate EXODUS coordinate at Cartesian index ({gx}, {gy}, {gz}); "
                f"use grid_mode='sample' to resample."
            )
        index_grid[gx, gy, gz] = src_i

    return x_1d, y_1d, z_1d, index_grid


def _domain_mask(
    query_3d: np.ndarray,
    coords_m: np.ndarray,
    *,
    points_m: np.ndarray | None = None,
    connectivities: Sequence[np.ndarray] | None = None,
    atol: float = 1e-9,
) -> np.ndarray:
    ref = points_m if points_m is not None and len(points_m) > 0 else coords_m
    lo = np.min(ref, axis=0) - atol
    hi = np.max(ref, axis=0) + atol
    in_bbox = np.all((query_3d >= lo) & (query_3d <= hi), axis=1)

    if connectivities and points_m is not None:
        inside_any = np.zeros(len(query_3d), dtype=bool)
        active = np.nonzero(in_bbox)[0]
        if active.size == 0:
            return inside_any
        q_act = query_3d[active]
        hit = np.zeros(len(q_act), dtype=bool)
        for conn in connectivities:
            cell_pts = points_m[conn]
            c_lo = np.min(cell_pts, axis=1) - atol
            c_hi = np.max(cell_pts, axis=1) + atol
            chunk = 2048
            for start in range(0, len(c_lo), chunk):
                sl = slice(start, min(start + chunk, len(c_lo)))
                inside_chunk = np.all(
                    (q_act[:, None, :] >= c_lo[None, sl, :])
                    & (q_act[:, None, :] <= c_hi[None, sl, :]),
                    axis=2,
                )
                hit |= np.any(inside_chunk, axis=1)
        inside_any[active] = hit
        return inside_any
    return in_bbox


def sample_exodus_to_grid(
    coords_m: np.ndarray,
    values: np.ndarray,
    grid_spec: RegularGridSpec,
    *,
    method: Literal["linear", "nearest"] = "linear",
    points_m: np.ndarray | None = None,
    connectivities: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Explicitly resample EXODUS values onto ``grid_spec``.

    Points outside the source domain are assigned ``np.nan`` and marked
    ``False`` in the returned valid mask. Values that are already ``NaN``
    (e.g. FAVOR-blocked cells) remain ``NaN`` and invalid under ``nearest``;
    under ``linear``, only finite source entities contribute or NaN propagates.

    Returns
    -------
    data : ndarray
        Resampled array of shape ``(nx, ny, nz)`` or ``(nx, ny, nz, ncomp)``.
    valid_mask : ndarray of bool
        Boolean mask of valid samples inside the domain.
    method_used : str
        Name of the interpolation method applied.

    """
    if method not in {"linear", "nearest"}:
        raise ValueError(
            f"Unsupported sample method {method!r}; expected 'linear' or 'nearest'"
        )

    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    nx, ny, nz = grid_spec.shape_xyz
    gx, gy, gz = np.meshgrid(grid_spec.x, grid_spec.y, grid_spec.z, indexing="ij")
    query_3d = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    spans = np.ptp(coords_m, axis=0)
    active_dims = [d for d in range(3) if spans[d] > 1e-12]

    in_domain = _domain_mask(
        query_3d,
        coords_m,
        points_m=points_m,
        connectivities=connectivities,
    )
    if connectivities is None:
        for d in range(3):
            if d not in active_dims:
                in_domain &= np.abs(query_3d[:, d] - float(coords_m[0, d])) <= 1e-9

    val_arr = np.asarray(values, dtype=np.float64)
    is_vec = val_arr.ndim == 2
    n_comp = val_arr.shape[1] if is_vec else 1
    vals_2d = val_arr if is_vec else val_arr[:, np.newaxis]

    out_flat = np.full((len(query_3d), n_comp), np.nan, dtype=np.float64)
    valid_flat = np.zeros(len(query_3d), dtype=bool)

    if len(active_dims) == 0:
        if np.all(np.isfinite(vals_2d[0])):
            out_flat[in_domain] = vals_2d[0]
            valid_flat[in_domain] = True
    elif method == "nearest" or len(coords_m) < (len(active_dims) + 1):
        if method == "linear" and len(coords_m) < (len(active_dims) + 1):
            raise ValueError(
                f"Linear interpolation requires at least {len(active_dims) + 1} "
                f"points, got {len(coords_m)}; use sample_method='nearest'."
            )
        interp = NearestNDInterpolator(coords_m[:, active_dims], vals_2d)
        if np.any(in_domain):
            res = interp(query_3d[in_domain][:, active_dims])
            out_flat[in_domain] = res
            valid_flat[in_domain] = np.all(np.isfinite(res), axis=1)
    else:
        if len(active_dims) == 1:
            dim = active_dims[0]
            order = np.argsort(coords_m[:, dim])
            x_src = coords_m[order, dim]
            y_src = vals_2d[order]
            if np.any(in_domain):
                q_1d = query_3d[in_domain, dim]
                for c in range(n_comp):
                    out_flat[in_domain, c] = np.interp(
                        q_1d, x_src, y_src[:, c], left=np.nan, right=np.nan
                    )
                valid_flat[in_domain] = np.all(np.isfinite(out_flat[in_domain]), axis=1)
        else:
            interp = LinearNDInterpolator(
                coords_m[:, active_dims], vals_2d, fill_value=np.nan
            )
            if np.any(in_domain):
                res = interp(query_3d[in_domain][:, active_dims])
                out_flat[in_domain] = res
                valid_flat[in_domain] = np.all(np.isfinite(res), axis=1)

    mask = valid_flat.reshape((nx, ny, nz))
    if is_vec:
        return out_flat.reshape((nx, ny, nz, n_comp)), mask, f"scipy_{method}"
    return out_flat[:, 0].reshape((nx, ny, nz)), mask, f"scipy_{method}"
