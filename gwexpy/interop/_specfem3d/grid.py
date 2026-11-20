"""Cartesian grid validation and explicit resampling for SPECFEM3D fields."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from astropy import units as u

__all__ = [
    "RegularGridSpec",
    "infer_exact_cartesian_grid",
    "sample_specfem_to_grid",
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
            f"SPECFEM3D mesh does not form an exact Cartesian tensor product: "
            f"coordinates deviate from axis grid by {max_err:.3e} m; "
            "use grid_mode='sample' with an explicit RegularGridSpec to resample."
        )
    return centers, nearest_idx


def infer_exact_cartesian_grid(
    coords_m: np.ndarray,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate that deduplicated coordinates form an exact Cartesian tensor-product grid."""
    pts = np.asarray(coords_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"coords_m must have shape (N, 3), got {pts.shape}")
    n_pts = pts.shape[0]
    if n_pts == 0:
        raise ValueError("Cannot infer Cartesian grid from zero points")

    x_1d, ix = _cluster_axis(pts[:, 0], atol=atol, rtol=rtol)
    y_1d, iy = _cluster_axis(pts[:, 1], atol=atol, rtol=rtol)
    z_1d, iz = _cluster_axis(pts[:, 2], atol=atol, rtol=rtol)

    nx, ny, nz = len(x_1d), len(y_1d), len(z_1d)
    if nx * ny * nz != n_pts:
        raise ValueError(
            f"SPECFEM3D mesh does not form an exact Cartesian tensor product: "
            f"{n_pts} unique points != {nx}*{ny}*{nz}={nx * ny * nz}. "
            "Use grid_mode='sample' with an explicit RegularGridSpec to resample."
        )

    index_grid = np.full((nx, ny, nz), -1, dtype=np.int64)
    for src_i in range(n_pts):
        gx, gy, gz = int(ix[src_i]), int(iy[src_i]), int(iz[src_i])
        if index_grid[gx, gy, gz] != -1:
            raise ValueError(
                f"Duplicate SPECFEM3D coordinate mapped to grid index ({gx}, {gy}, {gz}); "
                "use grid_mode='sample' to resample."
            )
        index_grid[gx, gy, gz] = src_i
    return x_1d, y_1d, z_1d, index_grid


def _element_aabb_mask(
    query_3d: np.ndarray,
    raw_coords_m: np.ndarray,
    *,
    cell_blocks: Sequence[tuple[str, np.ndarray]] = (),
    atol: float = 1e-7,
) -> np.ndarray:
    """Compute domain containment mask from per-element bounding boxes or global bbox."""
    if raw_coords_m.ndim == 3:
        # GLL elements: (nspec, ngll, 3) -> exact per-element AABB
        e_lo = np.min(raw_coords_m, axis=1) - atol
        e_hi = np.max(raw_coords_m, axis=1) + atol
        g_lo = np.min(e_lo, axis=0)
        g_hi = np.max(e_hi, axis=0)
        in_bbox = np.all((query_3d >= g_lo) & (query_3d <= g_hi), axis=1)
        active = np.nonzero(in_bbox)[0]
        if active.size == 0:
            return in_bbox
        q_act = query_3d[active]
        hit = np.zeros(len(q_act), dtype=bool)
        chunk = 1024
        for start in range(0, len(e_lo), chunk):
            sl = slice(start, min(start + chunk, len(e_lo)))
            hit |= np.any(
                np.all(
                    (q_act[:, None, :] >= e_lo[None, sl, :])
                    & (q_act[:, None, :] <= e_hi[None, sl, :]),
                    axis=2,
                ),
                axis=1,
            )
        out = np.zeros(len(query_3d), dtype=bool)
        out[active] = hit
        return out

    if cell_blocks:
        g_lo = np.min(raw_coords_m, axis=0) - atol
        g_hi = np.max(raw_coords_m, axis=0) + atol
        in_bbox = np.all((query_3d >= g_lo) & (query_3d <= g_hi), axis=1)
        active = np.nonzero(in_bbox)[0]
        if active.size == 0:
            return in_bbox
        q_act = query_3d[active]
        hit = np.zeros(len(q_act), dtype=bool)
        for _kind, conn in cell_blocks:
            cpts = raw_coords_m[conn]
            c_lo = np.min(cpts, axis=1) - atol
            c_hi = np.max(cpts, axis=1) + atol
            chunk = 1024
            for start in range(0, len(c_lo), chunk):
                sl = slice(start, min(start + chunk, len(c_lo)))
                hit |= np.any(
                    np.all(
                        (q_act[:, None, :] >= c_lo[None, sl, :])
                        & (q_act[:, None, :] <= c_hi[None, sl, :]),
                        axis=2,
                    ),
                    axis=1,
                )
        out = np.zeros(len(query_3d), dtype=bool)
        out[active] = hit
        return out

    g_lo = np.min(raw_coords_m, axis=0) - atol
    g_hi = np.max(raw_coords_m, axis=0) + atol
    return np.all((query_3d >= g_lo) & (query_3d <= g_hi), axis=1)


def sample_specfem_to_grid(
    coords_m: np.ndarray,
    values: np.ndarray,
    grid_spec: RegularGridSpec,
    *,
    method: Literal["linear", "nearest"] = "linear",
    raw_coords_m: np.ndarray | None = None,
    cell_blocks: Sequence[tuple[str, np.ndarray]] = (),
) -> tuple[np.ndarray, np.ndarray, str]:
    """Explicitly resample SPECFEM3D coordinates onto ``grid_spec`` with ``NaN`` outside domain."""
    if method not in {"linear", "nearest"}:
        raise ValueError(
            f"Unsupported sample method {method!r}; expected 'linear' or 'nearest'"
        )
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    nx, ny, nz = grid_spec.shape_xyz
    gx, gy, gz = np.meshgrid(grid_spec.x, grid_spec.y, grid_spec.z, indexing="ij")
    query_3d = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    ref_raw = raw_coords_m if raw_coords_m is not None else coords_m
    in_domain = _element_aabb_mask(query_3d, ref_raw, cell_blocks=cell_blocks)

    spans = np.ptp(coords_m, axis=0)
    active_dims = [d for d in range(3) if spans[d] > 1e-12]

    val_arr = np.asarray(values, dtype=np.float64)
    is_vec = val_arr.ndim == 2
    n_comp = val_arr.shape[1] if is_vec else 1
    vals_2d = val_arr if is_vec else val_arr[:, np.newaxis]

    out_flat = np.full((len(query_3d), n_comp), np.nan, dtype=np.float64)
    valid_flat = np.zeros(len(query_3d), dtype=bool)

    if method == "nearest" or len(coords_m) < (len(active_dims) + 1):
        interp = NearestNDInterpolator(coords_m[:, active_dims], vals_2d)
        if np.any(in_domain):
            res = interp(query_3d[in_domain][:, active_dims])
            out_flat[in_domain] = res
            valid_flat[in_domain] = np.all(np.isfinite(res), axis=1)
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
