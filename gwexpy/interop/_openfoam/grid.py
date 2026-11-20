"""Cartesian grid specification, exact tensor-product validation, and explicit resampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from astropy import units as u

__all__ = [
    "RegularGridSpec",
    "check_moving_mesh",
    "infer_exact_cartesian_grid",
    "sample_to_regular_grid",
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
    """Explicit target Cartesian grid specification in metres for ``mode='sample'``.

    Parameters
    ----------
    x, y, z : 1-D array-like or Quantity
        Strictly increasing coordinate arrays along each spatial axis (metres).

    """

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
        shape: tuple[int, int, int] | None = None,
        spacing: float | tuple[float, float, float] | None = None,
    ) -> RegularGridSpec:
        """Build a regular grid from inclusive axis bounds and shape or spacing."""
        if (shape is None) == (spacing is None):
            raise ValueError("Provide exactly one of 'shape' or 'spacing'")
        bounds = (xlim, ylim, zlim)
        axes: list[np.ndarray] = []
        if shape is not None:
            if len(shape) != 3:
                raise ValueError("shape must have length 3 (nx, ny, nz)")
            for (lo, hi), n in zip(bounds, shape, strict=True):
                if n < 1:
                    raise ValueError("grid shape dimensions must be >= 1")
                if n == 1:
                    axes.append(
                        np.array([0.5 * (float(lo) + float(hi))], dtype=np.float64)
                    )
                else:
                    axes.append(
                        np.linspace(float(lo), float(hi), int(n), dtype=np.float64)
                    )
        else:
            assert spacing is not None
            spacings = (
                (float(spacing), float(spacing), float(spacing))
                if isinstance(spacing, (int, float, np.number))
                else tuple(float(s) for s in spacing)
            )
            if len(spacings) != 3:
                raise ValueError("spacing must be scalar or length-3 tuple")
            for (lo, hi), ds in zip(bounds, spacings, strict=True):
                if ds <= 0.0:
                    raise ValueError("spacing must be positive")
                lo_f, hi_f = float(lo), float(hi)
                if np.isclose(lo_f, hi_f, rtol=0.0, atol=1e-12):
                    axes.append(np.array([lo_f], dtype=np.float64))
                else:
                    n_pts = max(2, int(np.floor((hi_f - lo_f) / ds + 0.5)) + 1)
                    axes.append(np.linspace(lo_f, hi_f, n_pts, dtype=np.float64))
        return cls(x=axes[0], y=axes[1], z=axes[2])


def _cluster_axis_coordinates(
    coords_1d: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster 1-D coordinates within tolerance into unique sorted grid coordinates."""
    if coords_1d.size == 0:
        raise ValueError("Cannot infer Cartesian grid from empty coordinate array")
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
    # Map each input coordinate to its nearest cluster index
    idx = np.searchsorted(centers, coords_1d)
    idx = np.clip(idx, 0, len(centers) - 1)
    left = np.clip(idx - 1, 0, len(centers) - 1)
    choose_left = np.abs(coords_1d - centers[left]) < np.abs(coords_1d - centers[idx])
    nearest_idx = np.where(choose_left, left, idx)
    max_err = float(np.max(np.abs(coords_1d - centers[nearest_idx])))
    if max_err > tol:
        raise ValueError(
            f"Coordinates deviate from clustered axis grid by {max_err:.3e} m (> tol={tol:.3e} m)"
        )
    return centers, nearest_idx


def infer_exact_cartesian_grid(
    coords_m: np.ndarray,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate that ``coords_m`` forms a complete Cartesian tensor-product grid.

    No interpolation is performed. Returns ``(x, y, z, index_grid)`` where
    ``index_grid`` has shape ``(nx, ny, nz)`` giving the source index at each
    Cartesian grid point.
    """
    pts = np.asarray(coords_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"coords_m must have shape (N, 3), got {pts.shape}")
    n_pts = pts.shape[0]
    if n_pts == 0:
        raise ValueError("Cannot infer Cartesian grid from zero points")
    if not np.isfinite(pts).all():
        raise ValueError("coords_m must be finite")

    x_1d, ix = _cluster_axis_coordinates(pts[:, 0], atol=atol, rtol=rtol)
    y_1d, iy = _cluster_axis_coordinates(pts[:, 1], atol=atol, rtol=rtol)
    z_1d, iz = _cluster_axis_coordinates(pts[:, 2], atol=atol, rtol=rtol)

    nx, ny, nz = len(x_1d), len(y_1d), len(z_1d)
    if nx * ny * nz != n_pts:
        raise ValueError(
            f"Mesh coordinates do not form an exact Cartesian tensor-product grid: "
            f"point count {n_pts} != nx*ny*nz ({nx}*{ny}*{nz}={nx * ny * nz}). "
            f"Use mode='sample' with an explicit RegularGridSpec to resample."
        )

    index_grid = np.full((nx, ny, nz), -1, dtype=np.int64)
    for src_i in range(n_pts):
        gx, gy, gz = int(ix[src_i]), int(iy[src_i]), int(iz[src_i])
        if index_grid[gx, gy, gz] != -1:
            raise ValueError(
                f"Duplicate coordinate mapped to Cartesian cell ({gx}, {gy}, {gz}); "
                f"use mode='sample' to resample onto a regular grid."
            )
        index_grid[gx, gy, gz] = src_i

    if np.any(index_grid < 0):
        raise ValueError(
            "Incomplete Cartesian grid (missing cells); use mode='sample' to resample."
        )
    return x_1d, y_1d, z_1d, index_grid


def _point_in_domain_mask(
    query_points: np.ndarray,
    coords_m: np.ndarray,
    *,
    points_m: np.ndarray | None = None,
    cell_blocks: Sequence[tuple[str, np.ndarray]] | None = None,
    atol: float = 1e-9,
) -> np.ndarray:
    """Compute a conservative domain mask for query points."""
    ref = points_m if points_m is not None and len(points_m) > 0 else coords_m
    lo = np.min(ref, axis=0) - atol
    hi = np.max(ref, axis=0) + atol
    in_bbox = np.all((query_points >= lo) & (query_points <= hi), axis=1)

    if cell_blocks and points_m is not None:
        # Check union of per-cell AABBs (handles concave/L-shaped Cartesian meshes cleanly)
        inside_any = np.zeros(len(query_points), dtype=bool)
        active_idx = np.nonzero(in_bbox)[0]
        if active_idx.size == 0:
            return inside_any
        q_act = query_points[active_idx]
        hit = np.zeros(len(q_act), dtype=bool)
        for _kind, conn in cell_blocks:
            cell_pts = points_m[conn]  # (n_cells, n_vert, 3)
            c_lo = np.min(cell_pts, axis=1) - atol
            c_hi = np.max(cell_pts, axis=1) + atol
            # Chunk cells to keep peak memory bounded
            chunk = 2048
            for start in range(0, len(c_lo), chunk):
                sl = slice(start, min(start + chunk, len(c_lo)))
                inside_chunk = np.all(
                    (q_act[:, None, :] >= c_lo[None, sl, :])
                    & (q_act[:, None, :] <= c_hi[None, sl, :]),
                    axis=2,
                )
                hit |= np.any(inside_chunk, axis=1)
        inside_any[active_idx] = hit
        return inside_any

    return in_bbox


def _sample_via_pyvista(
    pyvista_mesh: Any,
    field_name: str,
    association: str,
    grid_spec: RegularGridSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Use PyVista ``RectilinearGrid.sample`` to probe unstructured cells/points."""
    import pyvista as pv

    target = pv.RectilinearGrid(
        np.asarray(grid_spec.x, dtype=np.float64),
        np.asarray(grid_spec.y, dtype=np.float64),
        np.asarray(grid_spec.z, dtype=np.float64),
    )
    # Ensure requested array is present on pyvista_mesh
    sampled = target.sample(pyvista_mesh, pass_cell_data=True, pass_point_data=True)
    nx, ny, nz = grid_spec.shape_xyz

    mask_flat = np.asarray(sampled["vtkValidPointMask"], dtype=bool)
    mask = mask_flat.reshape((nx, ny, nz), order="F")

    if field_name not in sampled.point_data:
        raise KeyError(
            f"Field {field_name!r} (association={association}) was not transferred by PyVista sample"
        )
    vals_flat = np.asarray(sampled.point_data[field_name], dtype=np.float64)
    if vals_flat.ndim == 1:
        vals = vals_flat.reshape((nx, ny, nz), order="F")
        vals = np.where(mask, vals, np.nan)
    else:
        ncomp = vals_flat.shape[1]
        vals = vals_flat.reshape((nx, ny, nz, ncomp), order="F")
        vals = np.where(mask[..., np.newaxis], vals, np.nan)
    return vals, mask


def _sample_via_scipy(
    coords_m: np.ndarray,
    values: np.ndarray,
    grid_spec: RegularGridSpec,
    *,
    method: Literal["linear", "nearest"],
    points_m: np.ndarray | None = None,
    cell_blocks: Sequence[tuple[str, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample scattered coordinates onto ``grid_spec`` using SciPy."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    nx, ny, nz = grid_spec.shape_xyz
    gx, gy, gz = np.meshgrid(grid_spec.x, grid_spec.y, grid_spec.z, indexing="ij")
    query_3d = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    # Identify active (non-degenerate) spatial dimensions
    spans = np.ptp(coords_m, axis=0)
    active_dims = [d for d in range(3) if spans[d] > 1e-12]

    in_domain = _point_in_domain_mask(
        query_3d,
        coords_m,
        points_m=points_m,
        cell_blocks=cell_blocks,
    )

    # Check degenerate coordinates match query points within tolerance if cell_blocks absent
    if cell_blocks is None:
        for d in range(3):
            if d not in active_dims:
                ref_val = float(coords_m[0, d])
                in_domain &= np.abs(query_3d[:, d] - ref_val) <= 1e-9

    val_arr = np.asarray(values, dtype=np.float64)
    is_vec = val_arr.ndim == 2
    n_comp = val_arr.shape[1] if is_vec else 1
    vals_2d = val_arr if is_vec else val_arr[:, np.newaxis]

    out_shape = (len(query_3d), n_comp)
    out_flat = np.full(out_shape, np.nan, dtype=np.float64)
    valid_flat = np.zeros(len(query_3d), dtype=bool)

    if len(active_dims) == 0:
        # Single-point source
        out_flat[in_domain] = vals_2d[0]
        valid_flat[in_domain] = True
    elif method == "nearest" or len(coords_m) < (len(active_dims) + 1):
        if method == "linear" and len(coords_m) < (len(active_dims) + 1):
            raise ValueError(
                f"Linear interpolation requires at least {len(active_dims) + 1} "
                f"non-degenerate points, got {len(coords_m)}. Use method='nearest' or 'vtk_probe'."
            )
        interp = NearestNDInterpolator(coords_m[:, active_dims], vals_2d)
        if np.any(in_domain):
            out_flat[in_domain] = interp(query_3d[in_domain][:, active_dims])
            valid_flat[in_domain] = True
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
                valid_flat[in_domain] = np.isfinite(out_flat[in_domain, 0])
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
        return out_flat.reshape((nx, ny, nz, n_comp)), mask
    return out_flat[:, 0].reshape((nx, ny, nz)), mask


def sample_to_regular_grid(
    coords_m: np.ndarray,
    values: np.ndarray,
    grid_spec: RegularGridSpec,
    *,
    method: Literal["auto", "linear", "nearest", "vtk_probe"] = "auto",
    pyvista_mesh: Any = None,
    field_name: str | None = None,
    association: str = "cell",
    points_m: np.ndarray | None = None,
    cell_blocks: Sequence[tuple[str, np.ndarray]] | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Explicitly resample field values onto ``grid_spec``.

    Points outside the source mesh domain are assigned ``NaN`` and marked
    ``False`` in the returned boolean valid mask.

    Returns
    -------
    data : ndarray
        Shape ``(nx, ny, nz)`` for scalar or ``(nx, ny, nz, ncomp)`` for vector.
    valid_mask : ndarray of bool
        Shape ``(nx, ny, nz)``.
    resolved_method : str
        Name of the interpolation backend/method used.

    """
    if method not in {"auto", "linear", "nearest", "vtk_probe"}:
        raise ValueError(
            f"Unsupported sample method {method!r}; expected 'auto', 'linear', 'nearest', or 'vtk_probe'"
        )

    if method == "vtk_probe":
        if pyvista_mesh is None or field_name is None:
            raise ValueError(
                "method='vtk_probe' requires a PyVista mesh and field_name"
            )
        vals, mask = _sample_via_pyvista(
            pyvista_mesh, field_name, association, grid_spec
        )
        return vals, mask, "vtk_probe"

    if method == "auto":
        if pyvista_mesh is not None and field_name is not None:
            vals, mask = _sample_via_pyvista(
                pyvista_mesh, field_name, association, grid_spec
            )
            return vals, mask, "vtk_probe"
        vals, mask = _sample_via_scipy(
            coords_m,
            values,
            grid_spec,
            method="linear",
            points_m=points_m,
            cell_blocks=cell_blocks,
        )
        return vals, mask, "scipy_linear"

    vals, mask = _sample_via_scipy(
        coords_m,
        values,
        grid_spec,
        method=method,
        points_m=points_m,
        cell_blocks=cell_blocks,
    )
    return vals, mask, f"scipy_{method}"


def check_moving_mesh(
    snapshots: Sequence[Any],
    *,
    moving_mesh: Literal["error", "resample"] = "error",
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> bool:
    """Detect mesh motion across snapshots and enforce ``moving_mesh`` policy.

    Returns ``True`` if mesh geometry differs across snapshots, ``False`` if
    static. Raises ``ValueError`` when motion is detected and ``moving_mesh='error'``.
    """
    if moving_mesh not in {"error", "resample"}:
        raise ValueError(
            f"moving_mesh must be 'error' or 'resample', got {moving_mesh!r}"
        )
    if len(snapshots) <= 1:
        return False

    ref = snapshots[0]
    ref_pts = np.asarray(ref.points_m, dtype=np.float64)
    ref_centers = (
        np.asarray(ref.cell_centers_m, dtype=np.float64)
        if ref.cell_centers_m is not None
        else None
    )

    motion_detected = False
    for snap in snapshots[1:]:
        pts = np.asarray(snap.points_m, dtype=np.float64)
        if pts.shape != ref_pts.shape or not np.allclose(
            pts, ref_pts, rtol=rtol, atol=atol
        ):
            motion_detected = True
            break
        if ref_centers is not None or snap.cell_centers_m is not None:
            if ref_centers is None or snap.cell_centers_m is None:
                motion_detected = True
                break
            centers = np.asarray(snap.cell_centers_m, dtype=np.float64)
            if centers.shape != ref_centers.shape or not np.allclose(
                centers, ref_centers, rtol=rtol, atol=atol
            ):
                motion_detected = True
                break

    if motion_detected and moving_mesh == "error":
        raise ValueError(
            "Moving or deforming mesh detected across OpenFOAM time steps. "
            "Pass moving_mesh='resample' together with mode='sample' and an "
            "explicit RegularGridSpec (grid=...) to resample onto a fixed grid."
        )
    return motion_detected
