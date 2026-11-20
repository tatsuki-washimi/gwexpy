# Adapted from GraGra (https://github.com/tatsuki-washimi/gragra)
# Commit: ce3a9f0f7c686097dec2cb34aca80222a88d11e1
# Source: src/gragra/adapters/specfem/gll.py, src/gragra/adapters/specfem/schema.py
# Copyright (c) 2026 Tatsuki Washimi — MIT License
"""GLL quadrature rules, trilinear hex Jacobians, and GLL node deduplication."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Final, Literal

import numpy as np

__all__ = [
    "EXPECTED_NGLL",
    "N_CORNER_NODES",
    "deduplicate_gll_points",
    "element_quadrature_volume",
    "elements_quadrature_volume",
    "flatten_index",
    "gll_points_weights",
    "ngll_from_shape",
    "reshape_shape_zyx",
    "unflatten_index",
]

EXPECTED_NGLL: Final[tuple[int, int, int]] = (5, 5, 5)
N_CORNER_NODES: Final[int] = 8
_NEWTON_MAX_ITER: Final[int] = 100
_NEWTON_TOL: Final[float] = 1e-15
_ELEMENT_BLOCK_SIZE: Final[int] = 512

_REFERENCE_CORNERS: Final[np.ndarray] = np.array(
    [
        (-1.0, -1.0, -1.0),
        (+1.0, -1.0, -1.0),
        (+1.0, +1.0, -1.0),
        (-1.0, +1.0, -1.0),
        (-1.0, -1.0, +1.0),
        (+1.0, -1.0, +1.0),
        (+1.0, +1.0, +1.0),
        (-1.0, +1.0, +1.0),
    ],
    dtype=np.float64,
)
_REFERENCE_CORNERS.setflags(write=False)


def _validate_ngll_shape(ngll_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(ngll_shape) != 3:
        raise ValueError(f"ngll_shape must have length 3, got {ngll_shape!r}")
    nx, ny, nz = (int(v) for v in ngll_shape)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"ngll_shape entries must be positive, got {ngll_shape!r}")
    return nx, ny, nz


def ngll_from_shape(ngll_shape: tuple[int, int, int]) -> int:
    """Return total GLL point count ``NGLLX * NGLLY * NGLLZ``."""
    nx, ny, nz = _validate_ngll_shape(ngll_shape)
    return nx * ny * nz


def reshape_shape_zyx(ngll_shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return ``(NGLLZ, NGLLY, NGLLX)`` C-order reshape target (x innermost)."""
    nx, ny, nz = _validate_ngll_shape(ngll_shape)
    return (nz, ny, nx)


def flatten_index(i: int, j: int, k: int, ngll_shape: tuple[int, int, int]) -> int:
    """Map ``(i, j, k)`` to flat GLL index ``k*(ny*nx) + j*nx + i``."""
    nx, ny, nz = _validate_ngll_shape(ngll_shape)
    if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
        raise ValueError(f"(i, j, k)=({i}, {j}, {k}) out of bounds for {ngll_shape}")
    return k * (ny * nx) + j * nx + i


def unflatten_index(
    flat: int, ngll_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Inverse of :func:`flatten_index`, returning ``(i, j, k)``."""
    nx, ny, nz = _validate_ngll_shape(ngll_shape)
    total = nx * ny * nz
    if not (0 <= flat < total):
        raise ValueError(f"flat index {flat} out of range [0, {total})")
    k, rem = divmod(flat, ny * nx)
    j, i = divmod(rem, nx)
    return i, j, k


def _legendre_p_and_pm1(x: float, degree: int) -> tuple[float, float]:
    p_prev = 1.0
    p_curr = x
    for k in range(2, degree + 1):
        p_prev, p_curr = p_curr, ((2 * k - 1) * x * p_curr - (k - 1) * p_prev) / k
    return p_curr, p_prev


def _interior_gll_node(seed: float, degree: int) -> float:
    x = seed
    for _ in range(_NEWTON_MAX_ITER):
        p_n, p_nm1 = _legendre_p_and_pm1(x, degree)
        one_minus_x2 = 1.0 - x * x
        d_p = degree * (x * p_n - p_nm1) / (-one_minus_x2)
        d2_p = (2.0 * x * d_p - degree * (degree + 1) * p_n) / one_minus_x2
        step = d_p / d2_p
        x -= step
        if not (-1.0 < x < 1.0):
            raise RuntimeError(f"GLL Newton step left (-1, 1) at x={x!r}")
        if abs(step) <= _NEWTON_TOL:
            return x
    raise RuntimeError("GLL Newton iteration failed to converge")


@cache
def gll_points_weights(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute Gauss-Lobatto-Legendre abscissae and weights on ``[-1, 1]``."""
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    n = int(n)
    if n < 2:
        raise ValueError(f"GLL requires at least 2 points, got n={n}")

    degree = n - 1
    points = np.empty(n, dtype=np.float64)
    points[0] = -1.0
    points[n - 1] = +1.0
    if n % 2 == 1:
        points[(n - 1) // 2] = 0.0
    for idx in range(1, (n - 2) // 2 + 1):
        node = _interior_gll_node(-np.cos(np.pi * idx / degree), degree)
        points[idx] = node
        points[n - 1 - idx] = -node

    weights = np.empty(n, dtype=np.float64)
    for i in range(n):
        p_n, _ = _legendre_p_and_pm1(float(points[i]), degree)
        weights[i] = 2.0 / (n * degree * p_n * p_n)

    points.setflags(write=False)
    weights.setflags(write=False)
    return points, weights


@dataclass(frozen=True)
class _ReferenceGrid:
    dn_dxi: np.ndarray
    dn_deta: np.ndarray
    dn_dzeta: np.ndarray
    weights: np.ndarray


@cache
def _reference_grid_for_zyx(grid_shape: tuple[int, int, int]) -> _ReferenceGrid:
    n_z, n_y, n_x = grid_shape
    points_x, weights_x = gll_points_weights(n_x)
    points_y, weights_y = gll_points_weights(n_y)
    points_z, weights_z = gll_points_weights(n_z)

    xi = np.broadcast_to(points_x[None, None, :], grid_shape).ravel()
    eta = np.broadcast_to(points_y[None, :, None], grid_shape).ravel()
    zeta = np.broadcast_to(points_z[:, None, None], grid_shape).ravel()
    weights = (
        weights_z[:, None, None] * weights_y[None, :, None] * weights_x[None, None, :]
    ).ravel()

    corner_xi = _REFERENCE_CORNERS[:, 0]
    corner_eta = _REFERENCE_CORNERS[:, 1]
    corner_zeta = _REFERENCE_CORNERS[:, 2]

    one_plus_xi = 1.0 + xi[:, None] * corner_xi[None, :]
    one_plus_eta = 1.0 + eta[:, None] * corner_eta[None, :]
    one_plus_zeta = 1.0 + zeta[:, None] * corner_zeta[None, :]

    dn_dxi = 0.125 * corner_xi[None, :] * one_plus_eta * one_plus_zeta
    dn_deta = 0.125 * one_plus_xi * corner_eta[None, :] * one_plus_zeta
    dn_dzeta = 0.125 * one_plus_xi * one_plus_eta * corner_zeta[None, :]

    for arr in (dn_dxi, dn_deta, dn_dzeta, weights):
        arr.setflags(write=False)
    return _ReferenceGrid(
        dn_dxi=dn_dxi, dn_deta=dn_deta, dn_dzeta=dn_dzeta, weights=weights
    )


def _jacobian_det_batch(corners_batch: np.ndarray, ref: _ReferenceGrid) -> np.ndarray:
    centered = corners_batch - corners_batch.mean(axis=1, keepdims=True)
    dx_dxi = np.einsum("pa,eai->epi", ref.dn_dxi, centered)
    dx_deta = np.einsum("pa,eai->epi", ref.dn_deta, centered)
    dx_dzeta = np.einsum("pa,eai->epi", ref.dn_dzeta, centered)
    det_j = (
        dx_dxi[..., 0]
        * (dx_deta[..., 1] * dx_dzeta[..., 2] - dx_deta[..., 2] * dx_dzeta[..., 1])
        - dx_dxi[..., 1]
        * (dx_deta[..., 0] * dx_dzeta[..., 2] - dx_deta[..., 2] * dx_dzeta[..., 0])
        + dx_dxi[..., 2]
        * (dx_deta[..., 0] * dx_dzeta[..., 1] - dx_deta[..., 1] * dx_dzeta[..., 0])
    )
    return det_j


def elements_quadrature_volume(
    corners8: np.ndarray,
    ngll_shape: tuple[int, int, int] = EXPECTED_NGLL,
) -> np.ndarray:
    """Compute GLL quadrature volumes ``detJ * w_x * w_y * w_z`` for ``(nspec, 8, 3)`` corners."""
    corners = np.asarray(corners8, dtype=np.float64)
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise ValueError(f"corners8 must have shape (nspec, 8, 3), got {corners.shape}")
    if not np.isfinite(corners).all():
        raise ValueError("corners8 must contain finite coordinates")
    grid_shape = reshape_shape_zyx(ngll_shape)
    ref = _reference_grid_for_zyx(grid_shape)
    nspec = corners.shape[0]
    ngll = ngll_from_shape(ngll_shape)
    out = np.empty((nspec, ngll), dtype=np.float64)
    for start in range(0, nspec, _ELEMENT_BLOCK_SIZE):
        stop = min(start + _ELEMENT_BLOCK_SIZE, nspec)
        det_j = _jacobian_det_batch(corners[start:stop], ref)
        if np.any(det_j <= 0.0):
            raise ValueError(
                f"Non-positive Jacobian determinant in element range [{start}, {stop})"
            )
        out[start:stop] = det_j * ref.weights[None, :]
    out.setflags(write=False)
    return out


def element_quadrature_volume(
    corners8: np.ndarray,
    ngll_shape: tuple[int, int, int] = EXPECTED_NGLL,
) -> np.ndarray:
    """Compute GLL quadrature volumes ``(ngll,)`` for a single 8-corner hexahedron."""
    single = np.asarray(corners8, dtype=np.float64)
    if single.shape != (8, 3):
        raise ValueError(f"corners8 must have shape (8, 3), got {single.shape}")
    res = elements_quadrature_volume(single[np.newaxis, ...], ngll_shape=ngll_shape)[0]
    copy_res = np.array(res, dtype=np.float64, copy=True)
    copy_res.setflags(write=False)
    return copy_res


@dataclass(frozen=True)
class DeduplicatedGLLResult:
    """Result of GLL shared-face deduplication and continuity verification."""

    coordinates_m: np.ndarray
    values: np.ndarray
    inverse_indices: np.ndarray
    n_raw_points: int
    n_unique_points: int
    max_discrepancy: float


def deduplicate_gll_points(
    coordinates_m: np.ndarray,
    values: np.ndarray,
    *,
    coord_tol: float | None = None,
    verify_continuity: Literal["raise", "ignore"] = "raise",
    continuity_atol: float = 1e-6,
    continuity_rtol: float = 1e-5,
) -> DeduplicatedGLLResult:
    """Deduplicate shared GLL face/edge/corner points across SPECFEM3D elements.

    Parameters
    ----------
    coordinates_m : ndarray
        Shape ``(nspec, ngll, 3)`` or ``(N, 3)``.
    values : ndarray
        Shape ``(nspec, ngll, ...)``, ``(nt, nspec, ngll, ...)``, or ``(N, ...)``.
    coord_tol : float, optional
        Spatial tolerance (metres) for identifying coincident GLL coordinates.
        Defaults to ``1e-5 * max(1.0, max(|coords|))`` to absorb float32
        coordinate serialization rounding while keeping distinct GLL nodes apart.
    verify_continuity : {'raise', 'ignore'}, default 'raise'
        Whether to raise ``ValueError`` when coincident GLL nodes have
        values differing beyond ``continuity_atol + continuity_rtol * scale``.
    continuity_atol : float, default 1e-6
        Absolute tolerance for shared-GLL value continuity check.
    continuity_rtol : float, default 1e-5
        Relative tolerance for shared-GLL value continuity check.

    Returns
    -------
    DeduplicatedGLLResult
        Unique coordinates ``(n_unique, 3)``, deduplicated values, inverse
        index mapping, and observed maximum shared-point discrepancy.

    """
    coords = np.asarray(coordinates_m, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    if coords.ndim == 3:
        nspec, ngll, d = coords.shape
        if d != 3:
            raise ValueError(f"Expected trailing axis 3, got {coords.shape}")
        n_raw = nspec * ngll
        coords_flat = coords.reshape((n_raw, 3))
        if vals.shape[:2] == (nspec, ngll):
            has_time = False
            trailing = vals.shape[2:]
            vals_flat = vals.reshape((n_raw, *trailing))
        elif vals.ndim >= 3 and vals.shape[1:3] == (nspec, ngll):
            has_time = True
            nt = vals.shape[0]
            trailing = vals.shape[3:]
            vals_flat = vals.reshape((nt, n_raw, *trailing))
        else:
            raise ValueError(
                f"values shape {vals.shape} incompatible with coordinates shape {coords.shape}"
            )
    elif coords.ndim == 2 and coords.shape[1] == 3:
        n_raw = coords.shape[0]
        coords_flat = coords
        if vals.shape[0] == n_raw:
            has_time = False
            vals_flat = vals
        elif vals.ndim >= 2 and vals.shape[1] == n_raw:
            has_time = True
            vals_flat = vals
        else:
            raise ValueError(
                f"values shape {vals.shape} incompatible with coordinates shape {coords.shape}"
            )
    else:
        raise ValueError(
            f"coordinates_m must have shape (N, 3) or (nspec, ngll, 3), got {coords.shape}"
        )

    scale = max(1.0, float(np.max(np.abs(coords_flat)))) if n_raw > 0 else 1.0
    tol = float(coord_tol) if coord_tol is not None else 1e-5 * scale
    if tol <= 0.0:
        raise ValueError(f"coord_tol must be positive, got {tol}")

    # Quantize coordinates to bin width `tol` for O(N log N) exact hashing,
    # then verify exact pairwise match against bin representatives.
    keys = np.rint(coords_flat / tol).astype(np.int64)
    _, first_idx, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    # Sort representatives by first occurrence for deterministic ordering
    order = np.argsort(first_idx)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    first_idx = first_idx[order]
    inverse = remap[inverse]
    n_unique = len(first_idx)
    unique_coords = coords_flat[first_idx]

    # Compute min/max discrepancy across coincident points
    max_disc = 0.0
    if n_unique < n_raw:
        if has_time:
            ref_vals = vals_flat[:, first_idx]
            diffs = np.abs(vals_flat - ref_vals[:, inverse])
            max_disc = float(np.max(diffs)) if diffs.size else 0.0
            val_scale = float(np.max(np.abs(vals_flat))) if vals_flat.size else 0.0
            unique_vals = ref_vals
        else:
            ref_vals = vals_flat[first_idx]
            diffs = np.abs(vals_flat - ref_vals[inverse])
            max_disc = float(np.max(diffs)) if diffs.size else 0.0
            val_scale = float(np.max(np.abs(vals_flat))) if vals_flat.size else 0.0
            unique_vals = ref_vals

        limit = continuity_atol + continuity_rtol * val_scale
        if verify_continuity == "raise" and max_disc > limit:
            raise ValueError(
                f"Coincident GLL points disagree by {max_disc:.3e} (> tol={limit:.3e}); "
                "check GLL flatten order or set verify_continuity='ignore' for discontinuous material fields."
            )
    else:
        unique_vals = vals_flat

    return DeduplicatedGLLResult(
        coordinates_m=unique_coords,
        values=unique_vals,
        inverse_indices=inverse,
        n_raw_points=n_raw,
        n_unique_points=n_unique,
        max_discrepancy=max_disc,
    )
