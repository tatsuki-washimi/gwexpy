"""Provisional surface-defined physical fields with native geometry and association.

Status: **Provisional / Internal** (v0.3.0 U-5 Decision: Approach D + E2 + S1,
tracking `#902` and `#522`). This module is intentionally kept private and is
not re-exported in ``gwexpy.__init__`` or ``gwexpy.fields.__init__``.

Design Principles
-----------------
- **Thin value layer + optional delegation (Approach D):** GWexpy owns field
  values (`astropy.units.Quantity`), `axis0` (time or frequency coordinates +
  ``#725`` `time_reference`), entity `association` (``vertex``, ``face``,
  ``point``, ``quadrature``), `component_basis` / vector semantics,
  `valid_mask`, and `provenance`.
- **Optional backend delegation:** Mesh I/O and optional geometry backend calls
  delegate lazily to ``meshio`` and ``pyvista`` via
  :func:`gwexpy.interop._optional.require_optional`, while the core container
  and triangle/quad surface geometry run on pure NumPy with zero optional
  runtime dependencies.
- **#904 connection points:** Geometry (:class:`SurfaceGeometry` /
  :class:`MeshGeometry`) is an immutable object separate from the value
  container (:class:`SurfaceField`). Value ``axis0`` is distinct from any
  future time-varying geometry, and normals/area elements are derived dynamically
  from geometry rather than silently cached as static values.
- **Explicit conversion:** Rectilinear conversion distinguishes exact
  one-to-one mapping (:meth:`SurfaceField.to_rectilinear`) from explicit spatial
  sampling/interpolation (:meth:`SurfaceField.sample_to_grid`).

Attribution
-----------
The triangle/polygon orientation diagnostic structures and topological BFS
reorientation logic (:class:`OrientationReport`,
:func:`check_surface_orientation`, :func:`reorient_faces`) are adapted from
``gragra`` (Copyright (c) 2026 Tatsuki Washimi, MIT License).
"""

from __future__ import annotations

import copy
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from astropy import units as u

from ..interop._optional import require_optional
from ..time._reference import (
    TIME_INPUT_UNSET,
    TIME_REFERENCE_ABSOLUTE,
    TIME_REFERENCE_RELATIVE,
    classify_time_input,
    get_time_reference,
    set_time_reference,
)
from ..types.axis import coerce_1d_quantity
from .scalar import ScalarField
from .vector import VectorField

__all__ = [
    "MeshGeometry",
    "OrientationReport",
    "SurfaceField",
    "SurfaceGeometry",
    "SurfaceProvenance",
    "check_surface_orientation",
    "check_triangle_orientation",
    "reorient_faces",
    "reorient_triangles",
]

SurfaceAssociation = Literal["vertex", "face", "point", "quadrature"]
ComponentBasis = Literal["scalar", "cartesian", "tangent", "normal", "custom"]
MappingKind = Literal[
    "native", "exact_one_to_one", "interpolated", "selected", "computed"
]

_SURFACE_ASSOCIATIONS: tuple[str, ...] = ("vertex", "face", "point", "quadrature")
_COMPONENT_BASES: tuple[str, ...] = (
    "scalar",
    "cartesian",
    "tangent",
    "normal",
    "custom",
)
_CELL_VERTEX_COUNTS: dict[str, int] = {"triangle": 3, "quad": 4}
_AXIS_RTOL = 0.0
_AXIS_ATOL = 1e-12
_COLLINEAR_TOL = 1e-12


# ---------------------------------------------------------------------------
# Internal validation & array helpers
# ---------------------------------------------------------------------------


def _check_no_bool(value: Any, name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not be boolean")
    if isinstance(value, np.ndarray) and value.dtype.kind == "b":
        raise TypeError(f"{name} must not have boolean dtype")
    if hasattr(value, "dtype") and getattr(value.dtype, "kind", None) == "b":
        raise TypeError(f"{name} must not have boolean dtype")


def _freeze_array(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


def _as_readonly_vertices(
    vertices: Any,
    *,
    unit: u.UnitBase | str | None = None,
    name: str = "vertices",
) -> tuple[u.Quantity, int]:
    _check_no_bool(vertices, name)
    if isinstance(vertices, u.Quantity):
        arr = np.asarray(vertices.value, dtype=np.float64)
        src_unit = vertices.unit
        if unit is not None:
            target_unit = u.Unit(unit)
            arr = np.asarray(vertices.to_value(target_unit), dtype=np.float64)
            eff_unit = target_unit
        else:
            eff_unit = src_unit
    else:
        try:
            arr = np.asarray(vertices, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be numeric: {exc}") from exc
        eff_unit = u.m if unit is None else u.Unit(unit)

    if arr.ndim != 2 or arr.shape[1] not in (2, 3):
        raise ValueError(f"{name} must have shape (N, 2) or (N, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite coordinates")

    frozen = _freeze_array(arr)
    return u.Quantity(frozen, eff_unit, copy=False), int(arr.shape[1])


def _as_readonly_index_array(
    indices: Any, *, name: str, nv: int, ncols: int
) -> np.ndarray:
    _check_no_bool(indices, name)
    raw = np.asarray(indices)
    if raw.size > 0 and raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must be of integer dtype, got {raw.dtype}")
    try:
        arr = np.asarray(raw, dtype=np.intp)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be integer convertible: {exc}") from exc

    if arr.ndim != 2 or arr.shape[1] != ncols:
        raise ValueError(f"{name} must have shape (N, {ncols}), got {arr.shape}")
    if arr.size > 0:
        if np.any(arr < 0) or np.any(arr >= nv):
            raise ValueError(f"{name} indices must be in [0, {nv})")
        # Reject repeated vertices within the same face.
        for c1 in range(ncols):
            for c2 in range(c1 + 1, ncols):
                if np.any(arr[:, c1] == arr[:, c2]):
                    raise ValueError(
                        f"{name} must not contain repeated vertex indices within a face"
                    )
    return _freeze_array(arr)


def _as_json_safe(value: Any, *, path: str = "details") -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, u.UnitBase):
        return str(value)
    if isinstance(value, u.Quantity):
        if value.isscalar:
            val = float(value.value)
            if not math.isfinite(val):
                raise ValueError(f"{path} must not contain non-finite Quantity")
            return {"value": val, "unit": str(value.unit)}
        return {
            "value": [_as_json_safe(v, path=f"{path}[]") for v in value.value.tolist()],
            "unit": str(value.unit),
        }
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(f"{path} mapping keys must be strings")
            out[k] = _as_json_safe(v, path=f"{path}.{k}")
        return out
    if isinstance(value, (list, tuple)):
        return [_as_json_safe(v, path=f"{path}[]") for v in value]
    raise TypeError(
        f"{path} contains unsupported provenance value type {type(value).__name__}"
    )


def _canonical_unit(unit: u.UnitBase) -> u.UnitBase:
    decomposed = unit.decompose()
    canonical = u.dimensionless_unscaled
    for base, power in zip(decomposed.bases, decomposed.powers, strict=True):
        canonical *= base**power
    return canonical


def _quantities_close(
    left: u.Quantity,
    right: u.Quantity,
    *,
    rtol: float = _AXIS_RTOL,
    atol: float = _AXIS_ATOL,
) -> bool:
    if left.shape != right.shape:
        return False
    if not left.unit.is_equivalent(right.unit):
        return False
    cmp_unit = _canonical_unit(left.unit)
    lv = np.asarray(left.to_value(cmp_unit), dtype=np.float64)
    rv = np.asarray(right.to_value(cmp_unit), dtype=np.float64)
    return bool(np.allclose(lv, rv, rtol=rtol, atol=atol, equal_nan=False))


def _resolve_epoch_state(epoch: Any) -> tuple[Any, str]:
    kind = classify_time_input(epoch)
    if kind == "inf":
        raise ValueError(
            "epoch must not be infinite; pass a finite GPS time or omit it "
            "when the absolute time is unknown"
        )
    if epoch is TIME_INPUT_UNSET or kind in ("missing", "nan"):
        return None, TIME_REFERENCE_RELATIVE
    return epoch, TIME_REFERENCE_ABSOLUTE


def _format_epoch_value(epoch: Any) -> float | None:
    if epoch is None:
        return None
    if isinstance(epoch, u.Quantity):
        return float(epoch.to_value(u.s))
    if hasattr(epoch, "gps"):
        return float(epoch.gps)
    return float(epoch)


# ---------------------------------------------------------------------------
# Provenance dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceProvenance:
    """Frozen provenance record for surface fields and conversions.

    Attributes
    ----------
    source : str
        Origin of the field (e.g. ``"native"``, ``"meshio"``, ``"pyvista"``,
        ``"structured_grid"``).
    operation : str
        Most recent operation applied to create this object.
    mapping_kind : {'native', 'exact_one_to_one', 'interpolated', 'selected', 'computed'}
        Whether values are native to the geometry, mapped 1-to-1 without
        interpolation, explicitly interpolated, selected, or computed.
    interpolation_method : str or None
        Interpolation method when ``mapping_kind == 'interpolated'``.
    backend : str or None
        Optional backend used (e.g. ``'numpy'``, ``'meshio'``, ``'pyvista'``).
    history : tuple of str
        Ordered sequence of operations leading to this field.
    details : Mapping[str, Any]
        Read-only mapping of extra JSON-safe metadata.

    """

    source: str = "native"
    operation: str = "created"
    mapping_kind: MappingKind = "native"
    interpolation_method: str | None = None
    backend: str | None = None
    history: tuple[str, ...] = ()
    _details_items: tuple[tuple[str, Any], ...] = field(
        default=(), repr=False, compare=True
    )

    def __init__(
        self,
        source: str = "native",
        operation: str = "created",
        mapping_kind: MappingKind = "native",
        interpolation_method: str | None = None,
        backend: str | None = None,
        history: Sequence[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if mapping_kind not in (
            "native",
            "exact_one_to_one",
            "interpolated",
            "selected",
            "computed",
        ):
            raise ValueError(f"Unsupported mapping_kind: {mapping_kind!r}")
        hist_tuple = tuple(str(step) for step in history)
        if not hist_tuple:
            hist_tuple = (str(operation),)
        safe_details = _as_json_safe(dict(details)) if details else {}
        # Store immutable key-value pairs so the dataclass stays picklable.
        items = tuple(sorted((str(k), v) for k, v in safe_details.items()))
        object.__setattr__(self, "source", str(source))
        object.__setattr__(self, "operation", str(operation))
        object.__setattr__(self, "mapping_kind", mapping_kind)
        object.__setattr__(self, "interpolation_method", interpolation_method)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "history", hist_tuple)
        object.__setattr__(self, "_details_items", items)

    @property
    def details(self) -> Mapping[str, Any]:
        """Read-only mapping of extra JSON-safe provenance metadata."""
        return MappingProxyType(dict(self._details_items))

    @classmethod
    def coerce(
        cls, value: SurfaceProvenance | Mapping[str, Any] | None
    ) -> SurfaceProvenance:
        """Coerce a mapping or ``None`` into a :class:`SurfaceProvenance`."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(
                source=str(value.get("source", "native")),
                operation=str(value.get("operation", "created")),
                mapping_kind=value.get("mapping_kind", "native"),
                interpolation_method=value.get("interpolation_method"),
                backend=value.get("backend"),
                history=tuple(value.get("history", ())),
                details=value.get("details"),
            )
        raise TypeError(
            f"provenance must be SurfaceProvenance, mapping, or None, got {type(value).__name__}"
        )

    def with_step(
        self,
        operation: str,
        *,
        mapping_kind: MappingKind | None = None,
        interpolation_method: str | None = None,
        backend: str | None = None,
        **extra_details: Any,
    ) -> SurfaceProvenance:
        """Return a new frozen provenance record appending ``operation``."""
        merged_details = dict(self._details_items)
        if extra_details:
            merged_details.update(_as_json_safe(extra_details))
        return SurfaceProvenance(
            source=self.source,
            operation=operation,
            mapping_kind=self.mapping_kind if mapping_kind is None else mapping_kind,
            interpolation_method=(
                self.interpolation_method
                if interpolation_method is None
                else interpolation_method
            ),
            backend=self.backend if backend is None else backend,
            history=self.history + (operation,),
            details=merged_details,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-safe dictionary representation."""
        payload = {
            "schema": "gwexpy.fields.surface.provenance",
            "schema_version": 1,
            "source": self.source,
            "operation": self.operation,
            "mapping_kind": self.mapping_kind,
            "interpolation_method": self.interpolation_method,
            "backend": self.backend,
            "history": list(self.history),
            "details": dict(self._details_items),
        }
        json.dumps(payload, allow_nan=False, sort_keys=True)
        return copy.deepcopy(payload)


# ---------------------------------------------------------------------------
# Surface orientation diagnostics and reorientation (MIT adapted from gragra)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OrientationReport:
    """Diagnostic result for triangle/quad surface mesh orientation.

    Adapted from ``gragra.fields.surface_orientation.OrientationReport``
    (Copyright (c) 2026 Tatsuki Washimi, MIT License).

    Attributes
    ----------
    is_consistent : bool
        True when every edge shared by exactly two faces is traversed in
        opposite directions.
    is_closed : bool
        True when the mesh has faces and zero boundary or non-manifold edges.
    is_outward : bool or None
        Defined only when ``is_closed and is_consistent``; True when the signed
        enclosed volume is strictly positive.
    signed_volume : float
        Signed volume in ``length_unit**3`` over faces in connectivity order.
    inconsistent_edge_count : int
        Number of edges shared by two co-directed faces.
    boundary_edge_count : int
        Number of edges belonging to exactly one face.
    nonmanifold_edge_count : int
        Number of edges shared by three or more faces.
    inconsistent_face_indices : np.ndarray
        Sorted unique indices of faces participating in inconsistent edges.
    unit : astropy.units.UnitBase
        Length unit of the underlying vertex coordinates (volume unit is ``unit**3``).

    """

    is_consistent: bool
    is_closed: bool
    is_outward: bool | None
    signed_volume: float
    inconsistent_edge_count: int
    boundary_edge_count: int
    nonmanifold_edge_count: int
    inconsistent_face_indices: np.ndarray
    unit: u.UnitBase = u.m

    def __post_init__(self) -> None:
        arr = np.asarray(self.inconsistent_face_indices, dtype=np.intp).copy()
        arr.setflags(write=False)
        object.__setattr__(self, "inconsistent_face_indices", arr)

    @property
    def signed_volume_quantity(self) -> u.Quantity:
        """Signed volume as an :class:`~astropy.units.Quantity`."""
        return self.signed_volume * (self.unit**3)


def _triangulate_blocks_indices(
    cell_blocks: Sequence[tuple[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(triangles, parent_face_indices)`` across triangle/quad blocks."""
    tri_parts: list[np.ndarray] = []
    parent_parts: list[np.ndarray] = []
    face_offset = 0
    for cell_type, conn in cell_blocks:
        n_cells = conn.shape[0]
        if n_cells == 0:
            continue
        if cell_type == "triangle":
            tri_parts.append(conn)
            parent_parts.append(
                np.arange(face_offset, face_offset + n_cells, dtype=np.intp)
            )
        elif cell_type == "quad":
            t1 = conn[:, [0, 1, 2]]
            t2 = conn[:, [0, 2, 3]]
            tri_parts.append(t1)
            tri_parts.append(t2)
            p = np.arange(face_offset, face_offset + n_cells, dtype=np.intp)
            parent_parts.append(p)
            parent_parts.append(p)
        else:
            raise ValueError(f"Unsupported surface cell type: {cell_type!r}")
        face_offset += n_cells

    if not tri_parts:
        return np.empty((0, 3), dtype=np.intp), np.empty(0, dtype=np.intp)
    return np.concatenate(tri_parts, axis=0), np.concatenate(parent_parts, axis=0)


def check_surface_orientation(
    vertices: Any,
    cell_blocks: Sequence[tuple[str, Any]],
) -> OrientationReport:
    """Diagnose winding consistency and closedness of a triangle/quad surface.

    Adapted from ``gragra.fields.surface_orientation.check_triangle_orientation``
    (Copyright (c) 2026 Tatsuki Washimi, MIT License) and generalized to
    surfaces containing triangles and quads.
    """
    verts_q, emb_dim = _as_readonly_vertices(vertices)
    verts = np.asarray(verts_q.value, dtype=np.float64)
    nv = len(verts)

    normalized_blocks: list[tuple[str, np.ndarray]] = []
    for cell_type, conn in cell_blocks:
        if cell_type not in _CELL_VERTEX_COUNTS:
            raise ValueError(f"Unsupported surface cell type: {cell_type!r}")
        arr = _as_readonly_index_array(
            conn, name=f"{cell_type}_cells", nv=nv, ncols=_CELL_VERTEX_COUNTS[cell_type]
        )
        normalized_blocks.append((cell_type, arr))

    starts_list: list[np.ndarray] = []
    ends_list: list[np.ndarray] = []
    owners_list: list[np.ndarray] = []
    face_offset = 0
    for _, conn in normalized_blocks:
        nf = conn.shape[0]
        if nf == 0:
            continue
        k = conn.shape[1]
        face_ids = np.arange(face_offset, face_offset + nf, dtype=np.intp)
        for edge_idx in range(k):
            starts_list.append(conn[:, edge_idx])
            ends_list.append(conn[:, (edge_idx + 1) % k])
            owners_list.append(face_ids)
        face_offset += nf

    if face_offset == 0:
        return OrientationReport(
            is_consistent=True,
            is_closed=False,
            is_outward=None,
            signed_volume=0.0,
            inconsistent_edge_count=0,
            boundary_edge_count=0,
            nonmanifold_edge_count=0,
            inconsistent_face_indices=np.empty(0, dtype=np.intp),
            unit=verts_q.unit,
        )

    starts = np.concatenate(starts_list)
    ends = np.concatenate(ends_list)
    owners = np.concatenate(owners_list)

    key_lo = np.minimum(starts, ends)
    key_hi = np.maximum(starts, ends)
    signs = np.where(starts < ends, 1, -1)
    keys = np.stack([key_lo, key_hi], axis=1)

    uniq, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    inverse = np.asarray(inverse).reshape(-1)

    boundary_edge_count = int(np.count_nonzero(counts == 1))
    nonmanifold_edge_count = int(np.count_nonzero(counts >= 3))

    sign_sum = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(sign_sum, inverse, signs)
    inconsistent_edge_mask = (counts == 2) & (sign_sum != 0)
    inconsistent_edge_count = int(np.count_nonzero(inconsistent_edge_mask))

    entry_is_inconsistent = inconsistent_edge_mask[inverse]
    inconsistent_face_indices = np.unique(owners[entry_is_inconsistent]).astype(np.intp)

    is_consistent = inconsistent_edge_count == 0
    is_closed = boundary_edge_count == 0 and nonmanifold_edge_count == 0

    if emb_dim == 3:
        tris, _ = _triangulate_blocks_indices(normalized_blocks)
        a = verts[tris[:, 0]]
        b = verts[tris[:, 1]]
        c = verts[tris[:, 2]]
        signed_volume = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
        is_outward: bool | None = (
            bool(signed_volume > 0.0) if (is_closed and is_consistent) else None
        )
    else:
        signed_volume = 0.0
        is_outward = None

    return OrientationReport(
        is_consistent=is_consistent,
        is_closed=is_closed,
        is_outward=is_outward,
        signed_volume=signed_volume,
        inconsistent_edge_count=inconsistent_edge_count,
        boundary_edge_count=boundary_edge_count,
        nonmanifold_edge_count=nonmanifold_edge_count,
        inconsistent_face_indices=inconsistent_face_indices,
        unit=verts_q.unit,
    )


def check_triangle_orientation(
    vertices: Any,
    triangles: Any,
) -> OrientationReport:
    """Diagnose winding consistency and closedness of a triangle surface."""
    return check_surface_orientation(vertices, [("triangle", triangles)])


def reorient_faces(
    cell_blocks: Sequence[tuple[str, Any]],
) -> tuple[tuple[str, np.ndarray], ...]:
    """Consistently reorient triangle/quad surface face windings topologically.

    Adapted from ``gragra.fields.surface_orientation.reorient_triangles``
    (Copyright (c) 2026 Tatsuki Washimi, MIT License). Preserves the seed face
    winding per connected component and flips neighboring faces as needed. Raises
    ``ValueError`` if the surface is non-manifold or non-orientable (e.g. a
    Mobius strip).
    """
    validated: list[tuple[str, np.ndarray]] = []
    flat_faces: list[np.ndarray] = []
    block_slices: list[tuple[str, int, int]] = []
    offset = 0

    for cell_type, conn in cell_blocks:
        if cell_type not in _CELL_VERTEX_COUNTS:
            raise ValueError(f"Unsupported surface cell type: {cell_type!r}")
        _check_no_bool(conn, f"{cell_type}_cells")
        raw = np.asarray(conn)
        if raw.size > 0 and raw.dtype.kind not in "iu":
            raise TypeError(f"{cell_type}_cells must be of integer dtype")
        arr = np.array(raw, dtype=np.intp, copy=True)
        ncols = _CELL_VERTEX_COUNTS[cell_type]
        if arr.ndim != 2 or arr.shape[1] != ncols:
            raise ValueError(
                f"{cell_type}_cells must have shape (N, {ncols}), got {arr.shape}"
            )
        if arr.size > 0:
            if np.any(arr < 0):
                raise ValueError(f"{cell_type}_cells must not contain negative indices")
            for c1 in range(ncols):
                for c2 in range(c1 + 1, ncols):
                    if np.any(arr[:, c1] == arr[:, c2]):
                        raise ValueError(
                            "Surface faces must not contain repeated vertex indices"
                        )
        validated.append((cell_type, arr))
        for row in arr:
            flat_faces.append(row)
        block_slices.append((cell_type, offset, offset + arr.shape[0]))
        offset += arr.shape[0]

    nf = len(flat_faces)
    if nf == 0:
        return tuple((ct, _freeze_array(arr)) for ct, arr in validated)

    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for f_idx, face in enumerate(flat_faces):
        k = len(face)
        for local_idx in range(k):
            u_v = int(face[local_idx])
            v_v = int(face[(local_idx + 1) % k])
            edge_key = (u_v, v_v) if u_v < v_v else (v_v, u_v)
            edges.setdefault(edge_key, []).append((f_idx, local_idx))

    for edge_key, owners in edges.items():
        if len(owners) >= 3:
            raise ValueError(
                f"Surface mesh is non-manifold: edge {edge_key} shared by {len(owners)} faces"
            )

    visited = np.zeros(nf, dtype=bool)
    flip = np.zeros(nf, dtype=bool)

    for start_f in range(nf):
        if visited[start_f]:
            continue
        queue = [start_f]
        visited[start_f] = True
        flip[start_f] = False
        head = 0
        while head < len(queue):
            curr_f = queue[head]
            head += 1
            curr_verts = flat_faces[curr_f]
            curr_k = len(curr_verts)
            curr_flip = bool(flip[curr_f])

            for local_idx in range(curr_k):
                u_v = int(curr_verts[local_idx])
                v_v = int(curr_verts[(local_idx + 1) % curr_k])
                curr_dir = (v_v, u_v) if curr_flip else (u_v, v_v)
                edge_key = (u_v, v_v) if u_v < v_v else (v_v, u_v)
                for neigh_f, neigh_local_idx in edges[edge_key]:
                    if neigh_f == curr_f:
                        continue
                    n_verts = flat_faces[neigh_f]
                    n_k = len(n_verts)
                    nu = int(n_verts[neigh_local_idx])
                    nv = int(n_verts[(neigh_local_idx + 1) % n_k])
                    if visited[neigh_f]:
                        neigh_dir = (nv, nu) if flip[neigh_f] else (nu, nv)
                        if curr_dir == neigh_dir:
                            raise ValueError(
                                "Surface mesh is non-orientable "
                                "(winding inconsistency cannot be resolved)"
                            )
                    else:
                        u_dir, v_dir = curr_dir
                        if nu == u_dir and nv == v_dir:
                            flip[neigh_f] = True
                        elif nu == v_dir and nv == u_dir:
                            flip[neigh_f] = False
                        else:
                            raise ValueError("Topological error in edge sharing")
                        visited[neigh_f] = True
                        queue.append(neigh_f)

    out_blocks: list[tuple[str, np.ndarray]] = []
    for (cell_type, arr), (_, start, stop) in zip(validated, block_slices, strict=True):
        new_arr = arr.copy()
        local_flip = flip[start:stop]
        if np.any(local_flip):
            k = new_arr.shape[1]
            rev_order = [0] + list(range(k - 1, 0, -1))
            new_arr[local_flip] = new_arr[local_flip][:, rev_order]
        out_blocks.append((cell_type, _freeze_array(new_arr)))
    return tuple(out_blocks)


def reorient_triangles(triangles: Any) -> np.ndarray:
    """Consistently reorient a triangle connectivity table."""
    ((_, out),) = reorient_faces([("triangle", triangles)])
    return out


# ---------------------------------------------------------------------------
# Geometry Layer: MeshGeometry & SurfaceGeometry (#904 separate immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, init=False)
class MeshGeometry:
    """Immutable mesh geometry container separate from physical field values.

    This base geometry abstraction serves as a shared ``#522`` / ``#904``
    connection point: coordinates, topology, orientation convention, and
    optional quadrature locations live independently of any time/frequency axis
    or field quantity array.
    """

    vertices: u.Quantity
    cell_blocks: tuple[tuple[str, np.ndarray], ...]
    topological_dimension: int
    embedding_dimension: int
    coordinate_system: str
    orientation_convention: str
    global_vertex_ids: np.ndarray | None
    global_cell_ids: np.ndarray | None
    quadrature_points: u.Quantity | None
    quadrature_weights: u.Quantity | None
    quadrature_cell_ids: np.ndarray | None

    def __init__(
        self,
        vertices: Any,
        cell_blocks: Sequence[tuple[str, Any]] = (),
        *,
        topological_dimension: int,
        unit: u.UnitBase | str | None = None,
        coordinate_system: str = "cartesian",
        orientation_convention: str = "right_hand_rule",
        global_vertex_ids: Any = None,
        global_cell_ids: Any = None,
        quadrature_points: Any = None,
        quadrature_weights: Any = None,
        quadrature_cell_ids: Any = None,
    ) -> None:
        verts_q, emb_dim = _as_readonly_vertices(vertices, unit=unit, name="vertices")
        nv = len(verts_q)

        if isinstance(topological_dimension, bool) or not isinstance(
            topological_dimension, (int, np.integer)
        ):
            raise TypeError("topological_dimension must be an integer")
        topo_dim = int(topological_dimension)
        if topo_dim < 0 or topo_dim > emb_dim:
            raise ValueError(
                f"topological_dimension ({topo_dim}) must satisfy "
                f"0 <= topological_dimension <= embedding_dimension ({emb_dim})"
            )

        validated_blocks: list[tuple[str, np.ndarray]] = []
        n_cells = 0
        for block in cell_blocks:
            if not isinstance(block, (tuple, list)) or len(block) != 2:
                raise TypeError(
                    "Each cell_blocks entry must be a (cell_type, connectivity) pair"
                )
            cell_type, conn = block
            if cell_type not in _CELL_VERTEX_COUNTS:
                raise ValueError(f"Unsupported cell_type {cell_type!r}")
            ncols = _CELL_VERTEX_COUNTS[cell_type]
            arr = _as_readonly_index_array(
                conn, name=f"{cell_type}_cells", nv=nv, ncols=ncols
            )
            if arr.shape[0] > 0:
                validated_blocks.append((str(cell_type), arr))
                n_cells += int(arr.shape[0])

        gv_ids = _validate_optional_ids(global_vertex_ids, nv, "global_vertex_ids")
        gc_ids = _validate_optional_ids(global_cell_ids, n_cells, "global_cell_ids")

        q_pts: u.Quantity | None = None
        if quadrature_points is not None:
            q_pts, q_emb = _as_readonly_vertices(
                quadrature_points,
                unit=verts_q.unit,
                name="quadrature_points",
            )
            if q_emb != emb_dim:
                raise ValueError(
                    f"quadrature_points embedding dimension ({q_emb}) must match "
                    f"vertices ({emb_dim})"
                )

        q_weights: u.Quantity | None = None
        if quadrature_weights is not None:
            _check_no_bool(quadrature_weights, "quadrature_weights")
            expected_q = len(q_pts) if q_pts is not None else nv
            if isinstance(quadrature_weights, u.Quantity):
                w_arr = np.asarray(quadrature_weights.value, dtype=np.float64)
                w_unit = quadrature_weights.unit
            else:
                w_arr = np.asarray(quadrature_weights, dtype=np.float64)
                w_unit = verts_q.unit**topo_dim
            if w_arr.ndim != 1 or w_arr.shape[0] != expected_q:
                raise ValueError(
                    f"quadrature_weights must have shape ({expected_q},), got {w_arr.shape}"
                )
            if not np.all(np.isfinite(w_arr)):
                raise ValueError("quadrature_weights must be finite")
            q_weights = u.Quantity(_freeze_array(w_arr), w_unit, copy=False)

        q_cids: np.ndarray | None = None
        if quadrature_cell_ids is not None:
            expected_q = len(q_pts) if q_pts is not None else nv
            q_cids = _validate_optional_ids(
                quadrature_cell_ids, expected_q, "quadrature_cell_ids", max_val=n_cells
            )

        object.__setattr__(self, "vertices", verts_q)
        object.__setattr__(self, "cell_blocks", tuple(validated_blocks))
        object.__setattr__(self, "topological_dimension", topo_dim)
        object.__setattr__(self, "embedding_dimension", emb_dim)
        object.__setattr__(self, "coordinate_system", str(coordinate_system))
        object.__setattr__(self, "orientation_convention", str(orientation_convention))
        object.__setattr__(self, "global_vertex_ids", gv_ids)
        object.__setattr__(self, "global_cell_ids", gc_ids)
        object.__setattr__(self, "quadrature_points", q_pts)
        object.__setattr__(self, "quadrature_weights", q_weights)
        object.__setattr__(self, "quadrature_cell_ids", q_cids)

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the geometry."""
        return int(len(self.vertices))

    @property
    def n_cells(self) -> int:
        """Number of cells across all cell blocks."""
        return sum(int(conn.shape[0]) for _, conn in self.cell_blocks)

    @property
    def n_quadrature(self) -> int:
        """Number of quadrature locations."""
        if self.quadrature_points is not None:
            return int(len(self.quadrature_points))
        return self.n_vertices

    @property
    def length_unit(self) -> u.UnitBase:
        """Physical length unit of vertex coordinates."""
        return self.vertices.unit


def _validate_optional_ids(
    ids: Any,
    expected_len: int,
    name: str,
    *,
    max_val: int | None = None,
) -> np.ndarray | None:
    if ids is None:
        return None
    _check_no_bool(ids, name)
    raw = np.asarray(ids)
    if raw.size > 0 and raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must be of integer dtype, got {raw.dtype}")
    arr = np.asarray(raw, dtype=np.intp)
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        raise ValueError(f"{name} must have shape ({expected_len},), got {arr.shape}")
    if max_val is not None and arr.size > 0:
        if np.any(arr < 0) or np.any(arr >= max_val):
            raise ValueError(f"{name} entries must be in [0, {max_val})")
    return _freeze_array(arr)


@dataclass(frozen=True, init=False)
class SurfaceGeometry(MeshGeometry):
    """Immutable 2D surface geometry embedded in 2D or 3D space.

    Supports triangle meshes, quad meshes, mixed triangle/quad surfaces,
    structured surfaces, and point/quadrature surfaces. Geometric quantities
    (face normals, face areas, vertex areas, centroids) are computed on demand
    from coordinates and connectivity rather than silently cached.
    """

    structured_shape: tuple[int, int] | None
    structured_axes: tuple[u.Quantity, u.Quantity] | None
    structured_plane: tuple[int, int, int, u.Quantity] | None

    def __init__(
        self,
        vertices: Any,
        faces: Any = None,
        *,
        cell_blocks: Sequence[tuple[str, Any]] | None = None,
        unit: u.UnitBase | str | None = None,
        coordinate_system: str = "cartesian",
        orientation_convention: str = "right_hand_rule",
        global_vertex_ids: Any = None,
        global_cell_ids: Any = None,
        quadrature_points: Any = None,
        quadrature_weights: Any = None,
        quadrature_cell_ids: Any = None,
        structured_shape: tuple[int, int] | None = None,
        structured_axes: tuple[Any, Any] | None = None,
        structured_plane: tuple[int, int, int, Any] | None = None,
    ) -> None:
        if faces is not None and cell_blocks is not None:
            raise ValueError("Specify either 'faces' or 'cell_blocks', not both")

        blocks: Sequence[tuple[str, Any]]
        if cell_blocks is not None:
            blocks = cell_blocks
        elif faces is not None:
            _check_no_bool(faces, "faces")
            face_arr = np.asarray(faces)
            if face_arr.ndim != 2 or face_arr.shape[1] not in (3, 4):
                raise ValueError(
                    f"faces must have shape (N, 3) or (N, 4), got {face_arr.shape}"
                )
            ctype = "triangle" if face_arr.shape[1] == 3 else "quad"
            blocks = ((ctype, face_arr),)
        else:
            blocks = ()

        super().__init__(
            vertices=vertices,
            cell_blocks=blocks,
            topological_dimension=2,
            unit=unit,
            coordinate_system=coordinate_system,
            orientation_convention=orientation_convention,
            global_vertex_ids=global_vertex_ids,
            global_cell_ids=global_cell_ids,
            quadrature_points=quadrature_points,
            quadrature_weights=quadrature_weights,
            quadrature_cell_ids=quadrature_cell_ids,
        )

        s_shape: tuple[int, int] | None = None
        if structured_shape is not None:
            if len(structured_shape) != 2:
                raise ValueError("structured_shape must be a 2-tuple (nu, nv)")
            nu, nv = int(structured_shape[0]), int(structured_shape[1])
            if nu * nv != self.n_vertices:
                raise ValueError(
                    f"structured_shape {structured_shape} does not match "
                    f"n_vertices={self.n_vertices}"
                )
            s_shape = (nu, nv)

        s_axes: tuple[u.Quantity, u.Quantity] | None = None
        if structured_axes is not None:
            if s_shape is None:
                raise ValueError("structured_axes requires structured_shape to be set")
            if len(structured_axes) != 2:
                raise ValueError("structured_axes must have length 2")
            ax_u = coerce_1d_quantity(structured_axes[0], unit=self.length_unit)
            ax_v = coerce_1d_quantity(structured_axes[1], unit=self.length_unit)
            if len(ax_u) != s_shape[0] or len(ax_v) != s_shape[1]:
                raise ValueError(
                    f"structured_axes lengths ({len(ax_u)}, {len(ax_v)}) "
                    f"must match structured_shape {s_shape}"
                )
            ax_u = u.Quantity(_freeze_array(ax_u.value), ax_u.unit, copy=False)
            ax_v = u.Quantity(_freeze_array(ax_v.value), ax_v.unit, copy=False)
            s_axes = (ax_u, ax_v)

        s_plane: tuple[int, int, int, u.Quantity] | None = None
        if structured_plane is not None:
            u_idx, v_idx, fixed_idx, fixed_coord = structured_plane
            if sorted((int(u_idx), int(v_idx), int(fixed_idx))) != [0, 1, 2]:
                raise ValueError("structured_plane indices must permute (0, 1, 2)")
            fc_q = (
                fixed_coord.to(self.length_unit)
                if isinstance(fixed_coord, u.Quantity)
                else u.Quantity(float(fixed_coord), self.length_unit)
            )
            s_plane = (int(u_idx), int(v_idx), int(fixed_idx), fc_q)

        object.__setattr__(self, "structured_shape", s_shape)
        object.__setattr__(self, "structured_axes", s_axes)
        object.__setattr__(self, "structured_plane", s_plane)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_triangles(
        cls,
        vertices: Any,
        triangles: Any,
        **kwargs: Any,
    ) -> SurfaceGeometry:
        """Create a :class:`SurfaceGeometry` from vertices and triangle indices."""
        return cls(vertices=vertices, cell_blocks=[("triangle", triangles)], **kwargs)

    @classmethod
    def from_quads(
        cls,
        vertices: Any,
        quads: Any,
        **kwargs: Any,
    ) -> SurfaceGeometry:
        """Create a :class:`SurfaceGeometry` from vertices and quad indices."""
        return cls(vertices=vertices, cell_blocks=[("quad", quads)], **kwargs)

    @classmethod
    def from_points(
        cls,
        points: Any,
        *,
        weights: Any = None,
        unit: u.UnitBase | str | None = None,
        as_quadrature: bool = False,
        **kwargs: Any,
    ) -> SurfaceGeometry:
        """Create a face-less point or quadrature surface geometry."""
        if as_quadrature:
            return cls(
                vertices=points,
                unit=unit,
                quadrature_points=points,
                quadrature_weights=weights,
                **kwargs,
            )
        return cls(
            vertices=points,
            unit=unit,
            quadrature_weights=weights,
            **kwargs,
        )

    @classmethod
    def from_structured_grid(
        cls,
        x: Any,
        y: Any,
        z: Any = 0.0,
        *,
        unit: u.UnitBase | str | None = None,
        coordinate_system: str = "cartesian",
        orientation_convention: str = "right_hand_rule",
    ) -> SurfaceGeometry:
        """Create a structured quad surface from 1D rectilinear or 2D surface grids.

        When ``x`` and ``y`` are 1D coordinate arrays and ``z`` is constant, the
        returned geometry records exact planar-rectilinear metadata enabling
        lossless one-to-one conversion via :meth:`SurfaceField.to_rectilinear`.
        When ``z`` varies as a height field ``z = h(x, y)`` or when curvilinear
        2D arrays ``(X, Y, Z)`` are supplied, the surface remains genuinely
        curved and requires explicit :meth:`SurfaceField.sample_to_grid`.
        """
        eff_unit = u.m if unit is None else u.Unit(unit)
        if unit is None:
            for candidate in (x, y, z):
                if isinstance(candidate, u.Quantity):
                    eff_unit = candidate.unit
                    break

        def _to_val_arr(val: Any, name: str) -> np.ndarray:
            _check_no_bool(val, name)
            if isinstance(val, u.Quantity):
                arr = np.asarray(val.to_value(eff_unit), dtype=np.float64)
            else:
                arr = np.asarray(val, dtype=np.float64)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must contain only finite coordinates")
            return arr

        x_arr = _to_val_arr(x, "x")
        y_arr = _to_val_arr(y, "y")
        z_arr = _to_val_arr(z, "z")

        structured_axes: tuple[u.Quantity, u.Quantity] | None = None
        if x_arr.ndim == 1 and y_arr.ndim == 1:
            if len(x_arr) < 2 or len(y_arr) < 2:
                raise ValueError("1D structured axes x and y must have length >= 2")
            nx, ny = len(x_arr), len(y_arr)
            X, Y = np.meshgrid(x_arr, y_arr, indexing="ij")
            if z_arr.ndim == 0:
                Z = np.full((nx, ny), float(z_arr), dtype=np.float64)
            elif z_arr.shape == (nx, ny):
                Z = z_arr
            else:
                raise ValueError(
                    f"z must be scalar or have shape ({nx}, {ny}), got {z_arr.shape}"
                )
            if _is_strictly_monotonic(x_arr) and _is_strictly_monotonic(y_arr):
                structured_axes = (
                    u.Quantity(x_arr, eff_unit),
                    u.Quantity(y_arr, eff_unit),
                )
        elif x_arr.ndim == 2 and y_arr.ndim == 2:
            if x_arr.shape != y_arr.shape:
                raise ValueError(
                    f"2D x and y grids must have identical shapes, got {x_arr.shape} vs {y_arr.shape}"
                )
            nx, ny = x_arr.shape
            if nx < 2 or ny < 2:
                raise ValueError("2D structured grids must have shape >= (2, 2)")
            X, Y = x_arr, y_arr
            if z_arr.ndim == 0:
                Z = np.full((nx, ny), float(z_arr), dtype=np.float64)
            elif z_arr.shape == (nx, ny):
                Z = z_arr
            else:
                raise ValueError(
                    f"z must be scalar or have shape ({nx}, {ny}), got {z_arr.shape}"
                )
            x_1d = X[:, 0]
            y_1d = Y[0, :]
            if (
                np.allclose(X, x_1d[:, None], rtol=0.0, atol=_AXIS_ATOL)
                and np.allclose(Y, y_1d[None, :], rtol=0.0, atol=_AXIS_ATOL)
                and _is_strictly_monotonic(x_1d)
                and _is_strictly_monotonic(y_1d)
            ):
                structured_axes = (
                    u.Quantity(x_1d, eff_unit),
                    u.Quantity(y_1d, eff_unit),
                )
        else:
            raise ValueError(
                "x and y must both be 1D coordinate vectors or 2D grid arrays"
            )

        vertices = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        ii, jj = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
        v0 = ii * ny + jj
        v1 = (ii + 1) * ny + jj
        v2 = (ii + 1) * ny + (jj + 1)
        v3 = ii * ny + (jj + 1)
        quads = np.stack([v0, v1, v2, v3], axis=-1).reshape(-1, 4)

        structured_plane: tuple[int, int, int, u.Quantity] | None = None
        if structured_axes is not None and np.allclose(
            Z, Z.flat[0], rtol=0.0, atol=_AXIS_ATOL
        ):
            structured_plane = (0, 1, 2, u.Quantity(float(Z.flat[0]), eff_unit))

        return cls(
            vertices=u.Quantity(vertices, eff_unit),
            cell_blocks=(("quad", quads),),
            coordinate_system=coordinate_system,
            orientation_convention=orientation_convention,
            structured_shape=(nx, ny),
            structured_axes=structured_axes,
            structured_plane=structured_plane,
        )

    # ------------------------------------------------------------------
    # Properties & entity helpers
    # ------------------------------------------------------------------

    @property
    def n_faces(self) -> int:
        """Total number of surface faces across all blocks."""
        return self.n_cells

    @property
    def is_planar_rectilinear(self) -> bool:
        """Whether this surface is an exact planar axis-aligned rectilinear grid."""
        return (
            self.structured_shape is not None
            and self.structured_axes is not None
            and self.structured_plane is not None
        )

    @property
    def triangles(self) -> np.ndarray:
        """Triangle connectivity array if homogeneous triangles; else raise."""
        if not self.cell_blocks:
            return np.empty((0, 3), dtype=np.intp)
        if all(ct == "triangle" for ct, _ in self.cell_blocks):
            if len(self.cell_blocks) == 1:
                return self.cell_blocks[0][1]
            return _freeze_array(
                np.concatenate([c for _, c in self.cell_blocks], axis=0)
            )
        raise ValueError("SurfaceGeometry contains non-triangle cells")

    @property
    def quads(self) -> np.ndarray:
        """Quad connectivity array if homogeneous quads; else raise."""
        if not self.cell_blocks:
            return np.empty((0, 4), dtype=np.intp)
        if all(ct == "quad" for ct, _ in self.cell_blocks):
            if len(self.cell_blocks) == 1:
                return self.cell_blocks[0][1]
            return _freeze_array(
                np.concatenate([c for _, c in self.cell_blocks], axis=0)
            )
        raise ValueError("SurfaceGeometry contains non-quad cells")

    def entity_count(self, association: str) -> int:
        """Return number of entities for the given association."""
        if association not in _SURFACE_ASSOCIATIONS:
            raise ValueError(
                f"association must be one of {_SURFACE_ASSOCIATIONS}, got {association!r}"
            )
        if association in ("vertex", "point"):
            return self.n_vertices
        if association == "face":
            return self.n_faces
        return self.n_quadrature

    def entity_coordinates(self, association: str) -> u.Quantity:
        """Return physical coordinates of each entity for ``association``."""
        if association not in _SURFACE_ASSOCIATIONS:
            raise ValueError(
                f"association must be one of {_SURFACE_ASSOCIATIONS}, got {association!r}"
            )
        if association in ("vertex", "point"):
            return self.vertices.copy()
        if association == "face":
            return self.face_centroids()
        if self.quadrature_points is not None:
            return self.quadrature_points.copy()
        return self.vertices.copy()

    # ------------------------------------------------------------------
    # Geometry-derived dynamic quantities (never statically cached, #904)
    # ------------------------------------------------------------------

    def face_centroids(self) -> u.Quantity:
        """Compute area-weighted face centroids dynamically from vertices."""
        nf = self.n_faces
        if nf == 0:
            return u.Quantity(
                np.empty((0, self.embedding_dimension), dtype=np.float64),
                self.length_unit,
            )
        verts = np.asarray(self.vertices.value, dtype=np.float64)
        parts: list[np.ndarray] = []
        for cell_type, conn in self.cell_blocks:
            if cell_type == "triangle":
                a, b, c = verts[conn[:, 0]], verts[conn[:, 1]], verts[conn[:, 2]]
                parts.append((a + b + c) / 3.0)
            else:
                v0, v1, v2, v3 = (
                    verts[conn[:, 0]],
                    verts[conn[:, 1]],
                    verts[conn[:, 2]],
                    verts[conn[:, 3]],
                )
                c1 = (v0 + v1 + v2) / 3.0
                c2 = (v0 + v2 + v3) / 3.0
                a1 = _cross_norm(v1 - v0, v2 - v0)
                a2 = _cross_norm(v2 - v0, v3 - v0)
                denom = a1 + a2
                mean4 = 0.25 * (v0 + v1 + v2 + v3)
                centroids = np.where(
                    denom[:, None] > 0.0,
                    (a1[:, None] * c1 + a2[:, None] * c2)
                    / np.where(denom[:, None] > 0.0, denom[:, None], 1.0),
                    mean4,
                )
                parts.append(centroids)
        return u.Quantity(np.concatenate(parts, axis=0), self.length_unit)

    def face_vector_areas(self) -> u.Quantity:
        """Compute oriented vector area elements ``A_f * n_hat_f`` (3D only)."""
        if self.embedding_dimension != 3:
            raise ValueError(
                "face_vector_areas requires 3D embedded vertices "
                f"(embedding_dimension={self.embedding_dimension})"
            )
        nf = self.n_faces
        area_unit = self.length_unit**2
        if nf == 0:
            return u.Quantity(np.empty((0, 3), dtype=np.float64), area_unit)
        verts = np.asarray(self.vertices.value, dtype=np.float64)
        parts: list[np.ndarray] = []
        for cell_type, conn in self.cell_blocks:
            if cell_type == "triangle":
                a, b, c = verts[conn[:, 0]], verts[conn[:, 1]], verts[conn[:, 2]]
                parts.append(0.5 * np.cross(b - a, c - a))
            else:
                v0, v1, v2, v3 = (
                    verts[conn[:, 0]],
                    verts[conn[:, 1]],
                    verts[conn[:, 2]],
                    verts[conn[:, 3]],
                )
                parts.append(0.5 * np.cross(v2 - v0, v3 - v1))
        return u.Quantity(np.concatenate(parts, axis=0), area_unit)

    def face_areas(
        self, *, backend: Literal["numpy", "pyvista"] = "numpy"
    ) -> u.Quantity:
        """Compute scalar face areas dynamically from geometry."""
        area_unit = self.length_unit**2
        nf = self.n_faces
        if nf == 0:
            return u.Quantity(np.empty(0, dtype=np.float64), area_unit)
        if backend == "pyvista":
            pd = self.to_pyvista()
            sizes = pd.compute_cell_sizes(length=False, area=True, volume=False)
            return u.Quantity(
                np.asarray(sizes.cell_data["Area"], dtype=np.float64), area_unit
            )
        if backend != "numpy":
            raise ValueError(f"Unsupported geometry backend: {backend!r}")

        verts = np.asarray(self.vertices.value, dtype=np.float64)
        parts: list[np.ndarray] = []
        for cell_type, conn in self.cell_blocks:
            if cell_type == "triangle":
                a, b, c = verts[conn[:, 0]], verts[conn[:, 1]], verts[conn[:, 2]]
                parts.append(0.5 * _cross_norm(b - a, c - a))
            else:
                v0, v1, v2, v3 = (
                    verts[conn[:, 0]],
                    verts[conn[:, 1]],
                    verts[conn[:, 2]],
                    verts[conn[:, 3]],
                )
                a_012 = _cross_norm(v1 - v0, v2 - v0)
                a_023 = _cross_norm(v2 - v0, v3 - v0)
                a_123 = _cross_norm(v2 - v1, v3 - v1)
                a_130 = _cross_norm(v3 - v1, v0 - v1)
                parts.append(0.25 * (a_012 + a_023 + a_123 + a_130))
        return u.Quantity(np.concatenate(parts, axis=0), area_unit)

    def face_normals(
        self, *, backend: Literal["numpy", "pyvista"] = "numpy"
    ) -> np.ndarray:
        """Compute unit face normals ``(n_faces, 3)`` following right-hand winding."""
        if self.embedding_dimension != 3:
            raise ValueError(
                "face_normals requires 3D embedded vertices "
                f"(embedding_dimension={self.embedding_dimension})"
            )
        nf = self.n_faces
        if nf == 0:
            return np.empty((0, 3), dtype=np.float64)
        if backend == "pyvista":
            pd = self.to_pyvista()
            norms = pd.compute_normals(
                cell_normals=True,
                point_normals=False,
                split_vertices=False,
                flip_normals=False,
                consistent_normals=False,
                auto_orient_normals=False,
            )
            arr = np.asarray(norms.cell_data["Normals"], dtype=np.float64)
            lengths = np.linalg.norm(arr, axis=-1)
            if np.any(lengths <= _COLLINEAR_TOL):
                raise ValueError(
                    "Cannot compute unit normals for degenerate zero-area face(s)"
                )
            return arr / lengths[:, None]
        if backend != "numpy":
            raise ValueError(f"Unsupported geometry backend: {backend!r}")

        vec_areas = np.asarray(self.face_vector_areas().value, dtype=np.float64)
        norms = np.linalg.norm(vec_areas, axis=-1)
        scale = self._face_area_scale()
        tol = _COLLINEAR_TOL * scale
        if np.any((norms == 0.0) | (norms <= tol)):
            raise ValueError(
                "Cannot compute unit normals for degenerate zero-area face(s)"
            )
        return vec_areas / norms[:, None]

    def _face_area_scale(self) -> np.ndarray:
        verts = np.asarray(self.vertices.value, dtype=np.float64)
        parts: list[np.ndarray] = []
        for cell_type, conn in self.cell_blocks:
            if cell_type == "triangle":
                e1 = verts[conn[:, 1]] - verts[conn[:, 0]]
                e2 = verts[conn[:, 2]] - verts[conn[:, 0]]
                parts.append(
                    0.5 * np.linalg.norm(e1, axis=-1) * np.linalg.norm(e2, axis=-1)
                )
            else:
                d1 = verts[conn[:, 2]] - verts[conn[:, 0]]
                d2 = verts[conn[:, 3]] - verts[conn[:, 1]]
                parts.append(
                    0.5 * np.linalg.norm(d1, axis=-1) * np.linalg.norm(d2, axis=-1)
                )
        return np.concatenate(parts, axis=0)

    def vertex_areas(self) -> u.Quantity:
        """Compute barycentric dual cell areas per vertex (lumped surface area)."""
        area_unit = self.length_unit**2
        nv = self.n_vertices
        out = np.zeros(nv, dtype=np.float64)
        if self.n_faces == 0:
            return u.Quantity(out, area_unit)
        f_areas = np.asarray(self.face_areas().value, dtype=np.float64)
        offset = 0
        for cell_type, conn in self.cell_blocks:
            nf = conn.shape[0]
            block_areas = f_areas[offset : offset + nf]
            k = 3 if cell_type == "triangle" else 4
            contrib = block_areas / float(k)
            for col in range(k):
                np.add.at(out, conn[:, col], contrib)
            offset += nf
        return u.Quantity(out, area_unit)

    def vertex_normals(self) -> np.ndarray:
        """Compute area-weighted unit normals at vertices (3D only)."""
        if self.embedding_dimension != 3:
            raise ValueError(
                "vertex_normals requires 3D embedded vertices "
                f"(embedding_dimension={self.embedding_dimension})"
            )
        nv = self.n_vertices
        if nv == 0:
            return np.empty((0, 3), dtype=np.float64)
        if self.n_faces == 0:
            raise ValueError("Cannot compute vertex_normals on a surface with no faces")
        vec_areas = np.asarray(self.face_vector_areas().value, dtype=np.float64)
        accum = np.zeros((nv, 3), dtype=np.float64)
        offset = 0
        for _, conn in self.cell_blocks:
            nf = conn.shape[0]
            block_va = vec_areas[offset : offset + nf]
            for col in range(conn.shape[1]):
                np.add.at(accum, conn[:, col], block_va)
            offset += nf
        norms = np.linalg.norm(accum, axis=-1)
        max_norm = float(np.max(norms)) if norms.size else 0.0
        tol = _COLLINEAR_TOL * max(max_norm, 1e-30)
        if np.any(norms <= tol):
            raise ValueError(
                "Cannot compute vertex_normals: one or more vertices are unreferenced "
                "or have zero net incident normal"
            )
        return accum / norms[:, None]

    def total_area(
        self, *, backend: Literal["numpy", "pyvista"] = "numpy"
    ) -> u.Quantity:
        """Total surface area across all faces."""
        return u.Quantity(
            np.sum(self.face_areas(backend=backend).value), self.length_unit**2
        )

    def signed_volume(self) -> u.Quantity:
        """Signed enclosed volume for a closed 3D surface."""
        return self.check_orientation().signed_volume_quantity

    # ------------------------------------------------------------------
    # Orientation
    # ------------------------------------------------------------------

    def check_orientation(self) -> OrientationReport:
        """Diagnose winding consistency, closedness, and outward orientation."""
        return check_surface_orientation(self.vertices, self.cell_blocks)

    def flip_orientation(self) -> SurfaceGeometry:
        """Return a new :class:`SurfaceGeometry` with every face winding reversed."""
        flipped_blocks = []
        for cell_type, conn in self.cell_blocks:
            k = conn.shape[1]
            rev = [0] + list(range(k - 1, 0, -1))
            flipped_blocks.append((cell_type, conn[:, rev]))
        new_conv = (
            "inward"
            if self.orientation_convention == "outward"
            else (
                "outward"
                if self.orientation_convention == "inward"
                else self.orientation_convention
            )
        )
        return SurfaceGeometry(
            vertices=self.vertices,
            cell_blocks=flipped_blocks,
            coordinate_system=self.coordinate_system,
            orientation_convention=new_conv,
            global_vertex_ids=self.global_vertex_ids,
            global_cell_ids=self.global_cell_ids,
            quadrature_points=self.quadrature_points,
            quadrature_weights=self.quadrature_weights,
            quadrature_cell_ids=self.quadrature_cell_ids,
            structured_shape=self.structured_shape,
            structured_axes=self.structured_axes,
            structured_plane=self.structured_plane,
        )

    def with_consistent_orientation(self) -> SurfaceGeometry:
        """Return a new :class:`SurfaceGeometry` with consistent edge windings."""
        reoriented = reorient_faces(self.cell_blocks)
        return SurfaceGeometry(
            vertices=self.vertices,
            cell_blocks=reoriented,
            coordinate_system=self.coordinate_system,
            orientation_convention=self.orientation_convention,
            global_vertex_ids=self.global_vertex_ids,
            global_cell_ids=self.global_cell_ids,
            quadrature_points=self.quadrature_points,
            quadrature_weights=self.quadrature_weights,
            quadrature_cell_ids=self.quadrature_cell_ids,
            structured_shape=self.structured_shape,
            structured_axes=self.structured_axes,
            structured_plane=self.structured_plane,
        )

    def orient_outward(self) -> SurfaceGeometry:
        """Return a consistently wound outward-facing closed surface."""
        consistent = self.with_consistent_orientation()
        report = consistent.check_orientation()
        if not report.is_closed:
            raise ValueError("orient_outward requires a closed surface mesh")
        if report.signed_volume < 0.0:
            consistent = consistent.flip_orientation()
        return SurfaceGeometry(
            vertices=consistent.vertices,
            cell_blocks=consistent.cell_blocks,
            coordinate_system=consistent.coordinate_system,
            orientation_convention="outward",
            global_vertex_ids=consistent.global_vertex_ids,
            global_cell_ids=consistent.global_cell_ids,
            quadrature_points=consistent.quadrature_points,
            quadrature_weights=consistent.quadrature_weights,
            quadrature_cell_ids=consistent.quadrature_cell_ids,
            structured_shape=consistent.structured_shape,
            structured_axes=consistent.structured_axes,
            structured_plane=consistent.structured_plane,
        )

    # ------------------------------------------------------------------
    # Entity selection & comparison
    # ------------------------------------------------------------------

    def select_faces(self, selector: Any) -> SurfaceGeometry:
        """Return a compact :class:`SurfaceGeometry` keeping only selected faces.

        Face order follows ``selector``, unused vertices are pruned, and vertex
        indices are remapped cleanly.
        """
        idx = _normalize_entity_selector(selector, self.n_faces, name="face selector")
        if len(idx) == 0:
            return SurfaceGeometry(
                vertices=self.vertices[:0],
                cell_blocks=(),
                coordinate_system=self.coordinate_system,
                orientation_convention=self.orientation_convention,
                global_vertex_ids=(
                    self.global_vertex_ids[:0]
                    if self.global_vertex_ids is not None
                    else None
                ),
                global_cell_ids=(
                    self.global_cell_ids[:0]
                    if self.global_cell_ids is not None
                    else None
                ),
            )

        face_types: list[str] = []
        face_rows: list[np.ndarray] = []
        for cell_type, conn in self.cell_blocks:
            for row in conn:
                face_types.append(cell_type)
                face_rows.append(row)

        selected_rows = [face_rows[int(i)] for i in idx]
        selected_types = [face_types[int(i)] for i in idx]

        used_verts: list[int] = []
        old_to_new: dict[int, int] = {}
        for row in selected_rows:
            for vid in row:
                iv = int(vid)
                if iv not in old_to_new:
                    old_to_new[iv] = len(used_verts)
                    used_verts.append(iv)

        used_arr = np.asarray(used_verts, dtype=np.intp)
        new_vertices = self.vertices[used_arr]
        new_gv_ids = (
            self.global_vertex_ids[used_arr]
            if self.global_vertex_ids is not None
            else None
        )
        new_gc_ids = (
            self.global_cell_ids[idx] if self.global_cell_ids is not None else None
        )

        new_blocks: list[tuple[str, np.ndarray]] = []
        run_type = selected_types[0]
        run_rows: list[list[int]] = []
        for ftype, row in zip(selected_types, selected_rows, strict=True):
            remapped = [old_to_new[int(v)] for v in row]
            if ftype == run_type:
                run_rows.append(remapped)
            else:
                new_blocks.append((run_type, np.asarray(run_rows, dtype=np.intp)))
                run_type = ftype
                run_rows = [remapped]
        new_blocks.append((run_type, np.asarray(run_rows, dtype=np.intp)))

        return SurfaceGeometry(
            vertices=new_vertices,
            cell_blocks=new_blocks,
            coordinate_system=self.coordinate_system,
            orientation_convention=self.orientation_convention,
            global_vertex_ids=new_gv_ids,
            global_cell_ids=new_gc_ids,
        )

    def select_vertices(self, selector: Any) -> SurfaceGeometry:
        """Return a :class:`SurfaceGeometry` keeping selected vertices and contained faces."""
        idx = _normalize_entity_selector(
            selector, self.n_vertices, name="vertex selector"
        )
        new_vertices = self.vertices[idx]
        new_gv_ids = (
            self.global_vertex_ids[idx] if self.global_vertex_ids is not None else None
        )

        old_to_new = np.full(self.n_vertices, -1, dtype=np.intp)
        old_to_new[idx] = np.arange(len(idx), dtype=np.intp)

        kept_blocks: list[tuple[str, np.ndarray]] = []
        kept_face_indices: list[np.ndarray] = []
        face_offset = 0
        for cell_type, conn in self.cell_blocks:
            nf = conn.shape[0]
            remapped = old_to_new[conn]
            valid_face = np.all(remapped >= 0, axis=1)
            if np.any(valid_face):
                kept_blocks.append((cell_type, remapped[valid_face]))
                kept_face_indices.append(
                    np.nonzero(valid_face)[0].astype(np.intp) + face_offset
                )
            face_offset += nf

        new_gc_ids: np.ndarray | None = None
        if self.global_cell_ids is not None:
            if kept_face_indices:
                cat = np.concatenate(kept_face_indices)
                new_gc_ids = self.global_cell_ids[cat]
            else:
                new_gc_ids = self.global_cell_ids[:0]

        return SurfaceGeometry(
            vertices=new_vertices,
            cell_blocks=kept_blocks,
            coordinate_system=self.coordinate_system,
            orientation_convention=self.orientation_convention,
            global_vertex_ids=new_gv_ids,
            global_cell_ids=new_gc_ids,
        )

    def select_quadrature(self, selector: Any) -> SurfaceGeometry:
        """Return a :class:`SurfaceGeometry` keeping only selected quadrature points."""
        if self.quadrature_points is None:
            return self.select_vertices(selector)
        idx = _normalize_entity_selector(
            selector, self.n_quadrature, name="quadrature selector"
        )
        return SurfaceGeometry(
            vertices=self.vertices,
            cell_blocks=self.cell_blocks,
            coordinate_system=self.coordinate_system,
            orientation_convention=self.orientation_convention,
            global_vertex_ids=self.global_vertex_ids,
            global_cell_ids=self.global_cell_ids,
            quadrature_points=self.quadrature_points[idx],
            quadrature_weights=(
                self.quadrature_weights[idx]
                if self.quadrature_weights is not None
                else None
            ),
            quadrature_cell_ids=(
                self.quadrature_cell_ids[idx]
                if self.quadrature_cell_ids is not None
                else None
            ),
        )

    def copy(self) -> SurfaceGeometry:
        """Return a defensive deep copy of this geometry."""
        return SurfaceGeometry(
            vertices=self.vertices.copy(),
            cell_blocks=[(ct, c.copy()) for ct, c in self.cell_blocks],
            coordinate_system=self.coordinate_system,
            orientation_convention=self.orientation_convention,
            global_vertex_ids=(
                self.global_vertex_ids.copy()
                if self.global_vertex_ids is not None
                else None
            ),
            global_cell_ids=(
                self.global_cell_ids.copy()
                if self.global_cell_ids is not None
                else None
            ),
            quadrature_points=(
                self.quadrature_points.copy()
                if self.quadrature_points is not None
                else None
            ),
            quadrature_weights=(
                self.quadrature_weights.copy()
                if self.quadrature_weights is not None
                else None
            ),
            quadrature_cell_ids=(
                self.quadrature_cell_ids.copy()
                if self.quadrature_cell_ids is not None
                else None
            ),
            structured_shape=self.structured_shape,
            structured_axes=self.structured_axes,
            structured_plane=self.structured_plane,
        )

    def is_compatible(
        self,
        other: SurfaceGeometry,
        *,
        rtol: float = _AXIS_RTOL,
        atol: float = _AXIS_ATOL,
    ) -> bool:
        """Return True when two geometries have matching coordinates and topology."""
        if not isinstance(other, SurfaceGeometry):
            return False
        if self.embedding_dimension != other.embedding_dimension:
            return False
        if self.coordinate_system != other.coordinate_system:
            return False
        if not _quantities_close(self.vertices, other.vertices, rtol=rtol, atol=atol):
            return False
        if len(self.cell_blocks) != len(other.cell_blocks):
            return False
        for (ct_l, c_l), (ct_r, c_r) in zip(
            self.cell_blocks, other.cell_blocks, strict=True
        ):
            if ct_l != ct_r or c_l.shape != c_r.shape or not np.array_equal(c_l, c_r):
                return False
        if (self.quadrature_points is None) != (other.quadrature_points is None):
            return False
        if self.quadrature_points is not None and other.quadrature_points is not None:
            if not _quantities_close(
                self.quadrature_points, other.quadrature_points, rtol=rtol, atol=atol
            ):
                return False
        if (self.quadrature_weights is None) != (other.quadrature_weights is None):
            return False
        if self.quadrature_weights is not None and other.quadrature_weights is not None:
            if not _quantities_close(
                self.quadrature_weights, other.quadrature_weights, rtol=rtol, atol=atol
            ):
                return False
        return True

    # ------------------------------------------------------------------
    # Optional meshio / pyvista backend adapters
    # ------------------------------------------------------------------

    def to_meshio(self) -> Any:
        """Convert this geometry into a ``meshio.Mesh`` (optional dependency)."""
        meshio = require_optional("meshio")
        pts = np.asarray(self.vertices.value, dtype=np.float64)
        cells = [(ct, np.asarray(c, dtype=np.int64)) for ct, c in self.cell_blocks]
        point_data: dict[str, np.ndarray] = {}
        cell_data: dict[str, list[np.ndarray]] = {}
        if self.global_vertex_ids is not None:
            point_data["gwexpy_global_vertex_ids"] = np.asarray(
                self.global_vertex_ids, dtype=np.int64
            )
        if self.global_cell_ids is not None and self.cell_blocks:
            blocks: list[np.ndarray] = []
            offset = 0
            for _, c in self.cell_blocks:
                nf = c.shape[0]
                blocks.append(
                    np.asarray(
                        self.global_cell_ids[offset : offset + nf], dtype=np.int64
                    )
                )
                offset += nf
            cell_data["gwexpy_global_cell_ids"] = blocks
        return meshio.Mesh(
            points=pts,
            cells=cells,
            point_data=point_data,
            cell_data=cell_data,
        )

    @classmethod
    def from_meshio(
        cls,
        mesh_or_path: Any,
        *,
        unit: u.UnitBase | str = u.m,
        coordinate_system: str = "cartesian",
        orientation_convention: str = "right_hand_rule",
        allow_point_cloud: bool = False,
    ) -> SurfaceGeometry:
        """Create a :class:`SurfaceGeometry` from a ``meshio.Mesh`` or file path."""
        meshio = require_optional("meshio")
        mesh = (
            meshio.read(str(mesh_or_path))
            if isinstance(mesh_or_path, (str, Path))
            else mesh_or_path
        )
        points = np.asarray(mesh.points, dtype=np.float64)
        blocks: list[tuple[str, np.ndarray]] = []
        for block in getattr(mesh, "cells", ()):
            if block.type in ("triangle", "quad"):
                blocks.append((block.type, np.asarray(block.data, dtype=np.intp)))
            elif block.type in ("vertex", "line"):
                continue
            else:
                raise ValueError(
                    f"meshio mesh contains non-surface cell block {block.type!r}"
                )
        if not blocks and not allow_point_cloud:
            raise ValueError(
                "No 'triangle' or 'quad' surface cells found in meshio.Mesh"
            )
        gv_ids = getattr(mesh, "point_data", {}).get("gwexpy_global_vertex_ids")
        gc_ids_blocks = getattr(mesh, "cell_data", {}).get("gwexpy_global_cell_ids")
        gc_ids = (
            np.concatenate([np.asarray(b) for b in gc_ids_blocks])
            if gc_ids_blocks
            else None
        )
        return cls(
            vertices=points,
            cell_blocks=blocks,
            unit=unit,
            coordinate_system=coordinate_system,
            orientation_convention=orientation_convention,
            global_vertex_ids=gv_ids,
            global_cell_ids=gc_ids,
        )

    def to_pyvista(self) -> Any:
        """Convert this geometry into a ``pyvista.PolyData`` (optional dependency)."""
        pv = require_optional("pyvista")
        pts = np.asarray(self.vertices.value, dtype=np.float64)
        if self.embedding_dimension == 2:
            pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float64)])
        if not self.cell_blocks:
            pd = pv.PolyData(pts)
        else:
            face_parts: list[np.ndarray] = []
            for cell_type, conn in self.cell_blocks:
                k = 3 if cell_type == "triangle" else 4
                col_k = np.full((conn.shape[0], 1), k, dtype=np.int64)
                face_parts.append(
                    np.hstack([col_k, np.asarray(conn, dtype=np.int64)]).ravel()
                )
            pd = pv.PolyData(pts, faces=np.concatenate(face_parts))
        if self.global_vertex_ids is not None:
            pd.point_data["gwexpy_global_vertex_ids"] = np.asarray(
                self.global_vertex_ids, dtype=np.int64
            )
        if self.global_cell_ids is not None and self.cell_blocks:
            pd.cell_data["gwexpy_global_cell_ids"] = np.asarray(
                self.global_cell_ids, dtype=np.int64
            )
        return pd

    @classmethod
    def from_pyvista(
        cls,
        dataset_or_path: Any,
        *,
        unit: u.UnitBase | str = u.m,
        coordinate_system: str = "cartesian",
        orientation_convention: str = "right_hand_rule",
        allow_point_cloud: bool = False,
    ) -> SurfaceGeometry:
        """Create a :class:`SurfaceGeometry` from a PyVista dataset or file path."""
        pv = require_optional("pyvista")
        ds = (
            pv.read(str(dataset_or_path))
            if isinstance(dataset_or_path, (str, Path))
            else dataset_or_path
        )
        blocks, points = _extract_pyvista_surface_blocks(ds, pv)
        if not blocks and not allow_point_cloud:
            raise ValueError(
                "No triangle or quad surface cells found in PyVista dataset"
            )
        gv_ids = (
            ds.point_data["gwexpy_global_vertex_ids"]
            if "gwexpy_global_vertex_ids" in ds.point_data
            else None
        )
        gc_ids = (
            ds.cell_data["gwexpy_global_cell_ids"]
            if "gwexpy_global_cell_ids" in ds.cell_data
            else None
        )
        return cls(
            vertices=points,
            cell_blocks=blocks,
            unit=unit,
            coordinate_system=coordinate_system,
            orientation_convention=orientation_convention,
            global_vertex_ids=gv_ids,
            global_cell_ids=gc_ids,
        )


def _extract_pyvista_surface_blocks(
    ds: Any, pv: Any
) -> tuple[list[tuple[str, np.ndarray]], np.ndarray]:
    """Extract ordered ``(cell_type, conn)`` blocks and points from a PyVista mesh."""
    if isinstance(ds, pv.StructuredGrid):
        dims = tuple(int(d) for d in ds.dimensions)
        if 1 not in dims or dims.count(1) > 1:
            raise ValueError(
                f"StructuredGrid must represent a 2D surface (one dim == 1), got {dims}"
            )
        ds = ds.cast_to_unstructured_grid()

    points = np.asarray(ds.points, dtype=np.float64)
    if isinstance(ds, pv.PolyData):
        faces_flat = np.asarray(ds.faces, dtype=np.intp)
        if faces_flat.size == 0:
            return [], points
        cell_types: list[str] = []
        rows: list[np.ndarray] = []
        idx = 0
        while idx < len(faces_flat):
            n = int(faces_flat[idx])
            verts = faces_flat[idx + 1 : idx + 1 + n]
            idx += 1 + n
            if n == 3:
                cell_types.append("triangle")
                rows.append(verts)
            elif n == 4:
                cell_types.append("quad")
                rows.append(verts)
            else:
                raise ValueError(
                    f"Unsupported PolyData polygon with {n} vertices; only triangles "
                    "and quads are supported"
                )
        return _group_contiguous_cells(cell_types, rows), points

    vtk_tri = int(pv.CellType.TRIANGLE)
    vtk_quad = int(pv.CellType.QUAD)
    ctypes_arr = np.asarray(ds.celltypes)
    if ctypes_arr.size == 0:
        return [], points
    offsets = np.asarray(ds.offset, dtype=np.intp)
    conn_flat = np.asarray(ds.cell_connectivity, dtype=np.intp)

    cell_types = []
    rows = []
    for i, ctype_code in enumerate(ctypes_arr):
        code = int(ctype_code)
        seg = conn_flat[offsets[i] : offsets[i + 1]]
        if code == vtk_tri:
            cell_types.append("triangle")
            rows.append(seg)
        elif code == vtk_quad:
            cell_types.append("quad")
            rows.append(seg)
        else:
            raise ValueError(
                f"PyVista dataset contains unsupported VTK cell type code {code}"
            )
    return _group_contiguous_cells(cell_types, rows), points


def _group_contiguous_cells(
    cell_types: Sequence[str], rows: Sequence[np.ndarray]
) -> list[tuple[str, np.ndarray]]:
    if not cell_types:
        return []
    blocks: list[tuple[str, np.ndarray]] = []
    curr_type = cell_types[0]
    curr_rows: list[np.ndarray] = []
    for ct, r in zip(cell_types, rows, strict=True):
        if ct == curr_type:
            curr_rows.append(r)
        else:
            blocks.append((curr_type, np.stack(curr_rows, axis=0)))
            curr_type = ct
            curr_rows = [r]
    blocks.append((curr_type, np.stack(curr_rows, axis=0)))
    return blocks


def _cross_norm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = np.cross(a, b)
    if c.ndim == 1:
        return np.abs(c)
    return np.linalg.norm(c, axis=-1)


def _is_strictly_monotonic(arr: np.ndarray) -> bool:
    if arr.size <= 1:
        return True
    diffs = np.diff(arr)
    return bool(np.all(diffs > 0.0) or np.all(diffs < 0.0))


def _normalize_entity_selector(
    selector: Any, n_entities: int, *, name: str
) -> np.ndarray:
    if isinstance(selector, slice):
        return np.arange(*selector.indices(n_entities), dtype=np.intp)
    arr = np.asarray(selector)
    if arr.dtype.kind == "b":
        if arr.ndim != 1 or arr.shape[0] != n_entities:
            raise ValueError(
                f"{name} boolean mask must have shape ({n_entities},), got {arr.shape}"
            )
        return np.nonzero(arr)[0].astype(np.intp)
    if arr.dtype.kind not in "iu":
        raise TypeError(f"{name} must be slice, boolean mask, or integer indices")
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    idx = arr.astype(np.intp, copy=True)
    idx = np.where(idx < 0, idx + n_entities, idx)
    if np.any(idx < 0) or np.any(idx >= n_entities):
        raise IndexError(f"{name} index out of bounds for size {n_entities}")
    if len(np.unique(idx)) != len(idx):
        raise ValueError(f"{name} must not contain duplicate entity indices")
    return idx


def _normalize_axis0_selector(selector: Any, n0: int) -> np.ndarray:
    if isinstance(selector, slice):
        return np.arange(*selector.indices(n0), dtype=np.intp)
    if isinstance(selector, (int, np.integer)) and not isinstance(
        selector, (bool, np.bool_)
    ):
        idx_val = int(selector)
        if idx_val < 0:
            idx_val += n0
        if idx_val < 0 or idx_val >= n0:
            raise IndexError(f"axis0 index {selector} out of bounds for size {n0}")
        return np.array([idx_val], dtype=np.intp)
    arr = np.asarray(selector)
    if arr.dtype.kind == "b":
        if arr.ndim != 1 or arr.shape[0] != n0:
            raise ValueError(
                f"axis0 boolean mask must have shape ({n0},), got {arr.shape}"
            )
        return np.nonzero(arr)[0].astype(np.intp)
    if arr.dtype.kind not in "iu":
        raise TypeError("axis0 selector must be int, slice, bool mask, or int sequence")
    if arr.ndim != 1:
        raise ValueError(f"axis0 selector must be 1D, got shape {arr.shape}")
    idx = arr.astype(np.intp, copy=True)
    idx = np.where(idx < 0, idx + n0, idx)
    if np.any(idx < 0) or np.any(idx >= n0):
        raise IndexError(f"axis0 index out of bounds for size {n0}")
    return idx


# ---------------------------------------------------------------------------
# Value Layer: SurfaceField
# ---------------------------------------------------------------------------


class SurfaceField:
    """Physical scalar, vector, or tensor field defined on a 2D surface mesh.

    GWexpy owns the value layer (physical units via ``astropy.units.Quantity``,
    ``axis0`` time/frequency coordinate and ``#725`` ``time_reference``, entity
    ``association``, ``component_basis``, ``valid_mask``, and ``provenance``),
    while delegating geometry storage to a separate immutable
    :class:`SurfaceGeometry`.
    """

    __array_priority__ = 10_002
    __array_ufunc__ = None

    def __init__(
        self,
        data: Any,
        *,
        geometry: SurfaceGeometry,
        association: SurfaceAssociation = "vertex",
        unit: u.UnitBase | str | None = None,
        axis0: Any = None,
        axis0_domain: Literal["time", "frequency"] = "time",
        axis0_name: str | None = None,
        epoch: Any = TIME_INPUT_UNSET,
        component_basis: ComponentBasis | None = None,
        component_names: Sequence[str] | None = None,
        field_kind: Literal["auto", "scalar", "vector", "tensor"] = "auto",
        valid_mask: Any = None,
        provenance: SurfaceProvenance | Mapping[str, Any] | None = None,
        name: str | None = None,
        channel: Any = None,
        copy: bool = True,
    ) -> None:
        if not isinstance(geometry, SurfaceGeometry):
            raise TypeError(
                f"geometry must be SurfaceGeometry, got {type(geometry).__name__}"
            )
        if association not in _SURFACE_ASSOCIATIONS:
            raise ValueError(
                f"association must be one of {_SURFACE_ASSOCIATIONS}, got {association!r}"
            )
        if axis0_domain not in ("time", "frequency"):
            raise ValueError(
                f"axis0_domain must be 'time' or 'frequency', got {axis0_domain!r}"
            )
        if field_kind not in ("auto", "scalar", "vector", "tensor"):
            raise ValueError(f"Unsupported field_kind: {field_kind!r}")

        n_entities = geometry.entity_count(association)

        _check_no_bool(data, "data")
        if isinstance(data, u.Quantity):
            eff_unit = data.unit if unit is None else u.Unit(unit)
            raw = (
                np.asarray(data.to_value(eff_unit))
                if unit is not None
                else np.asarray(data.value)
            )
        else:
            raw = np.asarray(data)
            eff_unit = u.dimensionless_unscaled if unit is None else u.Unit(unit)

        if raw.dtype.kind not in "iufc":
            raise TypeError(f"data must have numeric dtype, got {raw.dtype}")
        if raw.dtype.kind in "iu":
            raw = raw.astype(np.float64)
        elif copy:
            raw = raw.copy()

        resolved_kind, values_norm = _normalize_surface_data_shape(
            raw,
            n_entities=n_entities,
            axis0=axis0,
            field_kind=field_kind,
            component_basis=component_basis,
            component_names=component_names,
        )
        n0 = int(values_norm.shape[0])
        comp_shape = tuple(int(s) for s in values_norm.shape[2:])
        tensor_rank = len(comp_shape)

        resolved_basis, resolved_names = _resolve_component_metadata(
            resolved_kind=resolved_kind,
            comp_shape=comp_shape,
            embedding_dim=geometry.embedding_dimension,
            topo_dim=geometry.topological_dimension,
            component_basis=component_basis,
            component_names=component_names,
        )

        if axis0 is None:
            default_unit = u.s if axis0_domain == "time" else u.Hz
            axis0_q = np.arange(n0, dtype=np.float64) * default_unit
        else:
            axis0_q = coerce_1d_quantity(axis0)
            if len(axis0_q) != n0:
                raise ValueError(
                    f"axis0 length ({len(axis0_q)}) does not match data axis0 size ({n0})"
                )
            if axis0_q.unit != u.dimensionless_unscaled:
                expected_ax_unit = u.s if axis0_domain == "time" else (1 / u.s)
                if not axis0_q.unit.is_equivalent(expected_ax_unit):
                    raise ValueError(
                        f"axis0_domain '{axis0_domain}' expects units equivalent to "
                        f"{expected_ax_unit}, got {axis0_q.unit}"
                    )
        axis0_q = u.Quantity(_freeze_array(axis0_q.value), axis0_q.unit, copy=False)

        eff_axis0_name = (
            str(axis0_name)
            if axis0_name is not None
            else ("t" if axis0_domain == "time" else "f")
        )

        eff_epoch, time_ref_state = _resolve_epoch_state(epoch)

        mask_arr = _normalize_valid_mask(
            valid_mask, n0=n0, n_entities=n_entities, data_shape=values_norm.shape
        )

        prov = SurfaceProvenance.coerce(provenance)

        self._geometry = geometry
        self._association: SurfaceAssociation = association
        self._values = _freeze_array(values_norm)
        self._unit = eff_unit
        self._axis0 = axis0_q
        self._axis0_domain: Literal["time", "frequency"] = axis0_domain
        self._axis0_name = eff_axis0_name
        self._epoch = eff_epoch
        self._tensor_rank = tensor_rank
        self._component_shape = comp_shape
        self._component_basis: ComponentBasis = resolved_basis
        self._component_names = resolved_names
        self._valid_mask = mask_arr
        self._provenance = prov
        self.name = name
        self.channel = channel
        set_time_reference(self, time_ref_state)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def geometry(self) -> SurfaceGeometry:
        """Immutable surface geometry associated with these values."""
        return self._geometry

    @property
    def association(self) -> SurfaceAssociation:
        """Geometric entity association (``vertex``, ``face``, ``point``, ``quadrature``)."""
        return self._association

    @property
    def topological_dimension(self) -> int:
        """Topological dimension of the surface (always 2)."""
        return self._geometry.topological_dimension

    @property
    def embedding_dimension(self) -> int:
        """Spatial embedding dimension (2 or 3)."""
        return self._geometry.embedding_dimension

    @property
    def tensor_rank(self) -> int:
        """Tensor rank of the field values (0=scalar, 1=vector, 2=tensor)."""
        return self._tensor_rank

    @property
    def component_shape(self) -> tuple[int, ...]:
        """Trailing component shape beyond ``(axis0, entity)``."""
        return self._component_shape

    @property
    def component_basis(self) -> ComponentBasis:
        """Physical basis of the components (e.g. ``cartesian``, ``tangent``)."""
        return self._component_basis

    @property
    def component_names(self) -> tuple[str, ...]:
        """Names of the vector/tensor components."""
        return self._component_names

    @property
    def values(self) -> u.Quantity:
        """Field values as a :class:`~astropy.units.Quantity` copy."""
        return u.Quantity(self._values.copy(), self._unit)

    @property
    def value(self) -> np.ndarray:
        """Read-only NumPy array view of the raw numerical values."""
        return self._values

    @property
    def unit(self) -> u.UnitBase:
        """Physical unit of the field values."""
        return self._unit

    @property
    def shape(self) -> tuple[int, ...]:
        """Full array shape ``(n_axis0, n_entities, *component_shape)``."""
        return self._values.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions of the stored values array."""
        return self._values.ndim

    @property
    def n_axis0(self) -> int:
        """Number of samples along ``axis0``."""
        return int(self._values.shape[0])

    @property
    def n_entities(self) -> int:
        """Number of geometric entities along the spatial entity axis."""
        return int(self._values.shape[1])

    @property
    def axis0(self) -> u.Quantity:
        """Coordinate values along ``axis0`` (time or frequency)."""
        return self._axis0.copy()

    @property
    def axis0_domain(self) -> Literal["time", "frequency"]:
        """Domain of ``axis0`` (``'time'`` or ``'frequency'``)."""
        return self._axis0_domain

    @property
    def axis0_name(self) -> str:
        """Name of ``axis0`` (default ``'t'`` or ``'f'``)."""
        return self._axis0_name

    @property
    def epoch(self) -> Any:
        """GPS epoch when ``time_reference == 'absolute'``, else ``None``."""
        return self._epoch

    @property
    def time_reference(self) -> str:
        """Timing reference state: ``'relative'`` or ``'absolute'`` (#725)."""
        return get_time_reference(self)

    @property
    def valid_mask(self) -> np.ndarray:
        """Read-only boolean mask of shape ``(n_axis0, n_entities)``."""
        return self._valid_mask

    @property
    def provenance(self) -> SurfaceProvenance:
        """Frozen provenance metadata for this surface field."""
        return self._provenance

    @property
    def entity_coordinates(self) -> u.Quantity:
        """Coordinates of each entity in the declared ``association``."""
        return self._geometry.entity_coordinates(self._association)

    def __repr__(self) -> str:
        shape_str = ", ".join(str(d) for d in self.shape)
        return (
            f"<SurfaceField({shape_str})@{self._axis0_domain}, "
            f"assoc={self._association!r}, basis={self._component_basis!r}, "
            f"unit={self._unit!s}>"
        )

    # ------------------------------------------------------------------
    # Time reference anchoring (#725)
    # ------------------------------------------------------------------

    def with_epoch(self, epoch: Any) -> SurfaceField:
        """Return a copy anchored at GPS ``epoch`` while leaving geometry untouched.

        Follows the ``#725`` / ``FieldBase.with_epoch`` contract: requires a
        relative time reference and finite GPS ``epoch``.
        """
        from gwpy.time import to_gps

        if self.time_reference != TIME_REFERENCE_RELATIVE:
            raise ValueError("with_epoch requires a relative time reference")
        if classify_time_input(epoch) != "finite":
            raise ValueError("epoch must be a finite GPS time")
        val = epoch.to_value(u.s) if isinstance(epoch, u.Quantity) else epoch
        anchor = float(to_gps(val))
        if not np.isfinite(anchor):
            raise ValueError("epoch must be a finite GPS time")

        if self._axis0_domain != "time":
            raise ValueError(
                "SurfaceField.with_epoch is only supported for axis0_domain='time'"
            )

        ax_unit = self._axis0.unit
        if ax_unit.is_equivalent(u.dimensionless_unscaled):
            offsets = np.asarray(self._axis0.value, dtype=np.float64) * u.s
        elif ax_unit.is_equivalent(u.s):
            offsets = self._axis0.to(u.s)
        else:
            raise ValueError(
                "axis0 must have time or dimensionless units to anchor with_epoch"
            )

        if len(offsets) >= 2:
            dt_val = float((offsets[1] - offsets[0]).to_value(u.s))
            ulp = abs(float(np.spacing(anchor)))
            if abs(dt_val) < 10 * ulp:
                warnings.warn(
                    f"dt={dt_val:g} s is below 10 ULP of epoch ({ulp:g} s); "
                    "absolute time coordinates may be rounded",
                    UserWarning,
                    stacklevel=2,
                )

        shifted_axis0 = offsets + anchor * u.s
        return self._rebuild(
            axis0=shifted_axis0,
            epoch=anchor,
            provenance=self._provenance.with_step("with_epoch", epoch=anchor),
        )

    # ------------------------------------------------------------------
    # Selection, slicing, copying (preserving geometry-value alignment)
    # ------------------------------------------------------------------

    def copy(self) -> SurfaceField:
        """Return a deep copy preserving geometry, values, mask, and provenance."""
        return SurfaceField(
            self._values.copy(),
            geometry=self._geometry.copy(),
            association=self._association,
            unit=self._unit,
            axis0=self._axis0.copy(),
            axis0_domain=self._axis0_domain,
            axis0_name=self._axis0_name,
            epoch=(
                self._epoch if self.time_reference == TIME_REFERENCE_ABSOLUTE else None
            ),
            component_basis=self._component_basis,
            component_names=self._component_names,
            field_kind=_kind_from_rank(self._tensor_rank),
            valid_mask=self._valid_mask.copy(),
            provenance=self._provenance,
            name=self.name,
            channel=copy.deepcopy(self.channel),
            copy=False,
        )

    def select_axis0(self, selector: Any) -> SurfaceField:
        """Select samples along ``axis0`` while keeping geometry intact."""
        idx0 = _normalize_axis0_selector(selector, self.n_axis0)
        return self._rebuild(
            data=self._values[idx0],
            axis0=self._axis0[idx0],
            valid_mask=self._valid_mask[idx0],
            provenance=self._provenance.with_step(
                "select_axis0", mapping_kind="selected", n_axis0=len(idx0)
            ),
        )

    def select_entities(self, selector: Any) -> SurfaceField:
        """Select or reorder entities while preserving value-to-entity correspondence."""
        idx = _normalize_entity_selector(
            selector, self.n_entities, name="entity selector"
        )
        if self._association == "face":
            new_geom = self._geometry.select_faces(idx)
        elif self._association in ("vertex", "point"):
            new_geom = self._geometry.select_vertices(idx)
        else:
            new_geom = self._geometry.select_quadrature(idx)

        return self._rebuild(
            data=self._values[:, idx],
            geometry=new_geom,
            valid_mask=self._valid_mask[:, idx],
            provenance=self._provenance.with_step(
                "select_entities",
                mapping_kind="selected",
                n_entities=len(idx),
            ),
        )

    def isel(
        self,
        *,
        axis0: Any = None,
        entity: Any = None,
    ) -> SurfaceField:
        """Index-select along ``axis0`` and/or ``entity`` while keeping alignment."""
        out = self
        if axis0 is not None:
            out = out.select_axis0(axis0)
        if entity is not None:
            out = out.select_entities(entity)
        return out

    def sel(
        self,
        coord: Any = None,
        *,
        method: Literal["nearest"] = "nearest",
        **kwargs: Any,
    ) -> SurfaceField:
        """Value-based selection along ``axis0`` (by coordinate value or slice)."""
        if method != "nearest":
            raise ValueError(f"Unsupported selection method: {method!r}")
        if kwargs:
            if coord is not None or len(kwargs) != 1 or self._axis0_name not in kwargs:
                raise ValueError(
                    f"sel accepts {self._axis0_name!r} or positional coordinate"
                )
            coord = kwargs[self._axis0_name]
        if coord is None:
            raise ValueError("sel requires a coordinate value or slice")

        ax_vals = self._axis0.value
        ax_unit = self._axis0.unit
        if isinstance(coord, slice):
            start_v = (
                coord.start.to_value(ax_unit)
                if isinstance(coord.start, u.Quantity)
                else coord.start
            )
            stop_v = (
                coord.stop.to_value(ax_unit)
                if isinstance(coord.stop, u.Quantity)
                else coord.stop
            )
            mask = np.ones(self.n_axis0, dtype=bool)
            if start_v is not None:
                mask &= ax_vals >= float(start_v)
            if stop_v is not None:
                mask &= ax_vals <= float(stop_v)
            return self.select_axis0(mask)

        target_v = (
            float(coord.to_value(ax_unit))
            if isinstance(coord, u.Quantity)
            else float(coord)
        )
        nearest_idx = int(np.argmin(np.abs(ax_vals - target_v)))
        return self.select_axis0(nearest_idx)

    def __getitem__(self, item: Any) -> SurfaceField:
        """Slice ``SurfaceField`` along ``(axis0, entity[, component])``."""
        if not isinstance(item, tuple):
            item = (item,)
        if len(item) > 3:
            raise IndexError(
                "SurfaceField supports indexing up to (axis0, entity, comp)"
            )
        out = self
        if len(item) >= 1 and not (
            isinstance(item[0], slice) and item[0] == slice(None)
        ):
            out = out.select_axis0(item[0])
        if len(item) >= 2 and not (
            isinstance(item[1], slice) and item[1] == slice(None)
        ):
            out = out.select_entities(item[1])
        if len(item) == 3 and not (
            isinstance(item[2], slice) and item[2] == slice(None)
        ):
            if out._tensor_rank != 1:
                raise IndexError(
                    "Component indexing is only supported for vector fields"
                )
            if isinstance(item[2], str):
                return out.component(item[2])
            if isinstance(item[2], (int, np.integer)):
                return out.component(int(item[2]))
            raise IndexError("Unsupported component selector")
        return out

    def component(self, key: str | int) -> SurfaceField:
        """Extract one scalar component from a vector ``SurfaceField``."""
        if self._tensor_rank != 1:
            raise ValueError(
                f"component() requires a vector field (tensor_rank=1), got rank {self._tensor_rank}"
            )
        if isinstance(key, str):
            if key not in self._component_names:
                raise KeyError(
                    f"Component {key!r} not found in {self._component_names}"
                )
            idx = self._component_names.index(key)
        elif isinstance(key, (int, np.integer)) and not isinstance(
            key, (bool, np.bool_)
        ):
            idx = int(key)
            if idx < 0:
                idx += self._component_shape[0]
            if idx < 0 or idx >= self._component_shape[0]:
                raise IndexError(f"Component index {key} out of range")
        else:
            raise TypeError("Component selector must be str or int")

        return SurfaceField(
            self._values[:, :, idx],
            geometry=self._geometry,
            association=self._association,
            unit=self._unit,
            axis0=self._axis0,
            axis0_domain=self._axis0_domain,
            axis0_name=self._axis0_name,
            epoch=(
                self._epoch if self.time_reference == TIME_REFERENCE_ABSOLUTE else None
            ),
            component_basis="scalar",
            field_kind="scalar",
            valid_mask=self._valid_mask,
            provenance=self._provenance.with_step(
                "component",
                mapping_kind="selected",
                component=self._component_names[idx],
            ),
            name=self.name,
            channel=self.channel,
            copy=True,
        )

    # ------------------------------------------------------------------
    # Alignment-checked arithmetic
    # ------------------------------------------------------------------

    def _validate_compatible(self, other: SurfaceField) -> None:
        if not isinstance(other, SurfaceField):
            raise TypeError(f"Expected SurfaceField, got {type(other).__name__}")
        if self.time_reference != other.time_reference:
            raise ValueError(
                f"Field time-reference mismatch: {self.time_reference} vs {other.time_reference}"
            )
        if self.time_reference == TIME_REFERENCE_ABSOLUTE and _format_epoch_value(
            self._epoch
        ) != _format_epoch_value(other._epoch):
            raise ValueError(f"Field epoch mismatch: {self._epoch} vs {other._epoch}")
        if self._axis0_domain != other._axis0_domain:
            raise ValueError(
                f"axis0_domain mismatch: {self._axis0_domain!r} vs {other._axis0_domain!r}"
            )
        if not _quantities_close(self._axis0, other._axis0):
            raise ValueError("axis0 coordinate mismatch between SurfaceFields")
        if self._association != other._association:
            raise ValueError(
                f"association mismatch: {self._association!r} vs {other._association!r}"
            )
        if not self._geometry.is_compatible(other._geometry):
            raise ValueError(
                "SurfaceGeometry mismatch between SurfaceFields (coordinates, "
                "entity ordering, or topology differ)"
            )
        if self._tensor_rank != other._tensor_rank:
            raise ValueError(
                f"tensor_rank mismatch: {self._tensor_rank} vs {other._tensor_rank}"
            )
        if self._component_basis != other._component_basis:
            raise ValueError(
                f"component_basis mismatch: {self._component_basis!r} vs {other._component_basis!r}"
            )
        if self._component_names != other._component_names:
            raise ValueError(
                f"component_names mismatch: {self._component_names} vs {other._component_names}"
            )

    def _binary_scalar_op(
        self, other: Any, op: str, *, reflected: bool = False
    ) -> SurfaceField:
        if isinstance(other, u.UnitBase):
            if op == "mul":
                new_unit = other * self._unit if reflected else self._unit * other
                return self._rebuild(unit=new_unit)
            if op == "truediv" and not reflected:
                return self._rebuild(unit=self._unit / other)
            if op == "truediv" and reflected:
                return self._rebuild(data=1.0 / self._values, unit=other / self._unit)
            return NotImplemented

        is_scalar = (
            isinstance(other, (int, float, complex, np.number))
            and not isinstance(other, (bool, np.bool_))
        ) or (isinstance(other, u.Quantity) and other.isscalar)
        if not is_scalar:
            return NotImplemented

        if isinstance(other, u.Quantity):
            o_val = other.value
            o_unit = other.unit
        else:
            o_val = other
            o_unit = u.dimensionless_unscaled

        if op in ("add", "sub"):
            if not self._unit.is_equivalent(o_unit):
                raise u.UnitConversionError(
                    f"Cannot add/subtract {o_unit} and {self._unit}"
                )
            o_converted = (
                float(other.to_value(self._unit))
                if isinstance(other, u.Quantity)
                else float(o_val)
            )
            if op == "add":
                new_vals = (
                    o_converted + self._values
                    if reflected
                    else self._values + o_converted
                )
            else:
                new_vals = (
                    o_converted - self._values
                    if reflected
                    else self._values - o_converted
                )
            return self._rebuild(data=new_vals, unit=self._unit)

        if op == "mul":
            new_vals = o_val * self._values if reflected else self._values * o_val
            new_unit = o_unit * self._unit if reflected else self._unit * o_unit
            return self._rebuild(data=new_vals, unit=new_unit)
        if op == "truediv":
            if reflected:
                new_vals = o_val / self._values
                new_unit = o_unit / self._unit
            else:
                new_vals = self._values / o_val
                new_unit = self._unit / o_unit
            return self._rebuild(data=new_vals, unit=new_unit)
        return NotImplemented

    def _binary_field_op(self, other: SurfaceField, op: str) -> SurfaceField:
        self._validate_compatible(other)
        combined_mask = self._valid_mask & other._valid_mask
        if op in ("add", "sub"):
            if not self._unit.is_equivalent(other._unit):
                raise u.UnitConversionError(
                    f"Cannot combine incompatible units {self._unit} and {other._unit}"
                )
            other_vals = u.Quantity(other._values, other._unit).to_value(self._unit)
            new_vals = (
                self._values + other_vals if op == "add" else self._values - other_vals
            )
            return self._rebuild(
                data=new_vals,
                unit=self._unit,
                valid_mask=combined_mask,
                provenance=self._provenance.with_step(op, mapping_kind="computed"),
            )
        if op == "mul":
            return self._rebuild(
                data=self._values * other._values,
                unit=self._unit * other._unit,
                valid_mask=combined_mask,
                provenance=self._provenance.with_step("mul", mapping_kind="computed"),
            )
        if op == "truediv":
            return self._rebuild(
                data=self._values / other._values,
                unit=self._unit / other._unit,
                valid_mask=combined_mask,
                provenance=self._provenance.with_step(
                    "truediv", mapping_kind="computed"
                ),
            )
        return NotImplemented

    def __pos__(self) -> SurfaceField:
        return self.copy()

    def __neg__(self) -> SurfaceField:
        return self._rebuild(data=-self._values)

    def __add__(self, other: Any) -> SurfaceField:
        if isinstance(other, SurfaceField):
            return self._binary_field_op(other, "add")
        return self._binary_scalar_op(other, "add", reflected=False)

    def __radd__(self, other: Any) -> SurfaceField:
        return self._binary_scalar_op(other, "add", reflected=True)

    def __sub__(self, other: Any) -> SurfaceField:
        if isinstance(other, SurfaceField):
            return self._binary_field_op(other, "sub")
        return self._binary_scalar_op(other, "sub", reflected=False)

    def __rsub__(self, other: Any) -> SurfaceField:
        return self._binary_scalar_op(other, "sub", reflected=True)

    def __mul__(self, other: Any) -> SurfaceField:
        if isinstance(other, SurfaceField):
            return self._binary_field_op(other, "mul")
        return self._binary_scalar_op(other, "mul", reflected=False)

    def __rmul__(self, other: Any) -> SurfaceField:
        return self._binary_scalar_op(other, "mul", reflected=True)

    def __truediv__(self, other: Any) -> SurfaceField:
        if isinstance(other, SurfaceField):
            return self._binary_field_op(other, "truediv")
        return self._binary_scalar_op(other, "truediv", reflected=False)

    def __rtruediv__(self, other: Any) -> SurfaceField:
        return self._binary_scalar_op(other, "truediv", reflected=True)

    # ------------------------------------------------------------------
    # Vector & geometry-dependent surface physics operations
    # ------------------------------------------------------------------

    def norm(self) -> SurfaceField:
        """Compute the Euclidean L2 norm across vector components."""
        if self._tensor_rank != 1:
            raise ValueError(
                f"norm() requires a vector field (tensor_rank=1), got rank {self._tensor_rank}"
            )
        mag = np.sqrt(np.sum(np.abs(self._values) ** 2, axis=-1))
        return SurfaceField(
            mag,
            geometry=self._geometry,
            association=self._association,
            unit=self._unit,
            axis0=self._axis0,
            axis0_domain=self._axis0_domain,
            axis0_name=self._axis0_name,
            epoch=(
                self._epoch if self.time_reference == TIME_REFERENCE_ABSOLUTE else None
            ),
            component_basis="scalar",
            field_kind="scalar",
            valid_mask=self._valid_mask,
            provenance=self._provenance.with_step("norm", mapping_kind="computed"),
            name=self.name,
            channel=self.channel,
        )

    def dot(self, other: SurfaceField) -> SurfaceField:
        """Compute the component-wise inner product with another vector field."""
        if not isinstance(other, SurfaceField):
            raise TypeError(f"dot expects SurfaceField, got {type(other).__name__}")
        if self._tensor_rank != 1 or other._tensor_rank != 1:
            raise ValueError("dot() is only defined for vector SurfaceFields")
        self._validate_compatible(other)
        prod = np.sum(self._values * other._values, axis=-1)
        return SurfaceField(
            prod,
            geometry=self._geometry,
            association=self._association,
            unit=self._unit * other._unit,
            axis0=self._axis0,
            axis0_domain=self._axis0_domain,
            axis0_name=self._axis0_name,
            epoch=(
                self._epoch if self.time_reference == TIME_REFERENCE_ABSOLUTE else None
            ),
            component_basis="scalar",
            field_kind="scalar",
            valid_mask=self._valid_mask & other._valid_mask,
            provenance=self._provenance.with_step("dot", mapping_kind="computed"),
            name=self.name,
            channel=self.channel,
        )

    def cross(self, other: SurfaceField) -> SurfaceField:
        """Compute the 3D Cartesian cross product with another vector field."""
        if not isinstance(other, SurfaceField):
            raise TypeError(f"cross expects SurfaceField, got {type(other).__name__}")
        if (
            self._tensor_rank != 1
            or other._tensor_rank != 1
            or self._component_basis != "cartesian"
            or other._component_basis != "cartesian"
            or self._component_shape != (3,)
        ):
            raise ValueError(
                "cross() is only defined for 3-component Cartesian vector fields"
            )
        self._validate_compatible(other)
        res = np.cross(self._values, other._values, axis=-1)
        return self._rebuild(
            data=res,
            unit=self._unit * other._unit,
            valid_mask=self._valid_mask & other._valid_mask,
            provenance=self._provenance.with_step("cross", mapping_kind="computed"),
        )

    def normal_component(
        self,
        *,
        location: Literal["face", "vertex"] | None = None,
        backend: Literal["numpy", "pyvista"] = "numpy",
    ) -> SurfaceField:
        """Project a 3D Cartesian vector field onto the surface unit normal.

        Uses a raw (non-conjugate) inner product ``F . n_hat`` so complex
        harmonic amplitudes remain linear.
        """
        if self._tensor_rank != 1 or self._component_basis != "cartesian":
            raise ValueError(
                "normal_component requires a Cartesian vector SurfaceField "
                f"(got tensor_rank={self._tensor_rank}, basis={self._component_basis!r})"
            )
        if self._component_shape != (3,) or self.embedding_dimension != 3:
            raise ValueError("normal_component requires 3D embedded vectors")
        if self._geometry.n_faces == 0:
            raise ValueError(
                "normal_component requires surface faces to define normals"
            )

        target_loc = location or ("face" if self._association == "face" else "vertex")
        if target_loc not in ("face", "vertex"):
            raise ValueError(f"location must be 'face' or 'vertex', got {target_loc!r}")

        if self._association == "face":
            if target_loc != "face":
                raise ValueError(
                    "Cannot evaluate face-associated field at vertices without explicit averaging"
                )
            normals = self._geometry.face_normals(backend=backend)
            proj = np.einsum("tek,ek->te", self._values, normals)
            return SurfaceField(
                proj,
                geometry=self._geometry,
                association="face",
                unit=self._unit,
                axis0=self._axis0,
                axis0_domain=self._axis0_domain,
                axis0_name=self._axis0_name,
                epoch=(
                    self._epoch
                    if self.time_reference == TIME_REFERENCE_ABSOLUTE
                    else None
                ),
                component_basis="scalar",
                field_kind="scalar",
                valid_mask=self._valid_mask,
                provenance=self._provenance.with_step(
                    "normal_component", mapping_kind="computed", backend=backend
                ),
            )

        if self._association != "vertex":
            raise ValueError(
                f"normal_component is supported for 'vertex' and 'face' associations, "
                f"got {self._association!r}"
            )

        if target_loc == "vertex":
            v_normals = self._geometry.vertex_normals()
            proj = np.einsum("tek,ek->te", self._values, v_normals)
            return SurfaceField(
                proj,
                geometry=self._geometry,
                association="vertex",
                unit=self._unit,
                axis0=self._axis0,
                axis0_domain=self._axis0_domain,
                axis0_name=self._axis0_name,
                epoch=(
                    self._epoch
                    if self.time_reference == TIME_REFERENCE_ABSOLUTE
                    else None
                ),
                component_basis="scalar",
                field_kind="scalar",
                valid_mask=self._valid_mask,
                provenance=self._provenance.with_step(
                    "normal_component", mapping_kind="computed", backend="numpy"
                ),
            )

        face_vals, face_mask = _average_vertex_to_face(
            self._values, self._valid_mask, self._geometry.cell_blocks
        )
        normals = self._geometry.face_normals(backend=backend)
        proj = np.einsum("tek,ek->te", face_vals, normals)
        return SurfaceField(
            proj,
            geometry=self._geometry,
            association="face",
            unit=self._unit,
            axis0=self._axis0,
            axis0_domain=self._axis0_domain,
            axis0_name=self._axis0_name,
            epoch=(
                self._epoch if self.time_reference == TIME_REFERENCE_ABSOLUTE else None
            ),
            component_basis="scalar",
            field_kind="scalar",
            valid_mask=face_mask,
            provenance=self._provenance.with_step(
                "normal_component", mapping_kind="computed", backend=backend
            ),
        )

    def surface_integral(
        self,
        *,
        oriented: bool | None = None,
        ignore_invalid: bool = False,
        weights: Any = None,
        backend: Literal["numpy", "pyvista"] = "numpy",
    ) -> u.Quantity:
        """Integrate the field over the surface geometry.

        - For **scalar fields**:
          - ``oriented=False`` (default): returns ``int_S f dS`` of shape ``(n0,)``.
          - ``oriented=True``: returns ``int_S f n_hat dS`` of shape ``(n0, 3)``.
        - For **3D Cartesian vector fields**:
          - ``oriented=True`` (default): returns the oriented surface flux
            ``int_S (F . n_hat) dS`` of shape ``(n0,)``. For vertex-associated
            vectors, face-mean values are dotted with exact face vector areas
            ``A_f = area_f * n_hat_f``, matching Gauss's divergence theorem to
            machine precision on polyhedral surfaces.
          - ``oriented=False``: returns the component-wise integral
            ``int_S F dS`` of shape ``(n0, n_comp)``.
        """
        if self._tensor_rank == 2:
            raise NotImplementedError(
                "surface_integral is not implemented for rank-2 tensor fields"
            )
        if not ignore_invalid and not np.all(self._valid_mask):
            raise ValueError(
                "SurfaceField contains invalid/masked samples; pass "
                "ignore_invalid=True to integrate over valid entities only"
            )

        area_unit = self._geometry.length_unit**2
        out_unit = self._unit * area_unit

        if self._tensor_rank == 1:
            do_oriented = True if oriented is None else bool(oriented)
            if do_oriented:
                return self._integrate_vector_flux(
                    ignore_invalid=ignore_invalid,
                    backend=backend,
                    out_unit=out_unit,
                )
            return self._integrate_unoriented(
                ignore_invalid=ignore_invalid,
                weights=weights,
                backend=backend,
                out_unit=out_unit,
            )

        do_oriented = False if oriented is None else bool(oriented)
        if do_oriented:
            return self._integrate_scalar_oriented(
                ignore_invalid=ignore_invalid,
                backend=backend,
                out_unit=out_unit,
            )
        return self._integrate_unoriented(
            ignore_invalid=ignore_invalid,
            weights=weights,
            backend=backend,
            out_unit=out_unit,
        )

    def _integrate_unoriented(
        self,
        *,
        ignore_invalid: bool,
        weights: Any,
        backend: Literal["numpy", "pyvista"],
        out_unit: u.UnitBase,
    ) -> u.Quantity:
        area_unit = self._geometry.length_unit**2
        if weights is not None:
            w_q = (
                weights.to(area_unit)
                if isinstance(weights, u.Quantity)
                else u.Quantity(weights, area_unit)
            )
            w_arr = np.asarray(w_q.value, dtype=np.float64)
            if w_arr.shape != (self.n_entities,):
                raise ValueError(
                    f"weights must have shape ({self.n_entities},), got {w_arr.shape}"
                )
        elif self._association == "face":
            if self._geometry.n_faces == 0:
                raise ValueError("Cannot integrate face field without surface faces")
            w_arr = np.asarray(
                self._geometry.face_areas(backend=backend).value, dtype=np.float64
            )
        elif self._association == "vertex":
            if self._geometry.n_faces == 0:
                raise ValueError("Cannot integrate vertex field without surface faces")
            w_arr = np.asarray(self._geometry.vertex_areas().value, dtype=np.float64)
        elif self._association == "quadrature":
            if self._geometry.quadrature_weights is None:
                raise ValueError(
                    "Integrating quadrature-associated field requires "
                    "geometry.quadrature_weights or explicit weights="
                )
            w_arr = np.asarray(
                self._geometry.quadrature_weights.to_value(area_unit), dtype=np.float64
            )
        else:
            raise ValueError(
                "Integrating point-associated field requires explicit weights="
            )

        vals = self._values
        if ignore_invalid:
            mask = self._valid_mask
            while mask.ndim < vals.ndim:
                mask = mask[..., None]
            vals = np.where(mask, vals, 0.0)

        if self._tensor_rank == 0:
            res = np.einsum("te,e->t", vals, w_arr)
        else:
            res = np.einsum("tek,e->tk", vals, w_arr)
        return u.Quantity(res, out_unit)

    def _integrate_scalar_oriented(
        self,
        *,
        ignore_invalid: bool,
        backend: Literal["numpy", "pyvista"],
        out_unit: u.UnitBase,
    ) -> u.Quantity:
        if self._geometry.n_faces == 0 or self.embedding_dimension != 3:
            raise ValueError(
                "Oriented scalar surface integral requires a 3D surface with faces"
            )
        if backend == "numpy":
            vec_areas = np.asarray(
                self._geometry.face_vector_areas().value, dtype=np.float64
            )
        else:
            f_areas = np.asarray(
                self._geometry.face_areas(backend=backend).value, dtype=np.float64
            )
            f_norms = self._geometry.face_normals(backend=backend)
            vec_areas = f_areas[:, None] * f_norms

        if self._association == "face":
            vals = self._values
            f_mask = self._valid_mask
        elif self._association == "vertex":
            vals, f_mask = _average_vertex_to_face(
                self._values, self._valid_mask, self._geometry.cell_blocks
            )
        else:
            raise ValueError(
                f"Oriented scalar surface integral requires 'face' or 'vertex' "
                f"association, got {self._association!r}"
            )
        if ignore_invalid:
            vals = np.where(f_mask, vals, 0.0)
        res = np.einsum("te,ek->tk", vals, vec_areas)
        return u.Quantity(res, out_unit)

    def _integrate_vector_flux(
        self,
        *,
        ignore_invalid: bool,
        backend: Literal["numpy", "pyvista"],
        out_unit: u.UnitBase,
    ) -> u.Quantity:
        if self._component_basis != "cartesian" or self._component_shape != (3,):
            raise ValueError(
                "Oriented vector flux integral requires a 3-component Cartesian vector field"
            )
        if self._geometry.n_faces == 0 or self.embedding_dimension != 3:
            raise ValueError(
                "Oriented vector flux integral requires a 3D surface with faces"
            )
        if backend == "numpy":
            vec_areas = np.asarray(
                self._geometry.face_vector_areas().value, dtype=np.float64
            )
        else:
            f_areas = np.asarray(
                self._geometry.face_areas(backend=backend).value, dtype=np.float64
            )
            f_norms = self._geometry.face_normals(backend=backend)
            vec_areas = f_areas[:, None] * f_norms

        if self._association == "face":
            vals = self._values
            f_mask = self._valid_mask
        elif self._association == "vertex":
            vals, f_mask = _average_vertex_to_face(
                self._values, self._valid_mask, self._geometry.cell_blocks
            )
        else:
            raise ValueError(
                f"Oriented vector flux integral requires 'face' or 'vertex' "
                f"association, got {self._association!r}"
            )
        if ignore_invalid:
            vals = np.where(f_mask[..., None], vals, 0.0)
        flux = np.einsum("tek,ek->t", vals, vec_areas)
        return u.Quantity(flux, out_unit)

    def flip_orientation(self) -> SurfaceField:
        """Return a copy whose surface geometry face windings are reversed."""
        if self._component_basis == "normal":
            raise ValueError(
                "Cannot flip geometry orientation ambiguously on a field stored "
                "in 'normal' basis; transform or negate explicitly"
            )
        return self._rebuild(
            geometry=self._geometry.flip_orientation(),
            provenance=self._provenance.with_step(
                "flip_orientation", mapping_kind="computed"
            ),
        )

    def with_consistent_orientation(self) -> SurfaceField:
        """Return a copy with topologically consistent face windings."""
        if self._component_basis == "normal":
            raise ValueError(
                "Cannot reorient faces automatically on a field stored in 'normal' basis"
            )
        return self._rebuild(
            geometry=self._geometry.with_consistent_orientation(),
            provenance=self._provenance.with_step(
                "with_consistent_orientation", mapping_kind="computed"
            ),
        )

    # ------------------------------------------------------------------
    # Explicit conversion to rectilinear ScalarField / VectorField
    # ------------------------------------------------------------------

    def to_rectilinear(self) -> ScalarField | VectorField:
        """Convert an exact planar-rectilinear structured surface to a regular Field.

        This method performs **no interpolation** and fails closed when the
        surface is curved, unstructured, or associated with points/quadrature.
        Use :meth:`sample_to_grid` for explicit spatial sampling/interpolation.
        """
        geom = self._geometry
        if not geom.is_planar_rectilinear:
            raise ValueError(
                "to_rectilinear() requires an exact planar axis-aligned structured "
                "surface (no silent flattening or interpolation is performed); "
                "use sample_to_grid(...) for explicit spatial sampling"
            )
        if self._association not in ("vertex", "face"):
            raise ValueError(
                f"to_rectilinear() supports 'vertex' or 'face' association, "
                f"got {self._association!r}"
            )
        if self._tensor_rank == 2:
            raise NotImplementedError(
                "to_rectilinear() is supported for scalar and vector fields"
            )

        assert geom.structured_shape is not None
        assert geom.structured_axes is not None
        assert geom.structured_plane is not None
        nu, nv = geom.structured_shape
        ax_u, ax_v = geom.structured_axes
        u_idx, v_idx, fixed_idx, fixed_coord = geom.structured_plane

        if self._association == "vertex":
            shape_uv = (nu, nv)
            eff_u, eff_v = ax_u, ax_v
        else:
            shape_uv = (nu - 1, nv - 1)
            eff_u = 0.5 * (ax_u[:-1] + ax_u[1:])
            eff_v = 0.5 * (ax_v[:-1] + ax_v[1:])

        fixed_axis = u.Quantity(
            [fixed_coord.to_value(geom.length_unit)], geom.length_unit
        )

        spatial_axes: list[u.Quantity] = [None, None, None]  # type: ignore[list-item]
        spatial_axes[u_idx] = eff_u
        spatial_axes[v_idx] = eff_v
        spatial_axes[fixed_idx] = fixed_axis

        prov = self._provenance.with_step(
            "to_rectilinear",
            mapping_kind="exact_one_to_one",
            interpolated=False,
            source_association=self._association,
        )

        mask_4d = self._valid_mask.reshape((self.n_axis0, shape_uv[0], shape_uv[1], 1))
        mask_4d = _permute_uv1_to_xyz(mask_4d, u_idx, v_idx, fixed_idx)

        def _make_sf(arr_2d: np.ndarray, comp_label: str | None = None) -> ScalarField:
            arr_uv1 = arr_2d.reshape((self.n_axis0, shape_uv[0], shape_uv[1], 1))
            arr_xyz = _permute_uv1_to_xyz(arr_uv1, u_idx, v_idx, fixed_idx)
            if not np.all(mask_4d):
                out_dtype = np.result_type(arr_xyz.dtype, np.float64)
                arr_xyz = np.where(mask_4d, arr_xyz.astype(out_dtype), np.nan)
            axis_names = (self._axis0_name, "x", "y", "z")
            sf = ScalarField(
                arr_xyz,
                unit=self._unit,
                axis0=self._axis0,
                axis1=spatial_axes[0],
                axis2=spatial_axes[1],
                axis3=spatial_axes[2],
                axis_names=axis_names,
                axis0_domain=self._axis0_domain,
                space_domain="real",
                epoch=(
                    self._epoch
                    if self.time_reference == TIME_REFERENCE_ABSOLUTE
                    else None
                ),
                name=comp_label or self.name,
                copy=True,
            )
            sf._gwex_provenance = prov.to_dict()
            sf._gwex_valid_mask = mask_4d.copy()
            return sf

        if self._tensor_rank == 0:
            return _make_sf(self._values)

        components = {
            cname: _make_sf(self._values[:, :, idx], comp_label=cname)
            for idx, cname in enumerate(self._component_names)
        }
        vf_basis: Literal["cartesian", "custom"] = (
            "cartesian" if self._component_basis == "cartesian" else "custom"
        )
        vf = VectorField(components, basis=vf_basis, validate=True)
        vf._gwex_provenance = prov.to_dict()  # type: ignore[attr-defined]
        vf._gwex_valid_mask = mask_4d.copy()  # type: ignore[attr-defined]
        return vf

    def sample_to_grid(
        self,
        axis1: Any,
        axis2: Any,
        axis3: Any = None,
        *,
        method: Literal["linear", "barycentric", "nearest"] = "linear",
        max_distance: Any = None,
        fill_value: float = np.nan,
        axis_names: Sequence[str] | None = None,
    ) -> ScalarField | VectorField:
        """Explicitly sample or interpolate this surface field onto a rectilinear grid.

        When ``axis3 is None``, the surface is sampled onto the 2D ``(axis1,
        axis2)`` plane (e.g. terrain/mirror height maps) and returned with a
        singleton length-1 ``axis3``. When ``axis3`` is supplied, sampling is
        performed in full 3D space via closest-point-on-surface projection and
        distance gating via ``max_distance``.

        Every conversion records ``mapping_kind='interpolated'`` and attaches
        ``_gwex_provenance`` and ``_gwex_valid_mask`` onto the resulting field.
        """
        if method not in ("linear", "barycentric", "nearest"):
            raise ValueError(
                f"method must be 'linear', 'barycentric', or 'nearest', got {method!r}"
            )
        if self._tensor_rank == 2:
            raise NotImplementedError(
                "sample_to_grid is supported for scalar and vector fields"
            )

        length_unit = self._geometry.length_unit
        ax1_q = coerce_1d_quantity(axis1, unit=length_unit)
        ax2_q = coerce_1d_quantity(axis2, unit=length_unit)
        is_2d_plane = axis3 is None
        if is_2d_plane:
            ax3_q = u.Quantity([0.0], length_unit)
        else:
            ax3_q = coerce_1d_quantity(axis3, unit=length_unit)

        max_dist_val: float | None
        if max_distance is None:
            max_dist_val = None
        else:
            _check_no_bool(max_distance, "max_distance")
            max_dist_val = (
                float(max_distance.to_value(length_unit))
                if isinstance(max_distance, u.Quantity)
                else float(max_distance)
            )
            if max_dist_val < 0.0 or not np.isfinite(max_dist_val):
                raise ValueError("max_distance must be finite and non-negative")

        nx, ny, nz = len(ax1_q), len(ax2_q), len(ax3_q)
        gx, gy, gz = np.meshgrid(ax1_q.value, ax2_q.value, ax3_q.value, indexing="ij")
        if is_2d_plane:
            query_2d = np.column_stack([gx[:, :, 0].ravel(), gy[:, :, 0].ravel()])
            sampled_vals, out_mask_flat = self._sample_2d_plane(
                query_2d,
                method=method,
                max_distance=max_dist_val,
                fill_value=fill_value,
            )
            out_shape_spatial = (nx, ny, 1)
        else:
            query_3d = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
            sampled_vals, out_mask_flat = self._sample_3d_surface(
                query_3d,
                method=method,
                max_distance=max_dist_val,
                fill_value=fill_value,
            )
            out_shape_spatial = (nx, ny, nz)

        out_mask_4d = out_mask_flat.reshape((self.n_axis0, *out_shape_spatial))
        valid_fraction = float(np.mean(out_mask_4d))

        prov = self._provenance.with_step(
            "sample_to_grid",
            mapping_kind="interpolated",
            interpolation_method=method,
            interpolated=True,
            source_association=self._association,
            plane_projection=is_2d_plane,
            max_distance=max_distance,
            valid_fraction=valid_fraction,
        )

        eff_axis_names = (
            tuple(str(n) for n in axis_names)
            if axis_names is not None
            else (self._axis0_name, "x", "y", "z")
        )

        def _pack_sf(
            arr_flat: np.ndarray, comp_label: str | None = None
        ) -> ScalarField:
            arr_4d = arr_flat.reshape((self.n_axis0, *out_shape_spatial))
            sf = ScalarField(
                arr_4d,
                unit=self._unit,
                axis0=self._axis0,
                axis1=ax1_q,
                axis2=ax2_q,
                axis3=ax3_q,
                axis_names=eff_axis_names,
                axis0_domain=self._axis0_domain,
                space_domain="real",
                epoch=(
                    self._epoch
                    if self.time_reference == TIME_REFERENCE_ABSOLUTE
                    else None
                ),
                name=comp_label or self.name,
                copy=True,
            )
            sf._gwex_provenance = prov.to_dict()
            sf._gwex_valid_mask = out_mask_4d.copy()
            return sf

        if self._tensor_rank == 0:
            return _pack_sf(sampled_vals)

        components = {
            cname: _pack_sf(sampled_vals[..., idx], comp_label=cname)
            for idx, cname in enumerate(self._component_names)
        }
        vf_basis: Literal["cartesian", "custom"] = (
            "cartesian" if self._component_basis == "cartesian" else "custom"
        )
        vf = VectorField(components, basis=vf_basis, validate=True)
        vf._gwex_provenance = prov.to_dict()  # type: ignore[attr-defined]
        vf._gwex_valid_mask = out_mask_4d.copy()  # type: ignore[attr-defined]
        return vf

    def _sample_2d_plane(
        self,
        query_2d: np.ndarray,
        *,
        method: str,
        max_distance: float | None,
        fill_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_q = len(query_2d)
        n0 = self.n_axis0
        tail_shape = self._component_shape
        out_dtype = np.result_type(self._values.dtype, np.float64)
        out = np.full((n0, n_q, *tail_shape), fill_value, dtype=out_dtype)
        out_mask = np.zeros((n0, n_q), dtype=bool)

        if method == "nearest" or self._geometry.n_faces == 0:
            coords = np.asarray(
                self._geometry.entity_coordinates(self._association).value[:, :2],
                dtype=np.float64,
            )
            if method != "nearest":
                from scipy.interpolate import LinearNDInterpolator

                for t in range(n0):
                    v_mask = self._valid_mask[t]
                    valid_pts = coords[v_mask]
                    if len(valid_pts) < 3:
                        continue
                    interp = LinearNDInterpolator(
                        valid_pts, self._values[t, v_mask], fill_value=np.nan
                    )
                    res_t = interp(query_2d)
                    ok = np.all(np.isfinite(res_t), axis=tuple(range(1, res_t.ndim)))
                    out[t, ok] = res_t[ok]
                    out_mask[t] = ok
                return out, out_mask

            dists, nn_idx = _nearest_neighbor_query(coords, query_2d)
            in_range = (
                dists <= max_distance
                if max_distance is not None
                else np.ones(n_q, dtype=bool)
            )
            for t in range(n0):
                ok = in_range & self._valid_mask[t, nn_idx]
                out[t, ok] = self._values[t, nn_idx[ok]]
                out_mask[t] = ok
            return out, out_mask

        verts_2d = np.asarray(self._geometry.vertices.value[:, :2], dtype=np.float64)
        tris, parent_faces = _triangulate_blocks_indices(self._geometry.cell_blocks)
        hit_tri, weights = _locate_points_in_2d_triangles(query_2d, verts_2d, tris)
        inside = hit_tri >= 0

        if self._association == "face":
            p_faces = parent_faces[hit_tri[inside]]
            for t in range(n0):
                ok_inside = self._valid_mask[t, p_faces]
                ok = np.zeros(n_q, dtype=bool)
                ok[inside] = ok_inside
                out[t, ok] = self._values[t, p_faces[ok_inside]]
                out_mask[t] = ok
            return out, out_mask

        if self._association == "vertex":
            tri_hits = tris[hit_tri[inside]]
            w_hits = weights[inside]
            for t in range(n0):
                v_ok = np.all(self._valid_mask[t, tri_hits], axis=1)
                ok = np.zeros(n_q, dtype=bool)
                ok[inside] = v_ok
                if np.any(v_ok):
                    th = tri_hits[v_ok]
                    wh = w_hits[v_ok]
                    if self._tensor_rank == 1:
                        wh = wh[:, :, None]
                    interp_v = (
                        wh[:, 0] * self._values[t, th[:, 0]]
                        + wh[:, 1] * self._values[t, th[:, 1]]
                        + wh[:, 2] * self._values[t, th[:, 2]]
                    )
                    out[t, ok] = interp_v
                out_mask[t] = ok
            return out, out_mask

        coords = np.asarray(
            self._geometry.entity_coordinates(self._association).value[:, :2],
            dtype=np.float64,
        )
        dists, nn_idx = _nearest_neighbor_query(coords, query_2d)
        in_range = (
            dists <= max_distance
            if max_distance is not None
            else np.ones(n_q, dtype=bool)
        )
        for t in range(n0):
            ok = in_range & self._valid_mask[t, nn_idx]
            out[t, ok] = self._values[t, nn_idx[ok]]
            out_mask[t] = ok
        return out, out_mask

    def _sample_3d_surface(
        self,
        query_3d: np.ndarray,
        *,
        method: str,
        max_distance: float | None,
        fill_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_q = len(query_3d)
        n0 = self.n_axis0
        tail_shape = self._component_shape
        out_dtype = np.result_type(self._values.dtype, np.float64)
        out = np.full((n0, n_q, *tail_shape), fill_value, dtype=out_dtype)
        out_mask = np.zeros((n0, n_q), dtype=bool)

        if method == "nearest" or self._geometry.n_faces == 0:
            if method != "nearest":
                raise ValueError(
                    "Linear/barycentric 3D surface sampling requires surface faces"
                )
            coords = np.asarray(
                self._geometry.entity_coordinates(self._association).value,
                dtype=np.float64,
            )
            if coords.shape[1] == 2:
                coords = np.column_stack([coords, np.zeros(len(coords))])
            dists, nn_idx = _nearest_neighbor_query(coords, query_3d)
            in_range = (
                dists <= max_distance
                if max_distance is not None
                else np.ones(n_q, dtype=bool)
            )
            for t in range(n0):
                ok = in_range & self._valid_mask[t, nn_idx]
                out[t, ok] = self._values[t, nn_idx[ok]]
                out_mask[t] = ok
            return out, out_mask

        verts_3d = np.asarray(self._geometry.vertices.value, dtype=np.float64)
        if verts_3d.shape[1] == 2:
            verts_3d = np.column_stack([verts_3d, np.zeros(len(verts_3d))])
        tris, parent_faces = _triangulate_blocks_indices(self._geometry.cell_blocks)
        closest_tri, bary, dists = _closest_point_on_3d_triangles(
            query_3d, verts_3d, tris
        )
        in_range = (
            dists <= max_distance
            if max_distance is not None
            else np.ones(n_q, dtype=bool)
        )

        if self._association == "face":
            p_faces = parent_faces[closest_tri]
            for t in range(n0):
                ok = in_range & self._valid_mask[t, p_faces]
                out[t, ok] = self._values[t, p_faces[ok]]
                out_mask[t] = ok
            return out, out_mask

        if self._association == "vertex":
            tri_hits = tris[closest_tri]
            for t in range(n0):
                ok = in_range & np.all(self._valid_mask[t, tri_hits], axis=1)
                if np.any(ok):
                    th = tri_hits[ok]
                    bh = bary[ok]
                    if self._tensor_rank == 1:
                        bh = bh[:, :, None]
                    out[t, ok] = (
                        bh[:, 0] * self._values[t, th[:, 0]]
                        + bh[:, 1] * self._values[t, th[:, 1]]
                        + bh[:, 2] * self._values[t, th[:, 2]]
                    )
                out_mask[t] = ok
            return out, out_mask

        raise ValueError(
            f"3D barycentric sampling is supported for 'vertex' and 'face' associations, "
            f"got {self._association!r}"
        )

    # ------------------------------------------------------------------
    # Optional meshio / pyvista interoperability
    # ------------------------------------------------------------------

    def to_meshio(
        self,
        field_name: str = "field",
        *,
        axis0_index: int | None = None,
    ) -> Any:
        """Export a snapshot of this field to a ``meshio.Mesh`` (optional backend)."""
        idx = self._resolve_export_axis0_index(axis0_index)
        mesh = self._geometry.to_meshio()
        snapshot = self._values[idx]
        if self._association in ("vertex", "point"):
            mesh.point_data[field_name] = np.asarray(snapshot)
            mesh.point_data[f"{field_name}_valid_mask"] = np.asarray(
                self._valid_mask[idx], dtype=np.uint8
            )
            return mesh
        if self._association == "face":
            blocks_data: list[np.ndarray] = []
            blocks_mask: list[np.ndarray] = []
            offset = 0
            for _, conn in self._geometry.cell_blocks:
                nf = conn.shape[0]
                blocks_data.append(np.asarray(snapshot[offset : offset + nf]))
                blocks_mask.append(
                    np.asarray(
                        self._valid_mask[idx, offset : offset + nf], dtype=np.uint8
                    )
                )
                offset += nf
            mesh.cell_data[field_name] = blocks_data
            mesh.cell_data[f"{field_name}_valid_mask"] = blocks_mask
            return mesh
        raise ValueError(
            f"to_meshio does not support association={self._association!r}"
        )

    @classmethod
    def from_meshio(
        cls,
        mesh_or_path: Any,
        field_name: str,
        *,
        association: SurfaceAssociation | None = None,
        unit: u.UnitBase | str | None = None,
        geometry_unit: u.UnitBase | str = u.m,
        axis0: Any = None,
        axis0_domain: Literal["time", "frequency"] = "time",
        epoch: Any = TIME_INPUT_UNSET,
        component_basis: ComponentBasis | None = None,
        component_names: Sequence[str] | None = None,
        allow_point_cloud: bool = False,
    ) -> SurfaceField:
        """Import a :class:`SurfaceField` from a ``meshio.Mesh`` or file path."""
        meshio = require_optional("meshio")
        mesh = (
            meshio.read(str(mesh_or_path))
            if isinstance(mesh_or_path, (str, Path))
            else mesh_or_path
        )
        geom = SurfaceGeometry.from_meshio(
            mesh, unit=geometry_unit, allow_point_cloud=allow_point_cloud
        )
        in_point = field_name in getattr(mesh, "point_data", {})
        in_cell = field_name in getattr(mesh, "cell_data", {})

        if association is None:
            if in_point and in_cell:
                raise ValueError(
                    f"Field {field_name!r} exists in both point_data and cell_data; "
                    "specify association='vertex' or association='face'"
                )
            if in_point:
                association = "vertex"
            elif in_cell:
                association = "face"
            else:
                raise KeyError(
                    f"Field {field_name!r} not found in meshio point_data or cell_data"
                )

        if association in ("vertex", "point"):
            if not in_point:
                raise KeyError(f"Field {field_name!r} not found in meshio.point_data")
            raw_vals = np.asarray(mesh.point_data[field_name])
            mask_key = f"{field_name}_valid_mask"
            raw_mask = (
                np.asarray(mesh.point_data[mask_key], dtype=bool)
                if mask_key in mesh.point_data
                else None
            )
        elif association == "face":
            if not in_cell:
                raise KeyError(f"Field {field_name!r} not found in meshio.cell_data")
            surf_blocks = [
                np.asarray(arr)
                for block, arr in zip(
                    mesh.cells, mesh.cell_data[field_name], strict=True
                )
                if block.type in ("triangle", "quad")
            ]
            raw_vals = np.concatenate(surf_blocks, axis=0)
            mask_key = f"{field_name}_valid_mask"
            if mask_key in mesh.cell_data:
                m_blocks = [
                    np.asarray(arr, dtype=bool)
                    for block, arr in zip(
                        mesh.cells, mesh.cell_data[mask_key], strict=True
                    )
                    if block.type in ("triangle", "quad")
                ]
                raw_mask = np.concatenate(m_blocks, axis=0)
            else:
                raw_mask = None
        else:
            raise ValueError(
                f"from_meshio supports 'vertex', 'face', or 'point', got {association!r}"
            )

        return cls(
            raw_vals,
            geometry=geom,
            association=association,
            unit=unit,
            axis0=axis0,
            axis0_domain=axis0_domain,
            epoch=epoch,
            component_basis=component_basis,
            component_names=component_names,
            valid_mask=raw_mask,
            provenance=SurfaceProvenance(
                source="meshio",
                operation="from_meshio",
                mapping_kind="native",
                backend="meshio",
                details={"field_name": field_name},
            ),
            name=field_name,
        )

    def to_pyvista(
        self,
        field_name: str = "field",
        *,
        axis0_index: int | None = None,
    ) -> Any:
        """Export a snapshot of this field to a ``pyvista.PolyData`` (optional backend)."""
        idx = self._resolve_export_axis0_index(axis0_index)
        pd = self._geometry.to_pyvista()
        snapshot = np.asarray(self._values[idx])
        mask_arr = np.asarray(self._valid_mask[idx], dtype=np.uint8)
        if self._association in ("vertex", "point"):
            pd.point_data[field_name] = snapshot
            pd.point_data[f"{field_name}_valid_mask"] = mask_arr
            return pd
        if self._association == "face":
            pd.cell_data[field_name] = snapshot
            pd.cell_data[f"{field_name}_valid_mask"] = mask_arr
            return pd
        raise ValueError(
            f"to_pyvista does not support association={self._association!r}"
        )

    @classmethod
    def from_pyvista(
        cls,
        dataset_or_path: Any,
        field_name: str,
        *,
        association: SurfaceAssociation | None = None,
        unit: u.UnitBase | str | None = None,
        geometry_unit: u.UnitBase | str = u.m,
        axis0: Any = None,
        axis0_domain: Literal["time", "frequency"] = "time",
        epoch: Any = TIME_INPUT_UNSET,
        component_basis: ComponentBasis | None = None,
        component_names: Sequence[str] | None = None,
        allow_point_cloud: bool = False,
    ) -> SurfaceField:
        """Import a :class:`SurfaceField` from a PyVista dataset or file path."""
        pv = require_optional("pyvista")
        ds = (
            pv.read(str(dataset_or_path))
            if isinstance(dataset_or_path, (str, Path))
            else dataset_or_path
        )
        geom = SurfaceGeometry.from_pyvista(
            ds, unit=geometry_unit, allow_point_cloud=allow_point_cloud
        )
        in_point = field_name in ds.point_data
        in_cell = field_name in ds.cell_data

        if association is None:
            if in_point and in_cell:
                raise ValueError(
                    f"Field {field_name!r} exists in both point_data and cell_data; "
                    "specify association='vertex' or association='face'"
                )
            if in_point:
                association = "vertex"
            elif in_cell:
                association = "face"
            else:
                raise KeyError(
                    f"Field {field_name!r} not found in PyVista point_data or cell_data"
                )

        if association in ("vertex", "point"):
            if not in_point:
                raise KeyError(f"Field {field_name!r} not found in PyVista point_data")
            raw_vals = np.asarray(ds.point_data[field_name])
            mask_key = f"{field_name}_valid_mask"
            raw_mask = (
                np.asarray(ds.point_data[mask_key], dtype=bool)
                if mask_key in ds.point_data
                else None
            )
        elif association == "face":
            if not in_cell:
                raise KeyError(f"Field {field_name!r} not found in PyVista cell_data")
            raw_vals = np.asarray(ds.cell_data[field_name])
            mask_key = f"{field_name}_valid_mask"
            raw_mask = (
                np.asarray(ds.cell_data[mask_key], dtype=bool)
                if mask_key in ds.cell_data
                else None
            )
        else:
            raise ValueError(
                f"from_pyvista supports 'vertex', 'face', or 'point', got {association!r}"
            )

        return cls(
            raw_vals,
            geometry=geom,
            association=association,
            unit=unit,
            axis0=axis0,
            axis0_domain=axis0_domain,
            epoch=epoch,
            component_basis=component_basis,
            component_names=component_names,
            valid_mask=raw_mask,
            provenance=SurfaceProvenance(
                source="pyvista",
                operation="from_pyvista",
                mapping_kind="native",
                backend="pyvista",
                details={"field_name": field_name},
            ),
            name=field_name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_export_axis0_index(self, axis0_index: int | None) -> int:
        if axis0_index is None:
            if self.n_axis0 != 1:
                raise ValueError(
                    f"SurfaceField has {self.n_axis0} axis0 samples; pass "
                    "axis0_index=... to export a single snapshot"
                )
            return 0
        idx = int(axis0_index)
        if idx < 0:
            idx += self.n_axis0
        if idx < 0 or idx >= self.n_axis0:
            raise IndexError(
                f"axis0_index {axis0_index} out of bounds for size {self.n_axis0}"
            )
        return idx

    def _rebuild(self, **overrides: Any) -> SurfaceField:
        epoch_arg = (
            overrides["epoch"]
            if "epoch" in overrides
            else (
                self._epoch if self.time_reference == TIME_REFERENCE_ABSOLUTE else None
            )
        )
        return SurfaceField(
            overrides.get("data", self._values),
            geometry=overrides.get("geometry", self._geometry),
            association=overrides.get("association", self._association),
            unit=overrides.get("unit", self._unit),
            axis0=overrides.get("axis0", self._axis0),
            axis0_domain=overrides.get("axis0_domain", self._axis0_domain),
            axis0_name=overrides.get("axis0_name", self._axis0_name),
            epoch=epoch_arg,
            component_basis=overrides.get("component_basis", self._component_basis),
            component_names=overrides.get("component_names", self._component_names),
            field_kind=overrides.get("field_kind", _kind_from_rank(self._tensor_rank)),
            valid_mask=overrides.get("valid_mask", self._valid_mask),
            provenance=overrides.get("provenance", self._provenance),
            name=overrides.get("name", self.name),
            channel=overrides.get("channel", self.channel),
            copy=True,
        )


# ---------------------------------------------------------------------------
# Internal normalization, projection, and sampling kernels
# ---------------------------------------------------------------------------


def _kind_from_rank(rank: int) -> Literal["scalar", "vector", "tensor"]:
    if rank == 0:
        return "scalar"
    if rank == 1:
        return "vector"
    return "tensor"


def _normalize_surface_data_shape(
    raw: np.ndarray,
    *,
    n_entities: int,
    axis0: Any,
    field_kind: str,
    component_basis: str | None,
    component_names: Sequence[str] | None,
) -> tuple[Literal["scalar", "vector", "tensor"], np.ndarray]:
    if raw.ndim == 0 or raw.ndim > 4:
        raise ValueError(
            f"SurfaceField data must have 1 to 4 dimensions, got {raw.ndim}D"
        )

    axis0_len = None
    if axis0 is not None:
        axis0_len = len(coerce_1d_quantity(axis0))

    if field_kind == "auto":
        if raw.ndim == 1:
            field_kind = "scalar"
        elif raw.ndim == 2:
            wants_vector = component_names is not None or component_basis in (
                "cartesian",
                "tangent",
                "normal",
                "custom",
            )
            if wants_vector:
                field_kind = "vector"
            elif axis0_len is not None:
                if raw.shape == (axis0_len, n_entities):
                    field_kind = "scalar"
                elif axis0_len == 1 and raw.shape[0] == n_entities:
                    field_kind = "vector"
                else:
                    raise ValueError(
                        f"2D data shape {raw.shape} is incompatible with "
                        f"axis0 length {axis0_len} and n_entities={n_entities}"
                    )
            elif raw.shape[1] == n_entities:
                field_kind = "scalar"
            elif raw.shape[0] == n_entities and raw.shape[1] in (1, 2, 3):
                field_kind = "vector"
            else:
                raise ValueError(
                    f"Cannot infer 2D data layout for shape {raw.shape} and n_entities={n_entities}"
                )
        elif raw.ndim == 3:
            if (
                (axis0_len is None or axis0_len == 1)
                and raw.shape[0] == n_entities
                and raw.shape[1] == raw.shape[2]
            ):
                field_kind = "tensor"
            else:
                field_kind = "vector"
        else:
            field_kind = "tensor"

    if field_kind == "scalar":
        if raw.ndim == 1:
            if raw.shape[0] != n_entities:
                raise ValueError(
                    f"1D scalar data length ({raw.shape[0]}) must match n_entities ({n_entities})"
                )
            return "scalar", raw[np.newaxis, :]
        if raw.ndim == 2:
            if raw.shape[1] != n_entities:
                raise ValueError(
                    f"Scalar data axis 1 size ({raw.shape[1]}) must match n_entities ({n_entities})"
                )
            return "scalar", raw
        raise ValueError(f"Scalar field expects 1D or 2D data, got {raw.ndim}D")

    if field_kind == "vector":
        if raw.ndim == 2:
            if raw.shape[0] != n_entities:
                raise ValueError(
                    f"2D vector data axis 0 size ({raw.shape[0]}) must match n_entities ({n_entities})"
                )
            return "vector", raw[np.newaxis, :, :]
        if raw.ndim == 3:
            if raw.shape[1] != n_entities:
                raise ValueError(
                    f"3D vector data axis 1 size ({raw.shape[1]}) must match n_entities ({n_entities})"
                )
            return "vector", raw
        raise ValueError(f"Vector field expects 2D or 3D data, got {raw.ndim}D")

    if raw.ndim == 3:
        if raw.shape[0] != n_entities:
            raise ValueError(
                f"3D tensor data axis 0 size ({raw.shape[0]}) must match n_entities ({n_entities})"
            )
        return "tensor", raw[np.newaxis, :, :, :]
    if raw.ndim == 4:
        if raw.shape[1] != n_entities:
            raise ValueError(
                f"4D tensor data axis 1 size ({raw.shape[1]}) must match n_entities ({n_entities})"
            )
        return "tensor", raw
    raise ValueError(f"Tensor field expects 3D or 4D data, got {raw.ndim}D")


def _resolve_component_metadata(
    *,
    resolved_kind: Literal["scalar", "vector", "tensor"],
    comp_shape: tuple[int, ...],
    embedding_dim: int,
    topo_dim: int,
    component_basis: ComponentBasis | None,
    component_names: Sequence[str] | None,
) -> tuple[ComponentBasis, tuple[str, ...]]:
    if component_basis is not None and component_basis not in _COMPONENT_BASES:
        raise ValueError(
            f"component_basis must be one of {_COMPONENT_BASES}, got {component_basis!r}"
        )

    if resolved_kind == "scalar":
        if component_basis is not None and component_basis != "scalar":
            raise ValueError(
                f"Scalar field cannot have component_basis={component_basis!r}"
            )
        if component_names:
            raise ValueError("Scalar field must not specify component_names")
        return "scalar", ()

    if resolved_kind == "vector":
        n_comp = comp_shape[0]
        if n_comp < 1:
            raise ValueError("Vector field must have at least 1 component")
        if component_basis is None:
            if n_comp == embedding_dim:
                basis: ComponentBasis = "cartesian"
            elif n_comp == topo_dim:
                basis = "tangent"
            else:
                basis = "custom"
        else:
            basis = component_basis
            if basis == "scalar":
                raise ValueError("Vector field cannot have component_basis='scalar'")
            if basis == "cartesian" and n_comp != embedding_dim:
                raise ValueError(
                    f"Cartesian vector on {embedding_dim}D surface requires "
                    f"{embedding_dim} components, got {n_comp}"
                )
            if basis == "tangent" and n_comp != topo_dim:
                raise ValueError(
                    f"Tangent vector on {topo_dim}D surface requires {topo_dim} "
                    f"components, got {n_comp}"
                )
            if basis == "normal" and n_comp != 1:
                raise ValueError(
                    f"Normal-component vector requires 1 component, got {n_comp}"
                )

        if component_names is None:
            if basis == "cartesian":
                names = ("x", "y", "z")[:n_comp]
            elif basis == "tangent":
                names = ("t1", "t2")[:n_comp]
            elif basis == "normal":
                names = ("n",)
            else:
                names = tuple(f"c{i}" for i in range(n_comp))
        else:
            names = tuple(str(n) for n in component_names)
            if len(names) != n_comp or len(set(names)) != n_comp:
                raise ValueError(
                    f"component_names must contain {n_comp} unique names, got {names}"
                )
        return basis, names

    d1, d2 = comp_shape
    if component_basis is None:
        if (d1, d2) == (embedding_dim, embedding_dim):
            t_basis: ComponentBasis = "cartesian"
        elif (d1, d2) == (topo_dim, topo_dim):
            t_basis = "tangent"
        else:
            t_basis = "custom"
    else:
        t_basis = component_basis
        if t_basis == "scalar":
            raise ValueError("Tensor field cannot have component_basis='scalar'")
        if t_basis == "cartesian" and (d1, d2) != (embedding_dim, embedding_dim):
            raise ValueError(
                f"Cartesian rank-2 tensor on {embedding_dim}D surface requires shape "
                f"({embedding_dim}, {embedding_dim}), got {(d1, d2)}"
            )
        if t_basis == "tangent" and (d1, d2) != (topo_dim, topo_dim):
            raise ValueError(
                f"Tangent rank-2 tensor on {topo_dim}D surface requires shape "
                f"({topo_dim}, {topo_dim}), got {(d1, d2)}"
            )
    if component_names is None:
        if t_basis == "cartesian":
            t_names = ("x", "y", "z")[:d1]
        elif t_basis == "tangent":
            t_names = ("t1", "t2")[:d1]
        else:
            t_names = tuple(f"c{i}" for i in range(max(d1, d2)))
    else:
        t_names = tuple(str(n) for n in component_names)
    return t_basis, t_names


def _normalize_valid_mask(
    valid_mask: Any,
    *,
    n0: int,
    n_entities: int,
    data_shape: tuple[int, ...],
) -> np.ndarray:
    if valid_mask is None:
        return _freeze_array(np.ones((n0, n_entities), dtype=bool))
    raw = np.asarray(valid_mask, dtype=bool)
    if raw.shape == (n_entities,):
        arr = np.broadcast_to(raw[np.newaxis, :], (n0, n_entities))
    elif raw.shape == (n0, n_entities):
        arr = raw
    elif raw.shape == data_shape and raw.ndim > 2:
        arr = np.all(raw, axis=tuple(range(2, raw.ndim)))
    else:
        raise ValueError(
            f"valid_mask must have shape ({n_entities},), ({n0}, {n_entities}), "
            f"or {data_shape}, got {raw.shape}"
        )
    return _freeze_array(arr)


def _average_vertex_to_face(
    values: np.ndarray,
    valid_mask: np.ndarray,
    cell_blocks: Sequence[tuple[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    n0 = values.shape[0]
    tail = values.shape[2:]
    face_vals_parts: list[np.ndarray] = []
    face_mask_parts: list[np.ndarray] = []
    for _, conn in cell_blocks:
        nf, _ = conn.shape
        if nf == 0:
            continue
        gathered = values[:, conn]
        face_vals_parts.append(np.mean(gathered, axis=2))
        mask_gathered = valid_mask[:, conn]
        face_mask_parts.append(np.all(mask_gathered, axis=2))
    if not face_vals_parts:
        return (
            np.empty((n0, 0, *tail), dtype=values.dtype),
            np.empty((n0, 0), dtype=bool),
        )
    return (
        np.concatenate(face_vals_parts, axis=1),
        np.concatenate(face_mask_parts, axis=1),
    )


def _permute_uv1_to_xyz(
    arr_uv1: np.ndarray, u_idx: int, v_idx: int, fixed_idx: int
) -> np.ndarray:
    src_of_spatial = [0, 0, 0]
    src_of_spatial[u_idx] = 1
    src_of_spatial[v_idx] = 2
    src_of_spatial[fixed_idx] = 3
    return np.transpose(arr_uv1, (0, *src_of_spatial))


def _nearest_neighbor_query(
    source_coords: np.ndarray, query_coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if len(source_coords) == 0:
        return (
            np.full(len(query_coords), np.inf, dtype=np.float64),
            np.zeros(len(query_coords), dtype=np.intp),
        )
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(source_coords)
        dists, idx = tree.query(query_coords)
        return np.asarray(dists, dtype=np.float64), np.asarray(idx, dtype=np.intp)
    except ImportError:
        diff = query_coords[:, None, :] - source_coords[None, :, :]
        dists_all = np.linalg.norm(diff, axis=-1)
        idx = np.argmin(dists_all, axis=1).astype(np.intp)
        dists = dists_all[np.arange(len(query_coords)), idx]
        return dists, idx


def _locate_points_in_2d_triangles(
    query: np.ndarray,
    verts_2d: np.ndarray,
    tris: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate 2D query points in surface triangles, returning ``(tri_idx, bary)``."""
    n_q = len(query)
    hit_tri = np.full(n_q, -1, dtype=np.intp)
    bary = np.zeros((n_q, 3), dtype=np.float64)
    if len(tris) == 0 or n_q == 0:
        return hit_tri, bary

    a = verts_2d[tris[:, 0]]
    b = verts_2d[tris[:, 1]]
    c = verts_2d[tris[:, 2]]

    v0 = b - a
    v1 = c - a
    denom = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]

    tri_min_x = np.minimum(np.minimum(a[:, 0], b[:, 0]), c[:, 0]) - 1e-12
    tri_max_x = np.maximum(np.maximum(a[:, 0], b[:, 0]), c[:, 0]) + 1e-12
    tri_min_y = np.minimum(np.minimum(a[:, 1], b[:, 1]), c[:, 1]) - 1e-12
    tri_max_y = np.maximum(np.maximum(a[:, 1], b[:, 1]), c[:, 1]) + 1e-12

    qx = query[:, 0]
    qy = query[:, 1]
    tol = 1e-12

    for t_idx in range(len(tris)):
        d = denom[t_idx]
        if abs(d) <= 1e-24:
            continue
        cand = (
            (hit_tri < 0)
            & (qx >= tri_min_x[t_idx])
            & (qx <= tri_max_x[t_idx])
            & (qy >= tri_min_y[t_idx])
            & (qy <= tri_max_y[t_idx])
        )
        if not np.any(cand):
            continue
        q_sub = query[cand]
        v2 = q_sub - a[t_idx]
        u_coord = (v2[:, 0] * v1[t_idx, 1] - v2[:, 1] * v1[t_idx, 0]) / d
        v_coord = (v0[t_idx, 0] * v2[:, 1] - v0[t_idx, 1] * v2[:, 0]) / d
        w_coord = 1.0 - u_coord - v_coord
        inside = (w_coord >= -tol) & (u_coord >= -tol) & (v_coord >= -tol)
        if np.any(inside):
            cand_indices = np.nonzero(cand)[0][inside]
            hit_tri[cand_indices] = t_idx
            bary[cand_indices, 0] = np.clip(w_coord[inside], 0.0, 1.0)
            bary[cand_indices, 1] = np.clip(u_coord[inside], 0.0, 1.0)
            bary[cand_indices, 2] = np.clip(v_coord[inside], 0.0, 1.0)
            bary[cand_indices] /= np.sum(bary[cand_indices], axis=1, keepdims=True)
    return hit_tri, bary


def _closest_point_on_3d_triangles(
    query: np.ndarray,
    verts_3d: np.ndarray,
    tris: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute exact closest triangle, barycentric coords, and distance in 3D.

    Implements Ericson's Voronoi-region closest-point-on-triangle projection
    vectorized across triangles per query chunk.
    """
    n_q = len(query)
    if len(tris) == 0 or n_q == 0:
        return (
            np.zeros(n_q, dtype=np.intp),
            np.zeros((n_q, 3), dtype=np.float64),
            np.full(n_q, np.inf, dtype=np.float64),
        )

    a = verts_3d[tris[:, 0]]
    b = verts_3d[tris[:, 1]]
    c = verts_3d[tris[:, 2]]
    ab = b - a
    ac = c - a

    best_tri = np.zeros(n_q, dtype=np.intp)
    best_bary = np.zeros((n_q, 3), dtype=np.float64)
    best_dist_sq = np.full(n_q, np.inf, dtype=np.float64)

    chunk_size = max(1, min(n_q, 256))
    for start in range(0, n_q, chunk_size):
        stop = min(n_q, start + chunk_size)
        p = query[start:stop, None, :]

        ap = p - a[None, :, :]
        d1 = np.sum(ab[None, :, :] * ap, axis=-1)
        d2 = np.sum(ac[None, :, :] * ap, axis=-1)

        bp = p - b[None, :, :]
        d3 = np.sum(ab[None, :, :] * bp, axis=-1)
        d4 = np.sum(ac[None, :, :] * bp, axis=-1)

        cp = p - c[None, :, :]
        d5 = np.sum(ab[None, :, :] * cp, axis=-1)
        d6 = np.sum(ac[None, :, :] * cp, axis=-1)

        w0 = np.zeros_like(d1)
        w1 = np.zeros_like(d1)
        w2 = np.zeros_like(d1)

        reg_a = (d1 <= 0.0) & (d2 <= 0.0)
        w0[reg_a] = 1.0

        reg_b = (~reg_a) & (d3 >= 0.0) & (d4 <= d3)
        w1[reg_b] = 1.0

        vc = d1 * d4 - d3 * d2
        reg_ab = (~reg_a) & (~reg_b) & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
        denom_ab = d1 - d3
        v_ab = np.where(
            reg_ab & (denom_ab != 0.0),
            d1 / np.where(denom_ab != 0.0, denom_ab, 1.0),
            0.0,
        )
        w0[reg_ab] = 1.0 - v_ab[reg_ab]
        w1[reg_ab] = v_ab[reg_ab]

        reg_c = (~reg_a) & (~reg_b) & (~reg_ab) & (d6 >= 0.0) & (d5 <= d6)
        w2[reg_c] = 1.0

        vb = d5 * d2 - d1 * d6
        reg_ac = (
            (~reg_a)
            & (~reg_b)
            & (~reg_ab)
            & (~reg_c)
            & (vb <= 0.0)
            & (d2 >= 0.0)
            & (d6 <= 0.0)
        )
        denom_ac = d2 - d6
        w_ac = np.where(
            reg_ac & (denom_ac != 0.0),
            d2 / np.where(denom_ac != 0.0, denom_ac, 1.0),
            0.0,
        )
        w0[reg_ac] = 1.0 - w_ac[reg_ac]
        w2[reg_ac] = w_ac[reg_ac]

        va = d3 * d6 - d5 * d4
        reg_bc = (
            (~reg_a)
            & (~reg_b)
            & (~reg_ab)
            & (~reg_c)
            & (~reg_ac)
            & (va <= 0.0)
            & ((d4 - d3) >= 0.0)
            & ((d5 - d6) >= 0.0)
        )
        denom_bc = (d4 - d3) + (d5 - d6)
        w_bc = np.where(
            reg_bc & (denom_bc != 0.0),
            (d4 - d3) / np.where(denom_bc != 0.0, denom_bc, 1.0),
            0.0,
        )
        w1[reg_bc] = 1.0 - w_bc[reg_bc]
        w2[reg_bc] = w_bc[reg_bc]

        reg_int = (~reg_a) & (~reg_b) & (~reg_ab) & (~reg_c) & (~reg_ac) & (~reg_bc)
        denom_int = va + vb + vc
        safe_denom = np.where(denom_int != 0.0, denom_int, 1.0)
        v_int = vb / safe_denom
        w_int = vc / safe_denom
        u_int = 1.0 - v_int - w_int
        w0[reg_int] = u_int[reg_int]
        w1[reg_int] = v_int[reg_int]
        w2[reg_int] = w_int[reg_int]

        closest = (
            w0[:, :, None] * a[None, :, :]
            + w1[:, :, None] * b[None, :, :]
            + w2[:, :, None] * c[None, :, :]
        )
        dist_sq = np.sum((p - closest) ** 2, axis=-1)
        tri_idx = np.argmin(dist_sq, axis=1)
        rows = np.arange(stop - start)
        best_tri[start:stop] = tri_idx
        best_dist_sq[start:stop] = dist_sq[rows, tri_idx]
        best_bary[start:stop, 0] = w0[rows, tri_idx]
        best_bary[start:stop, 1] = w1[rows, tri_idx]
        best_bary[start:stop, 2] = w2[rows, tri_idx]

    return best_tri, best_bary, np.sqrt(best_dist_sq)
