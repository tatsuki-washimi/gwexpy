# Adapted from GraGra (https://github.com/tatsuki-washimi/gragra)
# Commit: ce3a9f0f7c686097dec2cb34aca80222a88d11e1
# Source: src/gragra/adapters/openfoam_density.py
# Copyright (c) 2026 Tatsuki Washimi — MIT License
"""Native mesh-retaining data models and frozen provenance for OpenFOAM."""

from __future__ import annotations

import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from astropy import units as u

__all__ = [
    "FoamFieldData",
    "FoamSnapshot",
    "FoamVolumeMesh",
    "OpenFOAMDensitySnapshot",
    "OpenFOAMProvenance",
    "validate_legacy_vtk_geometry",
]

_DENSITY_UNITS = ("kg/m^3", "m", "m^3", "s")
_CALLER_KEYS = {
    "distribution",
    "version",
    "patch",
    "solver",
    "compressibility",
    "region",
    "reconstructed",
    "exporter",
    "export_command",
}
_PROFILE = {
    "distribution": "OpenFOAM Foundation",
    "version": "14",
    "patch": "20260724",
    "region": "region0",
    "reconstructed": True,
    "exporter": "foamToVTK",
}
_RESERVED = {
    "input_sha256",
    "points_dtype",
    "centre_dtype",
    "volume_dtype",
    "density_dtype",
    "cell_id_dtype",
}


def _readonly(a: Any, dtype: Any = None) -> np.ndarray:
    out = np.array(a, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _check_no_bool(value: Any, name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be real numeric, not bool")
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise TypeError(f"{name} must be real numeric, not bool")


def _real_array(
    name: str, value: Any, shape: tuple[int, ...] | None = None
) -> np.ndarray:
    _check_no_bool(value, name)
    raw = np.asarray(value)
    if raw.dtype.kind in "bOc" or not np.issubdtype(raw.dtype, np.number):
        raise TypeError(f"{name} must be real numeric")
    out = np.asarray(raw, dtype=np.float64)
    if shape is not None and out.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _validate_density_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CALLER_KEYS:
        raise ValueError("provenance must contain exactly the nine caller keys")
    if set(value) & _RESERVED:
        raise ValueError("caller may not override reserved provenance keys")
    p = dict(value)
    if any(p[k] != v for k, v in _PROFILE.items()):
        raise ValueError("provenance does not match the Foundation 14 profile")
    if not isinstance(p["reconstructed"], bool) or p["reconstructed"] is not True:
        raise ValueError("reconstructed must be the boolean True")
    if (
        not isinstance(p["solver"], str)
        or not p["solver"].strip()
        or not isinstance(p["export_command"], str)
        or not p["export_command"].strip()
    ):
        raise ValueError("solver and export_command must be non-empty strings")
    if p["compressibility"] not in {"compressible", "incompressible", "not-applicable"}:
        raise ValueError("invalid compressibility")
    if (
        p["solver"] == "manufactured-no-solver"
        and p["compressibility"] != "not-applicable"
    ):
        raise ValueError("manufactured-no-solver requires not-applicable")
    return p


def validate_legacy_vtk_geometry(
    points: np.ndarray,
    cells: Sequence[tuple[str, np.ndarray]],
    centres: np.ndarray,
    volumes: np.ndarray,
    eps: float,
) -> None:
    """Validate tetrahedron and affine-hexahedron geometry against native C and Vc.

    Adapted from GraGra's precision-aware ``_geometry`` check so float32
    legacy-ASCII VTK exports from ``foamToVTK`` are verified against exact
    trilinear/tetrahedral determinants and centroids.
    """
    for kind, _conn in cells:
        if kind not in {"tetra", "hexahedron"}:
            raise ValueError(f"unsupported cell block {kind!r}")
    index = 0
    for kind, conn in cells:
        for row in conn:
            q = points[row]
            h = float(np.max(np.linalg.norm(q[:, None] - q[None, :], axis=-1)))
            if kind == "tetra":
                vg = (
                    abs(
                        float(
                            np.linalg.det(
                                np.stack(
                                    (q[1] - q[0], q[2] - q[0], q[3] - q[0]), axis=1
                                )
                            )
                        )
                    )
                    / 6
                )
                cg = q.mean(axis=0)
                if 6 * vg <= 64 * eps * h**3:
                    raise ValueError("precision-degenerate tetra")
            else:
                p0, p1, p2, p3, p4, p5, p6, p7 = q

                def close(lhs: np.ndarray, rhs: np.ndarray, cell_h: float = h) -> bool:
                    return bool(
                        np.all(
                            np.abs(lhs - rhs)
                            <= 8 * eps * np.maximum(np.abs(lhs), np.abs(rhs))
                            + 32 * eps * cell_h
                        )
                    )

                affine = (
                    close(p2, p1 + p3 - p0)
                    and close(p5, p1 + p4 - p0)
                    and close(p7, p3 + p4 - p0)
                    and close(p6, p1 + p3 + p4 - 2 * p0)
                )
                if not affine:
                    raise ValueError("nonaffine hexahedron")
                vg = abs(
                    float(np.linalg.det(np.stack((p1 - p0, p3 - p0, p4 - p0), axis=1)))
                )
                cg = q.mean(axis=0)
                if vg <= 64 * eps * h**3:
                    raise ValueError("precision-degenerate hexahedron")
            ctol = (
                8 * eps * np.maximum(np.abs(centres[index]), np.abs(cg)) + 32 * eps * h
            )
            vtol = 32 * eps * max(abs(volumes[index]), abs(vg)) + 64 * eps * h**3
            if (
                not np.all(np.abs(centres[index] - cg) <= ctol)
                or abs(volumes[index] - vg) > vtol
            ):
                raise ValueError("native C/Vc fails fixed geometry tolerance")
            index += 1


@dataclass(frozen=True)
class FoamVolumeMesh:
    """Native volume-mesh representation retaining vertices and cell connectivity."""

    vertices_m: np.ndarray
    cell_blocks: tuple[tuple[str, np.ndarray], ...]

    def __post_init__(self) -> None:
        verts = _real_array("vertices_m", self.vertices_m)
        if verts.ndim != 2 or verts.shape[1] != 3:
            raise ValueError("vertices_m must have shape (N, 3)")
        frozen_blocks: list[tuple[str, np.ndarray]] = []
        for kind, conn in self.cell_blocks:
            arr = _readonly(conn, np.int64)
            frozen_blocks.append((str(kind), arr))
        object.__setattr__(self, "vertices_m", _readonly(verts, np.float64))
        object.__setattr__(self, "cell_blocks", tuple(frozen_blocks))


@dataclass(frozen=True)
class OpenFOAMProvenance:
    """Frozen provenance attached to OpenFOAM-derived GWexpy Fields via ``_gwex_openfoam``."""

    source_path: str
    region: str
    field_name: str
    foam_class: str | None
    dimensions: tuple[int | float, ...] | None
    unit_str: str
    pressure_kind: str | None
    association: Literal["cell", "point", "face"]
    grid_mode: Literal["exact", "sample"]
    moving_mesh: Literal["error", "resample"]
    mesh_motion_detected: bool
    time_names: tuple[str, ...]
    solver: str | None = None
    backend: str = "pyvista"
    input_sha256: str | None = None
    interpolation_method: str | None = None
    valid_fraction: float | None = None
    valid_mask: np.ndarray | None = field(default=None, compare=False, repr=False)
    extra: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.valid_mask is not None:
            object.__setattr__(self, "valid_mask", _readonly(self.valid_mask, np.bool_))

    def _slice_for_field(
        self,
        item: tuple[slice, slice, slice, slice],
        source_shape: tuple[int, ...],
        axis0: Any,
    ) -> OpenFOAMProvenance:
        """Align sampled validity and selected time names with a 4-D Field slice."""
        if self.valid_mask is not None and self.valid_mask.shape != source_shape:
            raise ValueError(
                "OpenFOAM provenance valid_mask shape does not match the source Field"
            )
        selected_times = self.time_names[item[0]]
        selected_mask = self.valid_mask[item] if self.valid_mask is not None else None
        fraction = (
            float(np.mean(selected_mask))
            if selected_mask is not None and selected_mask.size
            else None
        )
        return replace(
            self,
            time_names=selected_times,
            valid_mask=selected_mask,
            valid_fraction=fraction
            if selected_mask is not None
            else self.valid_fraction,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly metadata dictionary."""
        return {
            "source_path": self.source_path,
            "region": self.region,
            "field_name": self.field_name,
            "foam_class": self.foam_class,
            "dimensions": self.dimensions,
            "unit_str": self.unit_str,
            "pressure_kind": self.pressure_kind,
            "association": self.association,
            "grid_mode": self.grid_mode,
            "moving_mesh": self.moving_mesh,
            "mesh_motion_detected": self.mesh_motion_detected,
            "time_names": self.time_names,
            "solver": self.solver,
            "backend": self.backend,
            "input_sha256": self.input_sha256,
            "interpolation_method": self.interpolation_method,
            "valid_fraction": self.valid_fraction,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class FoamFieldData:
    """Native mesh field array with OpenFOAM class, dimensions, and Astropy unit."""

    name: str
    values: np.ndarray
    association: Literal["cell", "point", "face"]
    foam_class: str | None
    dimensions: tuple[int | float, ...] | None
    unit: u.UnitBase
    pressure_kind: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _readonly(self.values, np.float64))

    @property
    def is_vector(self) -> bool:
        return self.values.ndim == 2 and self.values.shape[1] == 3


@dataclass(frozen=True)
class FoamSnapshot:
    """Native mesh-retaining OpenFOAM snapshot for a single time step and region."""

    time_s: float
    time_name: str
    region: str
    points_m: np.ndarray
    cell_centers_m: np.ndarray | None
    cell_volumes_m3: np.ndarray | None
    cell_blocks: tuple[tuple[str, np.ndarray], ...]
    cell_ids: np.ndarray | None
    fields: Mapping[str, FoamFieldData]
    provenance: Mapping[str, Any]
    pyvista_mesh: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(self, "points_m", _readonly(self.points_m, np.float64))
        if self.cell_centers_m is not None:
            object.__setattr__(
                self, "cell_centers_m", _readonly(self.cell_centers_m, np.float64)
            )
        if self.cell_volumes_m3 is not None:
            object.__setattr__(
                self, "cell_volumes_m3", _readonly(self.cell_volumes_m3, np.float64)
            )
        if self.cell_ids is not None:
            object.__setattr__(self, "cell_ids", _readonly(self.cell_ids, np.int64))
        frozen_blocks = tuple(
            (str(kind), _readonly(conn, np.int64)) for kind, conn in self.cell_blocks
        )
        object.__setattr__(self, "cell_blocks", frozen_blocks)
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True, init=False)
class OpenFOAMDensitySnapshot:
    """Immutable native-centre density snapshot in declared world coordinates.

    Preserves GraGra's bounded Foundation 14 legacy-ASCII ``foamToVTK``
    density validation and semantics.
    """

    input_path: str
    density_unit: str
    coordinate_unit: str
    volume_unit: str
    time_unit: str
    mesh: FoamVolumeMesh
    positions_m: np.ndarray
    volumes_m3: np.ndarray
    density_kg_m3: np.ndarray
    cell_ids: np.ndarray
    time_s: float
    time_name: str
    field_name: str
    density_kind: str
    coordinate_frame: str
    coordinate_origin_m: np.ndarray
    provenance: Mapping[str, object]

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError(
            "OpenFOAMDensitySnapshot is loader-only; use load_openfoam_density()"
        )

    @classmethod
    def _from_validated_loader(
        cls,
        *,
        input_path: str,
        density_unit: str,
        coordinate_unit: str,
        volume_unit: str,
        time_unit: str,
        mesh: FoamVolumeMesh,
        positions_m: np.ndarray,
        volumes_m3: np.ndarray,
        density_kg_m3: np.ndarray,
        cell_ids: np.ndarray,
        time_s: float,
        time_name: str,
        field_name: str,
        coordinate_frame: str,
        coordinate_origin_m: np.ndarray,
        provenance: Mapping[str, object],
    ) -> OpenFOAMDensitySnapshot:
        """Build an immutable result after ``load_openfoam_density`` validation."""
        result = object.__new__(cls)
        for name, value in (
            ("input_path", str(input_path)),
            ("density_unit", density_unit),
            ("coordinate_unit", coordinate_unit),
            ("volume_unit", volume_unit),
            ("time_unit", time_unit),
            ("mesh", mesh),
            ("positions_m", _readonly(positions_m, np.float64)),
            ("volumes_m3", _readonly(volumes_m3, np.float64)),
            ("density_kg_m3", _readonly(density_kg_m3, np.float64)),
            ("cell_ids", _readonly(cell_ids, np.int64)),
            ("time_s", float(time_s)),
            ("time_name", time_name),
            ("field_name", field_name),
            ("density_kind", field_name),
            ("coordinate_frame", coordinate_frame),
            ("coordinate_origin_m", _readonly(coordinate_origin_m, np.float64)),
            ("provenance", MappingProxyType(dict(provenance))),
        ):
            object.__setattr__(result, name, value)
        return result

    def __post_init__(self) -> None:
        path = Path(self.input_path)
        if not path.is_absolute():
            raise ValueError("input_path must be an absolute path string")
        if (
            self.density_unit,
            self.coordinate_unit,
            self.volume_unit,
            self.time_unit,
        ) != _DENSITY_UNITS:
            raise ValueError("snapshot units must be canonical")
        if (
            self.field_name not in {"rho", "delta_rho"}
            or self.density_kind != self.field_name
        ):
            raise ValueError("field_name and density_kind must agree")
        if not isinstance(self.mesh, FoamVolumeMesh):
            raise TypeError("mesh must be a FoamVolumeMesh")
        n = sum(len(block[1]) for block in self.mesh.cell_blocks)
        positions = _real_array("positions_m", self.positions_m, (n, 3))
        volumes = _real_array("volumes_m3", self.volumes_m3, (n,))
        density = _real_array("density_kg_m3", self.density_kg_m3, (n,))
        ids_raw = np.asarray(self.cell_ids)
        if ids_raw.dtype.kind not in "iu":
            raise TypeError("cell_ids must be integer")
        ids = np.asarray(ids_raw, dtype=np.int64)
        if (
            ids.shape != (n,)
            or (ids < 0).any()
            or not np.array_equal(np.sort(ids), np.arange(n))
        ):
            raise ValueError("cell_ids must be a permutation of 0..N-1")
        if (volumes <= 0).any() or (self.field_name == "rho" and (density < 0).any()):
            raise ValueError("invalid snapshot density or volume")
        origin = _real_array("coordinate_origin_m", self.coordinate_origin_m, (3,))
        if (
            not isinstance(self.coordinate_frame, str)
            or not self.coordinate_frame.strip()
        ):
            raise ValueError("coordinate_frame must be non-empty")
        if (
            not isinstance(self.time_name, str)
            or not isinstance(self.time_s, numbers.Real)
            or isinstance(self.time_s, bool)
            or not np.isfinite(self.time_s)
        ):
            raise ValueError("snapshot time is invalid")
        try:
            if (
                not np.isfinite(float(self.time_name))
                or abs(float(self.time_name) - self.time_s) > 1e-12
            ):
                raise ValueError("snapshot time_name and time_s must agree")
        except ValueError:
            raise ValueError("snapshot time_name must be finite numeric") from None
        if (
            not isinstance(self.provenance, Mapping)
            or set(self.provenance) != _CALLER_KEYS | _RESERVED
        ):
            raise ValueError("snapshot provenance keys are invalid")
        caller = _validate_density_provenance(
            {k: self.provenance[k] for k in _CALLER_KEYS}
        )
        reserved = {k: self.provenance[k] for k in _RESERVED}
        if (
            not isinstance(reserved["input_sha256"], str)
            or len(reserved["input_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in reserved["input_sha256"])
            or any(
                not isinstance(reserved[k], str) for k in _RESERVED - {"input_sha256"}
            )
        ):
            raise ValueError("snapshot reserved provenance is invalid")
        expected_dtypes = {
            "points_dtype": "float32",
            "centre_dtype": "float32",
            "volume_dtype": "float32",
            "density_dtype": "float32",
            "cell_id_dtype": "int32",
        }
        if any(reserved[key] != value for key, value in expected_dtypes.items()):
            raise ValueError("snapshot payload dtype provenance is invalid")
        validate_legacy_vtk_geometry(
            self.mesh.vertices_m - origin,
            self.mesh.cell_blocks,
            positions - origin,
            volumes,
            float(np.finfo(np.float32).eps),
        )
        frozen = MappingProxyType({**caller, **reserved})
        for name, value in (
            ("positions_m", positions),
            ("volumes_m3", volumes),
            ("density_kg_m3", density),
            ("cell_ids", ids),
            ("coordinate_origin_m", origin),
        ):
            object.__setattr__(self, name, _readonly(value))
        object.__setattr__(self, "input_path", str(path))
        object.__setattr__(self, "time_s", float(self.time_s))
        object.__setattr__(self, "provenance", frozen)

    def assert_compatible_successor(
        self, next_snapshot: OpenFOAMDensitySnapshot
    ) -> None:
        """Verify that a subsequent snapshot shares the exact mesh and producer profile."""
        if not isinstance(next_snapshot, OpenFOAMDensitySnapshot):
            raise TypeError("next_snapshot must be OpenFOAMDensitySnapshot")
        if not next_snapshot.time_s > self.time_s:
            raise ValueError("successor time must be strictly increasing")
        attrs = (
            "field_name",
            "density_unit",
            "coordinate_unit",
            "volume_unit",
            "time_unit",
            "coordinate_frame",
        )
        arrays = ("positions_m", "volumes_m3", "cell_ids", "coordinate_origin_m")
        if any(getattr(self, x) != getattr(next_snapshot, x) for x in attrs) or any(
            not np.array_equal(getattr(self, x), getattr(next_snapshot, x))
            for x in arrays
        ):
            raise ValueError("successor geometry or profile differs")
        same_blocks = len(self.mesh.cell_blocks) == len(
            next_snapshot.mesh.cell_blocks
        ) and all(
            a[0] == b[0] and np.array_equal(a[1], b[1])
            for a, b in zip(
                self.mesh.cell_blocks, next_snapshot.mesh.cell_blocks, strict=True
            )
        )
        if not same_blocks or not np.array_equal(
            self.mesh.vertices_m, next_snapshot.mesh.vertices_m
        ):
            raise ValueError("successor mesh differs")
        keys = _PROFILE.keys() | {"solver", "compressibility"}
        if any(self.provenance[k] != next_snapshot.provenance[k] for k in keys):
            raise ValueError("successor producer profile differs")
