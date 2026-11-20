# Adapted from GraGra (https://github.com/tatsuki-washimi/gragra)
# Commit: ce3a9f0f7c686097dec2cb34aca80222a88d11e1
# Source: src/gragra/adapters/specfem/{schema.py, _validation.py, reader.py}
# Copyright (c) 2026 Tatsuki Washimi — MIT License
"""Canonical SPECFEM3D Cartesian GLL HDF5 reader (lazy ``h5py``)."""

from __future__ import annotations

import hashlib
import importlib
import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from astropy import units as u

from .gll import ngll_from_shape
from .model import SpecfemFieldSnapshot

__all__ = [
    "ADOPTED_ELEMENT_ID_METHOD",
    "IDOMAIN_ACOUSTIC",
    "IDOMAIN_ELASTIC",
    "IDOMAIN_POROELASTIC",
    "LAYOUT_SINGLE_FILE",
    "MESH_TYPE",
    "PATH_FIELD_DISPLACEMENT",
    "PATH_MATERIAL_DENSITY",
    "PATH_MESH_COORDINATES",
    "PATH_MESH_DOMAIN_ID",
    "PATH_MESH_ELEMENT_ID",
    "PATH_MESH_IS_PML",
    "PATH_MESH_MATERIAL_ID",
    "PATH_MESH_PARTITION_ID",
    "PATH_MESH_QUADRATURE_VOLUME",
    "PATH_TIME",
    "PATH_TIME_STEP",
    "SCHEMA_VERSION",
    "SpecfemGLLBlock",
    "SpecfemGLLDataset",
    "SpecfemGLLFileIdentity",
    "read_specfem_gll_hdf5",
    "read_specfem_gll_snapshot",
]

SCHEMA_VERSION: Final[int] = 1
MESH_TYPE: Final[str] = "gll"
LAYOUT_SINGLE_FILE: Final[str] = "single_file"
IDOMAIN_ACOUSTIC: Final[int] = 1
IDOMAIN_ELASTIC: Final[int] = 2
IDOMAIN_POROELASTIC: Final[int] = 3
ADOPTED_ELEMENT_ID_METHOD: Final[str] = "pre_partition_global_index"

PATH_MESH_COORDINATES: Final[str] = "/mesh/coordinates"
PATH_MESH_QUADRATURE_VOLUME: Final[str] = "/mesh/quadrature_volume"
PATH_MESH_ELEMENT_ID: Final[str] = "/mesh/element_id"
PATH_MESH_PARTITION_ID: Final[str] = "/mesh/partition_id"
PATH_MESH_DOMAIN_ID: Final[str] = "/mesh/domain_id"
PATH_MESH_MATERIAL_ID: Final[str] = "/mesh/material_id"
PATH_MESH_IS_PML: Final[str] = "/mesh/is_pml"
PATH_MATERIAL_DENSITY: Final[str] = "/material/density"
PATH_FIELD_DISPLACEMENT: Final[str] = "/field/displacement"
PATH_TIME: Final[str] = "/time"
PATH_TIME_STEP: Final[str] = "/time_step"
METADATA_GROUP: Final[str] = "/metadata"


def _readonly(array: np.ndarray) -> np.ndarray:
    copied = np.asarray(array).copy()
    copied.setflags(write=False)
    return copied


def _import_h5py() -> Any:
    try:
        return importlib.import_module("h5py")
    except ImportError as exc:
        raise ImportError(
            "The 'h5py' package is required for SPECFEM3D GLL HDF5 input, "
            "but is not installed. Install it via `pip install h5py`."
        ) from exc


def _materialize(dataset: Any) -> np.ndarray:
    result = np.empty(dataset.shape, dtype=dataset.dtype)
    dataset.read_direct(result)
    return result


@dataclass(frozen=True)
class SpecfemGLLFileIdentity:
    """Stable file identity captured when a canonical GLL HDF5 file is read."""

    path: Path
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True)
class SpecfemGLLBlock:
    """One contiguous element-id block from :class:`SpecfemGLLDataset`."""

    element_id: np.ndarray
    coordinates_m: np.ndarray
    mass_kg: np.ndarray
    displacement_m: np.ndarray | None


@dataclass(frozen=True)
class SpecfemGLLDataset:
    """Closed immutable canonical GLL dataset, ordered by element then GLL point."""

    file_identity: SpecfemGLLFileIdentity
    time_s: np.ndarray
    time_step: np.ndarray
    element_id: np.ndarray
    coordinates_m: np.ndarray
    quadrature_volume_m3: np.ndarray
    density_kg_m3: np.ndarray
    displacement_m: np.ndarray | None

    def iter_blocks(self, *, element_block_size: int | None = None):
        """Yield contiguous element-id blocks without merging shared GLL points."""
        n_elements = self.element_id.size
        if element_block_size is None:
            block_size = max(1, n_elements)
        else:
            if isinstance(element_block_size, (bool, np.bool_)):
                raise ValueError("element_block_size must be a positive integer")
            try:
                block_size = operator.index(element_block_size)
            except TypeError as error:
                raise ValueError(
                    "element_block_size must be a positive integer"
                ) from error
            if block_size <= 0:
                raise ValueError("element_block_size must be a positive integer")

        for start in range(0, n_elements, block_size):
            stop = min(start + block_size, n_elements)
            mass = _readonly(
                self.density_kg_m3[start:stop] * self.quadrature_volume_m3[start:stop]
            )
            displacement = self.displacement_m
            if displacement is not None:
                displacement = displacement[:, start:stop]
            yield SpecfemGLLBlock(
                element_id=self.element_id[start:stop],
                coordinates_m=self.coordinates_m[start:stop],
                mass_kg=mass,
                displacement_m=displacement,
            )


def _decode_utf8_scalar(dataset: Any, name: str, h5py_mod: Any) -> str:
    if dataset.shape != ():
        raise ValueError(f"/metadata/{name} must be a scalar dataset")
    info = h5py_mod.check_string_dtype(dataset.dtype)
    if info is None or info.encoding != "utf-8":
        raise ValueError(f"/metadata/{name} must be a UTF-8 string dataset")
    holder = np.empty((), dtype=dataset.dtype)
    dataset.read_direct(holder)
    val = holder[()]
    if isinstance(val, str):
        return val
    if isinstance(val, bytes):
        return val.decode("utf-8")
    raise ValueError(f"/metadata/{name} must be a UTF-8 string dataset")


def _read_meta_dict(metadata: Any, h5py_mod: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in metadata.attrs.items():
        if isinstance(v, np.generic):
            out[k] = v.item()
        elif isinstance(v, bytes):
            out[k] = v.decode("utf-8", errors="replace")
        elif isinstance(v, np.ndarray):
            out[k] = tuple(
                int(x) if np.issubdtype(v.dtype, np.integer) else float(x) for x in v
            )
        else:
            out[k] = v
    for k in metadata.keys():
        member = metadata[k]
        if hasattr(member, "dtype"):
            try:
                text = _decode_utf8_scalar(member, k, h5py_mod)
            except ValueError:
                continue
            if k in {
                "field_units",
                "build_configuration",
                "executables",
                "f_content_max_evidence",
                "antialias_filter_spec",
            }:
                try:
                    out[k] = json.loads(text)
                    continue
                except json.JSONDecodeError:
                    pass
            out[k] = text
    return out


def _validate_gll_h5(
    h5: Any,
    h5py_mod: Any,
    *,
    profile_density_range: tuple[float, float] | None,
    require_time_sampled: bool = True,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int]]:
    if METADATA_GROUP not in h5:
        raise ValueError("Missing required /metadata group")
    meta = _read_meta_dict(h5[METADATA_GROUP], h5py_mod)

    sampling = meta.get("output_sampling_kind")
    if require_time_sampled and sampling != "time_sampled":
        raise ValueError("SPECFEM reader requires output_sampling_kind='time_sampled'")
    if sampling not in {"time_sampled", "static"}:
        raise ValueError(f"Unsupported output_sampling_kind={sampling!r}")

    for key, expected, msg in (
        ("schema_version", SCHEMA_VERSION, "Unsupported SPECFEM GLL schema_version"),
        ("mesh_type", MESH_TYPE, "Expected metadata mesh_type='gll'"),
        ("layout", LAYOUT_SINGLE_FILE, "Expected metadata layout='single_file'"),
        (
            "solver_kind",
            "specfem3d_cartesian",
            "SPECFEM reader requires solver_kind='specfem3d_cartesian'",
        ),
        (
            "coordinate_convention",
            "ENU",
            "metadata coordinate_convention must be 'ENU'",
        ),
    ):
        if meta.get(key) != expected:
            raise ValueError(msg)

    field_units = meta.get("field_units")
    if isinstance(field_units, dict) and sampling == "time_sampled":
        if field_units.get("displacement") != "m":
            raise ValueError("metadata field_units.displacement must be 'm'")

    ngll_shape_raw = meta.get("ngll_shape")
    if ngll_shape_raw is None:
        raise ValueError("Missing required /metadata/ngll_shape")
    ngll_shape = tuple(int(v) for v in ngll_shape_raw)
    if len(ngll_shape) != 3:
        raise ValueError("ngll_shape must have length 3")
    ngll = ngll_from_shape((ngll_shape[0], ngll_shape[1], ngll_shape[2]))
    if int(meta.get("ngll", -1)) != ngll:
        raise ValueError("metadata ngll must match metadata ngll_shape")

    for req_path in (
        PATH_MESH_COORDINATES,
        PATH_MESH_QUADRATURE_VOLUME,
        PATH_MESH_ELEMENT_ID,
        PATH_MESH_PARTITION_ID,
        PATH_MESH_DOMAIN_ID,
        PATH_MESH_MATERIAL_ID,
        PATH_MESH_IS_PML,
        PATH_MATERIAL_DENSITY,
    ):
        if req_path not in h5:
            raise ValueError(f"Missing required dataset {req_path!r}")

    coords_ds = h5[PATH_MESH_COORDINATES]
    if coords_ds.ndim != 3 or coords_ds.shape[-1] != 3:
        raise ValueError("/mesh/coordinates must have shape (element, gll, 3)")
    n_elements, file_ngll, _ = coords_ds.shape
    if n_elements == 0:
        raise ValueError("SPECFEM GLL input must contain at least one element")
    if file_ngll != ngll:
        raise ValueError(f"GLL axis length {file_ngll} != ngll_shape product {ngll}")

    elem_ids = _materialize(h5[PATH_MESH_ELEMENT_ID]).astype(np.int64)
    if not np.array_equal(elem_ids, np.arange(n_elements, dtype=np.int64)):
        raise ValueError("/mesh/element_id must be ascending 0..N-1")

    is_pml = _materialize(h5[PATH_MESH_IS_PML])
    if is_pml.dtype != np.bool_:
        raise ValueError("/mesh/is_pml dtype must be bool")
    if np.any(is_pml):
        raise ValueError("PML elements are not supported by this SPECFEM reader")

    dom = _materialize(h5[PATH_MESH_DOMAIN_ID])
    if not np.all(dom == IDOMAIN_ELASTIC):
        raise ValueError("SPECFEM reader requires a single elastic domain")

    mat = _materialize(h5[PATH_MESH_MATERIAL_ID])
    if np.any(mat < 1):
        raise ValueError("/mesh/material_id must be one-based and positive")

    vol = _materialize(h5[PATH_MESH_QUADRATURE_VOLUME])
    rho = _materialize(h5[PATH_MATERIAL_DENSITY])
    if not np.isfinite(vol).all() or np.any(vol <= 0.0):
        raise ValueError("quadrature volume must be finite and positive")
    if not np.isfinite(rho).all() or np.any(rho <= 0.0):
        raise ValueError("density must be finite and positive")
    if profile_density_range is not None:
        lo, hi = float(profile_density_range[0]), float(profile_density_range[1])
        if np.any(rho < lo) or np.any(rho > hi):
            raise ValueError(
                f"density outside supplied profile_density_range ({lo}, {hi})"
            )

    if sampling == "time_sampled":
        for req_path in (PATH_FIELD_DISPLACEMENT, PATH_TIME, PATH_TIME_STEP):
            if req_path not in h5:
                raise ValueError(f"Missing required dataset {req_path!r}")
        disp_ds = h5[PATH_FIELD_DISPLACEMENT]
        if disp_ds.ndim != 4 or disp_ds.shape[1:] != (n_elements, ngll, 3):
            raise ValueError(
                "/field/displacement must have shape (time, element, gll, 3)"
            )
        time_s = _materialize(h5[PATH_TIME]).astype(np.float64)
        time_step = _materialize(h5[PATH_TIME_STEP]).astype(np.int64)
        if time_s.size == 0 or not np.isfinite(time_s).all():
            raise ValueError("time must be finite and non-empty")
        if time_s.size > 1 and (
            np.any(np.diff(time_s) <= 0.0) or np.any(np.diff(time_step) <= 0)
        ):
            raise ValueError("/time and /time_step must be strictly increasing")
        dt_s = float(meta["dt_s"])
        t0 = float(meta["t0"])
        if not np.allclose(time_s, time_step.astype(np.float64) * dt_s - t0):
            raise ValueError("/time must equal time_step * dt_s - t0")
    else:
        time_s = np.array([0.0], dtype=np.float64)
        time_step = np.array([0], dtype=np.int64)

    meta["_pml_hash_verified"] = hashlib.sha256(
        is_pml.astype(np.uint8).tobytes()
    ).hexdigest()
    return (
        meta,
        time_s,
        time_step,
        elem_ids,
        (ngll_shape[0], ngll_shape[1], ngll_shape[2]),
    )


def read_specfem_gll_hdf5(
    path: str | Path,
    *,
    profile_density_range: tuple[float, float] | None = None,
) -> SpecfemGLLDataset:
    """Read an accepted canonical SPECFEM GLL HDF5 file into :class:`SpecfemGLLDataset`."""
    h5py = _import_h5py()
    resolved_path = Path(path).expanduser().resolve()
    with h5py.File(resolved_path, "r") as h5:
        _, time_s, time_step, elem_ids, _ = _validate_gll_h5(
            h5,
            h5py,
            profile_density_range=profile_density_range,
            require_time_sampled=True,
        )
        coordinates = _materialize(h5[PATH_MESH_COORDINATES])
        quadrature_volume = _materialize(h5[PATH_MESH_QUADRATURE_VOLUME])
        density = _materialize(h5[PATH_MATERIAL_DENSITY])
        displacement = _materialize(h5[PATH_FIELD_DISPLACEMENT])

    stat = resolved_path.stat()
    return SpecfemGLLDataset(
        file_identity=SpecfemGLLFileIdentity(
            path=resolved_path,
            device=stat.st_dev,
            inode=stat.st_ino,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        ),
        time_s=_readonly(time_s),
        time_step=_readonly(time_step),
        element_id=_readonly(elem_ids),
        coordinates_m=_readonly(coordinates),
        quadrature_volume_m3=_readonly(quadrature_volume),
        density_kg_m3=_readonly(density),
        displacement_m=_readonly(displacement),
    )


def read_specfem_gll_snapshot(
    path: str | Path,
    *,
    profile_density_range: tuple[float, float] | None = None,
    require_time_sampled: bool = False,
) -> SpecfemFieldSnapshot:
    """Read a canonical SPECFEM3D GLL HDF5 file into :class:`SpecfemFieldSnapshot`."""
    h5py = _import_h5py()
    resolved_path = Path(path).expanduser().resolve()
    with h5py.File(resolved_path, "r") as h5:
        meta, time_s, _, elem_ids, ngll_shape = _validate_gll_h5(
            h5,
            h5py,
            profile_density_range=profile_density_range,
            require_time_sampled=require_time_sampled,
        )
        coordinates = _materialize(h5[PATH_MESH_COORDINATES]).astype(np.float64)
        quadrature_volume = _materialize(h5[PATH_MESH_QUADRATURE_VOLUME]).astype(
            np.float64
        )
        density = _materialize(h5[PATH_MATERIAL_DENSITY]).astype(np.float64)
        part_ids = _materialize(h5[PATH_MESH_PARTITION_ID]).astype(np.int64)
        dom_ids = _materialize(h5[PATH_MESH_DOMAIN_ID]).astype(np.int64)
        mat_ids = _materialize(h5[PATH_MESH_MATERIAL_ID]).astype(np.int64)
        is_pml = _materialize(h5[PATH_MESH_IS_PML]).astype(np.bool_)

        fields: dict[str, np.ndarray] = {"rho": density, "density": density}
        field_units: dict[str, u.UnitBase] = {
            "rho": u.kg / u.m**3,
            "density": u.kg / u.m**3,
        }
        if PATH_FIELD_DISPLACEMENT in h5:
            disp = _materialize(h5[PATH_FIELD_DISPLACEMENT]).astype(np.float64)
            fields["displacement"] = disp
            field_units["displacement"] = u.m
        if "/field" in h5:
            for fname in h5["/field"].keys():
                if fname not in fields:
                    fields[fname] = _materialize(h5[f"/field/{fname}"]).astype(
                        np.float64
                    )
                    field_units[fname] = u.dimensionless_unscaled

    return SpecfemFieldSnapshot(
        source_path=str(resolved_path),
        source_kind="gll_hdf5",
        times_s=time_s,
        coordinates_m=coordinates,
        fields=fields,
        field_units=field_units,
        association="gll",
        element_id=elem_ids,
        partition_id=part_ids,
        material_id=mat_ids,
        domain_id=dom_ids,
        is_pml=is_pml,
        ngll_shape=ngll_shape,
        quadrature_volume_m3=quadrature_volume,
        provenance=meta,
    )
