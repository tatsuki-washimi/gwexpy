# Adapted in part from GraGra (https://github.com/tatsuki-washimi/gragra)
# Commit: ce3a9f0f7c686097dec2cb34aca80222a88d11e1
# Source: src/gragra/adapters/openfoam_density.py
# Copyright (c) 2026 Tatsuki Washimi — MIT License
"""Backends and case discovery for OpenFOAM native cases and legacy-ASCII VTK files."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import numbers
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy import units as u

from .dimensions import (
    OpenFOAMHeader,
    classify_pressure_dimensions,
    parse_control_dict,
    parse_foam_header,
)
from .model import (
    _DENSITY_UNITS,
    FoamFieldData,
    FoamSnapshot,
    FoamVolumeMesh,
    OpenFOAMDensitySnapshot,
    _real_array,
    _validate_density_provenance,
    validate_legacy_vtk_geometry,
)

__all__ = [
    "LegacyVTKOpenFOAMBackend",
    "OpenFOAMCase",
    "PyVistaOpenFOAMBackend",
    "load_openfoam_density",
    "read_openfoam_vtk",
]

_VTK_TYPE_MAP: dict[int, str] = {
    1: "vertex",
    3: "line",
    5: "triangle",
    8: "pixel",
    9: "quad",
    10: "tetra",
    11: "voxel",
    12: "hexahedron",
    13: "wedge",
    14: "pyramid",
}

_TIME_DIR_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _require_pyvista() -> Any:
    try:
        return importlib.import_module("pyvista")
    except ImportError as exc:
        raise ImportError(
            "The 'pyvista' package is required for reading OpenFOAM cases via "
            "OpenFOAMReader, but is not installed. Install it via "
            "`pip install pyvista` or `pip install 'gwexpy[openfoam]'`."
        ) from exc


def _is_time_dir(name: str) -> bool:
    return bool(_TIME_DIR_RE.fullmatch(name))


def _vtk_dtype(token: str) -> np.dtype[Any]:
    low = token.lower()
    mapping: dict[str, np.dtype[Any]] = {
        "float": np.dtype(np.float32),
        "double": np.dtype(np.float64),
        "int": np.dtype(np.int32),
        "long": np.dtype(np.int64),
        "vtktypeint64": np.dtype(np.int64),
        "unsigned_int": np.dtype(np.uint32),
        "short": np.dtype(np.int16),
        "unsigned_short": np.dtype(np.uint16),
        "char": np.dtype(np.int8),
        "unsigned_char": np.dtype(np.uint8),
    }
    if low not in mapping:
        raise ValueError(f"Unsupported legacy VTK scalar/vector dtype {token!r}")
    return mapping[low]


@dataclass(frozen=True)
class ParsedLegacyVTK:
    """Parsed legacy-ASCII VTK ``UNSTRUCTURED_GRID`` payload."""

    points: np.ndarray
    cells: tuple[tuple[str, np.ndarray], ...]
    cell_data: dict[str, np.ndarray]
    point_data: dict[str, np.ndarray]
    sha256: str


def _parse_legacy_vtk_arrays(
    tokens: list[str],
    pos: int,
    n_items: int,
    target: dict[str, np.ndarray],
    selected: set[str] | None,
) -> int:
    """Parse VTK POINT_DATA or CELL_DATA attribute arrays into ``target``."""
    n_tok = len(tokens)
    while pos < n_tok:
        kw = tokens[pos].upper()
        if kw in {"POINT_DATA", "CELL_DATA"}:
            break
        if kw == "SCALARS":
            name = tokens[pos + 1]
            dt = _vtk_dtype(tokens[pos + 2])
            pos += 3
            ncomp = 1
            if pos < n_tok and tokens[pos].isdigit():
                ncomp = int(tokens[pos])
                pos += 1
            if pos < n_tok and tokens[pos].upper() == "LOOKUP_TABLE":
                pos += 2  # skip LOOKUP_TABLE <tableName>
            count = n_items * ncomp
            if pos + count > n_tok:
                raise ValueError(f"Truncated SCALARS {name!r} block in legacy VTK")
            if selected is not None and name not in selected:
                pos += count
                continue
            raw_vals = tokens[pos : pos + count]
            pos += count
            arr = np.array(
                [int(v) if dt.kind in "iu" else float(v) for v in raw_vals],
                dtype=dt,
            )
            if ncomp > 1:
                arr = arr.reshape((n_items, ncomp))
            target[name] = arr
            continue
        if kw == "VECTORS":
            name = tokens[pos + 1]
            dt = _vtk_dtype(tokens[pos + 2])
            pos += 3
            count = n_items * 3
            if pos + count > n_tok:
                raise ValueError(f"Truncated VECTORS {name!r} block in legacy VTK")
            if selected is not None and name not in selected:
                pos += count
                continue
            raw_vals = tokens[pos : pos + count]
            pos += count
            arr = np.array(
                [int(v) if dt.kind in "iu" else float(v) for v in raw_vals],
                dtype=dt,
            ).reshape((n_items, 3))
            target[name] = arr
            continue
        if kw == "TENSORS":
            name = tokens[pos + 1]
            dt = _vtk_dtype(tokens[pos + 2])
            pos += 3
            count = n_items * 9
            if pos + count > n_tok:
                raise ValueError(f"Truncated TENSORS {name!r} block in legacy VTK")
            if selected is not None and name not in selected:
                pos += count
                continue
            raw_vals = tokens[pos : pos + count]
            pos += count
            arr = np.array(
                [int(v) if dt.kind in "iu" else float(v) for v in raw_vals],
                dtype=dt,
            ).reshape((n_items, 9))
            target[name] = arr
            continue
        if kw == "FIELD":
            # FIELD <fieldName> <numArrays>
            pos += 2
            num_arrays = int(tokens[pos])
            pos += 1
            for _ in range(num_arrays):
                arr_name = tokens[pos]
                ncomp = int(tokens[pos + 1])
                ntup = int(tokens[pos + 2])
                dt = _vtk_dtype(tokens[pos + 3])
                pos += 4
                count = ncomp * ntup
                if pos + count > n_tok:
                    raise ValueError(
                        f"Truncated FIELD array {arr_name!r} in legacy VTK"
                    )
                if selected is not None and arr_name not in selected:
                    pos += count
                    continue
                raw_vals = tokens[pos : pos + count]
                pos += count
                arr = np.array(
                    [int(v) if dt.kind in "iu" else float(v) for v in raw_vals],
                    dtype=dt,
                )
                if ncomp > 1:
                    arr = arr.reshape((ntup, ncomp))
                target[arr_name] = arr
            continue
        raise ValueError(
            f"Unsupported legacy VTK data section keyword: {tokens[pos]!r}"
        )
    return pos


def list_legacy_vtk_arrays(path: Path) -> dict[str, dict[str, int]]:
    """Scan legacy ASCII VTK declarations without retaining numeric payloads."""
    arrays: dict[str, dict[str, int]] = {"cell": {}, "point": {}}
    with path.open(encoding="ascii") as stream:
        version, _, encoding, dataset = (stream.readline().strip() for _ in range(4))
        if (
            not version.startswith("# vtk DataFile Version")
            or encoding != "ASCII"
            or dataset != "DATASET UNSTRUCTURED_GRID"
        ):
            raise ValueError("requires legacy ASCII UNSTRUCTURED_GRID")
        tokens = (token for line in stream for token in line.split())

        def skip(count: int) -> None:
            for _ in range(count):
                next(tokens)

        association = "cell"
        items = 0
        try:
            while (section := next(tokens, None)) is not None:
                section = section.upper()
                if section == "POINTS":
                    count = int(next(tokens))
                    skip(1 + 3 * count)
                elif section == "CELLS":
                    skip(1)
                    skip(int(next(tokens)))
                elif section == "CELL_TYPES":
                    skip(int(next(tokens)))
                elif section in {"CELL_DATA", "POINT_DATA"}:
                    association = "cell" if section == "CELL_DATA" else "point"
                    items = int(next(tokens))
                elif section == "FIELD":
                    skip(1)
                    for _ in range(int(next(tokens))):
                        name = next(tokens)
                        components, tuples = int(next(tokens)), int(next(tokens))
                        arrays[association][name] = components
                        skip(1 + components * tuples)
                elif section in {"SCALARS", "VECTORS", "TENSORS"}:
                    name = next(tokens)
                    skip(1)
                    components = {"SCALARS": 1, "VECTORS": 3, "TENSORS": 9}[section]
                    if section == "SCALARS":
                        token = next(tokens)
                        if token.isdecimal():
                            components = int(token)
                            token = next(tokens)
                        if token.upper() != "LOOKUP_TABLE":
                            # The first value was consumed when the table is omitted.
                            skip(items * components - 1)
                        else:
                            skip(1 + items * components)
                    else:
                        skip(items * components)
                    arrays[association][name] = components
                else:
                    raise ValueError(f"Unsupported legacy VTK section: {section!r}")
        except StopIteration as exc:
            raise ValueError(f"Truncated legacy VTK file: {path}") from exc
    return arrays


def parse_legacy_ascii_vtk(
    path: str | Path, *, fields: set[str] | None = None
) -> ParsedLegacyVTK:
    """Parse a legacy-ASCII VTK ``DATASET UNSTRUCTURED_GRID`` file without meshio.

    Preserves exact declared on-disk dtypes (``float32``/``float64``/``int32``)
    so precision-aware geometry and profile validation matches GraGra.
    """
    p = Path(path)
    raw = p.read_bytes()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("requires legacy ASCII UNSTRUCTURED_GRID") from exc
    lines = text.splitlines()
    if (
        len(lines) < 4
        or not lines[0].startswith("# vtk DataFile Version")
        or lines[2].strip() != "ASCII"
        or lines[3].strip() != "DATASET UNSTRUCTURED_GRID"
    ):
        raise ValueError("requires legacy ASCII UNSTRUCTURED_GRID")

    tokens = "\n".join(lines[4:]).split()
    n_tok = len(tokens)
    pos = 0

    points: np.ndarray | None = None
    raw_cells: list[list[int]] | None = None
    cell_types: list[int] | None = None
    cell_data: dict[str, np.ndarray] = {}
    point_data: dict[str, np.ndarray] = {}

    while pos < n_tok:
        kw = tokens[pos].upper()
        if kw == "POINTS":
            n_pts = int(tokens[pos + 1])
            dt = _vtk_dtype(tokens[pos + 2])
            pos += 3
            count = n_pts * 3
            if pos + count > n_tok:
                raise ValueError("Truncated POINTS section in legacy VTK")
            points = np.array(
                [float(v) for v in tokens[pos : pos + count]], dtype=dt
            ).reshape((n_pts, 3))
            pos += count
        elif kw == "CELLS":
            n_cells = int(tokens[pos + 1])
            size = int(tokens[pos + 2])
            pos += 3
            if pos + size > n_tok:
                raise ValueError("Truncated CELLS section in legacy VTK")
            cell_tokens = tokens[pos : pos + size]
            pos += size
            raw_cells = []
            cpos = 0
            for _ in range(n_cells):
                nv = int(cell_tokens[cpos])
                cpos += 1
                raw_cells.append([int(v) for v in cell_tokens[cpos : cpos + nv]])
                cpos += nv
            if cpos != size:
                raise ValueError("CELLS token count mismatch in legacy VTK")
        elif kw == "CELL_TYPES":
            n_cells = int(tokens[pos + 1])
            pos += 2
            if pos + n_cells > n_tok:
                raise ValueError("Truncated CELL_TYPES section in legacy VTK")
            cell_types = [int(v) for v in tokens[pos : pos + n_cells]]
            pos += n_cells
        elif kw == "CELL_DATA":
            n_items = int(tokens[pos + 1])
            pos += 2
            pos = _parse_legacy_vtk_arrays(tokens, pos, n_items, cell_data, fields)
        elif kw == "POINT_DATA":
            n_items = int(tokens[pos + 1])
            pos += 2
            pos = _parse_legacy_vtk_arrays(tokens, pos, n_items, point_data, fields)
        else:
            raise ValueError(f"Unsupported legacy VTK section keyword: {tokens[pos]!r}")

    if points is None or raw_cells is None or cell_types is None:
        raise ValueError(
            "Legacy VTK UNSTRUCTURED_GRID must contain POINTS, CELLS, and CELL_TYPES"
        )
    if len(raw_cells) != len(cell_types):
        raise ValueError("CELLS and CELL_TYPES counts disagree in legacy VTK")

    # Group contiguous cell blocks by VTK cell type
    blocks: list[tuple[str, np.ndarray]] = []
    if len(cell_types) > 0:
        cur_type = cell_types[0]
        cur_rows = [raw_cells[0]]
        for ctype, row in zip(cell_types[1:], raw_cells[1:], strict=True):
            if ctype == cur_type:
                cur_rows.append(row)
            else:
                kind = _VTK_TYPE_MAP.get(cur_type, f"vtk_{cur_type}")
                blocks.append((kind, np.asarray(cur_rows, dtype=np.int64)))
                cur_type = ctype
                cur_rows = [row]
        kind = _VTK_TYPE_MAP.get(cur_type, f"vtk_{cur_type}")
        blocks.append((kind, np.asarray(cur_rows, dtype=np.int64)))

    return ParsedLegacyVTK(
        points=points,
        cells=tuple(blocks),
        cell_data=cell_data,
        point_data=point_data,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_openfoam_density(
    path: str | Path,
    *,
    field_name: str,
    time_s: float,
    time_name: str,
    coordinate_frame: str,
    coordinate_origin_m: Any,
    density_unit: str,
    coordinate_unit: str,
    volume_unit: str,
    time_unit: str,
    provenance: Mapping[str, Any],
) -> OpenFOAMDensitySnapshot:
    """Load ``rho`` or ``delta_rho`` from the bounded Foundation 14 legacy-ASCII VTK profile.

    Preserves GraGra's precision-aware ``_geometry`` check and ``float32``/``int32``
    dtype enforcement for ``foamToVTK`` exports.
    """
    if field_name not in {"rho", "delta_rho"}:
        raise ValueError("field_name must be 'rho' or 'delta_rho'")
    if (density_unit, coordinate_unit, volume_unit, time_unit) != _DENSITY_UNITS:
        raise ValueError("units must be canonical kg/m^3, m, m^3, s")
    if not isinstance(time_name, str):
        raise TypeError("time_name must be a string")
    try:
        parsed_t = float(time_name)
    except ValueError as exc:
        raise ValueError("time_name must be finite numeric") from exc
    if (
        not np.isfinite(parsed_t)
        or isinstance(time_s, bool)
        or not isinstance(time_s, numbers.Real)
        or not np.isfinite(time_s)
        or abs(float(time_s) - parsed_t) > 1e-12
    ):
        raise ValueError("time_s and time_name must agree")
    if not isinstance(coordinate_frame, str) or not coordinate_frame.strip():
        raise ValueError("coordinate_frame must be non-empty")
    origin = _real_array("coordinate_origin_m", coordinate_origin_m, (3,))
    p = Path(path).resolve()
    if p.suffix != ".vtk" or not p.name.endswith(f"_{time_name}.vtk"):
        raise ValueError("filename must end with _{time_name}.vtk")

    parsed = parse_legacy_ascii_vtk(p)
    if parsed.point_data:
        raise ValueError("point_data is not allowed")
    required = {field_name, "C", "Vc", "cellID"}
    if not required <= set(parsed.cell_data):
        raise ValueError("missing required cell_data")
    for kind, _conn in parsed.cells:
        if kind not in {"tetra", "hexahedron"}:
            raise ValueError(f"unsupported cell block {kind!r}")

    points = parsed.points
    density_raw = parsed.cell_data[field_name]
    centre_raw = parsed.cell_data["C"]
    volume_raw = parsed.cell_data["Vc"]
    ids_raw = parsed.cell_data["cellID"]

    if (
        any(x.dtype.kind in "bOc" for x in (density_raw, centre_raw, volume_raw))
        or ids_raw.dtype.kind not in "iu"
    ):
        raise TypeError("payload dtypes are invalid")
    if density_raw.ndim == 2 and density_raw.shape[1:] == (1,):
        density_raw = density_raw[:, 0]
    if volume_raw.ndim == 2 and volume_raw.shape[1:] == (1,):
        volume_raw = volume_raw[:, 0]
    if ids_raw.ndim == 2 and ids_raw.shape[1:] == (1,):
        ids_raw = ids_raw[:, 0]

    total_cell_count = sum(len(block[1]) for block in parsed.cells)
    if (
        density_raw.shape != (total_cell_count,)
        or centre_raw.shape != (total_cell_count, 3)
        or volume_raw.shape != (total_cell_count,)
        or ids_raw.shape != (total_cell_count,)
    ):
        raise ValueError("C, Vc, cellID, and density payloads have invalid shape")
    if any(
        x.dtype != np.dtype("float32")
        for x in (points, centre_raw, volume_raw, density_raw)
    ) or ids_raw.dtype != np.dtype("int32"):
        raise ValueError(
            "Foundation 14 profile requires float32 C/Vc/density and int32 "
            "cellID payloads"
        )

    density = _real_array(field_name, density_raw, (total_cell_count,))
    centres = _real_array("C", centre_raw, (total_cell_count, 3))
    volumes = _real_array("Vc", volume_raw, (total_cell_count,))
    if (volumes <= 0).any() or (field_name == "rho" and (density < 0).any()):
        raise ValueError("invalid density or Vc")
    ids = np.asarray(ids_raw, dtype=np.int64)
    if (ids < 0).any() or not np.array_equal(np.sort(ids), np.arange(total_cell_count)):
        raise ValueError("cellID must be a permutation of 0..N-1")

    eps = max(
        float(np.finfo(points.dtype).eps),
        float(np.finfo(centre_raw.dtype).eps),
        float(np.finfo(volume_raw.dtype).eps),
        float(np.finfo(density_raw.dtype).eps),
    )
    validate_legacy_vtk_geometry(
        np.asarray(points, dtype=np.float64), parsed.cells, centres, volumes, eps
    )
    world_points = _real_array(
        "world points", np.asarray(points, dtype=np.float64) + origin
    )
    world_centres = _real_array("world C", centres + origin, (len(density), 3))
    prov = _validate_density_provenance(provenance)
    prov.update(
        input_sha256=parsed.sha256,
        points_dtype=points.dtype.name,
        centre_dtype=centre_raw.dtype.name,
        volume_dtype=volume_raw.dtype.name,
        density_dtype=density_raw.dtype.name,
        cell_id_dtype=ids_raw.dtype.name,
    )
    return OpenFOAMDensitySnapshot._from_validated_loader(
        input_path=str(p),
        density_unit=density_unit,
        coordinate_unit=coordinate_unit,
        volume_unit=volume_unit,
        time_unit=time_unit,
        mesh=FoamVolumeMesh(world_points, parsed.cells),
        positions_m=world_centres,
        volumes_m3=volumes,
        density_kg_m3=density,
        cell_ids=ids,
        time_s=time_s,
        time_name=time_name,
        field_name=field_name,
        coordinate_frame=coordinate_frame,
        coordinate_origin_m=origin,
        provenance=prov,
    )


_DEFAULT_VTK_UNITS: dict[str, tuple[u.UnitBase, tuple[int, ...] | None, str | None]] = {
    "rho": (u.kg / u.m**3, (1, -3, 0, 0, 0, 0, 0), None),
    "delta_rho": (u.kg / u.m**3, (1, -3, 0, 0, 0, 0, 0), None),
    "U": (u.m / u.s, (0, 1, -1, 0, 0, 0, 0), None),
    "T": (u.K, (0, 0, 0, 1, 0, 0, 0), None),
    "p": (u.Pa, (1, -1, -2, 0, 0, 0, 0), "physical"),
    "p_rgh": (u.Pa, (1, -1, -2, 0, 0, 0, 0), "physical"),
}


def _infer_time_from_vtk_filename(path: Path) -> tuple[float, str]:
    stem = path.stem
    if "_" in stem:
        suffix = stem.rsplit("_", 1)[-1]
        if _is_time_dir(suffix):
            return float(suffix), suffix
    return 0.0, "0"


def _compute_cell_centers_from_blocks(
    points: np.ndarray, cell_blocks: tuple[tuple[str, np.ndarray], ...]
) -> np.ndarray:
    centers = []
    for _kind, conn in cell_blocks:
        centers.append(points[conn].mean(axis=1))
    if not centers:
        return np.empty((0, 3), dtype=np.float64)
    return np.vstack(centers)


def read_openfoam_vtk(
    path: str | Path,
    *,
    time_s: float | None = None,
    time_name: str | None = None,
    region: str = "region0",
    field_units: Mapping[str, u.UnitBase | str] | None = None,
    field_dimensions: Mapping[str, tuple[int | float, ...]] | None = None,
    validate_geometry: bool = True,
    _fields: Sequence[str] | None = None,
) -> FoamSnapshot:
    """Read an OpenFOAM legacy-ASCII VTK file into a :class:`FoamSnapshot`."""
    p = Path(path).resolve()
    selected = set(_fields) | {"C", "Vc", "cellID"} if _fields is not None else None
    parsed = parse_legacy_ascii_vtk(p, fields=selected)
    inferred_t, inferred_name = _infer_time_from_vtk_filename(p)
    t_val = inferred_t if time_s is None else float(time_s)
    t_name = inferred_name if time_name is None else str(time_name)

    points_f64 = np.asarray(parsed.points, dtype=np.float64)
    if "C" in parsed.cell_data:
        centers = np.asarray(parsed.cell_data["C"], dtype=np.float64)
    else:
        centers = _compute_cell_centers_from_blocks(points_f64, parsed.cells)

    volumes = (
        np.asarray(parsed.cell_data["Vc"], dtype=np.float64).ravel()
        if "Vc" in parsed.cell_data
        else None
    )
    cell_ids = (
        np.asarray(parsed.cell_data["cellID"], dtype=np.int64).ravel()
        if "cellID" in parsed.cell_data
        else None
    )

    if validate_geometry and "C" in parsed.cell_data and volumes is not None:
        all_supported = all(kind in {"tetra", "hexahedron"} for kind, _ in parsed.cells)
        if all_supported:
            eps = float(
                max(
                    np.finfo(parsed.points.dtype).eps,
                    np.finfo(parsed.cell_data["C"].dtype).eps,
                    np.finfo(parsed.cell_data["Vc"].dtype).eps,
                )
            )
            validate_legacy_vtk_geometry(
                points_f64, parsed.cells, centers, volumes, eps
            )

    skip_meta = {"C", "Vc", "cellID"}
    fields: dict[str, FoamFieldData] = {}
    for assoc, source_dict in (
        ("cell", parsed.cell_data),
        ("point", parsed.point_data),
    ):
        for name, arr in source_dict.items():
            if assoc == "cell" and name in skip_meta:
                continue
            if _fields is not None and name not in _fields:
                continue
            vals = np.asarray(arr, dtype=np.float64)
            if vals.ndim == 2 and vals.shape[1] == 1:
                vals = vals[:, 0]
            dims_tuple: tuple[int | float, ...] | None = None
            p_kind: str | None = None
            if field_units and name in field_units:
                unit_obj = u.Unit(field_units[name])
            elif name in _DEFAULT_VTK_UNITS:
                unit_obj, dims_tuple, p_kind = _DEFAULT_VTK_UNITS[name]
            else:
                unit_obj = u.dimensionless_unscaled
            if field_dimensions and name in field_dimensions:
                dims_tuple = tuple(field_dimensions[name])
            foam_cls = (
                "volVectorField"
                if vals.ndim == 2 and vals.shape[1] == 3
                else "volScalarField"
            )
            fields[name] = FoamFieldData(
                name=name,
                values=vals,
                association=assoc,  # type: ignore[arg-type]
                foam_class=foam_cls,
                dimensions=dims_tuple,
                unit=unit_obj,
                pressure_kind=p_kind,
            )

    return FoamSnapshot(
        time_s=t_val,
        time_name=t_name,
        region=region,
        points_m=points_f64,
        cell_centers_m=centers,
        cell_volumes_m3=volumes,
        cell_blocks=parsed.cells,
        cell_ids=cell_ids,
        fields=fields,
        provenance={
            "backend": "vtk_legacy_ascii",
            "input_path": str(p),
            "input_sha256": parsed.sha256,
        },
    )


class LegacyVTKOpenFOAMBackend:
    """Backend for reading OpenFOAM legacy-ASCII VTK snapshots."""

    def read_snapshot(self, path: str | Path, **kwargs: Any) -> FoamSnapshot:
        return read_openfoam_vtk(path, **kwargs)


class PyVistaOpenFOAMBackend:
    """Backend wrapping ``pyvista.OpenFOAMReader`` with lazy import and unit mapping."""

    def __init__(self, case_path: str | Path, *, skip_zero_time: bool = False) -> None:
        self.case_path = Path(case_path)
        self.skip_zero_time = bool(skip_zero_time)

    @contextlib.contextmanager
    def _open_reader(self) -> Any:
        pv = _require_pyvista()
        if self.case_path.is_file():
            reader = pv.OpenFOAMReader(str(self.case_path))
            reader.skip_zero_time = self.skip_zero_time
            reader.cell_to_point_creation = False
            yield reader
            return

        foam_files = sorted(self.case_path.glob("*.foam")) + sorted(
            self.case_path.glob("*.OpenFOAM")
        )
        if foam_files:
            reader = pv.OpenFOAMReader(str(foam_files[0]))
            reader.skip_zero_time = self.skip_zero_time
            reader.cell_to_point_creation = False
            yield reader
            return

        # PyVista OpenFOAMReader requires a *.foam marker file; create a
        # non-mutating symlink shadow directory inside a TemporaryDirectory.
        with tempfile.TemporaryDirectory(prefix="gwexpy_foam_") as tmpdir:
            tmp = Path(tmpdir)
            for child in self.case_path.iterdir():
                (tmp / child.name).symlink_to(child)
            marker = tmp / "case.foam"
            marker.write_text("", encoding="ascii")
            reader = pv.OpenFOAMReader(str(marker))
            reader.skip_zero_time = self.skip_zero_time
            reader.cell_to_point_creation = False
            yield reader

    def read_snapshot(
        self,
        *,
        time_s: float,
        time_name: str,
        region: str = "region0",
        fields: Sequence[str] | None = None,
        field_headers: Mapping[str, OpenFOAMHeader] | None = None,
        solver: str | None = None,
    ) -> FoamSnapshot:
        with self._open_reader() as reader:
            available_times = list(reader.time_values)
            if not available_times:
                raise ValueError(
                    f"No time steps reported by OpenFOAMReader for {self.case_path}"
                )
            # Snap to exact nearest reported time value
            diffs = [abs(float(t) - float(time_s)) for t in available_times]
            best_idx = int(np.argmin(diffs))
            if diffs[best_idx] > 1e-9:
                raise ValueError(
                    f"Requested time {time_s} not found in OpenFOAMReader time_values {available_times}"
                )
            reader.set_active_time_value(available_times[best_idx])

            patch_names = list(reader.patch_array_names)
            reader.disable_all_patch_arrays()
            if region == "region0":
                if "internalMesh" in patch_names:
                    reader.enable_patch_array("internalMesh")
                elif patch_names:
                    reader.enable_patch_array(patch_names[0])
            else:
                target_patch = f"{region}/internalMesh"
                if target_patch in patch_names:
                    reader.enable_patch_array(target_patch)
                else:
                    matched = [
                        p
                        for p in patch_names
                        if p == region or p.startswith(f"{region}/")
                    ]
                    if not matched:
                        raise ValueError(
                            f"Region {region!r} not found in OpenFOAM patches {patch_names}"
                        )
                    for m in matched:
                        reader.enable_patch_array(m)

            if fields is not None:
                wanted = set(fields)
                reader.disable_all_cell_arrays()
                reader.disable_all_point_arrays()
                for name in reader.cell_array_names:
                    if name in wanted:
                        reader.enable_cell_array(name)
                for name in reader.point_array_names:
                    if name in wanted:
                        reader.enable_point_array(name)
            else:
                reader.enable_all_cell_arrays()
                reader.enable_all_point_arrays()

            multiblock = reader.read()
            mesh = self._extract_region_mesh(multiblock, region)

            points = np.asarray(mesh.points, dtype=np.float64)
            centers = np.asarray(mesh.cell_centers().points, dtype=np.float64)
            cell_blocks = self._extract_cell_blocks(mesh)

            foam_fields: dict[str, FoamFieldData] = {}
            for assoc, data_dict in (
                ("cell", mesh.cell_data),
                ("point", mesh.point_data),
            ):
                for name in data_dict.keys():
                    if fields is not None and name not in fields:
                        continue
                    arr = np.asarray(data_dict[name], dtype=np.float64)
                    hdr = field_headers.get(name) if field_headers else None
                    dims_tuple = (
                        hdr.dimensions.exponents if hdr and hdr.dimensions else None
                    )
                    unit_obj = (
                        hdr.dimensions.unit
                        if hdr and hdr.dimensions
                        else _DEFAULT_VTK_UNITS.get(
                            name, (u.dimensionless_unscaled, None, None)
                        )[0]
                    )
                    foam_cls = (
                        hdr.foam_class
                        if hdr
                        else (
                            "volVectorField"
                            if arr.ndim == 2 and arr.shape[1] == 3
                            else "volScalarField"
                        )
                    )
                    p_kind = (
                        hdr.pressure_kind
                        if hdr and hdr.pressure_kind is not None
                        else (
                            classify_pressure_dimensions(dims_tuple, field_name=name)
                            if dims_tuple is not None
                            else None
                        )
                    )
                    foam_fields[name] = FoamFieldData(
                        name=name,
                        values=arr,
                        association=assoc,  # type: ignore[arg-type]
                        foam_class=foam_cls,
                        dimensions=dims_tuple,
                        unit=unit_obj,
                        pressure_kind=p_kind,
                    )

            return FoamSnapshot(
                time_s=float(available_times[best_idx]),
                time_name=time_name,
                region=region,
                points_m=points,
                cell_centers_m=centers,
                cell_volumes_m3=None,
                cell_blocks=cell_blocks,
                cell_ids=None,
                fields=foam_fields,
                provenance={
                    "backend": "pyvista",
                    "case_path": str(self.case_path.resolve()),
                    "solver": solver,
                },
                pyvista_mesh=mesh,
            )

    @staticmethod
    def _extract_region_mesh(multiblock: Any, region: str) -> Any:
        keys = list(multiblock.keys())
        if region == "region0":
            if "internalMesh" in keys:
                return multiblock["internalMesh"]
            for k in keys:
                if multiblock[k] is not None:
                    return multiblock[k]
        if region in keys and multiblock[region] is not None:
            sub = multiblock[region]
            if hasattr(sub, "keys") and "internalMesh" in sub.keys():
                return sub["internalMesh"]
            return sub
        if "internalMesh" in keys and multiblock["internalMesh"] is not None:
            return multiblock["internalMesh"]
        raise ValueError(
            f"Could not locate mesh block for region {region!r} in OpenFOAMReader output {keys}"
        )

    @staticmethod
    def _extract_cell_blocks(mesh: Any) -> tuple[tuple[str, np.ndarray], ...]:
        if hasattr(mesh, "cells_dict"):
            blocks = []
            for ctype, conn in mesh.cells_dict.items():
                kind = _VTK_TYPE_MAP.get(int(ctype), f"vtk_{ctype}")
                blocks.append((kind, np.asarray(conn, dtype=np.int64)))
            return tuple(blocks)
        return ()


class OpenFOAMCase:
    """Case-directory discovery and lazy reading interface for OpenFOAM simulations."""

    def __init__(
        self,
        path: str | Path,
        *,
        skip_zero_time: bool = False,
        backend: Literal["auto", "pyvista", "vtk_legacy_ascii"] = "auto",
    ) -> None:
        raw_path = Path(path)
        if not raw_path.exists():
            raise FileNotFoundError(f"OpenFOAM source does not exist: {raw_path}")
        self.source_path = raw_path.resolve()
        self.skip_zero_time = bool(skip_zero_time)
        self.backend_preference = backend

        if self.source_path.is_file() and self.source_path.suffix == ".vtk":
            self.is_vtk_file = True
            self.case_dir = self.source_path.parent
        elif self.source_path.is_file() and self.source_path.suffix in {
            ".foam",
            ".OpenFOAM",
        }:
            self.is_vtk_file = False
            self.case_dir = self.source_path.parent
        elif self.source_path.is_dir():
            self.is_vtk_file = False
            self.case_dir = self.source_path
        else:
            raise ValueError(
                f"Expected OpenFOAM case directory, .foam file, or legacy .vtk file, got {self.source_path}"
            )

        self._control_dict: dict[str, Any] = {}
        cd_path = self.case_dir / "system" / "controlDict"
        if not self.is_vtk_file and cd_path.is_file():
            self._control_dict = parse_control_dict(cd_path)

    @property
    def solver(self) -> str | None:
        val = self._control_dict.get("application")
        return str(val) if val is not None else None

    @property
    def available_time_names(self) -> tuple[str, ...]:
        """Exact time directory names (or VTK timestamp name), sorted ascending."""
        if self.is_vtk_file:
            _, tname = _infer_time_from_vtk_filename(self.source_path)
            return (tname,)
        pairs: list[tuple[float, str]] = []
        for entry in self.case_dir.iterdir():
            if not entry.is_dir() or not _is_time_dir(entry.name):
                continue
            tval = float(entry.name)
            if self.skip_zero_time and tval == 0.0:
                continue
            pairs.append((tval, entry.name))
        pairs.sort(key=lambda item: item[0])
        return tuple(name for _, name in pairs)

    @property
    def available_times(self) -> tuple[float, ...]:
        """Sorted numeric simulation times (seconds)."""
        return tuple(float(name) for name in self.available_time_names)

    @property
    def available_regions(self) -> tuple[str, ...]:
        """Discovered mesh regions (``('region0',)`` for single-region cases)."""
        if self.is_vtk_file:
            return ("region0",)
        const_dir = self.case_dir / "constant"
        regions: list[str] = []
        if (const_dir / "polyMesh").is_dir():
            regions.append("region0")
        if const_dir.is_dir():
            for child in sorted(const_dir.iterdir()):
                if (
                    child.is_dir()
                    and child.name != "polyMesh"
                    and (child / "polyMesh").is_dir()
                ):
                    regions.append(child.name)
        return tuple(regions) if regions else ("region0",)

    def available_fields(
        self,
        *,
        time: float | str | None = None,
        region: str = "region0",
    ) -> dict[str, OpenFOAMHeader]:
        """Discover available fields and parse their ``FoamFile`` headers/dimensions."""
        if self.is_vtk_file:
            return {
                name: OpenFOAMHeader(
                    foam_class="volVectorField" if count == 3 else "volScalarField",
                    object_name=name,
                    pressure_kind=_DEFAULT_VTK_UNITS.get(name, (None, None, None))[2],
                )
                for association, arrays in list_legacy_vtk_arrays(
                    self.source_path
                ).items()
                for name, count in arrays.items()
                if association != "cell" or name not in {"C", "Vc", "cellID"}
            }

        tname = self._resolve_time_name(time)
        field_dir = self.case_dir / tname
        if region != "region0" and (field_dir / region).is_dir():
            field_dir = field_dir / region
        if not field_dir.is_dir():
            return {}

        headers: dict[str, OpenFOAMHeader] = {}
        for item in sorted(field_dir.iterdir()):
            if not item.is_file():
                continue
            if item.name.startswith(".") or item.name in {"uniform"}:
                continue
            try:
                hdr = parse_foam_header(item)
            except Exception:
                continue
            if hdr.foam_class is not None or hdr.dimensions is not None:
                headers[item.name] = hdr
        return headers

    def _resolve_time_name(self, time: float | str | None) -> str:
        names = self.available_time_names
        if not names:
            raise ValueError(
                f"No time directories found in OpenFOAM case {self.case_dir}"
            )
        if time is None:
            return names[0]
        if isinstance(time, str) and time in names:
            return time
        tval = float(time)
        for name in names:
            if abs(float(name) - tval) <= 1e-12:
                return name
        raise ValueError(
            f"Requested time {time!r} not found in available times {names}"
        )

    def read_snapshot(
        self,
        *,
        time: float | str | None = None,
        region: str = "region0",
        fields: Sequence[str] | None = None,
    ) -> FoamSnapshot:
        """Load a single native mesh-retaining :class:`FoamSnapshot`."""
        if self.is_vtk_file or self.backend_preference == "vtk_legacy_ascii":
            tname = self._resolve_time_name(time) if not self.is_vtk_file else None
            return read_openfoam_vtk(
                self.source_path,
                time_s=float(tname) if tname is not None else None,
                time_name=tname,
                region=region,
                _fields=fields,
            )

        tname = self._resolve_time_name(time)
        headers = self.available_fields(time=tname, region=region)
        backend = PyVistaOpenFOAMBackend(
            self.source_path, skip_zero_time=self.skip_zero_time
        )
        return backend.read_snapshot(
            time_s=float(tname),
            time_name=tname,
            region=region,
            fields=fields,
            field_headers=headers,
            solver=self.solver,
        )

    def read_snapshots(
        self,
        *,
        times: Iterable[float | str] | None = None,
        region: str = "region0",
        fields: Sequence[str] | None = None,
    ) -> list[FoamSnapshot]:
        """Load a list of snapshots across selected (or all) time steps."""
        if times is None:
            target_names = list(self.available_time_names)
        else:
            target_names = [self._resolve_time_name(t) for t in times]
        return [
            self.read_snapshot(time=tname, region=region, fields=fields)
            for tname in target_names
        ]
