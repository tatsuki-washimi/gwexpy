# Adapted from GraGra (https://github.com/tatsuki-washimi/gragra)
# Commit: ce3a9f0f7c686097dec2cb34aca80222a88d11e1
# Source: src/gragra/adapters/specfem/specfem_bin.py
# Copyright (c) 2026 Tatsuki Washimi — MIT License
"""Fortran unformatted binary readers and MPI partition merger for SPECFEM3D."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from .gll import EXPECTED_NGLL, ngll_from_shape
from .model import SpecfemFieldSnapshot
from .units import resolve_field_unit

__all__ = [
    "DatabaseContents",
    "DatabaseElement",
    "DatabaseMaterial",
    "PartArray",
    "RecordLayout",
    "detect_displacement_layout",
    "detect_record_layout",
    "gather_element_coordinates",
    "merge_specfem_partitions",
    "parse_database",
    "read_coords",
    "read_displacement_snapshots",
    "read_element_scalar_bin",
    "read_ibool",
    "read_part_array_vtk",
    "read_rank_displacement_snapshots",
    "read_records",
    "read_rho",
    "recover_element_ids",
]

_MARKER_CANDIDATES: Final[tuple[tuple[str, int], ...]] = (
    ("<", 4),
    (">", 4),
    ("<", 8),
    (">", 8),
)


@dataclass(frozen=True)
class RecordLayout:
    """Auto-detected Fortran unformatted sequential record layout."""

    endianness: str
    record_marker_bytes: int

    @property
    def marker_dtype(self) -> np.dtype[Any]:
        return np.dtype(f"{self.endianness}i{self.record_marker_bytes}")


def _split_records(data: bytes, endian: str, marker_bytes: int) -> list[bytes] | None:
    marker_dtype = np.dtype(f"{endian}i{marker_bytes}")
    size = len(data)
    offset = 0
    payloads: list[bytes] = []
    while offset < size:
        if offset + marker_bytes > size:
            return None
        head = int(np.frombuffer(data, dtype=marker_dtype, count=1, offset=offset)[0])
        if head < 0:
            return None
        payload_start = offset + marker_bytes
        payload_end = payload_start + head
        tail_end = payload_end + marker_bytes
        if tail_end > size:
            return None
        tail = int(
            np.frombuffer(data, dtype=marker_dtype, count=1, offset=payload_end)[0]
        )
        if tail != head:
            return None
        payloads.append(data[payload_start:payload_end])
        offset = tail_end
    return payloads


def detect_record_layout(
    data: bytes, *, source: str = "<bytes>"
) -> tuple[RecordLayout, list[bytes]]:
    """Auto-detect Fortran unformatted ``(endianness, record_marker_bytes)``."""
    for endian, marker_bytes in _MARKER_CANDIDATES:
        payloads = _split_records(data, endian, marker_bytes)
        if payloads is not None:
            return RecordLayout(endian, marker_bytes), payloads
    raise ValueError(
        f"{source}: could not auto-detect Fortran unformatted record layout"
    )


def read_records(path: str | Path) -> tuple[RecordLayout, list[bytes]]:
    """Read an entire Fortran unformatted sequential file."""
    p = Path(path)
    return detect_record_layout(p.read_bytes(), source=str(p))


def _require_single_record(path: str | Path, payloads: list[bytes]) -> bytes:
    if len(payloads) != 1:
        raise ValueError(
            f"{path}: expected exactly one Fortran record, found {len(payloads)}"
        )
    return payloads[0]


def _detect_float_size(path: str | Path, payload: bytes, expected_count: int) -> int:
    if expected_count <= 0:
        raise ValueError(f"expected_count must be positive, got {expected_count}")
    if len(payload) == expected_count * 4:
        return 4
    if len(payload) == expected_count * 8:
        return 8
    raise ValueError(
        f"{path}: payload length {len(payload)} does not match expected_count={expected_count}"
    )


def read_ibool(path: str | Path, ngll: int) -> tuple[np.ndarray, RecordLayout, int]:
    """Read ``proc*_ibool.bin`` into zero-based ``(nspec, ngll)`` int64 array."""
    if ngll <= 0:
        raise ValueError(f"ngll must be positive, got {ngll}")
    layout, payloads = read_records(path)
    payload = _require_single_record(path, payloads)
    integer_kind: int | None = None
    for candidate in (4, 8):
        denom = candidate * ngll
        if len(payload) % denom == 0 and len(payload) // denom > 0:
            integer_kind = candidate
            break
    if integer_kind is None:
        raise ValueError(
            f"{path}: payload length {len(payload)} is not divisible by ngll={ngll}"
        )
    nspec = len(payload) // (integer_kind * ngll)
    dtype = np.dtype(f"{layout.endianness}i{integer_kind}")
    ibool_one = (
        np.frombuffer(payload, dtype=dtype).reshape((nspec, ngll)).astype(np.int64)
    )
    min_val = int(ibool_one.min())
    if min_val < 1:
        raise ValueError(f"{path}: ibool contains non-positive value {min_val}")
    nglob = int(ibool_one.max())
    if len(np.unique(ibool_one)) != nglob:
        raise ValueError(f"{path}: ibool is not a surjection onto 1..{nglob}")
    return ibool_one - 1, layout, integer_kind


def read_coords(path: str | Path, nglob: int) -> tuple[np.ndarray, RecordLayout]:
    """Read ``proc*_{x,y,z}.bin`` into float64 ``(nglob,)`` array."""
    if nglob <= 0:
        raise ValueError(f"nglob must be positive, got {nglob}")
    layout, payloads = read_records(path)
    payload = _require_single_record(path, payloads)
    float_size = _detect_float_size(path, payload, nglob)
    dtype = np.dtype(f"{layout.endianness}f{float_size}")
    return np.frombuffer(payload, dtype=dtype).astype(np.float64), layout


def read_element_scalar_bin(
    path: str | Path, nspec: int, ngll: int
) -> tuple[np.ndarray, RecordLayout]:
    """Read an element-local GLL scalar array (``rho``, ``vp``, ``vs``, etc.)."""
    if nspec <= 0 or ngll <= 0:
        raise ValueError(f"nspec/ngll must be positive, got {nspec}, {ngll}")
    layout, payloads = read_records(path)
    payload = _require_single_record(path, payloads)
    float_size = _detect_float_size(path, payload, nspec * ngll)
    dtype = np.dtype(f"{layout.endianness}f{float_size}")
    arr = np.frombuffer(payload, dtype=dtype).astype(np.float64).reshape((nspec, ngll))
    if not np.isfinite(arr).all():
        raise ValueError(f"{path}: array contains non-finite values")
    return arr, layout


def read_rho(
    path: str | Path, nspec: int, ngll: int
) -> tuple[np.ndarray, RecordLayout]:
    """Read ``proc*_rho.bin`` enforcing finite positive values."""
    rho, layout = read_element_scalar_bin(path, nspec, ngll)
    if (rho <= 0.0).any():
        raise ValueError(f"{path}: rho must be strictly positive")
    return rho, layout


def gather_element_coordinates(ibool: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Gather ``(nspec, ngll, 3)`` GLL coordinates from ``ibool`` and ``(nglob, 3)`` coords."""
    if ibool.min() < 0 or ibool.max() >= len(xyz):
        raise ValueError("ibool index out of bounds for xyz array")
    return xyz[ibool]


_DISPL_FILENAME_RE: Final[re.Pattern[str]] = re.compile(r"^displ_([XYZ])_it(\d+)\.bin$")
_RANK_DISPL_FILENAME_RE: Final[re.Pattern[str]] = re.compile(
    r"^proc(\d+)_displ_([XYZ])_it(\d+)\.bin$"
)
DisplacementLayout = Literal["per_rank", "whole_mesh"]


def detect_displacement_layout(dirpath: str | Path) -> DisplacementLayout:
    """Detect whether ``dirpath`` stores per-rank or whole-mesh displacement snapshots."""
    p = Path(dirpath)
    if any(p.glob("proc*_displ_*_it*.bin")):
        return "per_rank"
    if any(p.glob("displ_*_it*.bin")):
        return "whole_mesh"
    raise ValueError(f"{p}: no displacement snapshot files found")


def _read_snapshot_records(
    by_it: dict[int, dict[str, Path]],
    nspec: int,
    ngll: int,
    *,
    missing_component_context: str = "",
) -> tuple[np.ndarray, list[int], RecordLayout]:
    expected_count = nspec * ngll
    layout_ref: RecordLayout | None = None
    its_sorted = sorted(by_it)
    snapshots: list[np.ndarray] = []
    for it in its_sorted:
        comps = by_it[it]
        missing = sorted({"X", "Y", "Z"} - set(comps))
        if missing:
            raise ValueError(
                f"{missing_component_context}it={it:06d}: missing component(s) {missing}"
            )
        axis_arrays = []
        for comp in ("X", "Y", "Z"):
            comp_path = comps[comp]
            layout, payloads = read_records(comp_path)
            payload = _require_single_record(comp_path, payloads)
            float_size = _detect_float_size(comp_path, payload, expected_count)
            dtype = np.dtype(f"{layout.endianness}f{float_size}")
            arr = (
                np.frombuffer(payload, dtype=dtype)
                .astype(np.float64)
                .reshape((nspec, ngll))
            )
            axis_arrays.append(arr)
            if layout_ref is None:
                layout_ref = layout
        snapshots.append(np.stack(axis_arrays, axis=-1))
    assert layout_ref is not None
    return np.stack(snapshots, axis=0), its_sorted, layout_ref


def read_displacement_snapshots(
    dirpath: str | Path, nspec: int, ngll: int, glob_pattern: str = "displ_*_it*.bin"
) -> tuple[np.ndarray, list[int], RecordLayout]:
    """Read whole-mesh ``displ_{X,Y,Z}_it*.bin`` snapshots."""
    p_dir = Path(dirpath)
    by_it: dict[int, dict[str, Path]] = {}
    for p in sorted(p_dir.glob(glob_pattern)):
        m = _DISPL_FILENAME_RE.match(p.name)
        if m is not None:
            by_it.setdefault(int(m.group(2)), {})[m.group(1)] = p
    if not by_it:
        raise ValueError(f"{p_dir}: no displ_{{X,Y,Z}}_it*.bin files matched")
    return _read_snapshot_records(by_it, nspec, ngll)


def read_rank_displacement_snapshots(
    dirpath: str | Path,
    rank: int,
    nspec_rank: int,
    ngll: int,
    glob_pattern: str | None = None,
) -> tuple[np.ndarray, list[int], RecordLayout]:
    """Read one rank's ``proc{rank:06d}_displ_{X,Y,Z}_it*.bin`` snapshots."""
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    p_dir = Path(dirpath)
    pattern = glob_pattern or f"proc{rank:06d}_displ_*_it*.bin"
    by_it: dict[int, dict[str, Path]] = {}
    for p in sorted(p_dir.glob(pattern)):
        m = _RANK_DISPL_FILENAME_RE.match(p.name)
        if m is not None and int(m.group(1)) == rank:
            by_it.setdefault(int(m.group(3)), {})[m.group(2)] = p
    if not by_it:
        raise ValueError(f"{p_dir}: no proc{rank:06d}_displ_*_it*.bin files matched")
    return _read_snapshot_records(
        by_it, nspec_rank, ngll, missing_component_context=f"rank {rank}, "
    )


@dataclass(frozen=True)
class DatabaseMaterial:
    values: np.ndarray

    @property
    def domain_id(self) -> float:
        return float(self.values[6])


@dataclass(frozen=True)
class DatabaseElement:
    iloc: int
    mat1: int
    mat2: int
    node_loc_ids: tuple[int, int, int, int, int, int, int, int]


@dataclass(frozen=True)
class DatabaseContents:
    nnodes_loc: int
    node_iloc: np.ndarray
    node_xyz: np.ndarray
    materials: tuple[DatabaseMaterial, ...]
    nspec_local: int
    elements: tuple[DatabaseElement, ...]
    layout: RecordLayout
    integer_kind: int


def parse_database(path: str | Path) -> DatabaseContents:
    """Parse a ``proc*_Database`` file's node, material, and element blocks."""
    layout, payloads = read_records(path)
    idx = 0

    def next_payload() -> bytes:
        nonlocal idx
        if idx >= len(payloads):
            raise ValueError(f"{path}: unexpected EOF while parsing Database")
        p = payloads[idx]
        idx += 1
        return p

    nnodes_payload = next_payload()
    if len(nnodes_payload) not in (4, 8):
        raise ValueError(
            f"{path}: invalid nnodes_loc record length {len(nnodes_payload)}"
        )
    integer_kind = len(nnodes_payload)
    int_dtype = np.dtype(f"{layout.endianness}i{integer_kind}")
    float_dtype = np.dtype(f"{layout.endianness}f8")

    nnodes_loc = int(np.frombuffer(nnodes_payload, dtype=int_dtype)[0])
    node_iloc = np.empty(nnodes_loc, dtype=np.int64)
    node_xyz = np.empty((nnodes_loc, 3), dtype=np.float64)
    for n in range(nnodes_loc):
        p = next_payload()
        if len(p) != integer_kind + 24:
            raise ValueError(f"{path}: unexpected node record length {len(p)}")
        node_iloc[n] = int(np.frombuffer(p, dtype=int_dtype, count=1)[0])
        node_xyz[n] = np.frombuffer(p, dtype=float_dtype, count=3, offset=integer_kind)

    ndef_p = next_payload()
    ndef, nundef = (int(v) for v in np.frombuffer(ndef_p, dtype=int_dtype, count=2))
    materials: list[DatabaseMaterial] = []
    for _ in range(ndef):
        p = next_payload()
        materials.append(
            DatabaseMaterial(
                values=np.frombuffer(p, dtype=float_dtype, count=17).copy()
            )
        )
    for _ in range(nundef):
        next_payload()

    nspec_local = int(np.frombuffer(next_payload(), dtype=int_dtype)[0])
    elements: list[DatabaseElement] = []
    for _ in range(nspec_local):
        vals = np.frombuffer(next_payload(), dtype=int_dtype, count=11)
        elements.append(
            DatabaseElement(
                iloc=int(vals[0]),
                mat1=int(vals[1]),
                mat2=int(vals[2]),
                node_loc_ids=(
                    int(vals[3]),
                    int(vals[4]),
                    int(vals[5]),
                    int(vals[6]),
                    int(vals[7]),
                    int(vals[8]),
                    int(vals[9]),
                    int(vals[10]),
                ),
            )
        )
    return DatabaseContents(
        nnodes_loc=nnodes_loc,
        node_iloc=node_iloc,
        node_xyz=node_xyz,
        materials=tuple(materials),
        nspec_local=nspec_local,
        elements=tuple(elements),
        layout=layout,
        integer_kind=integer_kind,
    )


@dataclass(frozen=True)
class PartArray:
    points: np.ndarray
    cells: np.ndarray
    elem_flag: np.ndarray


_VTK_POINTS_RE: Final[re.Pattern[str]] = re.compile(r"^POINTS\s+(\d+)\s+(\S+)\s*$")
_VTK_CELLS_RE: Final[re.Pattern[str]] = re.compile(r"^CELLS\s+(\d+)\s+(\d+)\s*$")
_VTK_CELL_DATA_RE: Final[re.Pattern[str]] = re.compile(r"^CELL_DATA\s+(\d+)\s*$")


def read_part_array_vtk(path: str | Path) -> PartArray:
    """Parse ``part_array.vtk`` (ASCII legacy VTK hex mesh with ``elem_flag`` rank ids)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    n_lines = len(lines)
    points: np.ndarray | None = None
    cells: np.ndarray | None = None
    elem_flag: np.ndarray | None = None

    i = 0
    while i < n_lines:
        line = lines[i].strip()
        m_pts = _VTK_POINTS_RE.match(line)
        if m_pts:
            n_pts = int(m_pts.group(1))
            vals: list[float] = []
            i += 1
            while len(vals) < n_pts * 3:
                vals.extend(float(v) for v in lines[i].split())
                i += 1
            points = np.array(vals, dtype=np.float64).reshape((n_pts, 3))
            continue
        m_cells = _VTK_CELLS_RE.match(line)
        if m_cells:
            n_cells = int(m_cells.group(1))
            rows = []
            i += 1
            for _ in range(n_cells):
                parts = [int(v) for v in lines[i].split()]
                if not parts or parts[0] != 8:
                    raise ValueError(
                        f"{path}: expected VTK_HEXAHEDRON (8), got {parts[:1]}"
                    )
                rows.append(parts[1:9])
                i += 1
            cells = np.array(rows, dtype=np.int64)
            continue
        m_cd = _VTK_CELL_DATA_RE.match(line)
        if m_cd:
            n_cd = int(m_cd.group(1))
            i += 1
            while i < n_lines and not lines[i].strip().upper().startswith("SCALARS"):
                i += 1
            i += 1
            if i < n_lines and lines[i].strip().upper().startswith("LOOKUP_TABLE"):
                i += 1
            vals_int: list[int] = []
            while len(vals_int) < n_cd:
                vals_int.extend(int(float(v)) for v in lines[i].split())
                i += 1
            elem_flag = np.array(vals_int, dtype=np.int64)
            continue
        i += 1

    if points is None or cells is None or elem_flag is None:
        raise ValueError(f"{path}: missing POINTS, CELLS, or CELL_DATA/elem_flag")
    return PartArray(points=points, cells=cells, elem_flag=elem_flag)


def recover_element_ids(part: PartArray, rank: int) -> np.ndarray:
    """Recover zero-based ``element_id`` array for ``rank`` from ``part_array.vtk``."""
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}")
    matches = np.nonzero(part.elem_flag == rank)[0]
    if matches.size == 0:
        raise ValueError(f"No elements found for rank {rank} in elem_flag")
    return np.sort(matches).astype(np.int64)


_PROC_DB_RE: Final[re.Pattern[str]] = re.compile(r"^proc(\d+)_Database$")


def discover_partition_scalars(path: Path) -> tuple[str, ...]:
    """Discover the supported scalar arrays from rank-zero file declarations."""
    return tuple(
        name
        for name in ("rho", "vp", "vs", "qkappa", "qmu")
        if (path / f"proc000000_{name}.bin").is_file()
    )


def merge_specfem_partitions(
    databases_dir: str | Path,
    *,
    ngll_shape: tuple[int, int, int] = EXPECTED_NGLL,
    fields: Sequence[str] | None = ("rho",),
    include_displacement: bool = False,
    dt_s: float = 1.0,
    t0_s: float = 0.0,
) -> SpecfemFieldSnapshot:
    """Merge MPI-partitioned SPECFEM3D ``DATABASES_MPI`` files into global element order.

    Reads ``proc*_Database``, ``proc*_ibool.bin``, ``proc*_{x,y,z}.bin``,
    requested scalar ``proc*_<field>.bin`` arrays, optional
    ``part_array.vtk`` for partition-independent global ``element_id``
    recovery, and optional per-rank/whole-mesh displacement movie snapshots.
    """
    db_dir = Path(databases_dir).resolve()
    if not db_dir.is_dir():
        raise FileNotFoundError(f"SPECFEM3D databases directory not found: {db_dir}")

    ngll = ngll_from_shape(ngll_shape)
    ranks: list[int] = []
    for p in sorted(db_dir.iterdir()):
        m = _PROC_DB_RE.match(p.name)
        if m is not None and p.is_file():
            ranks.append(int(m.group(1)))
    ranks.sort()
    if not ranks:
        raise FileNotFoundError(f"No proc*_Database files found in {db_dir}")
    if ranks != list(range(len(ranks))):
        raise ValueError(
            f"Incomplete MPI rank files in {db_dir}: found {ranks}, expected 0..{len(ranks) - 1}"
        )
    nproc = len(ranks)

    part_path = db_dir / "part_array.vtk"
    part_array = read_part_array_vtk(part_path) if part_path.is_file() else None
    elem_id_method = (
        "pre_partition_global_index" if part_array is not None else "rank_local_order"
    )

    # Determine which scalar bin fields exist across rank 0
    if fields is None:
        target_fields = discover_partition_scalars(db_dir)
    else:
        target_fields = tuple(fields)

    rank_coords: list[np.ndarray] = []
    rank_elem_ids: list[np.ndarray] = []
    rank_part_ids: list[np.ndarray] = []
    rank_mat_ids: list[np.ndarray] = []
    rank_dom_ids: list[np.ndarray] = []
    rank_scalars: dict[str, list[np.ndarray]] = {f: [] for f in target_fields}
    rank_displs: list[np.ndarray] = []
    displ_its: list[int] | None = None

    has_per_rank_displ = include_displacement and any(
        db_dir.glob("proc*_displ_*_it*.bin")
    )
    has_whole_displ = (
        include_displacement
        and not has_per_rank_displ
        and any(db_dir.glob("displ_*_it*.bin"))
    )

    running_offset = 0
    layout_ref: RecordLayout | None = None
    int_kind_ref: int | None = None

    for r in ranks:
        prefix = f"proc{r:06d}"
        db = parse_database(db_dir / f"{prefix}_Database")
        ibool, layout, int_kind = read_ibool(db_dir / f"{prefix}_ibool.bin", ngll)
        if layout_ref is None:
            layout_ref = layout
            int_kind_ref = int_kind

        nspec_r = ibool.shape[0]
        if nspec_r != db.nspec_local:
            raise ValueError(
                f"Rank {r}: ibool element count {nspec_r} != Database nspec_local {db.nspec_local}"
            )
        nglob_r = int(ibool.max()) + 1
        rx, _ = read_coords(db_dir / f"{prefix}_x.bin", nglob_r)
        ry, _ = read_coords(db_dir / f"{prefix}_y.bin", nglob_r)
        rz, _ = read_coords(db_dir / f"{prefix}_z.bin", nglob_r)
        xyz_r = np.column_stack([rx, ry, rz])
        rank_coords.append(gather_element_coordinates(ibool, xyz_r))

        if part_array is not None:
            eids = recover_element_ids(part_array, r)
            if len(eids) != nspec_r:
                raise ValueError(
                    f"Rank {r}: part_array elem_flag count {len(eids)} != nspec_r {nspec_r}"
                )
        else:
            eids = np.arange(running_offset, running_offset + nspec_r, dtype=np.int64)
        running_offset += nspec_r

        rank_elem_ids.append(eids)
        rank_part_ids.append(np.full(nspec_r, r, dtype=np.int64))
        mat_arr = np.array([el.mat1 for el in db.elements], dtype=np.int64)
        dom_arr = np.array(
            [int(round(db.materials[el.mat1 - 1].domain_id)) for el in db.elements],
            dtype=np.int64,
        )
        rank_mat_ids.append(mat_arr)
        rank_dom_ids.append(dom_arr)

        for fname in target_fields:
            bin_path = db_dir / f"{prefix}_{fname}.bin"
            if fname == "rho":
                arr, _ = read_rho(bin_path, nspec_r, ngll)
            else:
                arr, _ = read_element_scalar_bin(bin_path, nspec_r, ngll)
            rank_scalars[fname].append(arr)

        if has_per_rank_displ:
            displ_r, its_r, _ = read_rank_displacement_snapshots(
                db_dir, r, nspec_r, ngll
            )
            if displ_its is None:
                displ_its = its_r
            elif displ_its != its_r:
                raise ValueError(
                    f"Rank {r}: displacement time step list {its_r} != {displ_its}"
                )
            rank_displs.append(displ_r)

    all_eids = np.concatenate(rank_elem_ids)
    n_total = len(all_eids)
    if not np.array_equal(np.sort(all_eids), np.arange(n_total, dtype=np.int64)):
        raise ValueError(
            "Recovered global element IDs do not form a bijection onto 0..N-1"
        )
    inv_perm = np.argsort(all_eids)

    coords_global = np.concatenate(rank_coords, axis=0)[inv_perm]
    elem_id_sorted = all_eids[inv_perm]
    part_id_sorted = np.concatenate(rank_part_ids)[inv_perm]
    mat_id_sorted = np.concatenate(rank_mat_ids)[inv_perm]
    dom_id_sorted = np.concatenate(rank_dom_ids)[inv_perm]

    field_arrays: dict[str, np.ndarray] = {}
    field_units: dict[str, Any] = {}
    for fname in target_fields:
        field_arrays[fname] = np.concatenate(rank_scalars[fname], axis=0)[inv_perm]
        field_units[fname] = resolve_field_unit(fname)

    if has_per_rank_displ and displ_its is not None:
        displ_merged = np.concatenate(rank_displs, axis=1)[:, inv_perm]
        field_arrays["displacement"] = displ_merged
        field_units["displacement"] = resolve_field_unit("displacement")
        times_arr = np.asarray(displ_its, dtype=np.float64) * float(dt_s) - float(t0_s)
    elif has_whole_displ:
        displ_whole, its_w, _ = read_displacement_snapshots(db_dir, n_total, ngll)
        field_arrays["displacement"] = displ_whole
        field_units["displacement"] = resolve_field_unit("displacement")
        times_arr = np.asarray(its_w, dtype=np.float64) * float(dt_s) - float(t0_s)
    else:
        times_arr = np.array([0.0], dtype=np.float64)

    return SpecfemFieldSnapshot(
        source_path=str(db_dir),
        source_kind="partition_bin",
        times_s=times_arr,
        coordinates_m=coords_global,
        fields=field_arrays,
        field_units=field_units,
        association="gll",
        element_id=elem_id_sorted,
        partition_id=part_id_sorted,
        material_id=mat_id_sorted,
        domain_id=dom_id_sorted,
        ngll_shape=ngll_shape,
        provenance={
            "nproc": nproc,
            "element_id_method": elem_id_method,
            "endianness": layout_ref.endianness if layout_ref else "<",
            "record_marker_bytes": layout_ref.record_marker_bytes if layout_ref else 4,
            "integer_kind": int_kind_ref or 4,
        },
    )
