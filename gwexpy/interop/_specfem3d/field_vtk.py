"""Readers for SPECFEM3D legacy ASCII VTK, XML VTU, and partitioned PVTU files."""

from __future__ import annotations

import importlib
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy import units as u

from gwexpy.interop._openfoam.backend import (
    list_legacy_vtk_arrays,
    parse_legacy_ascii_vtk,
)

from .model import SpecfemFieldSnapshot
from .units import resolve_field_unit

__all__ = [
    "read_specfem_pvtu",
    "read_specfem_vtk",
    "read_specfem_vtu",
]


def _require_pyvista_for_vtu() -> Any:
    try:
        return importlib.import_module("pyvista")
    except ImportError as exc:
        raise ImportError(
            "The 'pyvista' package is required for reading binary/appended XML "
            "VTU or PVTU files, but is not installed. Install it via "
            "`pip install pyvista` or `pip install 'gwexpy[specfem3d]'`."
        ) from exc


def _infer_vtk_field_name(path: Path, raw_name: str) -> str:
    """Resolve SPECFEM generic array names (like ``gll_data``) from filename when helpful."""
    if raw_name.lower() in {"gll_data", "elem_val", "scalars_"}:
        stem = path.stem
        for suffix in ("_vp", "_vs", "_rho", "_qkappa", "_qmu", "_jacobian_min"):
            if stem.endswith(suffix):
                return suffix.lstrip("_")
    return raw_name


def list_specfem_vtk_arrays(
    path: Path, association: Literal["point", "cell"] | None
) -> dict[str, int]:
    """List names and component counts from VTK array declarations."""
    first_piece = None
    if path.suffix.lower() == ".vtk":
        arrays = list_legacy_vtk_arrays(path)
    else:
        arrays = {"point": {}, "cell": {}}
        parent = None
        with path.open("rb") as stream:
            for event, elem in ET.iterparse(stream, events=("start", "end")):
                tag = elem.tag
                if event == "start":
                    if tag == "AppendedData":
                        break
                    if tag == "Piece" and first_piece is None:
                        first_piece = elem.get("Source")
                    if tag in {"PointData", "PPointData", "CellData", "PCellData"}:
                        parent = "point" if "Point" in tag else "cell"
                    if parent and tag in {"DataArray", "PDataArray"}:
                        name = elem.get("Name", f"{parent}_field")
                        arrays[parent][name] = int(elem.get("NumberOfComponents", "1"))
                else:
                    if tag in {"PointData", "PPointData", "CellData", "PCellData"}:
                        parent = None
                    if tag == "PUnstructuredGrid" or (
                        tag == "Piece" and first_piece is None
                    ):
                        break
                    elem.clear()
    selected = association or ("point" if arrays["point"] else "cell")
    if not arrays[selected] and path.suffix.lower() == ".pvtu" and first_piece:
        piece = path.parent / first_piece
        if piece.suffix.lower() == ".vtu":
            return list_specfem_vtk_arrays(piece, association)
    return {
        _infer_vtk_field_name(path, name): count
        for name, count in arrays[selected].items()
    }


def _read_xml_ascii_vtu(
    path: Path,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    tuple[tuple[str, np.ndarray], ...],
]:
    """Parse an ASCII-encoded XML ``.vtu`` UnstructuredGrid without PyVista."""
    tree = ET.parse(path)
    root = tree.getroot()
    ug = root.find("UnstructuredGrid")
    if ug is None:
        raise ValueError(f"{path}: missing <UnstructuredGrid> element")
    piece = ug.find("Piece")
    if piece is None:
        raise ValueError(f"{path}: missing <Piece> element")

    def _parse_data_array(elem: ET.Element) -> np.ndarray:
        fmt = elem.attrib.get("format", "ascii").lower()
        if fmt != "ascii":
            raise NotImplementedError(
                f"Non-ASCII DataArray format={fmt!r} requires pyvista"
            )
        text = (elem.text or "").strip()
        dtype_str = elem.attrib.get("type", "Float64").lower()
        dt: Any = np.int64 if "int" in dtype_str else np.float64
        arr = np.fromstring(text, sep=" ", dtype=dt)
        ncomp = int(elem.attrib.get("NumberOfComponents", "1"))
        if ncomp > 1:
            arr = arr.reshape((-1, ncomp))
        return arr

    pts_elem = piece.find("./Points/DataArray")
    if pts_elem is None:
        raise ValueError(f"{path}: missing Points/DataArray")
    points = _parse_data_array(pts_elem).astype(np.float64)

    point_data: dict[str, np.ndarray] = {}
    pd_elem = piece.find("PointData")
    if pd_elem is not None:
        for da in pd_elem.findall("DataArray"):
            name = da.attrib.get("Name", "point_field")
            point_data[name] = _parse_data_array(da).astype(np.float64)

    cell_data: dict[str, np.ndarray] = {}
    cd_elem = piece.find("CellData")
    if cd_elem is not None:
        for da in cd_elem.findall("DataArray"):
            name = da.attrib.get("Name", "cell_field")
            cell_data[name] = _parse_data_array(da).astype(np.float64)

    blocks: tuple[tuple[str, np.ndarray], ...] = ()
    cells_elem = piece.find("Cells")
    if cells_elem is not None:
        conn_el = next(
            (
                da
                for da in cells_elem.findall("DataArray")
                if da.attrib.get("Name") == "connectivity"
            ),
            None,
        )
        off_el = next(
            (
                da
                for da in cells_elem.findall("DataArray")
                if da.attrib.get("Name") == "offsets"
            ),
            None,
        )
        if conn_el is not None and off_el is not None:
            conn_flat = _parse_data_array(conn_el).astype(np.int64)
            offsets = _parse_data_array(off_el).astype(np.int64)
            if len(offsets) > 0 and offsets[0] == 8 and np.all(np.diff(offsets) == 8):
                blocks = (("hexahedron", conn_flat.reshape((len(offsets), 8))),)

    return points, point_data, cell_data, blocks


def _read_vtu_via_pyvista(
    path: Path,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    tuple[tuple[str, np.ndarray], ...],
]:
    pv = _require_pyvista_for_vtu()
    mesh = pv.read(str(path))
    points = np.asarray(mesh.points, dtype=np.float64)
    point_data = {
        k: np.asarray(mesh.point_data[k], dtype=np.float64)
        for k in mesh.point_data.keys()
    }
    cell_data = {
        k: np.asarray(mesh.cell_data[k], dtype=np.float64)
        for k in mesh.cell_data.keys()
    }
    blocks: list[tuple[str, np.ndarray]] = []
    if hasattr(mesh, "cells_dict"):
        for ctype, conn in mesh.cells_dict.items():
            kind = (
                "hexahedron"
                if int(ctype) == 12
                else ("tetra" if int(ctype) == 10 else f"vtk_{ctype}")
            )
            blocks.append((kind, np.asarray(conn, dtype=np.int64)))
    return points, point_data, cell_data, tuple(blocks)


def read_specfem_vtk(
    path: str | Path,
    *,
    time_s: float = 0.0,
    association: Literal["point", "cell"] | None = None,
    field_units: Mapping[str, u.UnitBase | str] | None = None,
) -> SpecfemFieldSnapshot:
    """Read a SPECFEM3D legacy-ASCII VTK file (e.g. ``proc*_vp.vtk``, ``proc*_mesh.vtk``)."""
    p = Path(path).resolve()
    parsed = parse_legacy_ascii_vtk(p)
    points = np.asarray(parsed.points, dtype=np.float64)

    has_point = bool(parsed.point_data)
    has_cell = bool(parsed.cell_data)
    resolved_assoc: Literal["point", "cell"]
    if association is not None:
        resolved_assoc = association
    elif has_point:
        resolved_assoc = "point"
    elif has_cell:
        resolved_assoc = "cell"
    else:
        raise ValueError(
            f"{p}: legacy VTK file contains neither POINT_DATA nor CELL_DATA"
        )

    raw_dict = parsed.point_data if resolved_assoc == "point" else parsed.cell_data
    coords = (
        points
        if resolved_assoc == "point"
        else np.vstack([points[conn].mean(axis=1) for _, conn in parsed.cells])
    )

    fields: dict[str, np.ndarray] = {}
    units_out: dict[str, u.UnitBase] = {}
    for raw_name, arr in raw_dict.items():
        name = _infer_vtk_field_name(p, raw_name)
        vals = np.asarray(arr, dtype=np.float64)
        if vals.ndim == 2 and vals.shape[1] == 1:
            vals = vals[:, 0]
        fields[name] = vals
        if field_units and name in field_units:
            units_out[name] = u.Unit(field_units[name])
        else:
            units_out[name] = resolve_field_unit(name)

    return SpecfemFieldSnapshot(
        source_path=str(p),
        source_kind="vtk",
        times_s=np.array([float(time_s)], dtype=np.float64),
        coordinates_m=coords,
        fields=fields,
        field_units=units_out,
        association=resolved_assoc,
        cell_blocks=parsed.cells,
        provenance={"input_sha256": parsed.sha256},
    )


def read_specfem_vtu(
    path: str | Path,
    *,
    time_s: float = 0.0,
    association: Literal["point", "cell"] | None = None,
    field_units: Mapping[str, u.UnitBase | str] | None = None,
) -> SpecfemFieldSnapshot:
    """Read a SPECFEM3D XML ``.vtu`` file (pure ASCII XML or lazy PyVista)."""
    p = Path(path).resolve()
    try:
        points, point_data, cell_data, blocks = _read_xml_ascii_vtu(p)
        backend_used = "xml_ascii"
    except (NotImplementedError, ET.ParseError):
        points, point_data, cell_data, blocks = _read_vtu_via_pyvista(p)
        backend_used = "pyvista"

    resolved_assoc: Literal["point", "cell"]
    if association is not None:
        resolved_assoc = association
    elif point_data:
        resolved_assoc = "point"
    elif cell_data:
        resolved_assoc = "cell"
    else:
        raise ValueError(f"{p}: VTU file contains neither PointData nor CellData")

    raw_dict = point_data if resolved_assoc == "point" else cell_data
    if resolved_assoc == "point":
        coords = points
    else:
        coords = (
            np.vstack([points[conn].mean(axis=1) for _, conn in blocks])
            if blocks
            else points
        )

    fields: dict[str, np.ndarray] = {}
    units_out: dict[str, u.UnitBase] = {}
    for raw_name, arr in raw_dict.items():
        name = _infer_vtk_field_name(p, raw_name)
        vals = np.asarray(arr, dtype=np.float64)
        if vals.ndim == 2 and vals.shape[1] == 1:
            vals = vals[:, 0]
        fields[name] = vals
        units_out[name] = (
            u.Unit(field_units[name])
            if field_units and name in field_units
            else resolve_field_unit(name)
        )

    return SpecfemFieldSnapshot(
        source_path=str(p),
        source_kind="vtu",
        times_s=np.array([float(time_s)], dtype=np.float64),
        coordinates_m=coords,
        fields=fields,
        field_units=units_out,
        association=resolved_assoc,
        cell_blocks=blocks,
        provenance={"backend": backend_used},
    )


def read_specfem_pvtu(
    path: str | Path,
    *,
    time_s: float = 0.0,
    association: Literal["point", "cell"] | None = None,
    field_units: Mapping[str, u.UnitBase | str] | None = None,
) -> SpecfemFieldSnapshot:
    """Read a partitioned SPECFEM3D ``.pvtu`` file and merge all piece ``.vtu`` shards."""
    p = Path(path).resolve()
    tree = ET.parse(p)
    root = tree.getroot()
    pug = root.find("PUnstructuredGrid")
    if pug is None:
        raise ValueError(f"{p}: missing <PUnstructuredGrid> element")
    pieces = [
        el.attrib["Source"] for el in pug.findall("Piece") if "Source" in el.attrib
    ]
    if not pieces:
        raise ValueError(f"{p}: no <Piece Source=.../> entries found")

    piece_snaps = [
        read_specfem_vtu(
            p.parent / rel,
            time_s=time_s,
            association=association,
            field_units=field_units,
        )
        for rel in pieces
    ]
    resolved_assoc = piece_snaps[0].association
    coords = np.vstack([s.coordinates_m for s in piece_snaps])
    common_fields = sorted(set(piece_snaps[0].fields.keys()))
    merged_fields: dict[str, np.ndarray] = {}
    for fname in common_fields:
        merged_fields[fname] = np.concatenate(
            [s.fields[fname] for s in piece_snaps], axis=0
        )

    return SpecfemFieldSnapshot(
        source_path=str(p),
        source_kind="pvtu",
        times_s=np.array([float(time_s)], dtype=np.float64),
        coordinates_m=coords,
        fields=merged_fields,
        field_units=dict(piece_snaps[0].field_units),
        association=resolved_assoc,
        provenance={"n_pieces": len(piece_snaps), "pieces": tuple(pieces)},
    )
