"""EXODUS II readers using lazy ``netCDF4`` and ``pyvista`` backends."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .flow3d_profile import Flow3DExodusProfile  # noqa: F401 (registers exo.flow3d)
from .masks import compute_flow3d_masks
from .model import ExodusBlock, ExodusSnapshot, ExodusVariable
from .profiles import ExodusProfile, find_vector_components, get_exodus_profile
from .units import resolve_coordinate_scale, resolve_variable_unit

__all__ = ["ExodusCase", "ExodusIIBackend"]

_VTK_INTERNAL_ARRAYS = frozenset(
    {
        "ObjectId",
        "GlobalNodeId",
        "GlobalElementId",
        "vtkOriginalCellIds",
        "vtkOriginalPointIds",
    }
)


def _selected_variables(
    profile: ExodusProfile, requested: tuple[str, ...], available: tuple[str, ...]
) -> set[str]:
    """Keep selected fields and the auxiliary arrays used for physical masks."""
    selected = {name for name in available if profile.is_mask_variable(name)}
    for qualified in requested:
        name = qualified.rsplit("/", 1)[-1]
        canonical = profile.canonical_name(name)
        selected.update(
            item
            for item in available
            if item.lower() == name.lower() or profile.canonical_name(item) == canonical
        )
        components = find_vector_components(profile, available, name)
        if components is not None:
            selected.update(components)
    return selected


def _require_netcdf4() -> Any:
    try:
        return importlib.import_module("netCDF4")
    except ImportError as exc:
        raise ImportError(
            "The 'netCDF4' package is required for reading EXODUS II files, "
            "but is not installed. Install it via `pip install netCDF4` or "
            "`pip install 'gwexpy[exodus]'`."
        ) from exc


def _require_pyvista() -> Any:
    try:
        return importlib.import_module("pyvista")
    except ImportError as exc:
        raise ImportError(
            "The 'pyvista' package is required for the PyVista EXODUS II backend, "
            "but is not installed. Install it via `pip install pyvista` or "
            "`pip install 'gwexpy[exodus]'`."
        ) from exc


def _select_backend(backend: Literal["auto", "netcdf4", "pyvista"]) -> str:
    if backend == "netcdf4":
        _require_netcdf4()
        return "netcdf4"
    if backend == "pyvista":
        _require_pyvista()
        return "pyvista"
    if backend != "auto":
        raise ValueError(
            f"Unsupported EXODUS II backend {backend!r}; expected 'auto', 'netcdf4', or 'pyvista'"
        )
    try:
        _require_netcdf4()
        return "netcdf4"
    except ImportError:
        pass
    try:
        _require_pyvista()
        return "pyvista"
    except ImportError as exc:
        raise ImportError(
            "Reading EXODUS II files requires 'netCDF4' or 'pyvista', neither of "
            "which is installed. Install via `pip install netCDF4` or "
            "`pip install 'gwexpy[exodus]'`."
        ) from exc


def _decode_char_table(var: Any) -> list[str]:
    """Decode EXODUS II 2-D ``S1`` character tables into clean Python strings."""
    raw = np.asarray(var[:])
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    out: list[str] = []
    for row in raw:
        chars: list[str] = []
        for item in row:
            if np.ma.is_masked(item):
                continue
            b = item.tobytes() if hasattr(item, "tobytes") else bytes(item)
            ch = b.decode("ascii", errors="replace")
            if ch == "\x00":
                break
            chars.append(ch)
        out.append("".join(chars).strip())
    return out


def _nc_array_float64(var: Any, idx: Any = slice(None)) -> np.ndarray:
    arr = var[idx]
    if np.ma.isMaskedArray(arr):
        arr = np.ma.filled(arr.astype(np.float64), np.nan)
    return np.asarray(arr, dtype=np.float64)


class ExodusIIBackend:
    """Backend for inspecting and loading EXODUS II (``.exo`` / ``.e``) files."""

    def __init__(
        self,
        path: str | Path,
        *,
        backend: Literal["auto", "netcdf4", "pyvista"] = "auto",
    ) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"EXODUS II file not found: {self.path}")
        self.backend_name = _select_backend(backend)

    def read_metadata(self) -> dict[str, Any]:
        if self.backend_name == "netcdf4":
            return self._read_metadata_nc()
        return self._read_metadata_pv()

    def read_snapshot(
        self,
        time_index: int = 0,
        *,
        profile: ExodusProfile,
        unit_system: str = "si",
        coordinate_unit: str | None = None,
        favor_threshold: float = 0.0,
        vof_threshold: float = 0.0,
        mask_blocked: bool = True,
        empty_fluid_mode: Literal["keep", "nan", "zero"] = "keep",
        _requested: tuple[str, ...] | None = None,
    ) -> ExodusSnapshot:
        if self.backend_name == "netcdf4":
            return self._read_snapshot_nc(
                time_index,
                profile=profile,
                unit_system=unit_system,
                coordinate_unit=coordinate_unit,
                favor_threshold=favor_threshold,
                vof_threshold=vof_threshold,
                mask_blocked=mask_blocked,
                empty_fluid_mode=empty_fluid_mode,
                _requested=_requested,
            )
        return self._read_snapshot_pv(
            time_index,
            profile=profile,
            unit_system=unit_system,
            coordinate_unit=coordinate_unit,
            favor_threshold=favor_threshold,
            vof_threshold=vof_threshold,
            mask_blocked=mask_blocked,
            empty_fluid_mode=empty_fluid_mode,
            _requested=_requested,
        )

    # ------------------------------------------------------------------
    # netCDF4 EXODUS II implementation
    # ------------------------------------------------------------------

    def _read_metadata_nc(self) -> dict[str, Any]:
        netCDF4 = _require_netcdf4()
        with netCDF4.Dataset(str(self.path), "r") as ds:
            title = str(ds.getncattr("title")) if "title" in ds.ncattrs() else None
            version = (
                float(ds.getncattr("version"))
                if "version" in ds.ncattrs()
                else (
                    float(ds.getncattr("api_version"))
                    if "api_version" in ds.ncattrs()
                    else None
                )
            )
            if "time_whole" in ds.variables and len(ds.variables["time_whole"]) > 0:
                times = tuple(
                    float(t) for t in _nc_array_float64(ds.variables["time_whole"])
                )
            else:
                times = (0.0,)

            n_blk = (
                len(ds.dimensions["num_el_blk"]) if "num_el_blk" in ds.dimensions else 0
            )
            if "eb_prop1" in ds.variables:
                blk_ids = [int(v) for v in ds.variables["eb_prop1"][:]]
            else:
                blk_ids = list(range(1, n_blk + 1))

            if "eb_names" in ds.variables:
                raw_names = _decode_char_table(ds.variables["eb_names"])
                blk_names = [
                    nm if nm else f"block_{bid}"
                    for nm, bid in zip(raw_names, blk_ids, strict=True)
                ]
            else:
                blk_names = [f"block_{bid}" for bid in blk_ids]

            elem_vars: dict[str, str | None] = {}
            elem_var_blocks: dict[str, list[str]] = {}
            if "name_elem_var" in ds.variables:
                e_names = _decode_char_table(ds.variables["name_elem_var"])
                tab = (
                    np.asarray(ds.variables["elem_var_tab"][:], dtype=int)
                    if "elem_var_tab" in ds.variables
                    else None
                )
                for k_idx, vname in enumerate(e_names, start=1):
                    if not vname:
                        continue
                    v_unit: str | None = None
                    active_blocks: list[str] = []
                    for b_idx, bname in enumerate(blk_names, start=1):
                        key = f"vals_elem_var{k_idx}eb{b_idx}"
                        if key in ds.variables and (
                            tab is None or int(tab[b_idx - 1, k_idx - 1]) != 0
                        ):
                            active_blocks.append(bname)
                            var_obj = ds.variables[key]
                            if v_unit is None and "units" in var_obj.ncattrs():
                                v_unit = str(var_obj.getncattr("units"))
                    elem_vars[vname] = v_unit
                    elem_var_blocks[vname] = active_blocks

            nod_vars: dict[str, str | None] = {}
            if "name_nod_var" in ds.variables:
                n_names = _decode_char_table(ds.variables["name_nod_var"])
                for k_idx, vname in enumerate(n_names, start=1):
                    if not vname:
                        continue
                    key = f"vals_nod_var{k_idx}"
                    v_unit = None
                    if key in ds.variables and "units" in ds.variables[key].ncattrs():
                        v_unit = str(ds.variables[key].getncattr("units"))
                    nod_vars[vname] = v_unit

            glo_vars: list[str] = []
            if "name_glo_var" in ds.variables:
                glo_vars = [
                    v for v in _decode_char_table(ds.variables["name_glo_var"]) if v
                ]

            coord_unit: str | None = None
            for cname in ("coordx", "coord"):
                if cname in ds.variables and "units" in ds.variables[cname].ncattrs():
                    coord_unit = str(ds.variables[cname].getncattr("units"))
                    break

        return {
            "title": title,
            "exodus_version": version,
            "times": times,
            "block_ids": tuple(blk_ids),
            "block_names": tuple(blk_names),
            "elem_vars": elem_vars,
            "elem_var_blocks": elem_var_blocks,
            "nod_vars": nod_vars,
            "glo_vars": glo_vars,
            "coord_unit": coord_unit,
        }

    def _read_snapshot_nc(
        self,
        time_index: int,
        *,
        profile: ExodusProfile,
        unit_system: str,
        coordinate_unit: str | None,
        favor_threshold: float,
        vof_threshold: float,
        mask_blocked: bool,
        empty_fluid_mode: Literal["keep", "nan", "zero"],
        _requested: tuple[str, ...] | None = None,
    ) -> ExodusSnapshot:
        netCDF4 = _require_netcdf4()
        meta = self._read_metadata_nc()
        wanted = (
            _selected_variables(
                profile, _requested, (*meta["elem_vars"], *meta["nod_vars"])
            )
            if _requested is not None
            else None
        )
        times = meta["times"]
        if time_index < 0 or time_index >= len(times):
            raise IndexError(f"time_index {time_index} out of range [0, {len(times)})")
        coord_scale, coord_src_unit = resolve_coordinate_scale(
            file_coord_unit=meta["coord_unit"],
            unit_system=unit_system,
            coordinate_unit=coordinate_unit,
        )

        with netCDF4.Dataset(str(self.path), "r") as ds:
            # Coordinates -> (num_nodes, 3) in metres
            n_nodes = len(ds.dimensions["num_nodes"])
            if "coordx" in ds.variables:
                cx = _nc_array_float64(ds.variables["coordx"])
                cy = (
                    _nc_array_float64(ds.variables["coordy"])
                    if "coordy" in ds.variables
                    else np.zeros(n_nodes, dtype=np.float64)
                )
                cz = (
                    _nc_array_float64(ds.variables["coordz"])
                    if "coordz" in ds.variables
                    else np.zeros(n_nodes, dtype=np.float64)
                )
                points_m = np.column_stack([cx, cy, cz]) * coord_scale
            elif "coord" in ds.variables:
                raw_coord = _nc_array_float64(ds.variables["coord"])
                if raw_coord.shape[0] <= 3 and raw_coord.shape[1] == n_nodes:
                    raw_coord = raw_coord.T
                points_m = np.zeros((n_nodes, 3), dtype=np.float64)
                ndim = min(3, raw_coord.shape[1])
                points_m[:, :ndim] = raw_coord[:, :ndim] * coord_scale
            else:
                raise ValueError(f"{self.path}: missing EXODUS coordinate arrays")

            blk_ids: tuple[int, ...] = meta["block_ids"]
            blk_names: tuple[str, ...] = meta["block_names"]
            eb_status = (
                [int(v) for v in ds.variables["eb_status"][:]]
                if "eb_status" in ds.variables
                else [1] * len(blk_ids)
            )

            e_names = (
                _decode_char_table(ds.variables["name_elem_var"])
                if "name_elem_var" in ds.variables
                else []
            )
            tab = (
                np.asarray(ds.variables["elem_var_tab"][:], dtype=int)
                if "elem_var_tab" in ds.variables
                else None
            )

            blocks: dict[str, ExodusBlock] = {}
            cell_data: dict[str, dict[str, np.ndarray]] = {}
            masks: dict[str, dict[str, np.ndarray]] = {}
            # Element and nodal variables may share a name but carry different
            # units. Keep their file metadata keyed by native association.
            elem_var_file_units: dict[str, str | None] = dict(meta["elem_vars"])
            nod_var_file_units: dict[str, str | None] = dict(meta["nod_vars"])

            for b_idx, (bid, bname, status) in enumerate(
                zip(blk_ids, blk_names, eb_status, strict=True), start=1
            ):
                conn_var = ds.variables[f"connect{b_idx}"]
                elem_type = (
                    str(conn_var.getncattr("elem_type"))
                    if "elem_type" in conn_var.ncattrs()
                    else "HEX8"
                )
                conn = np.asarray(conn_var[:], dtype=np.int64) - 1
                if conn.ndim == 1:
                    conn = conn[:, np.newaxis]
                node_indices = np.unique(conn)
                centers = points_m[conn].mean(axis=1)
                blocks[bname] = ExodusBlock(
                    block_id=bid,
                    name=bname,
                    elem_type=elem_type,
                    status=status,
                    connectivity=conn,
                    node_indices=node_indices,
                    cell_centers_m=centers,
                )

                block_wanted = wanted
                if _requested is not None and wanted is not None:
                    available = tuple(
                        name
                        for name, names in meta["elem_var_blocks"].items()
                        if bname in names
                    )
                    block_wanted = wanted | _selected_variables(
                        profile, _requested, (*available, *meta["nod_vars"])
                    )
                b_vars: dict[str, np.ndarray] = {}
                for k_idx, vname in enumerate(e_names, start=1):
                    if not vname or (
                        block_wanted is not None and vname not in block_wanted
                    ):
                        continue
                    var_key = f"vals_elem_var{k_idx}eb{b_idx}"
                    if var_key not in ds.variables:
                        continue
                    if tab is not None and int(tab[b_idx - 1, k_idx - 1]) == 0:
                        continue
                    vobj = ds.variables[var_key]
                    if vobj.ndim == 2:
                        arr = _nc_array_float64(vobj, (time_index, slice(None)))
                    else:
                        arr = _nc_array_float64(vobj)
                    b_vars[vname] = arr
                cell_data[bname] = b_vars

                # Compute per-block FAVOR / VOF masks
                favor_var = profile.find_favor_variable(list(b_vars.keys()))
                vof_var = profile.find_vof_variable(list(b_vars.keys()))
                mask_set = compute_flow3d_masks(
                    len(conn),
                    favor_values=b_vars.get(favor_var) if favor_var else None,
                    favor_variable=favor_var,
                    favor_threshold=favor_threshold,
                    vof_values=b_vars.get(vof_var) if vof_var else None,
                    vof_variable=vof_var,
                    vof_threshold=vof_threshold,
                    mask_blocked=mask_blocked,
                    empty_fluid_mode=empty_fluid_mode,
                )
                masks[bname] = {
                    "open_mask": mask_set.open_mask,
                    "fluid_mask": mask_set.fluid_mask,
                    "valid_mask": mask_set.valid_mask,
                }

            point_data: dict[str, np.ndarray] = {}
            if "name_nod_var" in ds.variables:
                n_names = _decode_char_table(ds.variables["name_nod_var"])
                for k_idx, vname in enumerate(n_names, start=1):
                    if not vname or (wanted is not None and vname not in wanted):
                        continue
                    var_key = f"vals_nod_var{k_idx}"
                    if var_key not in ds.variables:
                        continue
                    vobj = ds.variables[var_key]
                    if vobj.ndim == 2:
                        arr = _nc_array_float64(vobj, (time_index, slice(None)))
                    else:
                        arr = _nc_array_float64(vobj)
                    point_data[vname] = arr

            global_data: dict[str, float] = {}
            if "name_glo_var" in ds.variables and "vals_glo_var" in ds.variables:
                g_names = _decode_char_table(ds.variables["name_glo_var"])
                g_vals = _nc_array_float64(
                    ds.variables["vals_glo_var"], (time_index, slice(None))
                )
                for name, val in zip(g_names, g_vals, strict=False):
                    if name:
                        global_data[name] = float(val)

        return ExodusSnapshot(
            time_s=float(times[time_index]),
            time_index=time_index,
            points_m=points_m,
            blocks=blocks,
            cell_data=cell_data,
            point_data=point_data,
            global_data=global_data,
            masks=masks,
            provenance={
                "backend": "netcdf4",
                "title": meta["title"],
                "exodus_version": meta["exodus_version"],
                "coordinate_unit": str(coord_src_unit),
                "elem_var_file_units": elem_var_file_units,
                "nod_var_file_units": nod_var_file_units,
            },
        )

    # ------------------------------------------------------------------
    # PyVista ExodusIIReader implementation
    # ------------------------------------------------------------------

    def _open_pv_reader(self) -> Any:
        pv = _require_pyvista()
        reader = pv.ExodusIIReader(str(self.path))
        if hasattr(reader, "disable_displacements"):
            reader.disable_displacements()
        reader.enable_all_cell_arrays()
        reader.enable_all_point_arrays()
        reader.enable_all_global_arrays()
        return reader

    def _read_metadata_pv(self) -> dict[str, Any]:
        reader = self._open_pv_reader()
        times = (
            tuple(float(t) for t in reader.time_values)
            if reader.time_values
            else (0.0,)
        )
        reader.disable_all_cell_arrays()
        reader.disable_all_point_arrays()
        reader.disable_all_global_arrays()
        mb = reader.read()
        eb_mb = mb["Element Blocks"] if "Element Blocks" in mb.keys() else mb
        blk_names = tuple(str(k) for k in eb_mb.keys())
        blk_ids = tuple(range(1, len(blk_names) + 1))

        elem_vars: dict[str, str | None] = {}
        elem_var_blocks: dict[str, list[str]] = {}
        for cname in reader.cell_array_names:
            if cname in _VTK_INTERNAL_ARRAYS:
                continue
            elem_vars[cname] = None
            elem_var_blocks[cname] = list(blk_names)

        nod_vars = {
            pname: None
            for pname in reader.point_array_names
            if pname not in _VTK_INTERNAL_ARRAYS
        }
        glo_vars = [
            g for g in reader.global_array_names if g not in _VTK_INTERNAL_ARRAYS
        ]
        return {
            "title": None,
            "exodus_version": None,
            "times": times,
            "block_ids": blk_ids,
            "block_names": blk_names,
            "elem_vars": elem_vars,
            "elem_var_blocks": elem_var_blocks,
            "nod_vars": nod_vars,
            "glo_vars": glo_vars,
            "coord_unit": None,
        }

    def _read_snapshot_pv(
        self,
        time_index: int,
        *,
        profile: ExodusProfile,
        unit_system: str,
        coordinate_unit: str | None,
        favor_threshold: float,
        vof_threshold: float,
        mask_blocked: bool,
        empty_fluid_mode: Literal["keep", "nan", "zero"],
        _requested: tuple[str, ...] | None = None,
    ) -> ExodusSnapshot:
        reader = self._open_pv_reader()
        times = (
            tuple(float(t) for t in reader.time_values)
            if reader.time_values
            else (0.0,)
        )
        if time_index < 0 or time_index >= len(times):
            raise IndexError(f"time_index {time_index} out of range [0, {len(times)})")
        if reader.number_time_points > 0:
            reader.set_active_time_point(time_index)
        if _requested is not None:
            wanted = _selected_variables(
                profile,
                _requested,
                (*reader.cell_array_names, *reader.point_array_names),
            )
            for name in reader.cell_array_names:
                if name not in wanted:
                    reader.disable_cell_array(name)
            for name in reader.point_array_names:
                if name not in wanted:
                    reader.disable_point_array(name)
        mb = reader.read()
        eb_mb = mb["Element Blocks"] if "Element Blocks" in mb.keys() else mb

        coord_scale, coord_src_unit = resolve_coordinate_scale(
            file_coord_unit=None,
            unit_system=unit_system,
            coordinate_unit=coordinate_unit,
        )

        blocks: dict[str, ExodusBlock] = {}
        cell_data: dict[str, dict[str, np.ndarray]] = {}
        masks: dict[str, dict[str, np.ndarray]] = {}
        point_data: dict[str, np.ndarray] = {}

        # Extract combined points / point_data carefully across blocks
        first_mesh = None
        merged_points: list[np.ndarray] = []
        point_offset = 0
        point_data_chunks: dict[str, list[np.ndarray]] = {}
        all_share_same_points = True
        ref_points = None

        for b_idx, bname in enumerate(eb_mb.keys(), start=1):
            grid = eb_mb[bname]
            if grid is None:
                continue
            if first_mesh is None:
                first_mesh = grid
            pts = np.asarray(grid.points, dtype=np.float64) * coord_scale
            if ref_points is None:
                ref_points = pts
            elif pts.shape != ref_points.shape or not np.allclose(pts, ref_points):
                all_share_same_points = False

            centers = (
                np.asarray(grid.cell_centers().points, dtype=np.float64) * coord_scale
            )
            # Extract connectivity if cells_dict exists
            conn = np.empty((grid.n_cells, 0), dtype=np.int64)
            elem_type = "UNSTRUCTURED"
            if hasattr(grid, "cells_dict") and grid.cells_dict:
                first_ctype, first_conn = next(iter(grid.cells_dict.items()))
                conn = np.asarray(first_conn, dtype=np.int64)
                elem_type = f"VTK_{first_ctype}"

            blocks[bname] = ExodusBlock(
                block_id=b_idx,
                name=bname,
                elem_type=elem_type,
                status=1,
                connectivity=conn + (0 if all_share_same_points else point_offset),
                node_indices=np.unique(conn)
                + (0 if all_share_same_points else point_offset),
                cell_centers_m=centers,
            )
            merged_points.append(pts)
            point_offset += len(pts)

            b_vars: dict[str, np.ndarray] = {}
            for key in grid.cell_data.keys():
                if key in _VTK_INTERNAL_ARRAYS:
                    continue
                b_vars[key] = np.asarray(grid.cell_data[key], dtype=np.float64)
            cell_data[bname] = b_vars

            for key in grid.point_data.keys():
                if key in _VTK_INTERNAL_ARRAYS:
                    continue
                point_data_chunks.setdefault(key, []).append(
                    np.asarray(grid.point_data[key], dtype=np.float64)
                )

            favor_var = profile.find_favor_variable(list(b_vars.keys()))
            vof_var = profile.find_vof_variable(list(b_vars.keys()))
            mask_set = compute_flow3d_masks(
                grid.n_cells,
                favor_values=b_vars.get(favor_var) if favor_var else None,
                favor_variable=favor_var,
                favor_threshold=favor_threshold,
                vof_values=b_vars.get(vof_var) if vof_var else None,
                vof_variable=vof_var,
                vof_threshold=vof_threshold,
                mask_blocked=mask_blocked,
                empty_fluid_mode=empty_fluid_mode,
            )
            masks[bname] = {
                "open_mask": mask_set.open_mask,
                "fluid_mask": mask_set.fluid_mask,
                "valid_mask": mask_set.valid_mask,
            }

        if ref_points is None:
            points_m = np.empty((0, 3), dtype=np.float64)
        elif all_share_same_points:
            points_m = ref_points
            point_data = {k: v[0] for k, v in point_data_chunks.items()}
        else:
            points_m = np.vstack(merged_points)
            point_data = {
                k: np.concatenate(v, axis=0) for k, v in point_data_chunks.items()
            }

        return ExodusSnapshot(
            time_s=float(times[time_index]),
            time_index=time_index,
            points_m=points_m,
            blocks=blocks,
            cell_data=cell_data,
            point_data=point_data,
            global_data={},
            masks=masks,
            provenance={
                "backend": "pyvista",
                "title": None,
                "exodus_version": None,
                "coordinate_unit": str(coord_src_unit),
                "elem_var_file_units": {},
                "nod_var_file_units": {},
            },
            pyvista_multiblock=mb,
        )


class ExodusCase:
    """High-level discovery and snapshot interface for EXODUS II simulations."""

    def __init__(
        self,
        path: str | Path,
        *,
        format_name: str = "exo.flow3d",
        profile: ExodusProfile | None = None,
        unit_system: str = "si",
        coordinate_unit: str | None = None,
        backend: Literal["auto", "netcdf4", "pyvista"] = "auto",
    ) -> None:
        self.path = Path(path).resolve()
        self.profile = (
            profile if profile is not None else get_exodus_profile(format_name)
        )
        self.format_name = self.profile.format_name if profile is None else format_name
        self.unit_system = unit_system
        self.coordinate_unit = coordinate_unit
        self.backend = ExodusIIBackend(self.path, backend=backend)
        self._meta = self.backend.read_metadata()

    @property
    def available_times(self) -> tuple[float, ...]:
        return tuple(self._meta["times"])

    @property
    def available_blocks(self) -> tuple[str, ...]:
        return tuple(self._meta["block_names"])

    @property
    def available_variables(self) -> dict[str, ExodusVariable]:
        out: dict[str, ExodusVariable] = {}
        prof_units = self.profile.profile_units_map()
        for name, file_u in self._meta["elem_vars"].items():
            canon = self.profile.canonical_name(name)
            unit_obj, _, unit_src = resolve_variable_unit(
                canonical_name=canon,
                file_unit_str=file_u,
                unit_system=self.unit_system,
                convert_to_si=True,
                profile_units=prof_units,
            )
            out[name] = ExodusVariable(
                name=name,
                canonical_name=canon,
                association="cell",
                unit=unit_obj,
                unit_source=unit_src,
                block_names=tuple(self._meta["elem_var_blocks"].get(name, ())),
                quantity_kind=self.profile.quantity_kind(name),
            )
        for name, file_u in self._meta["nod_vars"].items():
            canon = self.profile.canonical_name(name)
            unit_obj, _, unit_src = resolve_variable_unit(
                canonical_name=canon,
                file_unit_str=file_u,
                unit_system=self.unit_system,
                convert_to_si=True,
                profile_units=prof_units,
            )
            out[name] = ExodusVariable(
                name=name,
                canonical_name=canon,
                association="point",
                unit=unit_obj,
                unit_source=unit_src,
                block_names=self.available_blocks,
                quantity_kind=self.profile.quantity_kind(name),
            )
        for name in self._meta["glo_vars"]:
            out[name] = ExodusVariable(
                name=name,
                canonical_name=self.profile.canonical_name(name),
                association="global",
                unit=None,
                unit_source="unknown",
            )
        return out

    def _resolve_time_indices(
        self,
        *,
        time: float | int | None = None,
        time_index: int | None = None,
        times: Iterable[float] | None = None,
    ) -> list[int]:
        avail = self.available_times
        if sum(x is not None for x in (time, time_index, times)) > 1:
            raise ValueError("Specify at most one of 'time', 'time_index', or 'times'")
        if time_index is not None:
            idx = int(time_index)
            if idx < 0:
                idx += len(avail)
            if idx < 0 or idx >= len(avail):
                raise IndexError(f"time_index {time_index} out of range")
            return [idx]
        if time is not None:
            t_req = float(time)
            diffs = [abs(t - t_req) for t in avail]
            best = int(np.argmin(diffs))
            if diffs[best] > 1e-9:
                raise ValueError(f"Time {time!r} not found in available times {avail}")
            return [best]
        if times is not None:
            indices: list[int] = []
            for t_req in times:
                diffs = [abs(t - float(t_req)) for t in avail]
                best = int(np.argmin(diffs))
                if diffs[best] > 1e-9:
                    raise ValueError(
                        f"Time {t_req!r} not found in available times {avail}"
                    )
                indices.append(best)
            return indices
        return list(range(len(avail)))

    def read_snapshot(
        self,
        time_index: int = 0,
        *,
        favor_threshold: float = 0.0,
        vof_threshold: float = 0.0,
        mask_blocked: bool = True,
        empty_fluid_mode: Literal["keep", "nan", "zero"] = "keep",
        _requested: tuple[str, ...] | None = None,
    ) -> ExodusSnapshot:
        """Load one native mesh-retaining :class:`ExodusSnapshot`."""
        return self.backend.read_snapshot(
            time_index,
            profile=self.profile,
            unit_system=self.unit_system,
            coordinate_unit=self.coordinate_unit,
            favor_threshold=favor_threshold,
            vof_threshold=vof_threshold,
            mask_blocked=mask_blocked,
            empty_fluid_mode=empty_fluid_mode,
            _requested=_requested,
        )

    def read_snapshots(
        self,
        *,
        time: float | None = None,
        time_index: int | None = None,
        times: Iterable[float] | None = None,
        favor_threshold: float = 0.0,
        vof_threshold: float = 0.0,
        mask_blocked: bool = True,
        empty_fluid_mode: Literal["keep", "nan", "zero"] = "keep",
        _requested: tuple[str, ...] | None = None,
    ) -> list[ExodusSnapshot]:
        """Load one or more snapshots selected by time or time index."""
        indices = self._resolve_time_indices(
            time=time, time_index=time_index, times=times
        )
        return [
            self.read_snapshot(
                idx,
                favor_threshold=favor_threshold,
                vof_threshold=vof_threshold,
                mask_blocked=mask_blocked,
                empty_fluid_mode=empty_fluid_mode,
                _requested=_requested,
            )
            for idx in indices
        ]
