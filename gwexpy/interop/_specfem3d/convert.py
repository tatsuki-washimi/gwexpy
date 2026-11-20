"""Conversion of SPECFEM3D fields and seismograms into GWexpy containers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy import units as u

from gwexpy.fields import FieldDict, ScalarField, VectorField
from gwexpy.timeseries import TimeSeriesDict

from .case import Specfem3DCase
from .field_hdf5 import (
    PATH_FIELD_DISPLACEMENT,
    PATH_MATERIAL_DENSITY,
    SpecfemGLLDataset,
    _import_h5py,
    read_specfem_gll_snapshot,
)
from .field_vtk import (
    list_specfem_vtk_arrays,
    read_specfem_pvtu,
    read_specfem_vtk,
    read_specfem_vtu,
)
from .gll import EXPECTED_NGLL, deduplicate_gll_points
from .grid import RegularGridSpec, infer_exact_cartesian_grid, sample_specfem_to_grid
from .model import Specfem3DProvenance, SpecfemFieldSnapshot
from .partitions import merge_specfem_partitions
from .seismogram_ascii import read_specfem_seismograms_ascii
from .units import resolve_field_unit

__all__ = [
    "from_specfem3d",
    "list_fields_specfem3d",
    "read_fielddict_specfem3d",
    "read_scalarfield_specfem3d",
    "read_seismograms_specfem3d",
    "read_vectorfield_specfem3d",
    "register_specfem3d_format",
    "register_specfem3d_formats",
]

_COMP_INDEX: dict[str, int] = {"x": 0, "y": 1, "z": 2}


def _snapshot_from_gll_dataset(ds: SpecfemGLLDataset) -> SpecfemFieldSnapshot:
    fields: dict[str, np.ndarray] = {
        "rho": ds.density_kg_m3,
        "density": ds.density_kg_m3,
    }
    units_map: dict[str, u.UnitBase] = {
        "rho": u.kg / u.m**3,
        "density": u.kg / u.m**3,
    }
    if ds.displacement_m is not None:
        fields["displacement"] = ds.displacement_m
        units_map["displacement"] = u.m
    return SpecfemFieldSnapshot(
        source_path=str(ds.file_identity.path),
        source_kind="gll_hdf5",
        times_s=ds.time_s,
        coordinates_m=ds.coordinates_m,
        fields=fields,
        field_units=units_map,
        association="gll",
        element_id=ds.element_id,
        quadrature_volume_m3=ds.quadrature_volume_m3,
    )


def _coerce_field_snapshot(
    source: str | Path | Specfem3DCase | SpecfemFieldSnapshot | SpecfemGLLDataset,
    *,
    ngll_shape: tuple[int, int, int] = EXPECTED_NGLL,
    fields: Sequence[str] | None = None,
    include_displacement: bool = False,
    association: Literal["point", "cell"] | None = None,
) -> SpecfemFieldSnapshot:
    if isinstance(source, SpecfemFieldSnapshot):
        return source
    if isinstance(source, SpecfemGLLDataset):
        return _snapshot_from_gll_dataset(source)
    if isinstance(source, Specfem3DCase):
        return source.read_field_snapshot(
            ngll_shape=ngll_shape,
            fields=fields,
            include_displacement=include_displacement,
            association=association,
        )

    p = Path(source).resolve()
    if p.is_dir():
        if (p / "DATA" / "Par_file").is_file() or (p / "OUTPUT_FILES").is_dir():
            return Specfem3DCase(p).read_field_snapshot(
                ngll_shape=ngll_shape,
                fields=fields,
                include_displacement=include_displacement,
                association=association,
            )
        return merge_specfem_partitions(
            p,
            ngll_shape=ngll_shape,
            fields=fields,
            include_displacement=include_displacement,
        )
    suf = p.suffix.lower()
    if suf in {".h5", ".hdf5"}:
        return read_specfem_gll_snapshot(p)
    if suf == ".vtk":
        return read_specfem_vtk(p, association=association)
    if suf == ".vtu":
        return read_specfem_vtu(p, association=association)
    if suf == ".pvtu":
        return read_specfem_pvtu(p, association=association)
    raise ValueError(f"Unsupported SPECFEM3D field source: {p}")


def _resolve_continuity_policy(
    field_name: str,
    verify_continuity: Literal["auto", "raise", "ignore"],
) -> Literal["raise", "ignore"]:
    if verify_continuity == "raise":
        return "raise"
    if verify_continuity == "ignore":
        return "ignore"
    low = field_name.lower()
    if low in {"displacement", "velocity", "acceleration"} or low.startswith(
        ("displ", "vel", "accel")
    ):
        return "raise"
    return "ignore"


def _prepare_field_values(
    snap: SpecfemFieldSnapshot,
    field: str,
    *,
    component: Literal["x", "y", "z"] | None,
    expected_vector: bool,
    deduplicate_gll: bool,
    verify_continuity: Literal["auto", "raise", "ignore"],
    continuity_atol: float,
) -> tuple[np.ndarray, np.ndarray, u.UnitBase, bool, bool, float | None]:
    if field not in snap.fields:
        raise KeyError(
            f"Field {field!r} not found in SPECFEM3D snapshot. Available: {sorted(snap.fields)}"
        )
    raw_arr = np.asarray(snap.fields[field], dtype=np.float64)
    unit_obj = snap.field_units.get(field, resolve_field_unit(field))

    # Determine whether raw_arr has a leading time dimension
    is_gll = snap.coordinates_m.ndim == 3
    if is_gll:
        nspec, ngll, _ = snap.coordinates_m.shape
        if raw_arr.shape[:2] == (nspec, ngll):
            has_time = False
            trailing = raw_arr.shape[2:]
        elif raw_arr.ndim >= 3 and raw_arr.shape[1:3] == (nspec, ngll):
            has_time = True
            trailing = raw_arr.shape[3:]
        else:
            raise ValueError(
                f"Field {field!r} shape {raw_arr.shape} incompatible with GLL coords {snap.coordinates_m.shape}"
            )
    else:
        npts = snap.coordinates_m.shape[0]
        if raw_arr.shape[0] == npts and not (
            raw_arr.ndim >= 2 and raw_arr.shape[1] == npts
        ):
            has_time = False
            trailing = raw_arr.shape[1:]
        elif raw_arr.ndim >= 2 and raw_arr.shape[1] == npts:
            has_time = True
            trailing = raw_arr.shape[2:]
        else:
            raise ValueError(
                f"Field {field!r} shape {raw_arr.shape} incompatible with coords {snap.coordinates_m.shape}"
            )

    is_vector = trailing == (3,)
    if expected_vector and not is_vector:
        raise ValueError(
            f"Expected 3-component vector field for {field!r}, got shape {raw_arr.shape}"
        )
    if not expected_vector and is_vector and component is None:
        raise ValueError(
            f"Field {field!r} is a 3-component vector field; pass component='x'|'y'|'z' "
            "or use read_vectorfield_specfem3d()"
        )
    if not is_vector and component is not None:
        raise ValueError(
            f"Field {field!r} is scalar; component={component!r} is invalid"
        )

    selected_arr = (
        raw_arr[..., _COMP_INDEX[component]]
        if (is_vector and component is not None)
        else raw_arr
    )

    max_disc: float | None = None
    did_dedup = False
    if deduplicate_gll and (is_gll or snap.source_kind == "pvtu"):
        policy = _resolve_continuity_policy(field, verify_continuity)
        dedup = deduplicate_gll_points(
            snap.coordinates_m,
            selected_arr,
            verify_continuity=policy,
            continuity_atol=continuity_atol,
        )
        coords_out = dedup.coordinates_m
        vals_out = dedup.values
        did_dedup = True
        max_disc = dedup.max_discrepancy
    else:
        if is_gll:
            nspec, ngll, _ = snap.coordinates_m.shape
            coords_out = snap.coordinates_m.reshape((nspec * ngll, 3))
            if has_time:
                vals_out = selected_arr.reshape(
                    (selected_arr.shape[0], nspec * ngll, *selected_arr.shape[3:])
                )
            else:
                vals_out = selected_arr.reshape((nspec * ngll, *selected_arr.shape[2:]))
        else:
            coords_out = snap.coordinates_m
            vals_out = selected_arr

    if not has_time:
        vals_out = vals_out[np.newaxis, ...]

    return coords_out, vals_out, unit_obj, has_time, did_dedup, max_disc


def _build_scalarfield_specfem(
    snap: SpecfemFieldSnapshot,
    *,
    field: str,
    component: Literal["x", "y", "z"] | None,
    grid_mode: Literal["exact", "sample"],
    grid: RegularGridSpec | None,
    sample_method: Literal["linear", "nearest"],
    deduplicate_gll: bool,
    verify_continuity: Literal["auto", "raise", "ignore"],
    continuity_atol: float,
    unit: Any | None,
) -> ScalarField:
    if grid_mode not in {"exact", "sample"}:
        raise ValueError(f"grid_mode must be 'exact' or 'sample', got {grid_mode!r}")
    if grid_mode == "sample" and grid is None:
        raise ValueError(
            "grid_mode='sample' requires an explicit RegularGridSpec via grid="
        )

    coords_m, vals_t, unit_obj, has_time, did_dedup, max_disc = _prepare_field_values(
        snap,
        field,
        component=component,
        expected_vector=False,
        deduplicate_gll=deduplicate_gll,
        verify_continuity=verify_continuity,
        continuity_atol=continuity_atol,
    )

    if unit is not None:
        target_unit = u.Unit(unit)
        if unit_obj != u.dimensionless_unscaled and not unit_obj.is_equivalent(
            target_unit
        ):
            raise u.UnitConversionError(
                f"Cannot convert {field!r} from {unit_obj} to {target_unit}"
            )
        scale = (
            1.0
            if unit_obj == u.dimensionless_unscaled
            else float(unit_obj.to(target_unit))
        )
    else:
        target_unit = unit_obj
        scale = 1.0

    times_arr = (
        snap.times_s
        if has_time
        else np.array([float(snap.times_s[0])], dtype=np.float64)
    )
    nt = vals_t.shape[0]

    if grid_mode == "exact":
        x_1d, y_1d, z_1d, idx_grid = infer_exact_cartesian_grid(coords_m)
        if grid is not None:
            if (
                len(x_1d) != len(grid.x)
                or len(y_1d) != len(grid.y)
                or len(z_1d) != len(grid.z)
                or not np.allclose(x_1d, grid.x, rtol=1e-5, atol=1e-6)
                or not np.allclose(y_1d, grid.y, rtol=1e-5, atol=1e-6)
                or not np.allclose(z_1d, grid.z, rtol=1e-5, atol=1e-6)
            ):
                raise ValueError(
                    "Native SPECFEM3D grid does not match target grid under grid_mode='exact'"
                )
        nx, ny, nz = len(x_1d), len(y_1d), len(z_1d)
        data_4d = np.empty((nt, nx, ny, nz), dtype=np.float64)
        for ti in range(nt):
            data_4d[ti] = vals_t[ti][idx_grid] * scale
        valid_mask_4d: np.ndarray | None = None
        valid_fraction: float | None = 1.0
        interp_method: str | None = None
    else:
        assert grid is not None
        x_1d, y_1d, z_1d = grid.x, grid.y, grid.z
        nx, ny, nz = grid.shape_xyz
        data_4d = np.empty((nt, nx, ny, nz), dtype=np.float64)
        valid_mask_4d = np.empty((nt, nx, ny, nz), dtype=bool)
        interp_method = None
        for ti in range(nt):
            sampled, mask_t, used_m = sample_specfem_to_grid(
                coords_m,
                vals_t[ti],
                grid,
                method=sample_method,
                raw_coords_m=snap.coordinates_m,
                cell_blocks=snap.cell_blocks,
            )
            data_4d[ti] = sampled * scale
            valid_mask_4d[ti] = mask_t
            interp_method = used_m
        valid_fraction = float(np.mean(valid_mask_4d))

    sf = ScalarField(
        data_4d,
        unit=target_unit,
        name=f"{field}_{component}" if component else field,
        axis0=times_arr * u.s,
        axis1=x_1d * u.m,
        axis2=y_1d * u.m,
        axis3=z_1d * u.m,
        axis_names=["t", "x", "y", "z"],
        axis0_domain="time",
        space_domain="real",
        copy=False,
    )
    sf._gwex_specfem3d = Specfem3DProvenance(
        source_path=snap.source_path,
        source_kind=snap.source_kind,  # type: ignore[arg-type]
        field_or_channel=f"{field}_{component}" if component else field,
        coordinate_convention=str(snap.provenance.get("coordinate_convention", "ENU")),
        association=snap.association,
        grid_mode=grid_mode,
        unit_str=str(target_unit),
        t0_s=float(times_arr[0]) if len(times_arr) else None,
        dt_s=float(snap.provenance["dt_s"]) if "dt_s" in snap.provenance else None,
        nproc=int(snap.provenance["nproc"]) if "nproc" in snap.provenance else None,
        ngll_shape=snap.ngll_shape,
        element_id_method=snap.provenance.get("element_id_method"),
        n_elements=int(len(snap.element_id)) if snap.element_id is not None else None,
        deduplicated_gll=did_dedup,
        max_shared_gll_discrepancy=max_disc,
        par_file_hash=snap.provenance.get("par_file_hash"),
        interpolation_method=interp_method,
        valid_fraction=valid_fraction,
        valid_mask=valid_mask_4d,
    )
    return sf


def read_scalarfield_specfem3d(
    source: str | Path | Specfem3DCase | SpecfemFieldSnapshot | SpecfemGLLDataset,
    field: str = "rho",
    *,
    component: Literal["x", "y", "z"] | None = None,
    grid_mode: Literal["exact", "sample"] = "exact",
    grid: RegularGridSpec | None = None,
    sample_method: Literal["linear", "nearest"] = "linear",
    deduplicate_gll: bool = True,
    verify_continuity: Literal["auto", "raise", "ignore"] = "auto",
    continuity_atol: float = 1e-6,
    ngll_shape: tuple[int, int, int] = EXPECTED_NGLL,
    include_displacement: bool = False,
    association: Literal["point", "cell"] | None = None,
    unit: Any | None = None,
) -> ScalarField:
    """Read a SPECFEM3D scalar field (or vector component) into a :class:`ScalarField`."""
    req_fields = None if field in {"displacement", "rho", "density"} else (field,)
    snap = _coerce_field_snapshot(
        source,
        ngll_shape=ngll_shape,
        fields=req_fields,
        include_displacement=include_displacement or field == "displacement",
        association=association,
    )
    return _build_scalarfield_specfem(
        snap,
        field=field,
        component=component,
        grid_mode=grid_mode,
        grid=grid,
        sample_method=sample_method,
        deduplicate_gll=deduplicate_gll,
        verify_continuity=verify_continuity,
        continuity_atol=continuity_atol,
        unit=unit,
    )


def read_vectorfield_specfem3d(
    source: str | Path | Specfem3DCase | SpecfemFieldSnapshot | SpecfemGLLDataset,
    field: str = "displacement",
    *,
    grid_mode: Literal["exact", "sample"] = "exact",
    grid: RegularGridSpec | None = None,
    sample_method: Literal["linear", "nearest"] = "linear",
    deduplicate_gll: bool = True,
    verify_continuity: Literal["auto", "raise", "ignore"] = "auto",
    continuity_atol: float = 1e-6,
    ngll_shape: tuple[int, int, int] = EXPECTED_NGLL,
    unit: Any | None = None,
) -> VectorField:
    """Read a SPECFEM3D 3-component vector wavefield into a :class:`VectorField`."""
    snap = _coerce_field_snapshot(
        source,
        ngll_shape=ngll_shape,
        fields=(),
        include_displacement=(field == "displacement"),
    )
    comps: dict[str, ScalarField] = {}
    for comp in ("x", "y", "z"):
        comps[comp] = _build_scalarfield_specfem(
            snap,
            field=field,
            component=comp,  # type: ignore[arg-type]
            grid_mode=grid_mode,
            grid=grid,
            sample_method=sample_method,
            deduplicate_gll=deduplicate_gll,
            verify_continuity=verify_continuity,
            continuity_atol=continuity_atol,
            unit=unit,
        )
    return VectorField(comps, basis="cartesian", validate=True)


def read_fielddict_specfem3d(
    source: str | Path | Specfem3DCase | SpecfemFieldSnapshot | SpecfemGLLDataset,
    fields: Sequence[str] | None = None,
    *,
    grid_mode: Literal["exact", "sample"] = "exact",
    grid: RegularGridSpec | None = None,
    sample_method: Literal["linear", "nearest"] = "linear",
    deduplicate_gll: bool = True,
    verify_continuity: Literal["auto", "raise", "ignore"] = "auto",
    continuity_atol: float = 1e-6,
    ngll_shape: tuple[int, int, int] = EXPECTED_NGLL,
    include_displacement: bool = True,
    _group_vectors: bool = False,
) -> FieldDict:
    """Read multiple SPECFEM3D fields into a :class:`FieldDict`."""
    req_fields: Sequence[str] | None = fields
    want_disp = include_displacement
    if fields is not None:
        want_disp = "displacement" in fields or (
            include_displacement and not _group_vectors
        )
        scalar_only = tuple(f for f in fields if f != "displacement")
        req_fields = scalar_only if scalar_only or _group_vectors else ("rho",)
    snap = _coerce_field_snapshot(
        source,
        ngll_shape=ngll_shape,
        fields=req_fields,
        include_displacement=want_disp,
    )
    target_fields = list(fields) if fields is not None else sorted(snap.fields.keys())
    out: dict[str, ScalarField | VectorField] = {}
    for fname in target_fields:
        arr = snap.fields.get(fname)
        if arr is None:
            raise KeyError(f"Field {fname!r} not found in snapshot")
        is_vec = arr.shape[-1] == 3 and (
            (snap.coordinates_m.ndim == 3 and arr.ndim >= 3)
            or (snap.coordinates_m.ndim == 2 and arr.ndim >= 2)
        )
        if is_vec:
            components: dict[str, ScalarField] = {}
            for comp in ("x", "y", "z"):
                components[comp] = _build_scalarfield_specfem(
                    snap,
                    field=fname,
                    component=comp,  # type: ignore[arg-type]
                    grid_mode=grid_mode,
                    grid=grid,
                    sample_method=sample_method,
                    deduplicate_gll=deduplicate_gll,
                    verify_continuity=verify_continuity,
                    continuity_atol=continuity_atol,
                    unit=None,
                )
            if _group_vectors:
                out[fname] = VectorField(components, basis="cartesian", validate=True)
            else:
                out.update(
                    (f"{fname}_{key}", value) for key, value in components.items()
                )
        else:
            out[fname] = _build_scalarfield_specfem(
                snap,
                field=fname,
                component=None,
                grid_mode=grid_mode,
                grid=grid,
                sample_method=sample_method,
                deduplicate_gll=deduplicate_gll,
                verify_continuity=verify_continuity,
                continuity_atol=continuity_atol,
                unit=None,
            )
    return FieldDict(out, validate=False)


def _discover_specfem_fields(
    source: str | Path | Specfem3DCase | SpecfemFieldSnapshot | SpecfemGLLDataset,
    *,
    association: Literal["point", "cell"] | None = None,
) -> dict[str, int]:
    """Return field names and component counts without materializing fields."""
    if isinstance(source, SpecfemFieldSnapshot):
        scalar_ndim = source.coordinates_m.ndim - 1
        return {
            name: int(array.shape[-1]) if array.ndim > scalar_ndim else 1
            for name, array in source.fields.items()
        }
    if isinstance(source, SpecfemGLLDataset):
        fields = {"rho": 1, "density": 1}
        if source.displacement_m is not None:
            fields["displacement"] = 3
        return fields

    path = source.root if isinstance(source, Specfem3DCase) else Path(source)
    if isinstance(source, Specfem3DCase) or (
        path.is_dir()
        and ((path / "DATA" / "Par_file").is_file() or (path / "OUTPUT_FILES").is_dir())
    ):
        case = source if isinstance(source, Specfem3DCase) else Specfem3DCase(path)
        if case.databases_dir is None:
            raise FileNotFoundError(f"No DATABASES_MPI directory found under {path}")
        path = case.databases_dir

    if path.is_dir():
        return _discover_partition_fields(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        h5py = _import_h5py()
        with h5py.File(path, "r") as handle:
            fields = {"rho": 1, "density": 1} if PATH_MATERIAL_DENSITY in handle else {}
            if PATH_FIELD_DISPLACEMENT in handle:
                fields["displacement"] = 3
            return fields
    if path.suffix.lower() in {".vtk", ".vtu", ".pvtu"}:
        return list_specfem_vtk_arrays(path, association)
    raise ValueError(f"Unsupported SPECFEM3D field source: {path}")


def _discover_partition_fields(path: Path) -> dict[str, int]:
    from .partitions import discover_partition_scalars

    fields = dict.fromkeys(discover_partition_scalars(path), 1)
    if any(path.glob("proc*_displ_*_it*.bin")) or any(path.glob("displ_*_it*.bin")):
        fields["displacement"] = 3
    return fields


def list_fields_specfem3d(
    source: str | Path | Specfem3DCase | SpecfemFieldSnapshot | SpecfemGLLDataset,
    *,
    ngll_shape: tuple[int, int, int] = EXPECTED_NGLL,
) -> dict[str, dict[str, Any]]:
    """List available SPECFEM3D fields and units for a snapshot, HDF5, VTK, or partition dir."""
    snap = _coerce_field_snapshot(
        source, ngll_shape=ngll_shape, fields=None, include_displacement=True
    )
    info: dict[str, dict[str, Any]] = {}
    for name, arr in snap.fields.items():
        info[name] = {
            "shape": arr.shape,
            "association": snap.association,
            "unit": str(snap.field_units.get(name, resolve_field_unit(name))),
            "times_s": tuple(float(t) for t in snap.times_s),
        }
    return info


def read_seismograms_specfem3d(
    source: str | Path | Sequence[str | Path],
    **kwargs: Any,
) -> TimeSeriesDict:
    """Read SPECFEM3D ASCII seismograms into a :class:`TimeSeriesDict`."""
    return read_specfem_seismograms_ascii(source, **kwargs)


def from_specfem3d(
    cls: type[ScalarField] | type[VectorField] | type[FieldDict] | type[TimeSeriesDict],
    source: Any,
    field: str | Sequence[str] | None = None,
    **kwargs: Any,
) -> ScalarField | VectorField | FieldDict | TimeSeriesDict:
    """Read SPECFEM3D data into the requested GWexpy Field or TimeSeriesDict container."""
    if issubclass(cls, TimeSeriesDict):
        return read_seismograms_specfem3d(source, **kwargs)
    if issubclass(cls, VectorField):
        fname = field if isinstance(field, str) else "displacement"
        return read_vectorfield_specfem3d(source, fname, **kwargs)
    if issubclass(cls, FieldDict):
        flist = [field] if isinstance(field, str) else field
        return read_fielddict_specfem3d(source, fields=flist, **kwargs)
    fname = field if isinstance(field, str) else "rho"
    return read_scalarfield_specfem3d(source, fname, **kwargs)


def register_specfem3d_formats(registry: Any = None) -> dict[str, dict[str, Any]]:
    """Return or register SPECFEM3D field format handlers for GWexpy Field I/O integration."""
    handlers = {
        "specfem3d": {
            "ScalarField": read_scalarfield_specfem3d,
            "VectorField": read_vectorfield_specfem3d,
            "FieldDict": read_fielddict_specfem3d,
            "list_fields": list_fields_specfem3d,
        },
        "h5.specfem3d": {
            "ScalarField": read_scalarfield_specfem3d,
            "VectorField": read_vectorfield_specfem3d,
            "FieldDict": read_fielddict_specfem3d,
            "list_fields": list_fields_specfem3d,
        },
        "vtk.specfem3d": {
            "ScalarField": read_scalarfield_specfem3d,
            "VectorField": read_vectorfield_specfem3d,
            "FieldDict": read_fielddict_specfem3d,
            "list_fields": list_fields_specfem3d,
        },
    }
    if registry is not None:
        for fmt, table in handlers.items():
            registry.register_reader(fmt, ScalarField, table["ScalarField"], force=True)
            registry.register_reader(fmt, VectorField, table["VectorField"], force=True)
            registry.register_reader(fmt, FieldDict, table["FieldDict"], force=True)
    return handlers


register_specfem3d_format = register_specfem3d_formats
