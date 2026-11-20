"""Conversion of OpenFOAM snapshots and cases into GWexpy Field objects."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy import units as u

from gwexpy.fields import FieldDict, ScalarField, VectorField

from .backend import OpenFOAMCase
from .dimensions import kinematic_to_physical_pressure
from .grid import (
    RegularGridSpec,
    check_moving_mesh,
    infer_exact_cartesian_grid,
    sample_to_regular_grid,
)
from .model import FoamFieldData, FoamSnapshot, OpenFOAMProvenance

__all__ = [
    "from_openfoam",
    "list_fields_openfoam",
    "read_fielddict_openfoam",
    "read_scalarfield_openfoam",
    "read_vectorfield_openfoam",
    "register_openfoam_format",
    "register_openfoam_formats",
]

_COMP_INDEX: dict[str, int] = {"x": 0, "y": 1, "z": 2}


def _resolve_grid_mode(
    mode: Literal["exact", "sample"] | None,
    grid_mode: Literal["exact", "sample"],
) -> Literal["exact", "sample"]:
    if mode is not None:
        if mode not in {"exact", "sample"}:
            raise ValueError(f"mode must be 'exact' or 'sample', got {mode!r}")
        if grid_mode != "exact" and grid_mode != mode:
            raise ValueError(f"Conflicting mode={mode!r} and grid_mode={grid_mode!r}")
        return mode
    if grid_mode not in {"exact", "sample"}:
        raise ValueError(f"grid_mode must be 'exact' or 'sample', got {grid_mode!r}")
    return grid_mode


def _coerce_snapshots(
    source: str | Path | OpenFOAMCase | FoamSnapshot | Sequence[FoamSnapshot],
    *,
    time: float | str | None = None,
    times: Iterable[float | str] | None = None,
    region: str = "region0",
    fields: Sequence[str] | None = None,
    skip_zero_time: bool = False,
    backend: Literal["auto", "pyvista", "vtk_legacy_ascii"] = "auto",
) -> tuple[list[FoamSnapshot], str, str | None]:
    if isinstance(source, FoamSnapshot):
        return (
            [source],
            str(
                source.provenance.get("input_path")
                or source.provenance.get("case_path")
                or ""
            ),
            source.provenance.get("solver"),
        )
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        snaps = list(source)
        if not snaps:
            raise ValueError("Empty FoamSnapshot sequence")
        first = snaps[0]
        return (
            snaps,
            str(
                first.provenance.get("input_path")
                or first.provenance.get("case_path")
                or ""
            ),
            first.provenance.get("solver"),
        )

    case = (
        source
        if isinstance(source, OpenFOAMCase)
        else OpenFOAMCase(source, skip_zero_time=skip_zero_time, backend=backend)
    )
    if time is not None and times is not None:
        raise ValueError("Provide at most one of 'time' or 'times'")
    if time is not None:
        snaps = [case.read_snapshot(time=time, region=region, fields=fields)]
    elif times is not None:
        snaps = case.read_snapshots(times=times, region=region, fields=fields)
    else:
        snaps = case.read_snapshots(region=region, fields=fields)
    return snaps, str(case.source_path), case.solver


def _resolve_association_coords(
    snap: FoamSnapshot,
    fdata: FoamFieldData,
    association: Literal["cell", "point"] | None,
) -> tuple[np.ndarray, Literal["cell", "point", "face"]]:
    if association is not None and association != fdata.association:
        raise ValueError(
            f"Field {fdata.name!r} is {fdata.association}-associated; "
            f"association={association!r} would reinterpret its values without "
            "an explicit cell/point conversion."
        )
    assoc = association if association is not None else fdata.association
    if assoc == "cell":
        if snap.cell_centers_m is None:
            raise ValueError(
                f"Field {fdata.name!r} has cell association but snapshot lacks cell_centers_m"
            )
        return snap.cell_centers_m, "cell"
    if assoc == "point":
        return snap.points_m, "point"
    raise ValueError(f"Unsupported association {assoc!r}")


def _extract_field_array(
    snap: FoamSnapshot,
    field: str,
    *,
    component: Literal["x", "y", "z"] | None = None,
    expected_vector: bool = False,
) -> tuple[FoamFieldData, np.ndarray]:
    if field not in snap.fields:
        raise KeyError(
            f"Field {field!r} not found in snapshot at t={snap.time_s} s. "
            f"Available: {sorted(snap.fields)}"
        )
    fdata = snap.fields[field]
    vals = np.asarray(fdata.values, dtype=np.float64)
    if expected_vector:
        if vals.ndim != 2 or vals.shape[1] != 3:
            raise ValueError(
                f"Expected 3-component vector field for {field!r}, got shape {vals.shape}"
            )
        return fdata, vals

    if vals.ndim == 2 and vals.shape[1] == 3:
        if component is None:
            raise ValueError(
                f"Field {field!r} is a 3-component vector field; pass "
                "component='x'|'y'|'z' or use read_vectorfield_openfoam()"
            )
        if component not in _COMP_INDEX:
            raise ValueError(
                f"Invalid component {component!r}; expected 'x', 'y', or 'z'"
            )
        return fdata, vals[:, _COMP_INDEX[component]]

    if component is not None:
        raise ValueError(
            f"Field {field!r} is scalar (shape {vals.shape}); component={component!r} is invalid"
        )
    return fdata, vals


def _build_scalarfield_from_snapshots(
    snaps: list[FoamSnapshot],
    *,
    field: str,
    source_path: str,
    solver: str | None,
    component: Literal["x", "y", "z"] | None,
    effective_mode: Literal["exact", "sample"],
    grid: RegularGridSpec | None,
    moving_mesh: Literal["error", "resample"],
    sample_method: Literal["auto", "linear", "nearest", "vtk_probe"],
    association: Literal["cell", "point"] | None,
    unit: Any | None,
    rho_ref: float | u.Quantity | None,
) -> ScalarField:
    motion_detected = check_moving_mesh(snaps, moving_mesh=moving_mesh)
    if motion_detected and effective_mode != "sample":
        raise ValueError(
            "Moving mesh detected with moving_mesh='resample'; mode='sample' and "
            "an explicit RegularGridSpec (grid=...) are required."
        )

    if effective_mode == "sample" and grid is None:
        raise ValueError(
            "mode='sample' requires an explicit RegularGridSpec via grid=..."
        )

    first_fdata, _ = _extract_field_array(snaps[0], field, component=component)
    first_coords, resolved_assoc = _resolve_association_coords(
        snaps[0], first_fdata, association
    )

    resolved_unit: u.UnitBase = first_fdata.unit
    pressure_kind = first_fdata.pressure_kind
    extra_meta: list[tuple[str, Any]] = []
    if component is not None:
        extra_meta.append(("component", component))

    if rho_ref is not None:
        if pressure_kind != "kinematic":
            raise ValueError(
                f"rho_ref is only applicable to kinematic pressure fields ([0 2 -2 0 0 0 0]), "
                f"got field {field!r} with pressure_kind={pressure_kind!r}"
            )
        resolved_unit = u.Pa
        pressure_kind = "physical_converted"
        rho_val = (
            float(rho_ref.to_value(u.kg / u.m**3))
            if isinstance(rho_ref, u.Quantity)
            else float(rho_ref)
        )
        extra_meta.append(("rho_ref_kg_m3", rho_val))

    if unit is not None:
        target_unit = u.Unit(unit)
        if not resolved_unit.is_equivalent(target_unit):
            # Allow explicit unit override if source was dimensionless
            if resolved_unit == u.dimensionless_unscaled:
                resolved_unit = target_unit
                scale_to_target = 1.0
            else:
                raise u.UnitConversionError(
                    f"Target unit {target_unit} is not equivalent to source unit {resolved_unit}"
                )
        else:
            scale_to_target = float(resolved_unit.to(target_unit))
            resolved_unit = target_unit
    else:
        scale_to_target = 1.0

    nt = len(snaps)
    times_arr = np.array([s.time_s for s in snaps], dtype=np.float64)
    time_names = tuple(s.time_name for s in snaps)

    if effective_mode == "exact":
        x_1d, y_1d, z_1d, idx_grid = infer_exact_cartesian_grid(first_coords)
        if grid is not None:
            if (
                len(x_1d) != len(grid.x)
                or len(y_1d) != len(grid.y)
                or len(z_1d) != len(grid.z)
                or not np.allclose(x_1d, grid.x, rtol=1e-6, atol=1e-8)
                or not np.allclose(y_1d, grid.y, rtol=1e-6, atol=1e-8)
                or not np.allclose(z_1d, grid.z, rtol=1e-6, atol=1e-8)
            ):
                raise ValueError(
                    "Native Cartesian coordinates do not match the supplied grid under mode='exact'; "
                    "use mode='sample' to resample onto the target grid."
                )
        nx, ny, nz = len(x_1d), len(y_1d), len(z_1d)
        data_4d = np.empty((nt, nx, ny, nz), dtype=np.float64)
        for ti, snap in enumerate(snaps):
            _, vals_1d = _extract_field_array(snap, field, component=component)
            if rho_ref is not None:
                vals_1d = np.asarray(
                    kinematic_to_physical_pressure(
                        vals_1d * (u.m**2 / u.s**2), rho_ref
                    ).to_value(u.Pa),
                    dtype=np.float64,
                )
            data_4d[ti] = vals_1d[idx_grid] * scale_to_target
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
        for ti, snap in enumerate(snaps):
            fdata_t, vals_1d = _extract_field_array(snap, field, component=component)
            coords_t, _ = _resolve_association_coords(snap, fdata_t, association)
            if rho_ref is not None:
                vals_1d = np.asarray(
                    kinematic_to_physical_pressure(
                        vals_1d * (u.m**2 / u.s**2), rho_ref
                    ).to_value(u.Pa),
                    dtype=np.float64,
                )
            pv_mesh = snap.pyvista_mesh
            pv_field = field
            if pv_mesh is not None and component is not None:
                # Extract scalar component onto transient mesh copy? Keep scipy or slice pv result.
                sampled_vec, mask_t, used_m = sample_to_regular_grid(
                    coords_t,
                    np.asarray(fdata_t.values, dtype=np.float64),
                    grid,
                    method=sample_method,
                    pyvista_mesh=pv_mesh,
                    field_name=pv_field,
                    association=resolved_assoc,
                    points_m=snap.points_m,
                    cell_blocks=snap.cell_blocks,
                )
                sampled_t = sampled_vec[..., _COMP_INDEX[component]]
                if rho_ref is not None:
                    sampled_t = np.asarray(
                        kinematic_to_physical_pressure(
                            sampled_t * (u.m**2 / u.s**2), rho_ref
                        ).to_value(u.Pa),
                        dtype=np.float64,
                    )
            else:
                sampled_t, mask_t, used_m = sample_to_regular_grid(
                    coords_t,
                    vals_1d,
                    grid,
                    method=sample_method,
                    pyvista_mesh=pv_mesh if component is None else None,
                    field_name=pv_field,
                    association=resolved_assoc,
                    points_m=snap.points_m,
                    cell_blocks=snap.cell_blocks,
                )
            data_4d[ti] = sampled_t * scale_to_target
            valid_mask_4d[ti] = mask_t
            interp_method = used_m
        valid_fraction = float(np.mean(valid_mask_4d))

    sf = ScalarField(
        data_4d,
        unit=resolved_unit,
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
    prov = OpenFOAMProvenance(
        source_path=source_path,
        region=snaps[0].region,
        field_name=field,
        foam_class=first_fdata.foam_class,
        dimensions=first_fdata.dimensions,
        unit_str=str(resolved_unit),
        pressure_kind=pressure_kind,
        association=resolved_assoc,
        grid_mode=effective_mode,
        moving_mesh=moving_mesh,
        mesh_motion_detected=motion_detected,
        time_names=time_names,
        solver=solver,
        backend=str(snaps[0].provenance.get("backend", "unknown")),
        input_sha256=snaps[0].provenance.get("input_sha256"),
        interpolation_method=interp_method,
        valid_fraction=valid_fraction,
        valid_mask=valid_mask_4d,
        extra=tuple(extra_meta),
    )
    sf._gwex_openfoam = prov
    return sf


def read_scalarfield_openfoam(
    source: str | Path | OpenFOAMCase | FoamSnapshot | Sequence[FoamSnapshot],
    field: str,
    *,
    time: float | str | None = None,
    times: Iterable[float | str] | None = None,
    region: str = "region0",
    component: Literal["x", "y", "z"] | None = None,
    mode: Literal["exact", "sample"] | None = None,
    grid_mode: Literal["exact", "sample"] = "exact",
    grid: RegularGridSpec | None = None,
    moving_mesh: Literal["error", "resample"] = "error",
    sample_method: Literal["auto", "linear", "nearest", "vtk_probe"] = "auto",
    association: Literal["cell", "point"] | None = None,
    unit: Any | None = None,
    rho_ref: float | u.Quantity | None = None,
    skip_zero_time: bool = False,
    backend: Literal["auto", "pyvista", "vtk_legacy_ascii"] = "auto",
) -> ScalarField:
    """Read an OpenFOAM scalar field (or vector component) into a :class:`ScalarField`."""
    effective_mode = _resolve_grid_mode(mode, grid_mode)
    snaps, src_path, solver = _coerce_snapshots(
        source,
        time=time,
        times=times,
        region=region,
        fields=[field],
        skip_zero_time=skip_zero_time,
        backend=backend,
    )
    return _build_scalarfield_from_snapshots(
        snaps,
        field=field,
        source_path=src_path,
        solver=solver,
        component=component,
        effective_mode=effective_mode,
        grid=grid,
        moving_mesh=moving_mesh,
        sample_method=sample_method,
        association=association,
        unit=unit,
        rho_ref=rho_ref,
    )


def read_vectorfield_openfoam(
    source: str | Path | OpenFOAMCase | FoamSnapshot | Sequence[FoamSnapshot],
    field: str = "U",
    *,
    time: float | str | None = None,
    times: Iterable[float | str] | None = None,
    region: str = "region0",
    mode: Literal["exact", "sample"] | None = None,
    grid_mode: Literal["exact", "sample"] = "exact",
    grid: RegularGridSpec | None = None,
    moving_mesh: Literal["error", "resample"] = "error",
    sample_method: Literal["auto", "linear", "nearest", "vtk_probe"] = "auto",
    association: Literal["cell", "point"] | None = None,
    unit: Any | None = None,
    skip_zero_time: bool = False,
    backend: Literal["auto", "pyvista", "vtk_legacy_ascii"] = "auto",
) -> VectorField:
    """Read an OpenFOAM ``volVectorField`` into a :class:`VectorField`."""
    effective_mode = _resolve_grid_mode(mode, grid_mode)
    snaps, src_path, solver = _coerce_snapshots(
        source,
        time=time,
        times=times,
        region=region,
        fields=[field],
        skip_zero_time=skip_zero_time,
        backend=backend,
    )
    _extract_field_array(snaps[0], field, expected_vector=True)

    components: dict[str, ScalarField] = {}
    for comp in ("x", "y", "z"):
        components[comp] = _build_scalarfield_from_snapshots(
            snaps,
            field=field,
            source_path=src_path,
            solver=solver,
            component=comp,  # type: ignore[arg-type]
            effective_mode=effective_mode,
            grid=grid,
            moving_mesh=moving_mesh,
            sample_method=sample_method,
            association=association,
            unit=unit,
            rho_ref=None,
        )
    return VectorField(components, basis="cartesian", validate=True)


def read_fielddict_openfoam(
    source: str | Path | OpenFOAMCase | FoamSnapshot | Sequence[FoamSnapshot],
    fields: Sequence[str] | None = None,
    *,
    time: float | str | None = None,
    times: Iterable[float | str] | None = None,
    region: str = "region0",
    mode: Literal["exact", "sample"] | None = None,
    grid_mode: Literal["exact", "sample"] = "exact",
    grid: RegularGridSpec | None = None,
    moving_mesh: Literal["error", "resample"] = "error",
    sample_method: Literal["auto", "linear", "nearest", "vtk_probe"] = "auto",
    association: Literal["cell", "point"] | None = None,
    skip_zero_time: bool = False,
    backend: Literal["auto", "pyvista", "vtk_legacy_ascii"] = "auto",
    _group_vectors: bool = False,
) -> FieldDict:
    """Read multiple OpenFOAM fields into a :class:`FieldDict`.

    Vector fields (3 components) are expanded into ``f"{name}_x"``,
    ``f"{name}_y"``, ``f"{name}_z"`` ``ScalarField`` entries so every value
    in the returned ``FieldDict`` conforms to GWexpy's ``ScalarField``
    collection contract while preserving heterogeneous physical units.
    """
    effective_mode = _resolve_grid_mode(mode, grid_mode)
    snaps, src_path, solver = _coerce_snapshots(
        source,
        time=time,
        times=times,
        region=region,
        fields=fields,
        skip_zero_time=skip_zero_time,
        backend=backend,
    )
    target_fields = (
        list(fields) if fields is not None else sorted(snaps[0].fields.keys())
    )
    out: dict[str, ScalarField | VectorField] = {}
    for fname in target_fields:
        fdata = snaps[0].fields.get(fname)
        if fdata is None:
            raise KeyError(f"Field {fname!r} not found in snapshot")
        if fdata.is_vector:
            components: dict[str, ScalarField] = {}
            for comp in ("x", "y", "z"):
                components[comp] = _build_scalarfield_from_snapshots(
                    snaps,
                    field=fname,
                    source_path=src_path,
                    solver=solver,
                    component=comp,  # type: ignore[arg-type]
                    effective_mode=effective_mode,
                    grid=grid,
                    moving_mesh=moving_mesh,
                    sample_method=sample_method,
                    association=association,
                    unit=None,
                    rho_ref=None,
                )
            if _group_vectors:
                out[fname] = VectorField(components, basis="cartesian", validate=True)
            else:
                out.update(
                    (f"{fname}_{key}", value) for key, value in components.items()
                )
        else:
            out[fname] = _build_scalarfield_from_snapshots(
                snaps,
                field=fname,
                source_path=src_path,
                solver=solver,
                component=None,
                effective_mode=effective_mode,
                grid=grid,
                moving_mesh=moving_mesh,
                sample_method=sample_method,
                association=association,
                unit=None,
                rho_ref=None,
            )
    return FieldDict(out, validate=False)


def list_fields_openfoam(
    source: str | Path | OpenFOAMCase,
    *,
    time: float | str | None = None,
    region: str = "region0",
    skip_zero_time: bool = False,
) -> dict[str, dict[str, Any]]:
    """List available fields and parsed metadata in an OpenFOAM case or VTK file."""
    case = (
        source
        if isinstance(source, OpenFOAMCase)
        else OpenFOAMCase(source, skip_zero_time=skip_zero_time)
    )
    headers = case.available_fields(time=time, region=region)
    info: dict[str, dict[str, Any]] = {}
    for name, hdr in headers.items():
        info[name] = {
            "class": hdr.foam_class,
            "dimensions": hdr.dimensions.exponents if hdr.dimensions else None,
            "unit": str(hdr.dimensions.unit) if hdr.dimensions else None,
            "pressure_kind": hdr.pressure_kind,
            "internal_field_kind": hdr.internal_field_kind,
        }
    return info


def from_openfoam(
    cls: type[ScalarField] | type[VectorField] | type[FieldDict],
    source: str | Path | OpenFOAMCase | FoamSnapshot,
    field: str | Sequence[str] | None = None,
    **kwargs: Any,
) -> ScalarField | VectorField | FieldDict:
    """Read OpenFOAM data into the requested GWexpy Field container."""
    if issubclass(cls, VectorField):
        fname = field if isinstance(field, str) else "U"
        return read_vectorfield_openfoam(source, fname, **kwargs)
    if issubclass(cls, FieldDict):
        flist = [field] if isinstance(field, str) else field
        return read_fielddict_openfoam(source, fields=flist, **kwargs)
    if not isinstance(field, str):
        raise ValueError(
            "ScalarField OpenFOAM read requires a single field name string"
        )
    return read_scalarfield_openfoam(source, field, **kwargs)


def register_openfoam_formats(registry: Any = None) -> dict[str, dict[str, Any]]:
    """Return or register OpenFOAM format handlers for GWexpy Field I/O integration.

    If ``registry`` is provided (e.g. ``gwpy.io.registry.default_registry`` or
    Area A1's Field registry), registers ``openfoam``, ``foam``, and
    ``vtk.openfoam`` readers for ``ScalarField``, ``VectorField``, and ``FieldDict``.
    """
    handlers = {
        "openfoam": {
            "ScalarField": read_scalarfield_openfoam,
            "VectorField": read_vectorfield_openfoam,
            "FieldDict": read_fielddict_openfoam,
            "list_fields": list_fields_openfoam,
        },
        "foam": {
            "ScalarField": read_scalarfield_openfoam,
            "VectorField": read_vectorfield_openfoam,
            "FieldDict": read_fielddict_openfoam,
            "list_fields": list_fields_openfoam,
        },
        "vtk.openfoam": {
            "ScalarField": read_scalarfield_openfoam,
            "VectorField": read_vectorfield_openfoam,
            "FieldDict": read_fielddict_openfoam,
            "list_fields": list_fields_openfoam,
        },
    }
    if registry is not None:
        for fmt, table in handlers.items():
            registry.register_reader(fmt, ScalarField, table["ScalarField"], force=True)
            registry.register_reader(fmt, VectorField, table["VectorField"], force=True)
            registry.register_reader(fmt, FieldDict, table["FieldDict"], force=True)
    return handlers


register_openfoam_format = register_openfoam_formats
