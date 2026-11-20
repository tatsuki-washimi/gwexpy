"""Conversion of EXODUS II / FLOW-3D snapshots into GWexpy Field objects."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy import units as u

from gwexpy.fields import FieldDict, ScalarField, VectorField

from .backend import ExodusCase
from .grid import (
    RegularGridSpec,
    infer_exact_cartesian_grid,
    sample_exodus_to_grid,
)
from .masks import Flow3DMaskSet, apply_flow3d_masks, compute_flow3d_masks
from .model import (
    ExodusBlock,
    ExodusProvenance,
    ExodusSnapshot,
    Flow3DProvenance,
)
from .profiles import ExodusProfile, find_vector_components, get_exodus_profile
from .units import resolve_variable_unit

__all__ = [
    "from_exodus",
    "list_fields_exodus",
    "read_fielddict_exodus",
    "read_scalarfield_exodus",
    "read_vectorfield_exodus",
    "register_exodus_format",
    "register_exodus_formats",
]


def _resolve_blocks(
    snap: ExodusSnapshot,
    block: str | int | Sequence[str | int] | None,
) -> list[ExodusBlock]:
    if not snap.blocks:
        raise ValueError("EXODUS II snapshot contains no element blocks")
    by_id = {blk.block_id: blk for blk in snap.blocks.values()}
    if block is None:
        active = [blk for blk in snap.blocks.values() if blk.status != 0]
        return active if active else list(snap.blocks.values())
    items = [block] if isinstance(block, (str, int, np.integer)) else list(block)
    resolved: list[ExodusBlock] = []
    for item in items:
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            bid = int(item)
            if bid not in by_id:
                raise KeyError(
                    f"Block ID {bid} not found. Available IDs: {sorted(by_id)}"
                )
            resolved.append(by_id[bid])
        elif isinstance(item, str):
            if item not in snap.blocks:
                raise KeyError(
                    f"Block {item!r} not found. Available: {list(snap.blocks)}"
                )
            resolved.append(snap.blocks[item])
        else:
            raise TypeError(f"Invalid block selector type: {type(item).__name__}")
    if not resolved:
        raise ValueError("No EXODUS blocks selected")
    return resolved


def _resolve_variable_name(
    available: Sequence[str],
    requested: str,
    profile: ExodusProfile,
) -> str:
    if requested in available:
        return requested
    low_map = {name.lower(): name for name in available}
    if requested.lower() in low_map:
        return low_map[requested.lower()]
    # Match canonical alias via profile
    req_canon = profile.canonical_name(requested)
    for cand in available:
        if profile.canonical_name(cand) == req_canon:
            return cand
    raise KeyError(
        f"Variable {requested!r} not found. Available variables: {list(available)}"
    )


def _check_disjoint_centers(blocks: Sequence[ExodusBlock], atol: float = 1e-9) -> None:
    if len(blocks) <= 1:
        return
    all_centers = np.vstack([b.cell_centers_m for b in blocks])
    # Check for duplicate coordinates across merged blocks
    rounded = np.round(all_centers / max(atol, 1e-12)).astype(np.int64)
    _, counts = np.unique(rounded, axis=0, return_counts=True)
    if np.any(counts > 1):
        raise ValueError(
            "Selected EXODUS blocks contain overlapping/duplicate cell centers and "
            "cannot be merged cleanly; select a single block or use block_mode='separate'."
        )


def _extract_entity_payload(
    snap: ExodusSnapshot,
    blocks: Sequence[ExodusBlock],
    variable: str,
    *,
    profile: ExodusProfile,
    association: Literal["cell", "point"] | None,
    block_mode: str,
    favor_threshold: float,
    vof_threshold: float,
    mask_blocked: bool,
    empty_fluid_mode: Literal["keep", "nan", "zero"],
) -> tuple[
    str,
    Literal["cell", "point"],
    np.ndarray,
    np.ndarray,
    Flow3DMaskSet,
    list[np.ndarray],
]:
    """Extract entity coords, masked values, masks, and connectivities for selected blocks."""
    cell_avail_per_block = [set(snap.cell_data.get(b.name, {}).keys()) for b in blocks]
    all_cell_names = sorted(set().union(*cell_avail_per_block))
    point_names = list(snap.point_data.keys())

    resolved_assoc: Literal["cell", "point"]
    if association is not None:
        resolved_assoc = association
    else:
        # Check cell variables first (profile default for FLOW-3D), then point
        try:
            _resolve_variable_name(all_cell_names, variable, profile)
            resolved_assoc = "cell"
        except KeyError:
            _resolve_variable_name(point_names, variable, profile)
            resolved_assoc = "point"

    if resolved_assoc == "cell":
        if len(blocks) > 1 and block_mode == "single":
            raise ValueError(
                f"Multiple EXODUS blocks {[b.name for b in blocks]} selected with block_mode='single'"
            )
        if len(blocks) > 1:
            _check_disjoint_centers(blocks)

        actual_names: list[str] = []
        coords_list: list[np.ndarray] = []
        vals_list: list[np.ndarray] = []
        open_list: list[np.ndarray] = []
        fluid_list: list[np.ndarray] = []
        valid_list: list[np.ndarray] = []
        conns: list[np.ndarray] = []
        favor_var_used: str | None = None
        vof_var_used: str | None = None

        for blk in blocks:
            b_vars = snap.cell_data.get(blk.name, {})
            try:
                actual_v = _resolve_variable_name(
                    list(b_vars.keys()), variable, profile
                )
            except KeyError as exc:
                raise ValueError(
                    f"Variable {variable!r} is missing on EXODUS block {blk.name!r}; "
                    f"blocks are incompatible for merge."
                ) from exc
            actual_names.append(actual_v)
            raw_vals = np.asarray(b_vars[actual_v], dtype=np.float64)
            b_masks = snap.masks.get(blk.name, {})
            open_m = np.asarray(
                b_masks.get("open_mask", np.ones(blk.n_elements, dtype=bool)),
                dtype=bool,
            )
            fluid_m = np.asarray(
                b_masks.get("fluid_mask", np.ones(blk.n_elements, dtype=bool)),
                dtype=bool,
            )
            valid_m = np.asarray(
                b_masks.get("valid_mask", np.ones(blk.n_elements, dtype=bool)),
                dtype=bool,
            )
            if favor_var_used is None:
                favor_var_used = profile.find_favor_variable(list(b_vars.keys()))
            if vof_var_used is None:
                vof_var_used = profile.find_vof_variable(list(b_vars.keys()))

            coords_list.append(blk.cell_centers_m)
            vals_list.append(raw_vals)
            open_list.append(open_m)
            fluid_list.append(fluid_m)
            valid_list.append(valid_m)
            if blk.connectivity.size > 0:
                conns.append(blk.connectivity)

        coords_m = np.vstack(coords_list)
        raw_merged = np.concatenate(vals_list, axis=0)
        mask_set = Flow3DMaskSet(
            open_mask=np.concatenate(open_list),
            fluid_mask=np.concatenate(fluid_list),
            valid_mask=np.concatenate(valid_list),
            favor_variable=favor_var_used,
            vof_variable=vof_var_used,
        )
        masked_vals = apply_flow3d_masks(
            raw_merged,
            mask_set,
            empty_fluid_mode=empty_fluid_mode,
            is_mask_variable=profile.is_mask_variable(actual_names[0]),
        )
        return (
            actual_names[0],
            "cell",
            coords_m,
            masked_vals,
            mask_set,
            conns,
        )

    # Point association
    actual_v = _resolve_variable_name(point_names, variable, profile)
    if len(blocks) == len(snap.blocks):
        node_idx = np.arange(len(snap.points_m), dtype=np.int64)
    else:
        node_idx = np.unique(np.concatenate([b.node_indices for b in blocks]))
    coords_m = snap.points_m[node_idx]
    raw_vals = np.asarray(snap.point_data[actual_v][node_idx], dtype=np.float64)

    favor_var = profile.find_favor_variable(point_names)
    vof_var = profile.find_vof_variable(point_names)
    mask_set = compute_flow3d_masks(
        len(node_idx),
        favor_values=snap.point_data[favor_var][node_idx] if favor_var else None,
        favor_variable=favor_var,
        favor_threshold=favor_threshold,
        vof_values=snap.point_data[vof_var][node_idx] if vof_var else None,
        vof_variable=vof_var,
        vof_threshold=vof_threshold,
        mask_blocked=mask_blocked,
        empty_fluid_mode=empty_fluid_mode,
    )
    masked_vals = apply_flow3d_masks(
        raw_vals,
        mask_set,
        empty_fluid_mode=empty_fluid_mode,
        is_mask_variable=profile.is_mask_variable(actual_v),
    )
    conns = [b.connectivity for b in blocks if b.connectivity.size > 0]
    return actual_v, "point", coords_m, masked_vals, mask_set, conns


def _build_scalarfield_from_exodus(
    snaps: list[ExodusSnapshot],
    *,
    source_path: str,
    format_name: str,
    profile: ExodusProfile,
    variable: str,
    block: str | int | Sequence[str | int] | None,
    block_mode: Literal["auto", "single", "separate", "merge"],
    grid_mode: Literal["exact", "sample"],
    grid: RegularGridSpec | None,
    sample_method: Literal["linear", "nearest"],
    association: Literal["cell", "point"] | None,
    unit_system: str,
    unit: Any | None,
    strict_units: bool,
    favor_threshold: float,
    vof_threshold: float,
    mask_blocked: bool,
    empty_fluid_mode: Literal["keep", "nan", "zero"],
) -> ScalarField:
    if grid_mode not in {"exact", "sample"}:
        raise ValueError(f"grid_mode must be 'exact' or 'sample', got {grid_mode!r}")
    if grid_mode == "sample" and grid is None:
        raise ValueError(
            "grid_mode='sample' requires an explicit RegularGridSpec via grid="
        )

    blocks = _resolve_blocks(snaps[0], block)
    if len(blocks) > 1 and block_mode == "auto":
        raise ValueError(
            "Multiple EXODUS blocks selected; choose one with block=, "
            "pass block_mode='merge' explicitly, or use "
            "read_fielddict_exodus() for separate fields."
        )
    if len(blocks) > 1 and block_mode == "separate":
        raise ValueError(
            "block_mode='separate' produces multiple fields; use read_fielddict_exodus() "
            "or pass a single block / block_mode='merge'."
        )
    effective_block_mode = "single" if len(blocks) == 1 else block_mode

    is_static = profile.is_static_variable(variable)
    is_surface = profile.is_surface_variable(variable)
    active_snaps = [snaps[0]] if (is_static and len(snaps) > 1) else snaps

    actual_var, resolved_assoc, first_coords, _, _, first_conns = (
        _extract_entity_payload(
            active_snaps[0],
            blocks,
            variable,
            profile=profile,
            association=association,
            block_mode=effective_block_mode,
            favor_threshold=favor_threshold,
            vof_threshold=vof_threshold,
            mask_blocked=mask_blocked,
            empty_fluid_mode=empty_fluid_mode,
        )
    )
    canonical = profile.canonical_name(actual_var)

    # Check static consistency for terrain_elevation if multiple steps were present
    if is_static and len(snaps) > 1:
        _, _, _, v0, _, _ = _extract_entity_payload(
            snaps[0],
            blocks,
            actual_var,
            profile=profile,
            association=resolved_assoc,
            block_mode=effective_block_mode,
            favor_threshold=favor_threshold,
            vof_threshold=vof_threshold,
            mask_blocked=mask_blocked,
            empty_fluid_mode=empty_fluid_mode,
        )
        for later in snaps[1:]:
            _, _, _, vk, _, _ = _extract_entity_payload(
                later,
                blocks,
                actual_var,
                profile=profile,
                association=resolved_assoc,
                block_mode=effective_block_mode,
                favor_threshold=favor_threshold,
                vof_threshold=vof_threshold,
                mask_blocked=mask_blocked,
                empty_fluid_mode=empty_fluid_mode,
            )
            if not np.allclose(v0, vk, rtol=1e-12, atol=1e-12, equal_nan=True):
                raise ValueError(
                    f"Static surface variable {actual_var!r} varies across EXODUS time steps"
                )

    unit_key = (
        "elem_var_file_units" if resolved_assoc == "cell" else "nod_var_file_units"
    )
    var_file_units = active_snaps[0].provenance.get(unit_key, {})
    if not var_file_units:
        # Compatibility with externally constructed snapshots made before
        # association-specific unit maps existed.
        var_file_units = active_snaps[0].provenance.get("var_file_units", {})
    file_unit_str = var_file_units.get(actual_var)
    target_unit, scale_factor, unit_source = resolve_variable_unit(
        canonical_name=canonical,
        file_unit_str=file_unit_str,
        unit_system=unit_system,
        unit_override=unit,
        convert_to_si=True,
        profile_units=profile.profile_units_map(),
    )
    if strict_units and unit_source == "dimensionless":
        raise ValueError(
            f"Unit for EXODUS variable {actual_var!r} is unresolved; "
            "provide unit= or a file units attribute."
        )

    nt = len(active_snaps)
    times_arr = np.array([s.time_s for s in active_snaps], dtype=np.float64)

    if grid_mode == "exact":
        x_1d, y_1d, z_1d, idx_grid = infer_exact_cartesian_grid(
            first_coords, force_surface_2d=is_surface
        )
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
                    "Native Cartesian EXODUS coordinates do not match grid under grid_mode='exact'"
                )
        nx, ny, nz = len(x_1d), len(y_1d), len(z_1d)
        data_4d = np.empty((nt, nx, ny, nz), dtype=np.float64)
        open_mask_4d = np.empty((nt, nx, ny, nz), dtype=bool)
        fluid_mask_4d = np.empty((nt, nx, ny, nz), dtype=bool)
        valid_mask_4d = np.empty((nt, nx, ny, nz), dtype=bool)
        favor_used: str | None = None
        vof_used: str | None = None
        for ti, snap in enumerate(active_snaps):
            cur_blocks = _resolve_blocks(snap, block)
            _, _, _, vals_1d, mset, _ = _extract_entity_payload(
                snap,
                cur_blocks,
                actual_var,
                profile=profile,
                association=resolved_assoc,
                block_mode=effective_block_mode,
                favor_threshold=favor_threshold,
                vof_threshold=vof_threshold,
                mask_blocked=mask_blocked,
                empty_fluid_mode=empty_fluid_mode,
            )
            if vals_1d.ndim != 1:
                raise ValueError(
                    f"Variable {actual_var!r} is multi-component ({vals_1d.shape}); "
                    "use read_vectorfield_exodus()."
                )
            data_4d[ti] = vals_1d[idx_grid] * scale_factor
            open_mask_4d[ti] = mset.open_mask[idx_grid]
            fluid_mask_4d[ti] = mset.fluid_mask[idx_grid]
            valid_mask_4d[ti] = mset.valid_mask[idx_grid]
            favor_used = mset.favor_variable
            vof_used = mset.vof_variable
        interp_method: str | None = None
        valid_fraction = float(np.mean(valid_mask_4d))
    else:
        assert grid is not None
        x_1d, y_1d, z_1d = grid.x, grid.y, grid.z
        nx, ny, nz = grid.shape_xyz
        if is_surface and nz != 1:
            raise ValueError(
                f"Surface variable {actual_var!r} requires grid with nz=1, got nz={nz}"
            )
        data_4d = np.empty((nt, nx, ny, nz), dtype=np.float64)
        valid_mask_4d = np.empty((nt, nx, ny, nz), dtype=bool)
        open_mask_4d = np.empty((nt, nx, ny, nz), dtype=bool)
        fluid_mask_4d = np.empty((nt, nx, ny, nz), dtype=bool)
        interp_method = None
        favor_used = None
        vof_used = None
        for ti, snap in enumerate(active_snaps):
            cur_blocks = _resolve_blocks(snap, block)
            _, _, coords_t, vals_1d, mset, conns_t = _extract_entity_payload(
                snap,
                cur_blocks,
                actual_var,
                profile=profile,
                association=resolved_assoc,
                block_mode=effective_block_mode,
                favor_threshold=favor_threshold,
                vof_threshold=vof_threshold,
                mask_blocked=mask_blocked,
                empty_fluid_mode=empty_fluid_mode,
            )
            if vals_1d.ndim != 1:
                raise ValueError(
                    f"Variable {actual_var!r} is multi-component ({vals_1d.shape}); "
                    "use read_vectorfield_exodus()."
                )
            sampled, dom_mask, used_m = sample_exodus_to_grid(
                coords_t,
                vals_1d,
                grid,
                method=sample_method,
                points_m=snap.points_m,
                connectivities=conns_t,
            )
            data_4d[ti] = sampled * scale_factor
            valid_mask_4d[ti] = dom_mask & np.isfinite(sampled)
            # FAVOR/VOF are categorical source attributes. Resample them with
            # nearest-neighbor selection, independently of value interpolation.
            sampled_masks, mask_domain, _ = sample_exodus_to_grid(
                coords_t,
                np.column_stack((mset.open_mask, mset.fluid_mask)).astype(float),
                grid,
                method="nearest",
                points_m=snap.points_m,
                connectivities=conns_t,
            )
            open_mask_4d[ti] = mask_domain & (sampled_masks[..., 0] >= 0.5)
            fluid_mask_4d[ti] = mask_domain & (sampled_masks[..., 1] >= 0.5)
            interp_method = used_m
            favor_used = mset.favor_variable
            vof_used = mset.vof_variable
        valid_fraction = float(np.mean(valid_mask_4d))

    sf = ScalarField(
        data_4d,
        unit=target_unit,
        name=actual_var,
        axis0=times_arr * u.s,
        axis1=x_1d * u.m,
        axis2=y_1d * u.m,
        axis3=z_1d * u.m,
        axis_names=["t", "x", "y", "z"],
        axis0_domain="time",
        space_domain="real",
        copy=False,
    )
    sf._gwex_exodus = ExodusProvenance(
        source_path=source_path,
        format_name=format_name,
        profile_name=profile.name,
        backend=str(active_snaps[0].provenance.get("backend", "unknown")),
        title=active_snaps[0].provenance.get("title"),
        exodus_version=active_snaps[0].provenance.get("exodus_version"),
        variable_name=actual_var,
        canonical_name=canonical,
        association=resolved_assoc,
        blocks=tuple(b.name for b in blocks),
        block_mode=effective_block_mode,
        grid_mode=grid_mode,
        unit_str=str(target_unit),
        unit_source=unit_source,
        coordinate_unit=str(active_snaps[0].provenance.get("coordinate_unit", "m")),
        time_steps=tuple(float(t) for t in times_arr),
        interpolation_method=interp_method,
        valid_fraction=valid_fraction,
        valid_mask=valid_mask_4d,
    )
    if profile.name == "flow3d":
        sf._gwex_flow3d = Flow3DProvenance(
            profile_version=getattr(profile, "profile_version", "flow3d-exodus-v1"),
            quantity_kind=profile.quantity_kind(actual_var),
            unit_system=unit_system,
            favor_variable=favor_used,
            vof_variable=vof_used,
            favor_threshold=favor_threshold,
            vof_threshold=vof_threshold,
            mask_blocked=mask_blocked,
            empty_fluid_mode=empty_fluid_mode,
            open_cell_fraction=float(np.mean(open_mask_4d)),
            fluid_cell_fraction=float(np.mean(fluid_mask_4d)),
            open_mask=open_mask_4d,
            fluid_mask=fluid_mask_4d,
            notes=(
                "Synthetic-verified FLOW-3D EXODUS II profile; physical 0 in open "
                "cells preserved distinctly from FAVOR-blocked invalid cells (NaN)."
            ),
        )
    return sf


def _open_exodus_snaps(
    source: str | Path | ExodusCase | ExodusSnapshot | Sequence[ExodusSnapshot],
    *,
    format_name: str,
    profile: ExodusProfile | None,
    unit_system: str,
    coordinate_unit: str | None,
    backend: Literal["auto", "netcdf4", "pyvista"],
    time: float | None,
    time_index: int | None,
    times: Iterable[float] | None,
    favor_threshold: float,
    vof_threshold: float,
    mask_blocked: bool,
    empty_fluid_mode: Literal["keep", "nan", "zero"],
    requested: tuple[str, ...] | None = None,
) -> tuple[list[ExodusSnapshot], str, ExodusProfile]:
    prof = profile if profile is not None else get_exodus_profile(format_name)
    if isinstance(source, ExodusSnapshot):
        return [source], str(source.provenance.get("source_path", "<snapshot>")), prof
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        snaps = list(source)
        if not snaps:
            raise ValueError("Empty ExodusSnapshot sequence")
        return snaps, str(snaps[0].provenance.get("source_path", "<snapshots>")), prof

    case = (
        source
        if isinstance(source, ExodusCase)
        else ExodusCase(
            source,
            format_name=format_name,
            profile=prof,
            unit_system=unit_system,
            coordinate_unit=coordinate_unit,
            backend=backend,
        )
    )
    snaps = case.read_snapshots(
        time=time,
        time_index=time_index,
        times=times,
        favor_threshold=favor_threshold,
        vof_threshold=vof_threshold,
        mask_blocked=mask_blocked,
        empty_fluid_mode=empty_fluid_mode,
        _requested=requested,
    )
    return snaps, str(case.path), case.profile


def read_scalarfield_exodus(
    source: str | Path | ExodusCase | ExodusSnapshot | Sequence[ExodusSnapshot],
    variable: str,
    *,
    format: str = "exo.flow3d",
    profile: ExodusProfile | None = None,
    block: str | int | Sequence[str | int] | None = None,
    block_mode: Literal["auto", "single", "separate", "merge"] = "auto",
    time: float | None = None,
    time_index: int | None = None,
    times: Iterable[float] | None = None,
    grid_mode: Literal["exact", "sample"] = "exact",
    grid: RegularGridSpec | None = None,
    sample_method: Literal["linear", "nearest"] = "linear",
    association: Literal["cell", "point"] | None = None,
    unit_system: str = "si",
    coordinate_unit: str | None = None,
    unit: Any | None = None,
    strict_units: bool = False,
    favor_threshold: float = 0.0,
    vof_threshold: float = 0.0,
    mask_blocked: bool = True,
    empty_fluid_mode: Literal["keep", "nan", "zero"] = "keep",
    backend: Literal["auto", "netcdf4", "pyvista"] = "auto",
) -> ScalarField:
    """Read an EXODUS II variable into a :class:`ScalarField`."""
    snaps, src_path, prof = _open_exodus_snaps(
        source,
        format_name=format,
        profile=profile,
        unit_system=unit_system,
        coordinate_unit=coordinate_unit,
        backend=backend,
        time=time,
        time_index=time_index,
        times=times,
        favor_threshold=favor_threshold,
        vof_threshold=vof_threshold,
        mask_blocked=mask_blocked,
        empty_fluid_mode=empty_fluid_mode,
        requested=(variable,),
    )
    return _build_scalarfield_from_exodus(
        snaps,
        source_path=src_path,
        format_name=format,
        profile=prof,
        variable=variable,
        block=block,
        block_mode=block_mode,
        grid_mode=grid_mode,
        grid=grid,
        sample_method=sample_method,
        association=association,
        unit_system=unit_system,
        unit=unit,
        strict_units=strict_units,
        favor_threshold=favor_threshold,
        vof_threshold=vof_threshold,
        mask_blocked=mask_blocked,
        empty_fluid_mode=empty_fluid_mode,
    )


def read_vectorfield_exodus(
    source: str | Path | ExodusCase | ExodusSnapshot | Sequence[ExodusSnapshot],
    variable: str | Sequence[str] = "velocity",
    *,
    format: str = "exo.flow3d",
    profile: ExodusProfile | None = None,
    block: str | int | Sequence[str | int] | None = None,
    block_mode: Literal["auto", "single", "separate", "merge"] = "auto",
    time: float | None = None,
    time_index: int | None = None,
    times: Iterable[float] | None = None,
    grid_mode: Literal["exact", "sample"] = "exact",
    grid: RegularGridSpec | None = None,
    sample_method: Literal["linear", "nearest"] = "linear",
    association: Literal["cell", "point"] | None = None,
    unit_system: str = "si",
    coordinate_unit: str | None = None,
    unit: Any | None = None,
    strict_units: bool = False,
    favor_threshold: float = 0.0,
    vof_threshold: float = 0.0,
    mask_blocked: bool = True,
    empty_fluid_mode: Literal["keep", "nan", "zero"] = "keep",
    backend: Literal["auto", "netcdf4", "pyvista"] = "auto",
) -> VectorField:
    """Read an EXODUS II vector field (e.g. ``(u, v, w)`` or ``(vx, vy, vz)``) as :class:`VectorField`."""
    snaps, src_path, prof = _open_exodus_snaps(
        source,
        format_name=format,
        profile=profile,
        unit_system=unit_system,
        coordinate_unit=coordinate_unit,
        backend=backend,
        time=time,
        time_index=time_index,
        times=times,
        favor_threshold=favor_threshold,
        vof_threshold=vof_threshold,
        mask_blocked=mask_blocked,
        empty_fluid_mode=empty_fluid_mode,
        requested=(variable,) if isinstance(variable, str) else tuple(variable),
    )
    snap0 = snaps[0]
    blocks = _resolve_blocks(snap0, block)

    if isinstance(variable, Sequence) and not isinstance(variable, str):
        if len(variable) != 3:
            raise ValueError(
                f"Expected 3 vector component variable names, got {variable!r}"
            )
        comp_vars = (str(variable[0]), str(variable[1]), str(variable[2]))
    else:
        # Discover available names from block cell_data or point_data
        if association == "point":
            avail = list(snap0.point_data.keys())
        elif association == "cell":
            avail = sorted(
                set().union(*(snap0.cell_data.get(b.name, {}).keys() for b in blocks))
            )
        else:
            avail = sorted(
                set(snap0.point_data.keys()).union(
                    *(snap0.cell_data.get(b.name, {}).keys() for b in blocks)
                )
            )
        found = find_vector_components(prof, avail, str(variable))
        if found is None:
            raise KeyError(
                f"Could not resolve 3 vector components for {variable!r} in {avail}"
            )
        comp_vars = found

    comps: dict[str, ScalarField] = {}
    for axis_label, comp_var in zip(("x", "y", "z"), comp_vars, strict=True):
        comps[axis_label] = _build_scalarfield_from_exodus(
            snaps,
            source_path=src_path,
            format_name=format,
            profile=prof,
            variable=comp_var,
            block=block,
            block_mode=block_mode,
            grid_mode=grid_mode,
            grid=grid,
            sample_method=sample_method,
            association=association,
            unit_system=unit_system,
            unit=unit,
            strict_units=strict_units,
            favor_threshold=favor_threshold,
            vof_threshold=vof_threshold,
            mask_blocked=mask_blocked,
            empty_fluid_mode=empty_fluid_mode,
        )
    return VectorField(comps, basis="cartesian", validate=True)


def read_fielddict_exodus(
    source: str | Path | ExodusCase | ExodusSnapshot | Sequence[ExodusSnapshot],
    variables: Sequence[str] | None = None,
    *,
    format: str = "exo.flow3d",
    profile: ExodusProfile | None = None,
    block: str | int | Sequence[str | int] | None = None,
    block_mode: Literal["auto", "single", "separate", "merge"] = "auto",
    time: float | None = None,
    time_index: int | None = None,
    times: Iterable[float] | None = None,
    grid_mode: Literal["exact", "sample"] = "exact",
    grid: RegularGridSpec | None = None,
    sample_method: Literal["linear", "nearest"] = "linear",
    association: Literal["cell", "point"] | None = None,
    unit_system: str = "si",
    coordinate_unit: str | None = None,
    strict_units: bool = False,
    favor_threshold: float = 0.0,
    vof_threshold: float = 0.0,
    mask_blocked: bool = True,
    empty_fluid_mode: Literal["keep", "nan", "zero"] = "keep",
    backend: Literal["auto", "netcdf4", "pyvista"] = "auto",
    _public_selection: bool = False,
) -> FieldDict:
    """Read multiple EXODUS II variables (and/or separate blocks) into a :class:`FieldDict`."""
    snaps, src_path, prof = _open_exodus_snaps(
        source,
        format_name=format,
        profile=profile,
        unit_system=unit_system,
        coordinate_unit=coordinate_unit,
        backend=backend,
        time=time,
        time_index=time_index,
        times=times,
        favor_threshold=favor_threshold,
        vof_threshold=vof_threshold,
        mask_blocked=mask_blocked,
        empty_fluid_mode=empty_fluid_mode,
        requested=tuple(variables)
        if _public_selection and variables is not None
        else None,
    )
    snap0 = snaps[0]
    blocks = _resolve_blocks(snap0, block)

    use_separate = block_mode == "separate" or (
        block_mode == "auto" and len(blocks) > 1
    )

    def build(
        variable: str,
        selected_block: str | int | Sequence[str | int] | None,
        selected_mode: Literal["auto", "single", "separate", "merge"],
    ) -> ScalarField:
        return _build_scalarfield_from_exodus(
            snaps,
            source_path=src_path,
            format_name=format,
            profile=prof,
            variable=variable,
            block=selected_block,
            block_mode=selected_mode,
            grid_mode=grid_mode,
            grid=grid,
            sample_method=sample_method,
            association=association,
            unit_system=unit_system,
            unit=None,
            strict_units=strict_units,
            favor_threshold=favor_threshold,
            vof_threshold=vof_threshold,
            mask_blocked=mask_blocked,
            empty_fluid_mode=empty_fluid_mode,
        )

    if _public_selection and variables is not None:
        return _select_exodus_fields(
            snap0,
            blocks,
            variables,
            separate=use_separate,
            association=association,
            profile=prof,
            build=build,
            block=block,
            block_mode=block_mode,
        )

    out: dict[str, ScalarField] = {}
    if use_separate:
        for blk in blocks:
            b_cell_vars = sorted(snap0.cell_data.get(blk.name, {}).keys())
            b_point_vars = sorted(snap0.point_data.keys())
            if variables is not None:
                vlist = list(variables)
            else:
                vlist = b_cell_vars + [v for v in b_point_vars if v not in b_cell_vars]
            for var in vlist:
                # Skip if variable not present on this block (when variables is None)
                if (
                    variables is None
                    and var not in b_cell_vars
                    and var not in b_point_vars
                ):
                    continue
                out[f"{blk.name}/{var}"] = build(var, blk.name, "single")
        return FieldDict(out, validate=False)

    # Single or merged block path
    if variables is None:
        all_cell = sorted(
            set().union(*(snap0.cell_data.get(b.name, {}).keys() for b in blocks))
        )
        all_point = sorted(snap0.point_data.keys())
        vlist = all_cell + [v for v in all_point if v not in all_cell]
    else:
        vlist = list(variables)

    for var in vlist:
        out[var] = build(var, block, block_mode)
    return FieldDict(out, validate=False)


def _select_exodus_fields(
    snapshot: ExodusSnapshot,
    blocks: list[ExodusBlock],
    names: Sequence[str],
    *,
    separate: bool,
    association: Literal["cell", "point"] | None,
    profile: ExodusProfile,
    build: Callable[..., ScalarField],
    block: str | int | Sequence[str | int] | None,
    block_mode: Literal["auto", "single", "separate", "merge"],
) -> FieldDict:
    out = FieldDict()
    for name in names:
        block_name, separator, variable = name.rpartition("/")
        if separator:
            selected = [item for item in blocks if item.name == block_name]
            if not selected:
                raise KeyError(f"Requested EXODUS block {block_name!r} is unavailable.")
        else:
            selected = blocks

        groups: list[
            tuple[
                str | int | Sequence[str | int] | None,
                Literal["auto", "single", "separate", "merge"],
                list[ExodusBlock],
            ]
        ] = (
            [(item.name, "single", [item]) for item in selected]
            if separator or separate
            else [(block, block_mode, selected)]
        )
        for target, mode, members in groups:
            available = set(snapshot.point_data) if association != "cell" else set()
            if association != "point":
                for member in members:
                    available.update(snapshot.cell_data.get(member.name, {}))
            components = (
                None
                if variable in available
                else find_vector_components(profile, sorted(available), variable)
            )
            key = f"{members[0].name}/{variable}" if separate or separator else name
            if components is None:
                out[key] = build(variable, target, mode)
            else:
                out[key] = VectorField(
                    {
                        axis: build(component, target, mode)
                        for axis, component in zip("xyz", components, strict=True)
                    },
                    basis="cartesian",
                )
    return out


def list_fields_exodus(
    source: str | Path | ExodusCase,
    *,
    format: str = "exo.flow3d",
    unit_system: str = "si",
    backend: Literal["auto", "netcdf4", "pyvista"] = "auto",
) -> dict[str, dict[str, Any]]:
    """List variables, blocks, associations, and resolved units in an EXODUS II file."""
    case = (
        source
        if isinstance(source, ExodusCase)
        else ExodusCase(
            source, format_name=format, unit_system=unit_system, backend=backend
        )
    )
    info: dict[str, dict[str, Any]] = {}
    for name, var in case.available_variables.items():
        info[name] = {
            "canonical_name": var.canonical_name,
            "association": var.association,
            "unit": str(var.unit) if var.unit is not None else None,
            "unit_source": var.unit_source,
            "blocks": var.block_names,
            "quantity_kind": var.quantity_kind,
        }
    return info


def from_exodus(
    cls: type[ScalarField] | type[VectorField] | type[FieldDict],
    source: str | Path | ExodusCase | ExodusSnapshot,
    variable: str | Sequence[str] | None = None,
    **kwargs: Any,
) -> ScalarField | VectorField | FieldDict:
    """Read EXODUS II data into the requested GWexpy Field container."""
    if issubclass(cls, VectorField):
        vname = variable if variable is not None else "velocity"
        return read_vectorfield_exodus(source, vname, **kwargs)
    if issubclass(cls, FieldDict):
        vlist = [variable] if isinstance(variable, str) else variable
        return read_fielddict_exodus(source, variables=vlist, **kwargs)
    if not isinstance(variable, str):
        raise ValueError(
            "ScalarField EXODUS read requires a single variable name string"
        )
    return read_scalarfield_exodus(source, variable, **kwargs)


def register_exodus_formats(registry: Any = None) -> dict[str, dict[str, Any]]:
    """Return or register EXODUS II format handlers for GWexpy Field I/O integration."""
    handlers = {
        "exo.flow3d": {
            "ScalarField": lambda src, var, **kw: read_scalarfield_exodus(
                src, var, format="exo.flow3d", **kw
            ),
            "VectorField": lambda src, var="velocity", **kw: read_vectorfield_exodus(
                src, var, format="exo.flow3d", **kw
            ),
            "FieldDict": lambda src, variables=None, **kw: read_fielddict_exodus(
                src, variables=variables, format="exo.flow3d", **kw
            ),
            "list_fields": lambda src, **kw: list_fields_exodus(
                src, format="exo.flow3d", **kw
            ),
        },
        "exo": {
            "ScalarField": lambda src, var, **kw: read_scalarfield_exodus(
                src, var, format="exo", **kw
            ),
            "VectorField": lambda src, var="velocity", **kw: read_vectorfield_exodus(
                src, var, format="exo", **kw
            ),
            "FieldDict": lambda src, variables=None, **kw: read_fielddict_exodus(
                src, variables=variables, format="exo", **kw
            ),
            "list_fields": lambda src, **kw: list_fields_exodus(
                src, format="exo", **kw
            ),
        },
    }
    if registry is not None:
        for fmt, table in handlers.items():
            registry.register_reader(fmt, ScalarField, table["ScalarField"], force=True)
            registry.register_reader(fmt, VectorField, table["VectorField"], force=True)
            registry.register_reader(fmt, FieldDict, table["FieldDict"], force=True)
    return handlers


register_exodus_format = register_exodus_formats
