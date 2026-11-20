"""Tests for EXODUS II and FLOW-3D EXODUS II reader (#516)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import netCDF4
import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import FieldDict, ScalarField, VectorField
from gwexpy.interop.exodus_ import (
    ExodusCase,
    ExodusProvenance,
    Flow3DProvenance,
    RegularGridSpec,
    list_fields_exodus,
    read_fielddict_exodus,
    read_scalarfield_exodus,
    read_vectorfield_exodus,
    register_exodus_formats,
)

HAS_PYVISTA = importlib.util.find_spec("pyvista") is not None


def _write_char_var(var, strings: list[str]) -> None:
    arr = np.zeros(var.shape, dtype="S1")
    for idx, text in enumerate(strings):
        encoded = text.encode("ascii")
        for j, byte_val in enumerate(encoded[: var.shape[1]]):
            arr[idx, j] = bytes([byte_val])
    var[:] = arr


def _build_hex_grid_points_and_conn(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    z_edges: np.ndarray,
    *,
    skew_x: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = len(x_edges) - 1, len(y_edges) - 1, len(z_edges) - 1
    gx, gy, gz = np.meshgrid(x_edges, y_edges, z_edges, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    if skew_x != 0.0:
        pts[:, 0] += skew_x * pts[:, 1]

    def node_id(ix: int, iy: int, iz: int) -> int:
        return ix * ((ny + 1) * (nz + 1)) + iy * (nz + 1) + iz + 1  # 1-based EXODUS

    conns = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                conns.append(
                    [
                        node_id(ix, iy, iz),
                        node_id(ix + 1, iy, iz),
                        node_id(ix + 1, iy + 1, iz),
                        node_id(ix, iy + 1, iz),
                        node_id(ix, iy, iz + 1),
                        node_id(ix + 1, iy, iz + 1),
                        node_id(ix + 1, iy + 1, iz + 1),
                        node_id(ix, iy + 1, iz + 1),
                    ]
                )
    return pts, np.asarray(conns, dtype=np.int32)


def _write_flow3d_3d_exo(
    path: Path,
    *,
    cgs: bool = False,
    skew_x: float = 0.0,
    bad_pressure_unit: str | None = None,
    duplicate_nodal_pressure: bool = False,
    include_unknown: bool = False,
) -> Path:
    scale = 100.0 if cgs else 1.0
    pts, conn = _build_hex_grid_points_and_conn(
        np.array([0.0, 1.0, 2.0]) * scale,
        np.array([0.0, 1.0, 2.0]) * scale,
        np.array([0.0, 1.0]) * scale,
        skew_x=skew_x,
    )
    # 4 hex cells:
    # cell 0 (x=0.5, y=0.5): open, fluid (vf=1.0, f=1.0, p=0.0 physical zero!)
    # cell 1 (x=0.5, y=1.5): open, fluid (vf=0.8, f=0.5, p=100)
    # cell 2 (x=1.5, y=0.5): open, dry/void (vf=1.0, f=0.0, p=0.0)
    # cell 3 (x=1.5, y=1.5): FAVOR solid obstacle (vf=0.0, f=0.0, p=9999 junk)
    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.api_version = np.float32(5.14)
        ds.version = np.float32(5.14)
        ds.floating_point_word_size = np.int32(8)
        ds.file_size = np.int32(1)
        ds.title = "FLOW-3D synthetic 3D EXODUS fixture"

        ds.createDimension("len_string", 33)
        ds.createDimension("len_name", 33)
        ds.createDimension("time_step", None)
        ds.createDimension("num_dim", 3)
        ds.createDimension("num_nodes", len(pts))
        ds.createDimension("num_elem", len(conn))
        ds.createDimension("num_el_blk", 1)
        ds.createDimension("num_el_in_blk1", len(conn))
        ds.createDimension("num_nod_per_el1", 8)
        ds.createDimension("num_elem_var", 8 if include_unknown else 7)
        ds.createDimension("num_nod_var", 2 if duplicate_nodal_pressure else 1)

        tw = ds.createVariable("time_whole", "f8", ("time_step",))
        tw[:] = [0.0, 0.25]

        eb_status = ds.createVariable("eb_status", "i4", ("num_el_blk",))
        eb_status[:] = [1]
        eb_prop1 = ds.createVariable("eb_prop1", "i4", ("num_el_blk",))
        eb_prop1.setncattr("name", "ID")
        eb_prop1[:] = [1]
        eb_names = ds.createVariable("eb_names", "S1", ("num_el_blk", "len_name"))
        _write_char_var(eb_names, ["fluid_domain"])

        cx = ds.createVariable("coordx", "f8", ("num_nodes",))
        cy = ds.createVariable("coordy", "f8", ("num_nodes",))
        cz = ds.createVariable("coordz", "f8", ("num_nodes",))
        cx[:], cy[:], cz[:] = pts[:, 0], pts[:, 1], pts[:, 2]
        if cgs:
            cx.setncattr("units", "cm")

        c_var = ds.createVariable(
            "connect1", "i4", ("num_el_in_blk1", "num_nod_per_el1")
        )
        c_var.setncattr("elem_type", "HEX8")
        c_var[:] = conn

        elem_names = ["density", "pressure", "u", "v", "w", "f", "vf"]
        if include_unknown:
            elem_names.append("unclassified_scalar")
        nev = ds.createVariable("name_elem_var", "S1", ("num_elem_var", "len_name"))
        _write_char_var(nev, elem_names)

        rho_t = np.array(
            [
                [1000.0, 1000.0, 1.2, 0.0],
                [1002.0, 1004.0, 1.2, 0.0],
            ]
        ) * (1e-3 if cgs else 1.0)
        p_t = np.array(
            [
                [0.0, 100.0, 0.0, 9999.0],
                [0.0, 120.0, 0.0, 9999.0],
            ]
        ) * (10.0 if cgs else 1.0)
        u_t = np.array(
            [
                [0.0, 2.0, 0.5, 99.0],
                [0.0, 3.0, 0.5, 99.0],
            ]
        ) * (100.0 if cgs else 1.0)
        v_t = np.array(
            [
                [0.0, -1.0, 0.0, 99.0],
                [0.0, -1.5, 0.0, 99.0],
            ]
        ) * (100.0 if cgs else 1.0)
        w_t = np.zeros_like(u_t)
        f_t = np.array([[1.0, 0.5, 0.0, 0.0], [1.0, 0.6, 0.0, 0.0]])
        vf_t = np.array([[1.0, 0.8, 1.0, 0.0], [1.0, 0.8, 1.0, 0.0]])

        arrays = [rho_t, p_t, u_t, v_t, w_t, f_t, vf_t]
        if include_unknown:
            arrays.append(np.ones_like(rho_t))
        for k_idx, (name, arr) in enumerate(
            zip(elem_names, arrays, strict=True), start=1
        ):
            v = ds.createVariable(
                f"vals_elem_var{k_idx}eb1", "f8", ("time_step", "num_el_in_blk1")
            )
            v[:] = arr
            if name == "pressure" and bad_pressure_unit is not None:
                v.setncattr("units", bad_pressure_unit)
            elif name == "pressure" and duplicate_nodal_pressure:
                v.setncattr("units", "Pa")

        nnv = ds.createVariable("name_nod_var", "S1", ("num_nod_var", "len_name"))
        _write_char_var(
            nnv,
            ["temperature", "pressure"]
            if duplicate_nodal_pressure
            else ["temperature"],
        )
        vn1 = ds.createVariable("vals_nod_var1", "f8", ("time_step", "num_nodes"))
        vn1.setncattr("units", "K")
        vn1[0, :] = np.full(len(pts), 290.0)
        vn1[1, :] = np.full(len(pts), 295.0)
        if duplicate_nodal_pressure:
            vn2 = ds.createVariable("vals_nod_var2", "f8", ("time_step", "num_nodes"))
            vn2.setncattr("units", "kPa")
            vn2[:] = 1.5
    return path


def _write_surface_2d_exo(path: Path) -> Path:
    # 2x2 QUAD4 surface mesh in x=[0,2], y=[0,2]
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.0, 1.0, 2.0])
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel()])

    def nid(ix: int, iy: int) -> int:
        return ix * 3 + iy + 1

    conn = np.array(
        [
            [nid(0, 0), nid(1, 0), nid(1, 1), nid(0, 1)],
            [nid(0, 1), nid(1, 1), nid(1, 2), nid(0, 2)],
            [nid(1, 0), nid(2, 0), nid(2, 1), nid(1, 1)],
            [nid(1, 1), nid(2, 1), nid(2, 2), nid(1, 2)],
        ],
        dtype=np.int32,
    )
    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.api_version = np.float32(5.14)
        ds.version = np.float32(5.14)
        ds.title = "FLOW-3D 2D shallow water fixture"
        ds.createDimension("len_name", 33)
        ds.createDimension("time_step", None)
        ds.createDimension("num_dim", 2)
        ds.createDimension("num_nodes", len(pts))
        ds.createDimension("num_elem", 4)
        ds.createDimension("num_el_blk", 1)
        ds.createDimension("num_el_in_blk1", 4)
        ds.createDimension("num_nod_per_el1", 4)
        ds.createDimension("num_elem_var", 2)

        tw = ds.createVariable("time_whole", "f8", ("time_step",))
        tw[:] = [0.0, 1.0]
        eb_status = ds.createVariable("eb_status", "i4", ("num_el_blk",))
        eb_status[:] = [1]
        eb_prop1 = ds.createVariable("eb_prop1", "i4", ("num_el_blk",))
        eb_prop1.setncattr("name", "ID")
        eb_prop1[:] = [1]
        eb_names = ds.createVariable("eb_names", "S1", ("num_el_blk", "len_name"))
        _write_char_var(eb_names, ["surface"])

        cx = ds.createVariable("coordx", "f8", ("num_nodes",))
        cy = ds.createVariable("coordy", "f8", ("num_nodes",))
        cx[:], cy[:] = pts[:, 0], pts[:, 1]

        cvar = ds.createVariable(
            "connect1", "i4", ("num_el_in_blk1", "num_nod_per_el1")
        )
        cvar.setncattr("elem_type", "QUAD4")
        cvar[:] = conn

        nev = ds.createVariable("name_elem_var", "S1", ("num_elem_var", "len_name"))
        _write_char_var(nev, ["flow_depth", "terrain_elevation"])
        v1 = ds.createVariable(
            "vals_elem_var1eb1", "f8", ("time_step", "num_el_in_blk1")
        )
        v1[:] = [[0.5, 0.0, 1.2, 0.3], [0.7, 0.1, 1.5, 0.4]]
        v2 = ds.createVariable(
            "vals_elem_var2eb1", "f8", ("time_step", "num_el_in_blk1")
        )
        v2[:] = [[10.0, 11.0, 12.0, 13.0], [10.0, 11.0, 12.0, 13.0]]
    return path


def _write_multiblock_exo(path: Path, *, overlapping: bool = False) -> Path:
    # Block 1: x in [0, 1], y in [0, 1], z in [0, 1]
    # Block 2: x in [1, 2] (disjoint) OR x in [0, 1] (overlapping)
    pts1, conn1 = _build_hex_grid_points_and_conn(
        np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    x2 = np.array([0.0, 1.0]) if overlapping else np.array([1.0, 2.0])
    pts2, conn2 = _build_hex_grid_points_and_conn(
        x2, np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    pts = np.vstack([pts1, pts2])
    conn2 = conn2 + len(pts1)

    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.api_version = np.float32(5.14)
        ds.version = np.float32(5.14)
        ds.createDimension("len_name", 33)
        ds.createDimension("time_step", None)
        ds.createDimension("num_dim", 3)
        ds.createDimension("num_nodes", len(pts))
        ds.createDimension("num_el_blk", 2)
        ds.createDimension("num_el_in_blk1", 1)
        ds.createDimension("num_nod_per_el1", 8)
        ds.createDimension("num_el_in_blk2", 1)
        ds.createDimension("num_nod_per_el2", 8)
        ds.createDimension("num_elem_var", 1)

        tw = ds.createVariable("time_whole", "f8", ("time_step",))
        tw[:] = [0.0]
        eb_status = ds.createVariable("eb_status", "i4", ("num_el_blk",))
        eb_status[:] = [1, 1]
        eb_prop1 = ds.createVariable("eb_prop1", "i4", ("num_el_blk",))
        eb_prop1.setncattr("name", "ID")
        eb_prop1[:] = [10, 20]
        eb_names = ds.createVariable("eb_names", "S1", ("num_el_blk", "len_name"))
        _write_char_var(eb_names, ["blk_left", "blk_right"])

        cx = ds.createVariable("coordx", "f8", ("num_nodes",))
        cy = ds.createVariable("coordy", "f8", ("num_nodes",))
        cz = ds.createVariable("coordz", "f8", ("num_nodes",))
        cx[:], cy[:], cz[:] = pts[:, 0], pts[:, 1], pts[:, 2]

        c1 = ds.createVariable("connect1", "i4", ("num_el_in_blk1", "num_nod_per_el1"))
        c1.setncattr("elem_type", "HEX8")
        c1[:] = conn1
        c2 = ds.createVariable("connect2", "i4", ("num_el_in_blk2", "num_nod_per_el2"))
        c2.setncattr("elem_type", "HEX8")
        c2[:] = conn2

        nev = ds.createVariable("name_elem_var", "S1", ("num_elem_var", "len_name"))
        _write_char_var(nev, ["pressure"])
        v1 = ds.createVariable(
            "vals_elem_var1eb1", "f8", ("time_step", "num_el_in_blk1")
        )
        v1[:] = [[101.0]]
        v2 = ds.createVariable(
            "vals_elem_var1eb2", "f8", ("time_step", "num_el_in_blk2")
        )
        v2[:] = [[202.0]]
    return path


def test_flow3d_3d_favor_masks_and_physical_zero(tmp_path):
    exo = _write_flow3d_3d_exo(tmp_path / "flow3d.exo")

    info = list_fields_exodus(exo, format="exo.flow3d")
    assert info["pressure"]["association"] == "cell"
    assert info["pressure"]["unit"] == "Pa"
    assert info["temperature"]["association"] == "point"
    assert info["temperature"]["unit"] == "K"

    # Read pressure in exact mode
    sf_p = read_scalarfield_exodus(
        exo, "pressure", format="exo.flow3d", grid_mode="exact"
    )
    assert isinstance(sf_p, ScalarField)
    assert sf_p.shape == (2, 2, 2, 1)
    assert sf_p.unit == u.Pa
    assert np.allclose(sf_p._axis0_index.to_value(u.s), [0.0, 0.25])
    assert np.allclose(sf_p._axis1_index.to_value(u.m), [0.5, 1.5])
    assert np.allclose(sf_p._axis2_index.to_value(u.m), [0.5, 1.5])

    # Cell (0, 0): open & physical 0.0 -> MUST stay 0.0 (not NaN!)
    assert sf_p.value[0, 0, 0, 0] == pytest.approx(0.0)
    # Cell (0, 1): open -> 100.0 / 120.0
    assert sf_p.value[0, 0, 1, 0] == pytest.approx(100.0)
    assert sf_p.value[1, 0, 1, 0] == pytest.approx(120.0)
    # Cell (1, 0): open & dry (f=0, vf=1) -> keeps physical 0.0 under default empty_fluid_mode='keep'
    assert sf_p.value[0, 1, 0, 0] == pytest.approx(0.0)
    # Cell (1, 1): FAVOR-blocked (vf=0) -> MUST be NaN!
    assert np.isnan(sf_p.value[0, 1, 1, 0])
    assert np.isnan(sf_p.value[1, 1, 1, 0])

    # Provenance auto-propagates on slicing
    assert isinstance(sf_p._gwex_exodus, ExodusProvenance)
    assert isinstance(sf_p._gwex_flow3d, Flow3DProvenance)
    assert sf_p._gwex_flow3d.open_cell_fraction == pytest.approx(0.75)
    assert sf_p._gwex_flow3d.fluid_cell_fraction == pytest.approx(0.5)
    sliced = sf_p[1, 1:2, :, :]
    assert sliced._gwex_exodus.time_steps == (0.25,)
    assert sliced._gwex_exodus.valid_mask.shape == sliced.shape
    assert np.array_equal(
        sliced._gwex_exodus.valid_mask, sf_p._gwex_exodus.valid_mask[1:2, 1:2]
    )
    assert sliced._gwex_exodus.valid_fraction == pytest.approx(0.5)
    assert sliced._gwex_flow3d.open_mask.shape == sliced.shape
    assert sliced._gwex_flow3d.fluid_mask.shape == sliced.shape
    assert np.array_equal(
        sliced._gwex_flow3d.open_mask, sf_p._gwex_flow3d.open_mask[1:2, 1:2]
    )
    assert np.array_equal(
        sliced._gwex_flow3d.fluid_mask, sf_p._gwex_flow3d.fluid_mask[1:2, 1:2]
    )
    assert sliced._gwex_flow3d.open_cell_fraction == pytest.approx(0.5)
    assert sliced._gwex_flow3d.fluid_cell_fraction == pytest.approx(0.0)

    # With empty_fluid_mode='nan', cell (1, 0) (f=0) becomes NaN too while cell (0, 0) stays 0.0
    sf_p_wet = read_scalarfield_exodus(
        exo, "pressure", format="exo.flow3d", empty_fluid_mode="nan"
    )
    assert sf_p_wet.value[0, 0, 0, 0] == pytest.approx(0.0)
    assert np.isnan(sf_p_wet.value[0, 1, 0, 0])
    assert np.isnan(sf_p_wet.value[0, 1, 1, 0])

    # VectorField velocity from (u, v, w)
    vf_u = read_vectorfield_exodus(exo, "velocity", format="exo.flow3d")
    assert isinstance(vf_u, VectorField)
    assert set(vf_u.keys()) == {"x", "y", "z"}
    assert vf_u["x"].unit == u.m / u.s
    assert vf_u["x"].value[0, 0, 0, 0] == pytest.approx(0.0)
    assert vf_u["x"].value[1, 0, 1, 0] == pytest.approx(3.0)
    assert vf_u["y"].value[1, 0, 1, 0] == pytest.approx(-1.5)
    assert np.isnan(vf_u["x"].value[0, 1, 1, 0])

    # Point-associated nodal variable
    sf_t = read_scalarfield_exodus(exo, "temperature", format="exo.flow3d")
    assert sf_t.shape == (2, 3, 3, 2)
    assert sf_t.unit == u.K
    assert sf_t._gwex_exodus.association == "point"
    assert np.allclose(sf_t.value[0], 290.0)
    assert np.allclose(sf_t.value[1], 295.0)

    # Generic format="exo" works without attaching _gwex_flow3d
    sf_generic = read_scalarfield_exodus(exo, "pressure", format="exo")
    assert isinstance(sf_generic._gwex_exodus, ExodusProvenance)
    assert not hasattr(sf_generic, "_gwex_flow3d")


def test_flow3d_2d_surface_flow_depth_and_terrain_elevation(tmp_path):
    exo = _write_surface_2d_exo(tmp_path / "surface.exo")
    sf_h = read_scalarfield_exodus(exo, "flow_depth", format="exo.flow3d")
    assert sf_h.shape == (2, 2, 2, 1)
    assert sf_h.unit == u.m
    assert sf_h.value[0, 0, 1, 0] == pytest.approx(0.0)  # dry cell keeps physical 0
    assert sf_h.value[1, 1, 0, 0] == pytest.approx(1.5)

    # terrain_elevation is static (1, nx, ny, 1) even though file has 2 time steps
    sf_z = read_scalarfield_exodus(exo, "terrain_elevation", format="exo.flow3d")
    assert sf_z.shape == (1, 2, 2, 1)
    assert sf_z.unit == u.m
    assert np.allclose(sf_z.value[0, :, :, 0], [[10.0, 11.0], [12.0, 13.0]])


def test_exodus_multiblock_merge_separate_and_overlap_reject(tmp_path):
    exo_disjoint = _write_multiblock_exo(tmp_path / "disjoint.exo", overlapping=False)
    with pytest.raises(ValueError, match="Multiple EXODUS blocks selected"):
        read_scalarfield_exodus(exo_disjoint, "pressure")
    sf_merged = read_scalarfield_exodus(
        exo_disjoint, "pressure", format="exo.flow3d", block_mode="merge"
    )
    assert sf_merged.shape == (1, 2, 1, 1)
    assert np.allclose(sf_merged.value[0, :, 0, 0], [101.0, 202.0])
    assert sf_merged._gwex_exodus.blocks == ("blk_left", "blk_right")

    # Select single block by ID or name
    sf_blk2 = read_scalarfield_exodus(exo_disjoint, "pressure", block=20)
    assert sf_blk2.shape == (1, 1, 1, 1)
    assert sf_blk2.value[0, 0, 0, 0] == pytest.approx(202.0)

    # Separate in FieldDict
    fd_sep = read_fielddict_exodus(
        exo_disjoint, variables=["pressure"], block_mode="separate"
    )
    assert isinstance(fd_sep, FieldDict)
    assert set(fd_sep.keys()) == {"blk_left/pressure", "blk_right/pressure"}

    # Overlapping blocks rejected when merging
    exo_overlap = _write_multiblock_exo(tmp_path / "overlap.exo", overlapping=True)
    with pytest.raises(ValueError, match="overlapping/duplicate cell centers"):
        read_scalarfield_exodus(exo_overlap, "pressure", block_mode="merge")


def test_exodus_skewed_mesh_exact_rejects_and_sample_masks_outside_nan(tmp_path):
    exo_skew = _write_flow3d_3d_exo(tmp_path / "skew.exo", skew_x=0.4)
    with pytest.raises(ValueError, match="exact Cartesian tensor product"):
        read_scalarfield_exodus(exo_skew, "density", grid_mode="exact")

    grid = RegularGridSpec(
        x=np.array([-0.5, 0.8, 1.8, 3.5]),
        y=np.array([0.5, 1.5]),
        z=np.array([0.5]),
    )
    sf_samp = read_scalarfield_exodus(
        exo_skew,
        "density",
        format="exo.flow3d",
        grid_mode="sample",
        grid=grid,
        sample_method="nearest",
    )
    assert sf_samp.shape == (2, 4, 2, 1)
    assert np.all(np.isnan(sf_samp.value[:, 0, :, 0]))
    assert np.all(np.isnan(sf_samp.value[:, 3, :, 0]))
    assert sf_samp._gwex_exodus.grid_mode == "sample"
    assert sf_samp._gwex_exodus.valid_fraction < 1.0


def test_exodus_cgs_conversion_and_incompatible_unit_failure(tmp_path):
    exo_cgs = _write_flow3d_3d_exo(tmp_path / "cgs.exo", cgs=True)
    sf_rho = read_scalarfield_exodus(
        exo_cgs, "density", format="exo.flow3d", unit_system="cgs"
    )
    assert sf_rho.unit == u.kg / u.m**3
    assert sf_rho.value[0, 0, 0, 0] == pytest.approx(1000.0)
    assert np.allclose(sf_rho._axis1_index.to_value(u.m), [0.5, 1.5])

    sf_p = read_scalarfield_exodus(
        exo_cgs, "pressure", format="exo.flow3d", unit_system="cgs"
    )
    assert sf_p.unit == u.Pa
    assert sf_p.value[0, 0, 1, 0] == pytest.approx(100.0)

    exo_bad = _write_flow3d_3d_exo(tmp_path / "bad_u.exo", bad_pressure_unit="K")
    with pytest.raises(u.UnitConversionError):
        read_scalarfield_exodus(exo_bad, "pressure", format="exo.flow3d")


def test_same_named_cell_and_point_variables_keep_distinct_file_units(tmp_path):
    exo = _write_flow3d_3d_exo(
        tmp_path / "same_name.exo", duplicate_nodal_pressure=True
    )
    cell = read_scalarfield_exodus(exo, "pressure", association="cell")
    point = read_scalarfield_exodus(exo, "pressure", association="point")
    assert cell.unit == u.Pa
    assert cell.value[0, 0, 1, 0] == pytest.approx(100.0)
    assert cell._gwex_exodus.unit_source == "file_attribute"
    assert point.unit == u.Pa
    assert np.allclose(point.value, 1500.0)
    assert point._gwex_exodus.unit_source == "file_attribute"


def test_strict_units_rejects_unresolved_exodus_variable(tmp_path):
    exo = _write_flow3d_3d_exo(tmp_path / "unknown.exo", include_unknown=True)
    default = read_scalarfield_exodus(exo, "unclassified_scalar")
    assert default.unit == u.dimensionless_unscaled
    with pytest.raises(ValueError, match="Unit .* is unresolved"):
        read_scalarfield_exodus(exo, "unclassified_scalar", strict_units=True)
    with pytest.raises(ValueError, match="Unit .* is unresolved"):
        read_fielddict_exodus(exo, variables=["unclassified_scalar"], strict_units=True)
    overridden = read_scalarfield_exodus(
        exo, "unclassified_scalar", strict_units=True, unit=u.kg
    )
    assert overridden.unit == u.kg
    assert overridden._gwex_exodus.unit_source == "override"
    known = read_vectorfield_exodus(exo, strict_units=True)
    assert set(known.keys()) == {"x", "y", "z"}


def test_sampled_flow3d_masks_preserve_blocked_and_dry_cells(tmp_path):
    exo = _write_flow3d_3d_exo(tmp_path / "masks.exo")
    grid = RegularGridSpec(
        x=np.array([-0.5, 0.5, 1.5]),
        y=np.array([0.5, 1.5]),
        z=np.array([0.5]),
    )
    pressure = read_scalarfield_exodus(
        exo,
        "pressure",
        format="exo.flow3d",
        grid_mode="sample",
        grid=grid,
        sample_method="nearest",
    )
    flow = pressure._gwex_flow3d
    assert not flow.open_mask[0, 0, 0, 0]  # outside source domain
    assert flow.open_mask[0, 1, 0, 0]
    assert flow.fluid_mask[0, 1, 0, 0]
    assert flow.open_mask[0, 2, 0, 0]  # dry but open
    assert not flow.fluid_mask[0, 2, 0, 0]
    assert not flow.open_mask[0, 2, 1, 0]  # FAVOR-blocked
    assert not flow.fluid_mask[0, 2, 1, 0]
    assert pressure.value[0, 2, 0, 0] == pytest.approx(0.0)
    assert np.isnan(pressure.value[0, 2, 1, 0])


def test_exodus_missing_dependencies_import_error(tmp_path):
    exo = _write_flow3d_3d_exo(tmp_path / "deps.exo")
    with patch.dict(sys.modules, {"netCDF4": None, "pyvista": None}):
        with pytest.raises(ImportError, match="gwexpy\\[exodus\\]"):
            ExodusCase(exo, backend="auto")
    with patch.dict(sys.modules, {"pyvista": None}):
        with pytest.raises(ImportError, match="gwexpy\\[exodus\\]"):
            ExodusCase(exo, backend="pyvista")


@pytest.mark.skipif(not HAS_PYVISTA, reason="pyvista not installed in environment")
def test_exodus_pyvista_vs_netcdf4_backend_parity(tmp_path):
    exo = _write_flow3d_3d_exo(tmp_path / "parity.exo")
    sf_nc = read_scalarfield_exodus(exo, "pressure", backend="netcdf4")
    sf_pv = read_scalarfield_exodus(exo, "pressure", backend="pyvista")
    assert sf_nc.shape == sf_pv.shape
    assert np.allclose(sf_nc.value, sf_pv.value, equal_nan=True)
    assert sf_pv._gwex_exodus.backend == "pyvista"


def test_register_exodus_formats_returns_handlers():
    handlers = register_exodus_formats()
    assert set(handlers) == {"exo.flow3d", "exo"}
    assert "ScalarField" in handlers["exo.flow3d"]
    assert "VectorField" in handlers["exo.flow3d"]
    assert "FieldDict" in handlers["exo.flow3d"]
