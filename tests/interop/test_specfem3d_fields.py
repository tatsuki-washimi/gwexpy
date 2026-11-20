# Synthetic SPECFEM3D partition generator adapted from GraGra
# (https://github.com/tatsuki-washimi/gragra)
# Commit: ce3a9f0f7c686097dec2cb34aca80222a88d11e1
# Source: tests/_specfem_synthetic.py
# Copyright (c) 2026 Tatsuki Washimi — MIT License
"""Tests for SPECFEM3D GLL quadrature, partition merge, HDF5, VTK/PVTU, and Field conversion (#517)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import FieldDict, ScalarField, VectorField, list_fields
from gwexpy.interop.specfem3d import (
    RegularGridSpec,
    Specfem3DCase,
    Specfem3DProvenance,
    SpecfemGLLDataset,
    deduplicate_gll_points,
    element_quadrature_volume,
    elements_quadrature_volume,
    gll_points_weights,
    list_fields_specfem3d,
    merge_specfem_partitions,
    read_fielddict_specfem3d,
    read_scalarfield_specfem3d,
    read_specfem_gll_hdf5,
    read_specfem_pvtu,
    read_specfem_vtk,
    read_specfem_vtu,
    read_vectorfield_specfem3d,
    register_specfem3d_formats,
)

HAS_PYVISTA = importlib.util.find_spec("pyvista") is not None
REAL_RUN_DIR = Path(
    "/home/washimi/HDD8TB/gragra-research/2026-09-12-wave4-6/S/runs/L12000-regenerated-attempt4"
)

_GLL5 = np.array(
    [-1.0, -np.sqrt(3.0 / 7.0), 0.0, np.sqrt(3.0 / 7.0), 1.0], dtype=np.float64
)
_REF_CORNERS = np.array(
    [
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    ],
    dtype=np.float64,
)


def _box_corners(
    xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float
) -> np.ndarray:
    return np.array(
        [
            (xmin, ymin, zmin),
            (xmax, ymin, zmin),
            (xmax, ymax, zmin),
            (xmin, ymax, zmin),
            (xmin, ymin, zmax),
            (xmax, ymin, zmax),
            (xmax, ymax, zmax),
            (xmin, ymax, zmax),
        ],
        dtype=np.float64,
    )


def _element_gll_grid(corners8: np.ndarray) -> np.ndarray:
    grid = np.empty((125, 3), dtype=np.float64)
    for k in range(5):
        zeta = _GLL5[k]
        for j in range(5):
            eta = _GLL5[j]
            for i in range(5):
                xi = _GLL5[i]
                total = np.zeros(3, dtype=np.float64)
                for idx in range(8):
                    xi_i, eta_i, zeta_i = _REF_CORNERS[idx]
                    w = (
                        0.125
                        * (1.0 + xi * xi_i)
                        * (1.0 + eta * eta_i)
                        * (1.0 + zeta * zeta_i)
                    )
                    total = total + w * corners8[idx]
                grid[k * 25 + j * 5 + i] = total
    return grid


def _build_global_6_elements() -> list[dict]:
    rot_th = np.deg2rad(25.0)
    c, s = np.cos(rot_th), np.sin(rot_th)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    e3 = _box_corners(-30.0, 30.0, -40.0, 40.0, -20.0, 20.0) @ rz.T + np.array(
        [3000.0, 3000.0, 3000.0]
    )
    e4 = _box_corners(-20.0, 20.0, -25.0, 25.0, -17.5, 17.5) + np.array(
        [4000.0, 4000.0, 4000.0]
    )
    e4[6] += np.array([6.0, -4.0, 3.0])
    e5 = _box_corners(-22.5, 22.5, -27.5, 27.5, -32.5, 32.5) + np.array(
        [5000.0, 5000.0, 5000.0]
    )
    e5[0] += np.array([-3.0, 2.0, -5.0])
    return [
        {
            "corners8": _box_corners(0.0, 100.0, 0.0, 200.0, 0.0, 400.0),
            "material_id": 1,
            "rho": 2000.0,
        },
        {
            "corners8": _box_corners(100.0, 200.0, 0.0, 200.0, 0.0, 400.0),
            "material_id": 1,
            "rho": 2000.0,
        },
        {
            "corners8": _box_corners(2000.0, 2050.0, 2000.0, 2180.0, 2000.0, 2090.0),
            "material_id": 2,
            "rho": 3000.0,
        },
        {"corners8": e3, "material_id": 2, "rho": 3000.0},
        {"corners8": e4, "material_id": 1, "rho": 2000.0},
        {"corners8": e5, "material_id": 2, "rho": 3000.0},
    ]


def _fortran_record(payload: bytes) -> bytes:
    m = np.array([len(payload)], dtype="<i4").tobytes()
    return m + payload + m


def _write_synthetic_partition_tree(
    db_dir: Path,
    *,
    variant: str = "a",
    with_displacement: bool = True,
) -> Path:
    db_dir.mkdir(parents=True, exist_ok=True)
    elements = _build_global_6_elements()
    rank_map = (
        {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
        if variant == "a"
        else {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 0}
    )

    # Assign global anchor nodes (sharing E0/E1 face)
    corner_gids: list[list[int]] = [[-1] * 8 for _ in range(6)]
    anchor_coords: list[np.ndarray] = []
    nid = 0
    for loc in range(8):
        corner_gids[0][loc] = nid
        anchor_coords.append(elements[0]["corners8"][loc])
        nid += 1
    share_map = {0: 1, 3: 2, 4: 5, 7: 6}
    for loc in range(8):
        if loc in share_map:
            corner_gids[1][loc] = corner_gids[0][share_map[loc]]
        else:
            corner_gids[1][loc] = nid
            anchor_coords.append(elements[1]["corners8"][loc])
            nid += 1
    for e_idx in range(2, 6):
        for loc in range(8):
            corner_gids[e_idx][loc] = nid
            anchor_coords.append(elements[e_idx]["corners8"][loc])
            nid += 1
    anchor_arr = np.array(anchor_coords, dtype=np.float64)

    # part_array.vtk
    vtk_lines = [
        "# vtk DataFile Version 3.0",
        "synthetic part_array",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {len(anchor_arr)} double",
    ]
    for pt in anchor_arr:
        vtk_lines.append(f"{pt[0]:.17g} {pt[1]:.17g} {pt[2]:.17g}")
    vtk_lines.append(f"CELLS 6 {6 * 9}")
    for row in corner_gids:
        vtk_lines.append("8 " + " ".join(str(v) for v in row))
    vtk_lines.append("CELL_TYPES 6")
    vtk_lines.extend(["12"] * 6)
    vtk_lines.append("CELL_DATA 6")
    vtk_lines.append("SCALARS elem_flag int 1")
    vtk_lines.append("LOOKUP_TABLE default")
    vtk_lines.extend(str(rank_map[e]) for e in range(6))
    (db_dir / "part_array.vtk").write_text(
        "\n".join(vtk_lines) + "\n", encoding="ascii"
    )

    for rank in (0, 1):
        prefix = f"proc{rank:06d}"
        eids = sorted(e for e, r in rank_map.items() if r == rank)
        used_nodes = sorted({g for e in eids for g in corner_gids[e]})
        loc_of = {g: idx + 1 for idx, g in enumerate(used_nodes)}

        payloads = [np.array([len(used_nodes)], dtype="<i4").tobytes()]
        for idx, gid in enumerate(used_nodes, start=1):
            payloads.append(
                np.array([idx], dtype="<i4").tobytes()
                + anchor_arr[gid].astype("<f8").tobytes()
            )
        payloads.append(np.array([2, 0], dtype="<i4").tobytes())
        for mat_id, rho_v in ((1, 2000.0), (2, 3000.0)):
            mrec = np.zeros(17, dtype="<f8")
            mrec[0] = rho_v
            mrec[6] = 2.0  # IDOMAIN_ELASTIC
            payloads.append(mrec.tobytes())
        payloads.append(np.array([len(eids)], dtype="<i4").tobytes())
        for lidx, gid in enumerate(eids, start=1):
            mat1 = elements[gid]["material_id"]
            nlocs = [loc_of[g] for g in corner_gids[gid]]
            payloads.append(np.array([lidx, mat1, 0, *nlocs], dtype="<i4").tobytes())
        (db_dir / f"{prefix}_Database").write_bytes(
            b"".join(_fortran_record(p) for p in payloads)
        )

        # GLL coordinates & ibool (deduplicate E0/E1 shared face when on same rank)
        xyz_list: list[np.ndarray] = []
        coord_to_gid: dict[tuple[float, float, float], int] = {}
        ibool_r = np.empty((len(eids), 125), dtype=np.int32)
        rho_r = np.empty((len(eids), 125), dtype=np.float32)
        for le, gid in enumerate(eids):
            grid = _element_gll_grid(elements[gid]["corners8"])
            rho_r[le, :] = np.float32(elements[gid]["rho"])
            for p_idx in range(125):
                key = tuple(float(v) for v in np.round(grid[p_idx], 8))
                if key not in coord_to_gid:
                    coord_to_gid[key] = len(xyz_list) + 1
                    xyz_list.append(grid[p_idx])
                ibool_r[le, p_idx] = coord_to_gid[key]

        xyz_arr = np.asarray(xyz_list, dtype=np.float32)
        (db_dir / f"{prefix}_ibool.bin").write_bytes(_fortran_record(ibool_r.tobytes()))
        for axis_i, cname in enumerate(("x", "y", "z")):
            (db_dir / f"{prefix}_{cname}.bin").write_bytes(
                _fortran_record(xyz_arr[:, axis_i].tobytes())
            )
        (db_dir / f"{prefix}_rho.bin").write_bytes(_fortran_record(rho_r.tobytes()))

        if with_displacement:
            for it in (1, 2):
                # Continuous linear displacement field u(x, y, z, t) = [1e-4*it*x, -2e-4*y, 5e-5*z]
                g_coords = np.stack(
                    [_element_gll_grid(elements[gid]["corners8"]) for gid in eids],
                    axis=0,
                )
                ux = (1e-4 * it * g_coords[..., 0]).astype(np.float32)
                uy = (-2e-4 * g_coords[..., 1]).astype(np.float32)
                uz = (5e-5 * g_coords[..., 2]).astype(np.float32)
                for comp, carr in (("X", ux), ("Y", uy), ("Z", uz)):
                    (db_dir / f"{prefix}_displ_{comp}_it{it:06d}.bin").write_bytes(
                        _fortran_record(carr.tobytes())
                    )
    return db_dir


def _write_canonical_gll_hdf5(
    path: Path,
    *,
    coords_m: np.ndarray,
    corners8: np.ndarray,
    rho: np.ndarray,
    displacement_m: np.ndarray | None = None,
) -> Path:
    nspec, ngll, _ = coords_m.shape
    qvol = elements_quadrature_volume(corners8, (5, 5, 5))
    sampling = "time_sampled" if displacement_m is not None else "static"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("/mesh/coordinates", data=coords_m.astype(np.float64))
        h5.create_dataset("/mesh/quadrature_volume", data=qvol.astype(np.float64))
        h5.create_dataset("/mesh/element_id", data=np.arange(nspec, dtype=np.int64))
        h5.create_dataset("/mesh/partition_id", data=np.zeros(nspec, dtype=np.int32))
        h5.create_dataset("/mesh/domain_id", data=np.full(nspec, 2, dtype=np.int32))
        h5.create_dataset("/mesh/material_id", data=np.ones(nspec, dtype=np.int32))
        h5.create_dataset("/mesh/is_pml", data=np.zeros(nspec, dtype=bool))
        h5.create_dataset("/material/density", data=rho.astype(np.float64))
        if displacement_m is not None:
            nt = displacement_m.shape[0]
            h5.create_dataset(
                "/field/displacement", data=displacement_m.astype(np.float64)
            )
            steps = np.arange(nt, dtype=np.int64)
            dt_s = 0.01
            t0 = 0.0
            h5.create_dataset("/time_step", data=steps)
            h5.create_dataset("/time", data=steps * dt_s - t0)
        mg = h5.create_group("/metadata")
        mg.attrs["schema_version"] = 1
        mg.attrs["mesh_type"] = "gll"
        mg.attrs["layout"] = "single_file"
        mg.attrs["output_sampling_kind"] = sampling
        mg.attrs["solver_kind"] = "specfem3d_cartesian"
        mg.attrs["coordinate_convention"] = "ENU"
        mg.attrs["ngll_shape"] = np.array([5, 5, 5], dtype=np.int64)
        mg.attrs["ngll"] = 125
        mg.attrs["dt_s"] = 0.01
        mg.attrs["t0"] = 0.0
        dt_str = h5py.string_dtype(encoding="utf-8")
        mg.create_dataset(
            "field_units", data=json.dumps({"displacement": "m"}), dtype=dt_str
        )
    return path


def test_gll_quadrature_and_shared_face_deduplication():
    pts, wts = gll_points_weights(5)
    assert np.allclose(pts, _GLL5, atol=1e-14)
    assert np.sum(wts) == pytest.approx(2.0, abs=1e-14)

    box = _box_corners(0.0, 100.0, 0.0, 200.0, 0.0, 400.0)
    qv = element_quadrature_volume(box, (5, 5, 5))
    assert np.sum(qv) == pytest.approx(100.0 * 200.0 * 400.0, rel=1e-13)

    # E0 + E1 share x=100 face -> 250 raw points deduplicate to 225 unique (9 x 5 x 5)
    e0 = _element_gll_grid(box)
    e1 = _element_gll_grid(_box_corners(100.0, 200.0, 0.0, 200.0, 0.0, 400.0))
    coords = np.stack([e0, e1], axis=0)
    vals = coords[..., 0] + 2.0 * coords[..., 1]

    dedup = deduplicate_gll_points(coords, vals, verify_continuity="raise")
    assert dedup.n_raw_points == 250
    assert dedup.n_unique_points == 225
    assert dedup.max_discrepancy < 1e-12

    # Perturbing one shared face node triggers continuity ValueError
    vals_bad = vals.copy()
    vals_bad[1, 0] += 10.0
    with pytest.raises(ValueError, match="Coincident GLL points disagree"):
        deduplicate_gll_points(coords, vals_bad, verify_continuity="raise")


def test_partition_merge_invariant_across_variants_a_and_b(tmp_path):
    dir_a = _write_synthetic_partition_tree(tmp_path / "db_a", variant="a")
    dir_b = _write_synthetic_partition_tree(tmp_path / "db_b", variant="b")

    snap_a = merge_specfem_partitions(
        dir_a, include_displacement=True, dt_s=0.01, t0_s=0.0
    )
    snap_b = merge_specfem_partitions(
        dir_b, include_displacement=True, dt_s=0.01, t0_s=0.0
    )

    assert np.array_equal(snap_a.element_id, np.arange(6))
    assert np.array_equal(snap_b.element_id, np.arange(6))
    assert np.allclose(snap_a.coordinates_m, snap_b.coordinates_m)
    assert np.allclose(snap_a.fields["rho"], snap_b.fields["rho"])
    assert np.allclose(snap_a.fields["displacement"], snap_b.fields["displacement"])
    assert snap_a.provenance["nproc"] == 2
    assert snap_a.provenance["element_id_method"] == "pre_partition_global_index"


def test_gll_hdf5_exact_on_cartesian_box_and_sample_on_unstructured(tmp_path):
    # 1. Axis-aligned E0 + E1 GLL HDF5 -> grid_mode="exact" succeeds with (2, 9, 5, 5)
    elements = _build_global_6_elements()[:2]
    corners2 = np.stack([e["corners8"] for e in elements], axis=0)
    coords2 = np.stack([_element_gll_grid(c) for c in corners2], axis=0)
    rho2 = np.full((2, 125), 2000.0, dtype=np.float64)
    disp2 = np.stack(
        [
            np.stack(
                [
                    coords2[..., 0] * 1e-4,
                    -coords2[..., 1] * 1e-4,
                    coords2[..., 2] * 1e-4,
                ],
                axis=-1,
            ),
            np.stack(
                [
                    coords2[..., 0] * 2e-4,
                    -coords2[..., 1] * 2e-4,
                    coords2[..., 2] * 2e-4,
                ],
                axis=-1,
            ),
        ],
        axis=0,
    )
    h5_cart = _write_canonical_gll_hdf5(
        tmp_path / "cart.h5",
        coords_m=coords2,
        corners8=corners2,
        rho=rho2,
        displacement_m=disp2,
    )
    with patch("gwexpy.interop._specfem3d.convert.read_specfem_gll_snapshot") as load:
        assert list_fields(h5_cart, format="h5.specfem3d") == [
            "rho",
            "density",
            "displacement",
        ]
        load.assert_not_called()

    ds = read_specfem_gll_hdf5(h5_cart)
    assert isinstance(ds, SpecfemGLLDataset)
    blocks = list(ds.iter_blocks(element_block_size=1))
    assert len(blocks) == 2

    vf_exact = read_vectorfield_specfem3d(h5_cart, "displacement", grid_mode="exact")
    assert isinstance(vf_exact, VectorField)
    assert vf_exact["x"].shape == (2, 9, 5, 5)
    assert vf_exact["x"].unit == u.m
    assert isinstance(vf_exact["x"]._gwex_specfem3d, Specfem3DProvenance)
    assert vf_exact["x"]._gwex_specfem3d.deduplicated_gll is True
    assert vf_exact["x"]._gwex_specfem3d.grid_mode == "exact"

    # Provenance auto-propagates on slicing
    sliced = vf_exact["x"][:, :4, :, :]
    assert getattr(sliced, "_gwex_specfem3d", None) == vf_exact["x"]._gwex_specfem3d

    later = vf_exact["x"][1]
    assert later._gwex_specfem3d.t0_s == pytest.approx(
        later._axis0_index[0].to_value(u.s)
    )
    assert later._gwex_specfem3d.t0_s != vf_exact["x"]._gwex_specfem3d.t0_s

    # 2. Full 6-element mesh (with rotated E3 and warped E4/E5) -> exact rejects, sample works
    dir_a = _write_synthetic_partition_tree(tmp_path / "db_6", variant="a")
    with pytest.raises(ValueError, match="exact Cartesian tensor product"):
        read_scalarfield_specfem3d(dir_a, "rho", grid_mode="exact")

    grid = RegularGridSpec(
        x=np.array([-50.0, 50.0, 150.0, 6000.0]),
        y=np.array([50.0, 150.0]),
        z=np.array([100.0, 300.0]),
    )
    sf_samp = read_scalarfield_specfem3d(
        dir_a, "rho", grid_mode="sample", grid=grid, sample_method="linear"
    )
    assert sf_samp.shape == (1, 4, 2, 2)
    assert np.all(np.isnan(sf_samp.value[0, 0, :, :]))  # x=-50 outside
    assert np.all(np.isnan(sf_samp.value[0, 3, :, :]))  # x=6000 outside
    assert np.allclose(sf_samp.value[0, 1:3, :, :], 2000.0)
    assert sf_samp._gwex_specfem3d.grid_mode == "sample"
    assert sf_samp._gwex_specfem3d.valid_fraction == pytest.approx(0.5)
    selected = sf_samp[:, 1:3, :, :]
    assert selected._gwex_specfem3d.valid_mask.shape == selected.shape
    assert np.array_equal(
        selected._gwex_specfem3d.valid_mask,
        sf_samp._gwex_specfem3d.valid_mask[:, 1:3],
    )
    assert selected._gwex_specfem3d.valid_fraction == pytest.approx(1.0)

    # FieldDict
    fd = read_fielddict_specfem3d(h5_cart, grid_mode="exact")
    assert isinstance(fd, FieldDict)
    assert "rho" in fd and "displacement_x" in fd
    info = list_fields_specfem3d(h5_cart)
    assert info["displacement"]["unit"] == "m"


def test_vtk_ascii_and_pvtu_piece_merging(tmp_path):
    # Legacy ASCII VTK without numComp on SCALARS (matches real SPECFEM3D proc*_vp.vtk)
    vtk_path = tmp_path / "proc000000_vp.vtk"
    vtk_path.write_text(
        """# vtk DataFile Version 3.1
material model VTK file
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 4 float
0 0 0
1 0 0
0 1 0
1 1 0
CELLS 1 5
4 0 1 3 2
CELL_TYPES 1
9
POINT_DATA 4
SCALARS gll_data float
LOOKUP_TABLE default
3000 3100 3050 3150
""",
        encoding="ascii",
    )
    snap_vtk = read_specfem_vtk(vtk_path)
    assert "vp" in snap_vtk.fields
    sf_vp = read_scalarfield_specfem3d(vtk_path, "vp", grid_mode="exact")
    assert sf_vp.shape == (1, 2, 2, 1)
    assert sf_vp.unit == u.m / u.s

    # A vector array retains its Cartesian components and source association.
    vector_path = tmp_path / "wavefield_vector.vtk"
    vector_path.write_text(
        """# vtk DataFile Version 3.1
SPECFEM3D vector field
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 4 float
0 0 0
1 0 0
0 1 0
1 1 0
CELLS 1 5
4 0 1 3 2
CELL_TYPES 1
9
POINT_DATA 4
VECTORS displacement float
0 1 2
1 2 3
2 3 4
3 4 5
""",
        encoding="ascii",
    )
    snap_vector = read_specfem_vtk(vector_path)
    assert snap_vector.association == "point"
    assert snap_vector.fields["displacement"].shape == (4, 3)
    vector = read_vectorfield_specfem3d(vector_path, "displacement", grid_mode="exact")
    assert set(vector.keys()) == {"x", "y", "z"}
    assert vector["x"].unit == u.m
    assert np.array_equal(vector["x"].value[0, :, :, 0], [[0, 2], [1, 3]])
    assert np.array_equal(vector["z"].value[0, :, :, 0], [[2, 4], [3, 5]])
    assert vector["x"]._gwex_specfem3d.association == "point"

    # ASCII XML .vtu + .pvtu partition merge
    for idx, x_off in ((0, 0.0), (1, 1.0)):
        (tmp_path / f"piece_{idx}.vtu").write_text(
            f"""<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
<UnstructuredGrid>
<Piece NumberOfPoints="2" NumberOfCells="0">
<Points>
<DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii">
{x_off} 0 0 {x_off} 1 0
</DataArray>
</Points>
<PointData>
<DataArray type="Float64" Name="rho" format="ascii">
2500.0 2600.0
</DataArray>
</PointData>
</Piece>
</UnstructuredGrid>
</VTKFile>
""",
            encoding="ascii",
        )
    pvtu_path = tmp_path / "field.pvtu"
    pvtu_path.write_text(
        """<?xml version="1.0"?>
<VTKFile type="PUnstructuredGrid" version="0.1" byte_order="LittleEndian">
<PUnstructuredGrid>
<Piece Source="piece_0.vtu"/>
<Piece Source="piece_1.vtu"/>
</PUnstructuredGrid>
</VTKFile>
""",
        encoding="ascii",
    )
    snap_pvtu = read_specfem_pvtu(pvtu_path)
    assert snap_pvtu.coordinates_m.shape == (4, 3)
    sf_pvtu = read_scalarfield_specfem3d(pvtu_path, "rho", grid_mode="exact")
    assert sf_pvtu.shape == (1, 2, 2, 1)
    assert sf_pvtu.unit == u.kg / u.m**3
    with patch("gwexpy.interop._specfem3d.convert._coerce_field_snapshot") as load:
        assert list_fields(vtk_path, format="vtk.specfem3d") == ["vp"]
        assert list_fields(vector_path, format="vtk.specfem3d") == ["displacement"]
        assert list_fields(pvtu_path, format="vtk.specfem3d") == ["rho"]
        load.assert_not_called()


def test_missing_optional_deps_errors(tmp_path):
    with patch.dict(sys.modules, {"h5py": None}):
        with pytest.raises(ImportError, match="h5py"):
            read_specfem_gll_hdf5(tmp_path / "dummy.h5")

    appended_vtu = tmp_path / "bin.vtu"
    appended_vtu.write_text(
        """<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
<UnstructuredGrid>
<Piece NumberOfPoints="1" NumberOfCells="0">
<Points><DataArray type="Float32" Name="Points" NumberOfComponents="3" format="appended" offset="0"/></Points>
</Piece>
</UnstructuredGrid>
</VTKFile>
""",
        encoding="ascii",
    )
    with patch.dict(sys.modules, {"pyvista": None}):
        with pytest.raises(ImportError, match="gwexpy\\[specfem3d\\]"):
            read_specfem_vtu(appended_vtu)


def test_register_specfem3d_formats_returns_handlers():
    handlers = register_specfem3d_formats()
    assert set(handlers) == {"specfem3d", "h5.specfem3d", "vtk.specfem3d"}


@pytest.mark.skipif(
    not REAL_RUN_DIR.is_dir(),
    reason="Local real SPECFEM3D run output directory not present",
)
def test_real_local_specfem3d_run_partitions_and_vtk():
    """Genuine producer verification against real 4-rank SPECFEM3D DATABASES_MPI_RUN."""
    db_run = REAL_RUN_DIR / "OUTPUT_FILES" / "DATABASES_MPI_RUN"
    snap = merge_specfem_partitions(db_run, fields=("rho", "vp"))
    assert snap.provenance["nproc"] == 4
    assert snap.provenance["element_id_method"] == "pre_partition_global_index"
    assert snap.coordinates_m.shape == (27648, 125, 3)
    assert snap.fields["rho"].shape == (27648, 125)
    assert np.all(snap.fields["rho"] > 0.0)
    assert np.all(snap.fields["vp"] > 0.0)
    assert snap.source_kind == "partition_bin"
    assert snap.association == "gll"
    assert snap.ngll_shape == (5, 5, 5)
    assert snap.field_units["rho"] == u.kg / u.m**3
    assert snap.field_units["vp"] == u.m / u.s
    assert snap.partition_id is not None
    assert np.array_equal(np.unique(snap.partition_id), [0, 1, 2, 3])
    assert snap.material_id is not None
    assert not snap.coordinates_m.flags.writeable
    assert not snap.fields["rho"].flags.writeable

    # Real legacy ASCII GLL VTK from rank 0
    vp_vtk = db_run / "proc000000_vp.vtk"
    snap_vtk = read_specfem_vtk(vp_vtk)
    assert snap_vtk.coordinates_m.shape == (461041, 3)
    assert "vp" in snap_vtk.fields
    assert snap_vtk.association == "point"
    assert snap_vtk.field_units["vp"] == u.m / u.s
    assert snap_vtk.provenance["input_sha256"]

    # Real appended-XML VTU from rank 0 (uses PyVista when installed)
    jac_vtu = db_run / "proc000000_jacobian_min.vtu"
    if HAS_PYVISTA:
        snap_vtu = read_specfem_vtu(jac_vtu)
        assert snap_vtu.association == "cell"
        assert "jacobian_min" in snap_vtu.fields
        assert len(snap_vtu.fields["jacobian_min"]) == 6912
