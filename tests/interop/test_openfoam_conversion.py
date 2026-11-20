# Synthetic VTK fixture adapted from GraGra (https://github.com/tatsuki-washimi/gragra)
# Commit: ce3a9f0f7c686097dec2cb34aca80222a88d11e1
# Source: tests/_openfoam_synthetic.py
# Copyright (c) 2026 Tatsuki Washimi — MIT License
"""Conversion and backend integration tests for OpenFOAM readers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import FieldDict, ScalarField, VectorField
from gwexpy.interop.openfoam_ import (
    OpenFOAMCase,
    OpenFOAMDensitySnapshot,
    OpenFOAMProvenance,
    RegularGridSpec,
    list_fields_openfoam,
    load_openfoam_density,
    read_fielddict_openfoam,
    read_openfoam_vtk,
    read_scalarfield_openfoam,
    read_vectorfield_openfoam,
    register_openfoam_formats,
)

HAS_PYVISTA = importlib.util.find_spec("pyvista") is not None


def _write_two_hex(
    path: Path,
    *,
    density=(-1.0, 2.0),
    centres=None,
    ids=(0, 1),
    extra: bool = False,
    x_offset: float = 0.0,
) -> Path:
    if centres is None:
        centres = ((0.5 + x_offset, 0.5, 0.5), (1.5 + x_offset, 0.5, 0.5))
    points = "\n".join(
        f"{x} {y} {z}"
        for x, y, z in (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
            (2, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (2, 1, 1),
        )
        for x in (x + x_offset,)
    )
    cells = "0 1 2 3 4 5 6 7\n1 8 9 2 5 10 11 6"

    def scalar(name, values, kind="float"):
        return f"SCALARS {name} {kind} 1\nLOOKUP_TABLE default\n" + "\n".join(
            map(str, values)
        )

    c = "VECTORS C float\n" + "\n".join(" ".join(map(str, row)) for row in centres)
    text = "# vtk DataFile Version 3.0\nOpenFOAM\nASCII\nDATASET UNSTRUCTURED_GRID\n"
    text += (
        f"POINTS 12 float\n{points}\nCELLS 2 18\n8 {cells.splitlines()[0]}\n"
        f"8 {cells.splitlines()[1]}\nCELL_TYPES 2\n12\n12\nCELL_DATA 2\n"
    )
    if extra:
        text += scalar("p", (1, 2)) + "\n" + scalar("p_rgh", (3, 4)) + "\n"
        text += "VECTORS U float\n0 0 0\n1 2 3\n" + scalar("T", (280, 281)) + "\n"
    text += (
        scalar("cellID", ids, "int")
        + "\n"
        + scalar("rho", (999, 1002))
        + "\n"
        + scalar("delta_rho", density)
        + "\n"
        + c
        + "\n"
        + scalar("Vc", (1, 1))
        + "\n"
    )
    path.write_text(text, encoding="ascii")
    return path


def _default_density_provenance():
    return {
        "distribution": "OpenFOAM Foundation",
        "version": "14",
        "patch": "20260724",
        "solver": "manufactured-no-solver",
        "compressibility": "not-applicable",
        "region": "region0",
        "reconstructed": True,
        "exporter": "foamToVTK",
        "export_command": "foamToVTK -ascii",
    }


def _build_synthetic_openfoam_case(
    case_dir: Path,
    *,
    kinematic_p: bool = False,
    moving_mesh_dx: float = 0.0,
    multiregion: bool = False,
) -> Path:
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    (case_dir / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
    (case_dir / "system" / "controlDict").write_text(
        """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.5;
deltaT          0.5;
writeControl    timeStep;
writeInterval   1;
""",
        encoding="ascii",
    )

    def write_polymesh(pm_dir: Path, dx: float = 0.0) -> None:
        pm_dir.mkdir(parents=True, exist_ok=True)
        # 2x1x1 hex mesh covering [0+dx, 2+dx] x [0, 1] x [0, 1]
        (pm_dir / "points").write_text(
            f"""FoamFile
{{
    version 2.0; format ascii; class vectorField; object points;
}}
12
(
({0.0 + dx} 0 0)
({1.0 + dx} 0 0)
({1.0 + dx} 1 0)
({0.0 + dx} 1 0)
({0.0 + dx} 0 1)
({1.0 + dx} 0 1)
({1.0 + dx} 1 1)
({0.0 + dx} 1 1)
({2.0 + dx} 0 0)
({2.0 + dx} 1 0)
({2.0 + dx} 0 1)
({2.0 + dx} 1 1)
)
""",
            encoding="ascii",
        )
        # Internal face between cell 0 and cell 1 must come BEFORE boundary faces in OpenFOAM!
        (pm_dir / "faces").write_text(
            """FoamFile
{
    version 2.0; format ascii; class faceList; object faces;
}
11
(
4(1 2 6 5)
4(0 3 2 1)
4(4 5 6 7)
4(0 1 5 4)
4(2 3 7 6)
4(0 4 7 3)
4(1 2 9 8)
4(5 10 11 6)
4(1 8 10 5)
4(2 6 11 9)
4(8 9 11 10)
)
""",
            encoding="ascii",
        )
        (pm_dir / "owner").write_text(
            """FoamFile
{
    version 2.0; format ascii; class labelList; object owner;
}
11
(
0
0
0
0
0
0
1
1
1
1
1
)
""",
            encoding="ascii",
        )
        (pm_dir / "neighbour").write_text(
            """FoamFile
{
    version 2.0; format ascii; class labelList; object neighbour;
}
1
(
1
)
""",
            encoding="ascii",
        )
        (pm_dir / "boundary").write_text(
            """FoamFile
{
    version 2.0; format ascii; class polyBoundaryMesh; object boundary;
}
1
(
    walls
    {
        type wall;
        nFaces 10;
        startFace 1;
    }
)
""",
            encoding="ascii",
        )

    write_polymesh(case_dir / "constant" / "polyMesh", dx=0.0)
    if multiregion:
        write_polymesh(case_dir / "constant" / "solid" / "polyMesh", dx=0.0)

    p_dims = "[0 2 -2 0 0 0 0]" if kinematic_p else "[1 -1 -2 0 0 0 0]"

    for tname, p_vals, u_vals, rho_vals in (
        ("0", (100.0, 200.0), ((1.0, 0.0, 0.0), (2.0, 0.5, -0.5)), (1000.0, 1005.0)),
        ("0.5", (110.0, 220.0), ((3.0, 1.0, 0.0), (4.0, 1.5, -1.5)), (1010.0, 1015.0)),
    ):
        tdir = case_dir / tname
        tdir.mkdir(parents=True, exist_ok=True)
        if moving_mesh_dx != 0.0 and tname != "0":
            write_polymesh(tdir / "polyMesh", dx=moving_mesh_dx)
        (tdir / "p").write_text(
            f"""FoamFile
{{
    version 2.0; format ascii; class volScalarField; object p;
}}
dimensions {p_dims};
internalField nonuniform List<scalar>
2
(
{p_vals[0]}
{p_vals[1]}
)
;
boundaryField {{ walls {{ type zeroGradient; }} }}
""",
            encoding="ascii",
        )
        (tdir / "rho").write_text(
            f"""FoamFile
{{
    version 2.0; format ascii; class volScalarField; object rho;
}}
dimensions [1 -3 0 0 0 0 0];
internalField nonuniform List<scalar>
2
(
{rho_vals[0]}
{rho_vals[1]}
)
;
boundaryField {{ walls {{ type zeroGradient; }} }}
""",
            encoding="ascii",
        )
        (tdir / "U").write_text(
            f"""FoamFile
{{
    version 2.0; format ascii; class volVectorField; object U;
}}
dimensions [0 1 -1 0 0 0 0];
internalField nonuniform List<vector>
2
(
({u_vals[0][0]} {u_vals[0][1]} {u_vals[0][2]})
({u_vals[1][0]} {u_vals[1][1]} {u_vals[1][2]})
)
;
boundaryField {{ walls {{ type fixedValue; value uniform (0 0 0); }} }}
""",
            encoding="ascii",
        )
        if multiregion:
            sdir = tdir / "solid"
            sdir.mkdir(parents=True, exist_ok=True)
            (sdir / "T").write_text(
                f"""FoamFile
{{
    version 2.0; format ascii; class volScalarField; object T;
}}
dimensions [0 0 0 1 0 0 0];
internalField nonuniform List<scalar>
2
(
{300.0 + float(tname) * 10}
{310.0 + float(tname) * 10}
)
;
boundaryField {{ walls {{ type zeroGradient; }} }}
""",
                encoding="ascii",
            )
    return case_dir


def test_gragra_density_snapshot_and_geometry_validation(tmp_path):
    vtk_path = _write_two_hex(tmp_path / "openfoam_0.5.vtk", density=(-1.0, 2.0))
    snap = load_openfoam_density(
        vtk_path,
        field_name="delta_rho",
        time_s=0.5,
        time_name="0.5",
        coordinate_frame="ENU",
        coordinate_origin_m=[10.0, 20.0, 30.0],
        density_unit="kg/m^3",
        coordinate_unit="m",
        volume_unit="m^3",
        time_unit="s",
        provenance=_default_density_provenance(),
    )
    assert isinstance(snap, OpenFOAMDensitySnapshot)
    assert np.allclose(snap.density_kg_m3, [-1.0, 2.0])
    assert np.allclose(snap.positions_m, [[10.5, 20.5, 30.5], [11.5, 20.5, 30.5]])
    assert np.allclose(snap.volumes_m3, [1.0, 1.0])
    assert len(snap.provenance["input_sha256"]) == 64

    # Corrupt cell centre should fail precision-aware _geometry validation
    bad_path = _write_two_hex(
        tmp_path / "bad_0.5.vtk",
        centres=((0.6, 0.5, 0.5), (1.5, 0.5, 0.5)),
    )
    with pytest.raises(ValueError, match="native C/Vc fails fixed geometry tolerance"):
        load_openfoam_density(
            bad_path,
            field_name="rho",
            time_s=0.5,
            time_name="0.5",
            coordinate_frame="ENU",
            coordinate_origin_m=[0.0, 0.0, 0.0],
            density_unit="kg/m^3",
            coordinate_unit="m",
            volume_unit="m^3",
            time_unit="s",
            provenance=_default_density_provenance(),
        )


def test_legacy_vtk_to_scalarfield_vectorfield_and_fielddict(tmp_path):
    vtk_path = _write_two_hex(tmp_path / "case_0.5.vtk", extra=True)
    snap = read_openfoam_vtk(vtk_path)
    assert snap.time_s == pytest.approx(0.5)
    assert set(snap.fields) >= {"rho", "delta_rho", "p", "p_rgh", "U", "T"}

    # Exact Cartesian grid conversion (2 x 1 x 1)
    sf_rho = read_scalarfield_openfoam(vtk_path, "rho", mode="exact")
    assert isinstance(sf_rho, ScalarField)
    assert sf_rho.shape == (1, 2, 1, 1)
    assert sf_rho.unit.is_equivalent(u.kg / u.m**3)
    assert np.allclose(sf_rho.value[:, :, 0, 0], [[999.0, 1002.0]])
    assert np.allclose(sf_rho._axis1_index.to_value(u.m), [0.5, 1.5])
    assert isinstance(sf_rho._gwex_openfoam, OpenFOAMProvenance)
    assert sf_rho._gwex_openfoam.grid_mode == "exact"
    # Native cell volumes and the exact Cartesian field give the same integral.
    assert snap.cell_volumes_m3 is not None
    native_mass = np.sum(snap.fields["rho"].values * snap.cell_volumes_m3)
    field_mass = np.sum(sf_rho.value) * 1.0  # 1 m^3 per Cartesian cell
    assert field_mass == pytest.approx(native_mass)
    assert len(snap.cell_blocks) == 1
    assert snap.fields["rho"].association == "cell"
    with pytest.raises(ValueError, match="would reinterpret its values"):
        read_scalarfield_openfoam(vtk_path, "rho", association="point")

    # Provenance auto-propagates on slicing
    sliced = sf_rho[:, :1, :, :]
    assert getattr(sliced, "_gwex_openfoam", None) == sf_rho._gwex_openfoam

    # Vector field U
    vf_u = read_vectorfield_openfoam(vtk_path, "U", mode="exact")
    assert isinstance(vf_u, VectorField)
    assert set(vf_u.keys()) == {"x", "y", "z"}
    assert vf_u["x"].unit.is_equivalent(u.m / u.s)
    assert np.allclose(vf_u["z"].value.ravel(), [0.0, 3.0])

    # FieldDict expands vector fields and keeps mixed units
    fd = read_fielddict_openfoam(vtk_path, fields=["rho", "T", "U"], mode="exact")
    assert isinstance(fd, FieldDict)
    assert set(fd.keys()) == {"rho", "T", "U_x", "U_y", "U_z"}
    assert fd["T"].unit == u.K

    # Explicit sample mode with NaN outside domain
    grid = RegularGridSpec(
        x=np.array([-0.5, 0.5, 1.5, 2.5]),
        y=np.array([0.5]),
        z=np.array([0.5]),
    )
    sf_sampled = read_scalarfield_openfoam(
        vtk_path, "rho", mode="sample", grid=grid, sample_method="linear"
    )
    assert sf_sampled.shape == (1, 4, 1, 1)
    assert np.isnan(sf_sampled.value[0, 0, 0, 0])
    assert np.isnan(sf_sampled.value[0, 3, 0, 0])
    assert np.allclose(sf_sampled.value[0, 1:3, 0, 0], [999.0, 1002.0])
    assert sf_sampled._gwex_openfoam.grid_mode == "sample"
    assert sf_sampled._gwex_openfoam.valid_fraction == pytest.approx(0.5)


def test_openfoam_case_discovery_and_missing_pyvista_error(tmp_path):
    case_dir = _build_synthetic_openfoam_case(
        tmp_path / "case_disc", kinematic_p=True, multiregion=True
    )
    case = OpenFOAMCase(case_dir, skip_zero_time=False)
    assert case.available_time_names == ("0", "0.5")
    assert case.available_times == (0.0, 0.5)
    assert case.available_regions == ("region0", "solid")
    assert case.solver == "simpleFoam"

    meta = list_fields_openfoam(case_dir)
    assert "p" in meta and meta["p"]["pressure_kind"] == "kinematic"
    assert meta["p"]["class"] == "volScalarField"
    assert meta["U"]["class"] == "volVectorField"

    meta_solid = list_fields_openfoam(case_dir, region="solid")
    assert "T" in meta_solid and meta_solid["T"]["unit"] == "K"

    # Missing pyvista raises informative ImportError with gwexpy[openfoam] hint
    with patch.dict(sys.modules, {"pyvista": None}):
        with pytest.raises(ImportError, match="gwexpy\\[openfoam\\]"):
            read_scalarfield_openfoam(case_dir, "p")


@pytest.mark.skipif(not HAS_PYVISTA, reason="pyvista not installed in environment")
def test_pyvista_openfoam_case_exact_sample_kinematic_and_moving_mesh(tmp_path):
    case_dir = _build_synthetic_openfoam_case(
        tmp_path / "case_pv", kinematic_p=True, multiregion=False
    )
    # 1. Exact mode across both timesteps (t=0, 0.5)
    sf_p_kin = read_scalarfield_openfoam(case_dir, "p", mode="exact")
    assert sf_p_kin.shape == (2, 2, 1, 1)
    assert sf_p_kin.unit.is_equivalent(u.m**2 / u.s**2)
    assert sf_p_kin._gwex_openfoam.pressure_kind == "kinematic"
    assert np.allclose(sf_p_kin.value[:, :, 0, 0], [[100.0, 200.0], [110.0, 220.0]])
    assert np.allclose(sf_p_kin._axis0_index.to_value(u.s), [0.0, 0.5])

    # Explicit rho_ref converts kinematic pressure to Pa
    sf_p_phys = read_scalarfield_openfoam(
        case_dir, "p", mode="exact", rho_ref=1000.0 * u.kg / u.m**3
    )
    assert sf_p_phys.unit == u.Pa
    assert sf_p_phys._gwex_openfoam.pressure_kind == "physical_converted"
    assert np.allclose(sf_p_phys.value[:, :, 0, 0], [[1.0e5, 2.0e5], [1.1e5, 2.2e5]])

    # Vector field U via PyVista
    vf_u = read_vectorfield_openfoam(case_dir, "U", mode="exact")
    assert isinstance(vf_u, VectorField)
    assert np.allclose(vf_u["x"].value[:, :, 0, 0], [[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(vf_u["z"].value[:, :, 0, 0], [[0.0, -0.5], [0.0, -1.5]])

    # Sample mode via PyVista vtk_probe
    grid = RegularGridSpec(
        x=np.array([-0.5, 0.5, 1.5, 2.5]),
        y=np.array([0.5]),
        z=np.array([0.5]),
    )
    sf_sampled = read_scalarfield_openfoam(
        case_dir, "rho", mode="sample", grid=grid, sample_method="vtk_probe"
    )
    assert sf_sampled.shape == (2, 4, 1, 1)
    assert np.all(np.isnan(sf_sampled.value[:, 0, 0, 0]))
    assert np.all(np.isnan(sf_sampled.value[:, 3, 0, 0]))
    assert np.allclose(
        sf_sampled.value[:, 1:3, 0, 0], [[1000.0, 1005.0], [1010.0, 1015.0]]
    )

    # Moving mesh: error by default, resample when moving_mesh='resample'
    moving_dir = _build_synthetic_openfoam_case(
        tmp_path / "case_moving", kinematic_p=False, moving_mesh_dx=1.0
    )
    with pytest.raises(ValueError, match="Moving or deforming mesh"):
        read_scalarfield_openfoam(moving_dir, "p", mode="exact", moving_mesh="error")

    fixed_grid = RegularGridSpec(
        x=np.array([0.5, 1.5, 2.5]),
        y=np.array([0.5]),
        z=np.array([0.5]),
    )
    sf_mov = read_scalarfield_openfoam(
        moving_dir,
        "p",
        mode="sample",
        grid=fixed_grid,
        moving_mesh="resample",
        sample_method="vtk_probe",
    )
    assert sf_mov._gwex_openfoam.mesh_motion_detected is True
    # At t=0 mesh covers [0, 2] -> x=2.5 is NaN; at t=0.5 mesh shifts to [1, 3] -> x=0.5 is NaN
    assert np.allclose(sf_mov.value[0, :2, 0, 0], [100.0, 200.0])
    assert np.isnan(sf_mov.value[0, 2, 0, 0])
    assert np.isnan(sf_mov.value[1, 0, 0, 0])
    assert np.allclose(sf_mov.value[1, 1:3, 0, 0], [110.0, 220.0])


def test_register_openfoam_formats_returns_handlers():
    handlers = register_openfoam_formats()
    assert set(handlers) == {"openfoam", "foam", "vtk.openfoam"}
    assert handlers["openfoam"]["ScalarField"] is read_scalarfield_openfoam
