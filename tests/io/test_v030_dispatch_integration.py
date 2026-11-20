"""End-to-end public dispatch integration tests for v0.3.0 wired readers."""

from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import FieldDict, ScalarField, VectorField, list_fields
from gwexpy.timeseries import TimeSeriesDict


def _make_minimal_openfoam_vtk(path: Path) -> Path:
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
    )
    text = (
        "# vtk DataFile Version 2.0\n"
        "foam\n"
        "ASCII\n"
        "DATASET UNSTRUCTURED_GRID\n"
        f"POINTS 12 float\n{points}\n"
        "CELLS 2 18\n"
        "8 0 1 2 3 4 5 6 7\n"
        "8 1 8 9 2 5 10 11 6\n"
        "CELL_TYPES 2\n12\n12\n"
        "CELL_DATA 2\n"
        "SCALARS p float 1\nLOOKUP_TABLE default\n100.0\n200.0\n"
        "SCALARS cellID int 1\nLOOKUP_TABLE default\n0\n1\n"
        "FIELD FieldData 1\n"
        "C 3 2 float\n0.5 0.5 0.5\n1.5 0.5 0.5\n"
        "VECTORS U float\n1 0 0\n2 0.5 -0.5\n"
    )
    path.write_text(text, encoding="ascii")
    return path


def _make_minimal_openfoam_case(case_dir: Path) -> Path:
    (case_dir / "system").mkdir(parents=True, exist_ok=True)
    pm = case_dir / "constant" / "polyMesh"
    pm.mkdir(parents=True, exist_ok=True)
    (case_dir / "system" / "controlDict").write_text(
        "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
        "startTime 0; endTime 0.5; deltaT 0.5;\n",
        encoding="ascii",
    )
    tdir = case_dir / "0"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "p").write_text(
        "FoamFile { version 2.0; format ascii; class volScalarField; object p; }\n"
        "dimensions [1 -1 -2 0 0 0 0];\n"
        "internalField nonuniform List<scalar>\n2\n(\n100.0\n200.0\n);\n"
        "boundaryField { walls { type zeroGradient; } }\n",
        encoding="ascii",
    )
    return case_dir


def _make_minimal_exodus(path: Path, block_names: tuple[str, ...] = ("fluid",)) -> Path:
    def write_char(var, items: list[str]) -> None:
        arr = np.zeros(var.shape, dtype="S1")
        for i, text in enumerate(items):
            for j, ch in enumerate(text.encode("ascii")[: var.shape[1]]):
                arr[i, j] = bytes([ch])
        var[:] = arr

    x_e = np.array([0.0, 1.0, 2.0])
    y_e = np.array([0.0, 1.0, 2.0])
    z_e = np.array([0.0, 1.0])
    gx, gy, gz = np.meshgrid(x_e, y_e, z_e, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    def nid(ix: int, iy: int, iz: int) -> int:
        return ix * 6 + iy * 2 + iz + 1

    conn = []
    for ix in range(2):
        for iy in range(2):
            conn.append(
                [
                    nid(ix, iy, 0),
                    nid(ix + 1, iy, 0),
                    nid(ix + 1, iy + 1, 0),
                    nid(ix, iy + 1, 0),
                    nid(ix, iy, 1),
                    nid(ix + 1, iy, 1),
                    nid(ix + 1, iy + 1, 1),
                    nid(ix, iy + 1, 1),
                ]
            )
    conn_arr = np.asarray(conn, dtype=np.int32)

    with netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC") as ds:
        ds.api_version = np.float32(5.14)
        ds.version = np.float32(5.14)
        ds.floating_point_word_size = np.int32(8)
        ds.file_size = np.int32(1)
        ds.title = "FLOW-3D minimal dispatch test"
        ds.createDimension("len_string", 33)
        ds.createDimension("len_name", 33)
        ds.createDimension("time_step", None)
        ds.createDimension("num_dim", 3)
        ds.createDimension("num_nodes", len(pts))
        ds.createDimension("num_elem", len(conn_arr))
        ds.createDimension("num_el_blk", len(block_names))
        for index in range(1, len(block_names) + 1):
            ds.createDimension(f"num_el_in_blk{index}", len(conn_arr))
            ds.createDimension(f"num_nod_per_el{index}", 8)
        ds.createDimension("num_elem_var", 6)

        tw = ds.createVariable("time_whole", "f8", ("time_step",))
        tw[:] = [0.0, 0.25]
        eb_status = ds.createVariable("eb_status", "i4", ("num_el_blk",))
        eb_status[:] = [1] * len(block_names)
        eb_prop1 = ds.createVariable("eb_prop1", "i4", ("num_el_blk",))
        eb_prop1.setncattr("name", "ID")
        eb_prop1[:] = list(range(1, len(block_names) + 1))
        eb_names = ds.createVariable("eb_names", "S1", ("num_el_blk", "len_name"))
        write_char(eb_names, block_names)

        for axis_name, col in zip(("coordx", "coordy", "coordz"), pts.T, strict=True):
            v = ds.createVariable(axis_name, "f8", ("num_nodes",))
            v[:] = col
        for index in range(1, len(block_names) + 1):
            cvar = ds.createVariable(
                f"connect{index}",
                "i4",
                (f"num_el_in_blk{index}", f"num_nod_per_el{index}"),
            )
            cvar.setncattr("elem_type", "HEX8")
            cvar[:] = conn_arr

        names = ["pressure", "u", "v", "w", "f", "vf"]
        nev = ds.createVariable("name_elem_var", "S1", ("num_elem_var", "len_name"))
        write_char(nev, names)
        vals = [
            np.array([[10.0, 20.0, 30.0, 40.0], [15.0, 25.0, 35.0, 45.0]]),
            np.array([[1.0, 2.0, 3.0, 4.0], [1.5, 2.5, 3.5, 4.5]]),
            np.zeros((2, 4)),
            np.zeros((2, 4)),
            np.ones((2, 4)),
            np.ones((2, 4)),
        ]
        for index in range(1, len(block_names) + 1):
            for idx, arr in enumerate(vals, start=1):
                ev = ds.createVariable(
                    f"vals_elem_var{idx}eb{index}",
                    "f8",
                    ("time_step", f"num_el_in_blk{index}"),
                )
                ev[:] = arr
    return path


def _make_minimal_specfem_partition(db_dir: Path) -> Path:
    db_dir.mkdir(parents=True, exist_ok=True)
    gll = np.array([-1.0, -np.sqrt(3.0 / 7.0), 0.0, np.sqrt(3.0 / 7.0), 1.0])
    gx, gy, gz = np.meshgrid(gll, gll, gll, indexing="ij")
    pts = np.column_stack(
        [
            ((gx + 1.0) * 50.0).ravel(),
            ((gy + 1.0) * 50.0).ravel(),
            ((gz + 1.0) * 50.0).ravel(),
        ]
    ).astype(np.float32)

    def rec(payload: bytes) -> bytes:
        marker = np.array([len(payload)], dtype="<i4").tobytes()
        return marker + payload + marker

    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
            [100.0, 100.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
            [100.0, 0.0, 100.0],
            [100.0, 100.0, 100.0],
            [0.0, 100.0, 100.0],
        ],
        dtype="<f8",
    )
    mrec = np.zeros(17, dtype="<f8")
    mrec[0] = 2600.0
    mrec[6] = 2.0  # IDOMAIN_ELASTIC
    db_payloads = [np.array([8], dtype="<i4").tobytes()]
    for idx in range(8):
        db_payloads.append(
            np.array([idx + 1], dtype="<i4").tobytes() + corners[idx].tobytes()
        )
    db_payloads.append(np.array([1, 0], dtype="<i4").tobytes())
    db_payloads.append(mrec.tobytes())
    db_payloads.append(np.array([1], dtype="<i4").tobytes())
    db_payloads.append(
        np.array([1, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype="<i4").tobytes()
    )
    (db_dir / "proc000000_Database").write_bytes(b"".join(rec(p) for p in db_payloads))

    ibool = np.arange(1, 126, dtype=np.int32).reshape(1, 125)
    rho = np.full((1, 125), 2600.0, dtype=np.float32)
    (db_dir / "proc000000_ibool.bin").write_bytes(rec(ibool.tobytes()))
    (db_dir / "proc000000_x.bin").write_bytes(rec(pts[:, 0].tobytes()))
    (db_dir / "proc000000_y.bin").write_bytes(rec(pts[:, 1].tobytes()))
    (db_dir / "proc000000_z.bin").write_bytes(rec(pts[:, 2].tobytes()))
    (db_dir / "proc000000_rho.bin").write_bytes(rec(rho.tobytes()))
    for comp, val in (("X", 0.01), ("Y", 0.0), ("Z", 0.0)):
        arr = np.full((1, 125), val, dtype=np.float32)
        (db_dir / f"proc000000_displ_{comp}_it000001.bin").write_bytes(
            rec(arr.tobytes())
        )
    return db_dir


def test_openfoam_public_field_dispatch(tmp_path: Path):
    case = _make_minimal_openfoam_case(tmp_path / "foam_case")
    assert "p" in list_fields(case, format="openfoam")

    vtk_path = _make_minimal_openfoam_vtk(tmp_path / "foam.vtk")
    sf = ScalarField.read(vtk_path, "p", format="openfoam", backend="vtk_legacy_ascii")
    assert isinstance(sf, ScalarField)
    assert np.allclose(sf.value.ravel(), [100.0, 200.0])

    vf = VectorField.read(vtk_path, "U", format="foam", backend="vtk_legacy_ascii")
    assert isinstance(vf, VectorField)
    assert set(vf.keys()) >= {"x", "y", "z"}

    fd = FieldDict.read(
        vtk_path, ["U", "p"], format="openfoam", backend="vtk_legacy_ascii"
    )
    assert list(fd) == ["U", "p"]
    assert isinstance(fd["U"], VectorField)
    np.testing.assert_array_equal(fd["U"]["x"].value, vf["x"].value)
    inferred = ScalarField.read(vtk_path, format="foam", backend="vtk_legacy_ascii")
    np.testing.assert_array_equal(inferred.value, sf.value)


def test_exodus_public_field_dispatch(tmp_path: Path):
    exo = _make_minimal_exodus(tmp_path / "flow3d.exo")
    avail = list_fields(exo, format="exo.flow3d")
    assert "pressure" in avail

    sf = ScalarField.read(exo, "pressure", format="exo.flow3d")
    assert isinstance(sf, ScalarField)
    assert sf.shape == (2, 2, 2, 1)

    fd = FieldDict.read(exo, ["pressure"], format="exo")
    assert list(fd.keys()) == ["pressure"]
    names = ["fluid/velocity", "fluid/pressure"]
    fields = FieldDict.read(exo, names, format="exo.flow3d", start=250 * u.ms)
    assert list(fields) == names
    assert isinstance(fields[names[0]], VectorField)
    for field in (fields[names[0]]["x"], fields[names[1]]):
        np.testing.assert_array_equal(field._axis0_index.to_value(u.s), [0.25])
    with pytest.raises(KeyError, match="absent"):
        FieldDict.read(exo, ["absent/pressure"], format="exo.flow3d")


def test_exodus_multiblock_selector_order(tmp_path: Path):
    exo = _make_minimal_exodus(tmp_path / "multi.exo", ("block_A", "block_B"))
    names = ["block_B/pressure", "block_A/velocity", "block_A/pressure"]
    fields = FieldDict.read(exo, names, format="exo.flow3d")
    assert list(fields) == names
    assert isinstance(fields[names[1]], VectorField)
    unqualified = FieldDict.read(exo, ["velocity", "pressure"], format="exo.flow3d")
    assert list(unqualified) == [
        "block_A/velocity",
        "block_B/velocity",
        "block_A/pressure",
        "block_B/pressure",
    ]


def test_exodus_public_read_skips_unrequested_payloads(tmp_path: Path, monkeypatch):
    from gwexpy.interop._exodus import backend

    exo = _make_minimal_exodus(tmp_path / "flow3d.exo")
    read_array = backend._nc_array_float64
    loaded = []

    def track(variable, *args, **kwargs):
        if variable.name.startswith("vals_elem_var"):
            loaded.append(variable.name)
        return read_array(variable, *args, **kwargs)

    monkeypatch.setattr(backend, "_nc_array_float64", track)
    names = list_fields(exo, format="exo.flow3d", backend="netcdf4")
    assert "pressure" in names
    assert loaded == []
    fields = FieldDict.read(exo, ["pressure"], format="exo.flow3d", backend="netcdf4")
    assert list(fields) == ["pressure"]
    # FLOW-3D retains the FAVOR and volume-fraction arrays used by its masks.
    assert set(loaded) == {
        "vals_elem_var1eb1",
        "vals_elem_var5eb1",
        "vals_elem_var6eb1",
    }


def test_specfem3d_public_field_and_seismogram_dispatch(tmp_path: Path):
    db_dir = _make_minimal_specfem_partition(tmp_path / "DATABASES_MPI")
    avail = list_fields(db_dir, format="specfem3d")
    assert "rho" in avail

    sf = ScalarField.read(db_dir, "rho", format="specfem3d")
    assert isinstance(sf, ScalarField)
    assert np.allclose(sf.value.ravel(), 2600.0)

    vf = VectorField.read(db_dir, "displacement", format="specfem3d")
    assert isinstance(vf, VectorField)

    fd = FieldDict.read(db_dir, ["rho", "displacement"], format="specfem3d")
    assert list(fd) == ["rho", "displacement"]
    assert isinstance(fd["displacement"], VectorField)
    np.testing.assert_array_equal(fd["displacement"]["x"].value, vf["x"].value)

    semd_dir = tmp_path / "OUTPUT_FILES"
    semd_dir.mkdir()
    semd_file = semd_dir / "AA.S0001.BXZ.semd"
    semd_file.write_text("0.0 1.0\n0.5 2.0\n1.0 3.0\n", encoding="ascii")

    tsd = TimeSeriesDict.read(semd_dir, format="semd.specfem3d")
    assert isinstance(tsd, TimeSeriesDict)
    assert "AA.S0001.BXZ" in tsd
    assert np.allclose(tsd["AA.S0001.BXZ"].value, [1.0, 2.0, 3.0])


def test_solver_discovery_skips_array_loaders(tmp_path: Path, monkeypatch):
    from gwexpy.interop._exodus.backend import ExodusIIBackend
    from gwexpy.interop._openfoam import backend
    from gwexpy.interop._specfem3d import convert

    def fail(*args, **kwargs):
        pytest.fail("Discovery called a field-array loader")

    exo = _make_minimal_exodus(tmp_path / "flow3d.exo")
    vtk_path = _make_minimal_openfoam_vtk(tmp_path / "foam.vtk")
    db_dir = _make_minimal_specfem_partition(tmp_path / "DATABASES_MPI")
    monkeypatch.setattr(backend, "parse_legacy_ascii_vtk", fail)
    monkeypatch.setattr(convert, "_coerce_field_snapshot", fail)
    monkeypatch.setattr(ExodusIIBackend, "read_snapshot", fail)

    assert list_fields(vtk_path, format="vtk.openfoam") == ["p", "U"]
    assert list_fields(db_dir, format="specfem3d") == ["rho", "displacement"]
    assert "velocity" in list_fields(exo, format="exo.flow3d")


def test_openfoam_vtk_selection_reaches_array_parser(tmp_path: Path, monkeypatch):
    from gwexpy.interop._openfoam import backend

    vtk_path = _make_minimal_openfoam_vtk(tmp_path / "foam.vtk")
    parse = backend.parse_legacy_ascii_vtk

    def assert_selected(path, *, fields):
        assert fields == {"p", "C", "Vc", "cellID"}
        vtk = parse(path, fields=fields)
        assert set(vtk.cell_data) == {"p", "C", "cellID"}
        return vtk

    monkeypatch.setattr(backend, "parse_legacy_ascii_vtk", assert_selected)
    fields = FieldDict.read(vtk_path, ["p"], format="openfoam")
    assert list(fields) == ["p"]
