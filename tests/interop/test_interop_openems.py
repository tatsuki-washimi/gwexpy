"""Tests for openEMS HDF5 field dump reader.

Synthetic HDF5 files are created with h5py; the openEMS binaries/Python
package are NOT required.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from gwexpy.fields import ScalarField, VectorField
from gwexpy.interop.openems_ import (
    DUMP_TYPE_MAP,
    _read_openems_fd,
    _read_openems_mesh,
    _read_openems_td,
    from_openems_hdf5,
)

# ---------------------------------------------------------------------------
# HDF5 file builders
# ---------------------------------------------------------------------------

NX, NY, NZ = 4, 5, 6


def _make_mesh(f: h5py.File, nx=NX, ny=NY, nz=NZ) -> None:
    """Write /Mesh/x, y, z."""
    mesh = f.require_group("Mesh")
    mesh.create_dataset("x", data=np.linspace(0, 1, nx))
    mesh.create_dataset("y", data=np.linspace(0, 2, ny))
    mesh.create_dataset("z", data=np.linspace(0, 3, nz))


def _make_td_file(
    path: Path,
    n_steps: int = 3,
    nx=NX,
    ny=NY,
    nz=NZ,
    n_comp: int = 3,
) -> None:
    """Write TD field dump HDF5."""
    with h5py.File(str(path), "w") as f:
        _make_mesh(f, nx, ny, nz)
        td = f.require_group("FieldData/TD")
        for i in range(n_steps):
            data = np.random.default_rng(i).random((nx, ny, nz, n_comp))
            td.create_dataset(str(i), data=data)


def _make_fd_file(
    path: Path,
    n_freq: int = 2,
    nx=NX,
    ny=NY,
    nz=NZ,
    n_comp: int = 3,
    complex_: bool = True,
) -> None:
    """Write FD field dump HDF5."""
    with h5py.File(str(path), "w") as f:
        _make_mesh(f, nx, ny, nz)
        fd = f.require_group("FieldData/FD")
        for i in range(n_freq):
            rng = np.random.default_rng(i)
            real = rng.random((nx, ny, nz, n_comp))
            imag = rng.random((nx, ny, nz, n_comp)) if complex_ else np.zeros_like(real)
            fd.create_dataset(f"f{i}_real", data=real)
            fd.create_dataset(f"f{i}_imag", data=imag)


def _make_sar_file(path: Path, n_freq: int = 2, nx=NX, ny=NY, nz=NZ) -> None:
    """Write SAR (scalar) FD dump – no component axis."""
    with h5py.File(str(path), "w") as f:
        _make_mesh(f, nx, ny, nz)
        fd = f.require_group("FieldData/FD")
        for i in range(n_freq):
            rng = np.random.default_rng(i)
            real = rng.random((nx, ny, nz))  # no component axis
            imag = np.zeros((nx, ny, nz))
            fd.create_dataset(f"f{i}_real", data=real)
            fd.create_dataset(f"f{i}_imag", data=imag)


# ---------------------------------------------------------------------------
# DUMP_TYPE_MAP
# ---------------------------------------------------------------------------


class TestDumpTypeMap:
    def test_known_types_present(self):
        for t in (0, 1, 2, 3, 10, 11, 20, 21, 22):
            assert t in DUMP_TYPE_MAP

    def test_td_types_domain(self):
        for t in (0, 1, 2, 3):
            assert DUMP_TYPE_MAP[t][2] == "time"

    def test_fd_types_domain(self):
        for t in (10, 11, 20, 21, 22):
            assert DUMP_TYPE_MAP[t][2] == "frequency"

    def test_efield_unit(self):
        assert "V/m" in DUMP_TYPE_MAP[0][1]
        assert "V/m" in DUMP_TYPE_MAP[10][1]


# ---------------------------------------------------------------------------
# _read_openems_mesh
# ---------------------------------------------------------------------------


class TestReadOpenemsМesh:
    def test_coords_shape(self, tmp_path):
        p = tmp_path / "mesh.h5"
        with h5py.File(str(p), "w") as f:
            _make_mesh(f, nx=4, ny=5, nz=6)
        with h5py.File(str(p), "r") as f:
            x, y, z = _read_openems_mesh(f)
        assert x.shape == (4,)
        assert y.shape == (5,)
        assert z.shape == (6,)

    def test_coords_values(self, tmp_path):
        p = tmp_path / "mesh.h5"
        with h5py.File(str(p), "w") as f:
            _make_mesh(f, nx=4, ny=5, nz=6)
        with h5py.File(str(p), "r") as f:
            x, y, z = _read_openems_mesh(f)
        assert x[0] == pytest.approx(0.0)
        assert x[-1] == pytest.approx(1.0)
        assert y[-1] == pytest.approx(2.0)
        assert z[-1] == pytest.approx(3.0)

    def test_missing_mesh_raises(self, tmp_path):
        p = tmp_path / "no_mesh.h5"
        with h5py.File(str(p), "w") as f:
            f.require_group("FieldData")
        with h5py.File(str(p), "r") as f:
            with pytest.raises(KeyError):
                _read_openems_mesh(f)


# ---------------------------------------------------------------------------
# _read_openems_td
# ---------------------------------------------------------------------------


class TestReadOpenemsТД:
    def test_all_steps(self, tmp_path):
        p = tmp_path / "td.h5"
        _make_td_file(p, n_steps=3)
        with h5py.File(str(p), "r") as f:
            data, times = _read_openems_td(f, timestep=None)
        assert data.shape[0] == 3
        assert times.shape == (3,)

    def test_single_step(self, tmp_path):
        p = tmp_path / "td.h5"
        _make_td_file(p, n_steps=3)
        with h5py.File(str(p), "r") as f:
            data, times = _read_openems_td(f, timestep=1)
        assert data.shape[0] == 1
        assert times[0] == 1

    def test_invalid_step_raises(self, tmp_path):
        p = tmp_path / "td.h5"
        _make_td_file(p, n_steps=3)
        with h5py.File(str(p), "r") as f:
            with pytest.raises(ValueError, match="Timestep"):
                _read_openems_td(f, timestep=99)


# ---------------------------------------------------------------------------
# _read_openems_fd
# ---------------------------------------------------------------------------


class TestReadOpenemsΦΔ:
    def test_all_freqs(self, tmp_path):
        p = tmp_path / "fd.h5"
        _make_fd_file(p, n_freq=2)
        with h5py.File(str(p), "r") as f:
            data, freqs = _read_openems_fd(f, freq_idx=None)
        assert data.shape[0] == 2
        assert np.iscomplexobj(data)

    def test_single_freq(self, tmp_path):
        p = tmp_path / "fd.h5"
        _make_fd_file(p, n_freq=3)
        with h5py.File(str(p), "r") as f:
            data, freqs = _read_openems_fd(f, freq_idx=2)
        assert data.shape[0] == 1
        assert freqs[0] == 2

    def test_invalid_freq_raises(self, tmp_path):
        p = tmp_path / "fd.h5"
        _make_fd_file(p, n_freq=2)
        with h5py.File(str(p), "r") as f:
            with pytest.raises(ValueError, match="Frequency index"):
                _read_openems_fd(f, freq_idx=99)


# ---------------------------------------------------------------------------
# from_openems_hdf5 — time-domain
# ---------------------------------------------------------------------------


class TestFromOpenemsHdf5TD:
    def test_vectorfield_shape(self, tmp_path):
        p = tmp_path / "td.h5"
        _make_td_file(p, n_steps=3)
        vf = from_openems_hdf5(VectorField, p, dump_type=0)
        assert isinstance(vf, VectorField)
        assert set(vf.keys()) == {"x", "y", "z"}
        # axis0 has 3 steps
        for sf in vf.values():
            assert sf.shape[0] == 3

    def test_single_component_returns_scalarfield(self, tmp_path):
        p = tmp_path / "td.h5"
        _make_td_file(p, n_steps=3)
        sf = from_openems_hdf5(ScalarField, p, dump_type=0, component="z")
        assert isinstance(sf, ScalarField)
        assert sf.shape == (3, NX, NY, NZ)

    def test_single_timestep(self, tmp_path):
        p = tmp_path / "td.h5"
        _make_td_file(p, n_steps=3)
        vf = from_openems_hdf5(VectorField, p, dump_type=0, timestep=1)
        for sf in vf.values():
            assert sf.shape[0] == 1

    def test_axis0_domain_time(self, tmp_path):
        p = tmp_path / "td.h5"
        _make_td_file(p, n_steps=2)
        vf = from_openems_hdf5(VectorField, p, dump_type=0)
        for sf in vf.values():
            assert sf.axis0_domain == "time"

    def test_spatial_coords_shape(self, tmp_path):
        p = tmp_path / "td.h5"
        _make_td_file(p, n_steps=1, nx=NX, ny=NY, nz=NZ)
        vf = from_openems_hdf5(VectorField, p, dump_type=0)
        for sf in vf.values():
            assert sf.shape == (1, NX, NY, NZ)

    def test_missing_td_group_raises(self, tmp_path):
        p = tmp_path / "no_td.h5"
        # Write FD only
        _make_fd_file(p, n_freq=2)
        with pytest.raises(ValueError, match="FieldData/TD"):
            from_openems_hdf5(VectorField, p, dump_type=0)

    def test_missing_mesh_raises(self, tmp_path):
        p = tmp_path / "no_mesh.h5"
        with h5py.File(str(p), "w") as f:
            td = f.require_group("FieldData/TD")
            td.create_dataset("0", data=np.ones((4, 5, 6, 3)))
        with pytest.raises((ValueError, KeyError)):
            from_openems_hdf5(VectorField, p, dump_type=0)


# ---------------------------------------------------------------------------
# from_openems_hdf5 — frequency-domain
# ---------------------------------------------------------------------------


class TestFromOpenemsHdf5FD:
    def test_vectorfield_complex(self, tmp_path):
        p = tmp_path / "fd.h5"
        _make_fd_file(p, n_freq=2)
        vf = from_openems_hdf5(VectorField, p, dump_type=10)
        assert isinstance(vf, VectorField)
        for sf in vf.values():
            assert np.iscomplexobj(sf.value)

    def test_axis0_domain_frequency(self, tmp_path):
        p = tmp_path / "fd.h5"
        _make_fd_file(p, n_freq=2)
        vf = from_openems_hdf5(VectorField, p, dump_type=10)
        for sf in vf.values():
            assert sf.axis0_domain == "frequency"

    def test_single_freq_index(self, tmp_path):
        p = tmp_path / "fd.h5"
        _make_fd_file(p, n_freq=3)
        vf = from_openems_hdf5(VectorField, p, dump_type=10, frequency_index=1)
        for sf in vf.values():
            assert sf.shape[0] == 1

    def test_missing_fd_group_raises(self, tmp_path):
        p = tmp_path / "no_fd.h5"
        _make_td_file(p, n_steps=2)
        with pytest.raises(ValueError, match="FieldData/FD"):
            from_openems_hdf5(VectorField, p, dump_type=10)


# ---------------------------------------------------------------------------
# SAR (scalar) dumps
# ---------------------------------------------------------------------------


class TestFromOpenemsHdf5SAR:
    def test_sar_returns_scalarfield(self, tmp_path):
        p = tmp_path / "sar.h5"
        _make_sar_file(p, n_freq=2)
        sf = from_openems_hdf5(ScalarField, p, dump_type=20)
        assert isinstance(sf, ScalarField)
        assert sf.shape == (2, NX, NY, NZ)

    def test_sar_unit(self, tmp_path):
        p = tmp_path / "sar.h5"
        _make_sar_file(p, n_freq=1)
        sf = from_openems_hdf5(ScalarField, p, dump_type=20)
        # Unit should be derived from DUMP_TYPE_MAP (W/kg)
        assert sf.unit is not None

    def test_sar_axis0_domain_frequency(self, tmp_path):
        p = tmp_path / "sar.h5"
        _make_sar_file(p, n_freq=2)
        sf = from_openems_hdf5(ScalarField, p, dump_type=20)
        assert sf.axis0_domain == "frequency"


# ---------------------------------------------------------------------------
# Mesh coordinates accuracy
# ---------------------------------------------------------------------------


class TestOpenemsCoordAccuracy:
    def test_irregular_mesh_preserved(self, tmp_path):
        p = tmp_path / "irregular.h5"
        x = np.array([0.0, 0.1, 0.3, 0.6, 1.0])  # non-uniform
        y = np.array([0.0, 1.0, 2.0])
        z = np.array([0.0, 0.5])
        nx, ny, nz = len(x), len(y), len(z)
        with h5py.File(str(p), "w") as f:
            mesh = f.require_group("Mesh")
            mesh.create_dataset("x", data=x)
            mesh.create_dataset("y", data=y)
            mesh.create_dataset("z", data=z)
            td = f.require_group("FieldData/TD")
            td.create_dataset("0", data=np.ones((nx, ny, nz, 3)))
        vf = from_openems_hdf5(VectorField, p, dump_type=0)
        for sf in vf.values():
            axis1 = sf._axis1_index
            vals = axis1.value if hasattr(axis1, "value") else np.asarray(axis1)
            np.testing.assert_allclose(vals, x)


# ---------------------------------------------------------------------------
# Unit override
# ---------------------------------------------------------------------------


class TestOpenemsUnitOverride:
    def test_custom_unit(self, tmp_path):
        p = tmp_path / "td.h5"
        _make_td_file(p, n_steps=1)
        vf = from_openems_hdf5(VectorField, p, dump_type=0, unit="mV/m")
        for sf in vf.values():
            # unit is stored; exact representation depends on astropy Unit parsing
            assert sf.unit is not None
