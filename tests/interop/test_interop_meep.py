"""Tests for Meep HDF5 field reader.

These tests create temporary HDF5 files with the Meep naming convention
(``<field>.r`` / ``<field>.i`` pairs, or real-only ``<field>``) using h5py,
which is available in the test environment.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest

from gwexpy.fields import ScalarField, VectorField
from gwexpy.interop.meep_ import (
    _build_complex_array,
    _build_spatial_coords,
    _parse_meep_datasets,
    from_meep_hdf5,
)

# ---------------------------------------------------------------------------
# Helpers to write test HDF5 files
# ---------------------------------------------------------------------------


def _write_real_field(path: Path, name: str, data: np.ndarray) -> None:
    """Write a real-only Meep-style HDF5 dataset."""
    with h5py.File(str(path), "w") as f:
        f.create_dataset(name, data=data)


def _write_complex_field(path: Path, name: str, data: np.ndarray) -> None:
    """Write a complex Meep-style HDF5 dataset as .r/.i pair."""
    with h5py.File(str(path), "w") as f:
        f.create_dataset(f"{name}.r", data=data.real)
        f.create_dataset(f"{name}.i", data=data.imag)


def _write_vector_fields(path: Path, components: dict[str, np.ndarray]) -> None:
    """Write multiple complex field components to one HDF5 file."""
    with h5py.File(str(path), "w") as f:
        for name, data in components.items():
            if np.iscomplex(data).any():
                f.create_dataset(f"{name}.r", data=data.real.astype(np.float64))
                f.create_dataset(f"{name}.i", data=data.imag.astype(np.float64))
            else:
                f.create_dataset(name, data=data.astype(np.float64))


# ---------------------------------------------------------------------------
# _parse_meep_datasets
# ---------------------------------------------------------------------------


class TestParseMeepDatasets:
    def test_complex_pair_detected(self, tmp_path):
        p = tmp_path / "field.h5"
        data = np.ones((4, 4, 4))
        _write_complex_field(p, "ex", data)
        with h5py.File(str(p), "r") as f:
            result = _parse_meep_datasets(f)
        assert "ex" in result
        assert result["ex"]["real"] == "ex.r"
        assert result["ex"]["imag"] == "ex.i"

    def test_real_only_detected(self, tmp_path):
        p = tmp_path / "field.h5"
        data = np.ones((4, 4, 4))
        _write_real_field(p, "ez", data)
        with h5py.File(str(p), "r") as f:
            result = _parse_meep_datasets(f)
        assert "ez" in result
        assert result["ez"]["imag"] is None

    def test_custom_name(self, tmp_path):
        p = tmp_path / "custom.h5"
        data = np.zeros((4, 4))
        _write_complex_field(p, "my_field", data + 0j)
        with h5py.File(str(p), "r") as f:
            result = _parse_meep_datasets(f)
        assert "my_field" in result
        assert result["my_field"]["real"] == "my_field.r"

    def test_multiple_components(self, tmp_path):
        p = tmp_path / "multi.h5"
        data = np.ones((4, 4, 4))
        with h5py.File(str(p), "w") as f:
            for name in ("ex", "ey", "ez"):
                f.create_dataset(f"{name}.r", data=data)
                f.create_dataset(f"{name}.i", data=data * 0.5)
        with h5py.File(str(p), "r") as f:
            result = _parse_meep_datasets(f)
        assert set(result.keys()) == {"ex", "ey", "ez"}

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.h5"
        with h5py.File(str(p), "w"):
            pass
        with h5py.File(str(p), "r") as f:
            result = _parse_meep_datasets(f)
        assert result == {}


# ---------------------------------------------------------------------------
# _build_complex_array
# ---------------------------------------------------------------------------


class TestBuildComplexArray:
    def test_complex_reconstruction(self, tmp_path):
        real = np.array([[1.0, 2.0], [3.0, 4.0]])
        imag = np.array([[0.5, -0.5], [1.5, -1.5]])
        p = tmp_path / "z.h5"
        with h5py.File(str(p), "w") as f:
            f.create_dataset("z.r", data=real)
            f.create_dataset("z.i", data=imag)
        with h5py.File(str(p), "r") as f:
            result = _build_complex_array(f, "z.r", "z.i")
        np.testing.assert_allclose(result.real, real)
        np.testing.assert_allclose(result.imag, imag)

    def test_real_only(self, tmp_path):
        data = np.array([1.0, 2.0, 3.0])
        p = tmp_path / "r.h5"
        with h5py.File(str(p), "w") as f:
            f.create_dataset("field", data=data)
        with h5py.File(str(p), "r") as f:
            result = _build_complex_array(f, "field", None)
        np.testing.assert_allclose(result, data)
        assert result.dtype == np.float64

    def test_shape_mismatch_raises(self, tmp_path):
        p = tmp_path / "bad.h5"
        with h5py.File(str(p), "w") as f:
            f.create_dataset("f.r", data=np.ones((4, 4)))
            f.create_dataset("f.i", data=np.ones((4, 5)))  # wrong shape
        with h5py.File(str(p), "r") as f:
            with pytest.raises(ValueError, match="Shape mismatch"):
                _build_complex_array(f, "f.r", "f.i")


# ---------------------------------------------------------------------------
# _build_spatial_coords
# ---------------------------------------------------------------------------


class TestBuildSpatialCoords:
    def test_1d_coords(self):
        coords = _build_spatial_coords((10,), resolution=5.0, origin=(0.0,))
        assert len(coords) == 1
        np.testing.assert_allclose(coords[0], np.arange(10) * 0.2)

    def test_3d_coords_with_origin(self):
        coords = _build_spatial_coords((4, 4, 4), resolution=4.0, origin=(1.0, 2.0, 3.0))
        assert len(coords) == 3
        assert coords[0][0] == pytest.approx(1.0)
        assert coords[1][0] == pytest.approx(2.0)
        assert coords[2][0] == pytest.approx(3.0)

    def test_spacing(self):
        coords = _build_spatial_coords((5,), resolution=2.0, origin=(0.0,))
        np.testing.assert_allclose(np.diff(coords[0]), 0.5)


# ---------------------------------------------------------------------------
# from_meep_hdf5 — ScalarField
# ---------------------------------------------------------------------------


class TestFromMeepHdf5Scalar:
    def test_3d_real_field(self, tmp_path):
        p = tmp_path / "ez.h5"
        data = np.ones((4, 4, 4))
        _write_real_field(p, "ez", data)
        sf = from_meep_hdf5(ScalarField, p)
        assert isinstance(sf, ScalarField)
        assert sf.shape == (1, 4, 4, 4)

    def test_3d_complex_field(self, tmp_path):
        p = tmp_path / "ex.h5"
        data = np.ones((4, 4, 4)) + 0.5j * np.ones((4, 4, 4))
        _write_complex_field(p, "ex", data)
        sf = from_meep_hdf5(ScalarField, p)
        assert np.iscomplexobj(sf.value)

    def test_2d_field_padded_to_4d(self, tmp_path):
        p = tmp_path / "ey.h5"
        data = np.ones((8, 8))
        _write_real_field(p, "ey", data)
        sf = from_meep_hdf5(ScalarField, p)
        assert sf.ndim == 4
        assert sf.shape == (1, 1, 8, 8)

    def test_1d_field_padded_to_4d(self, tmp_path):
        p = tmp_path / "field.h5"
        data = np.ones(16)
        _write_real_field(p, "field", data)
        sf = from_meep_hdf5(ScalarField, p)
        assert sf.shape == (1, 1, 1, 16)

    def test_spatial_coords_from_resolution(self, tmp_path):
        p = tmp_path / "ex.h5"
        data = np.ones((4, 4, 4))
        _write_real_field(p, "ex", data)
        sf = from_meep_hdf5(ScalarField, p, resolution=4.0)
        # Spacing should be 1/resolution = 0.25
        ax1 = sf._axis1_index
        ax1 = ax1.value if hasattr(ax1, "value") else np.asarray(ax1)
        np.testing.assert_allclose(np.diff(ax1), 0.25)

    def test_specific_component(self, tmp_path):
        p = tmp_path / "fields.h5"
        with h5py.File(str(p), "w") as f:
            f.create_dataset("ex.r", data=np.ones((4, 4, 4)))
            f.create_dataset("ex.i", data=np.zeros((4, 4, 4)))
            f.create_dataset("ey.r", data=np.ones((4, 4, 4)) * 2)
            f.create_dataset("ey.i", data=np.zeros((4, 4, 4)))
        sf = from_meep_hdf5(ScalarField, p, component="ex")
        assert isinstance(sf, ScalarField)

    def test_no_datasets_raises(self, tmp_path):
        p = tmp_path / "empty.h5"
        with h5py.File(str(p), "w"):
            pass
        with pytest.raises(ValueError, match="No field datasets"):
            from_meep_hdf5(ScalarField, p)

    def test_missing_component_raises(self, tmp_path):
        p = tmp_path / "ex.h5"
        _write_real_field(p, "ex", np.ones((4, 4, 4)))
        with pytest.raises(ValueError):
            from_meep_hdf5(ScalarField, p, component="hz")

    def test_missing_h5py_raises(self, tmp_path):
        p = tmp_path / "ex.h5"
        _write_real_field(p, "ex", np.ones((4, 4, 4)))
        with patch(
            "gwexpy.interop.meep_.require_optional",
            side_effect=ImportError("h5py not installed"),
        ):
            with pytest.raises(ImportError, match="h5py"):
                from_meep_hdf5(ScalarField, p)


# ---------------------------------------------------------------------------
# from_meep_hdf5 — VectorField
# ---------------------------------------------------------------------------


class TestFromMeepHdf5Vector:
    def test_ex_ey_ez_to_vectorfield(self, tmp_path):
        p = tmp_path / "efield.h5"
        data = np.ones((4, 4, 4))
        _write_vector_fields(p, {"ex": data, "ey": data * 2, "ez": data * 3})
        vf = from_meep_hdf5(VectorField, p)
        assert isinstance(vf, VectorField)
        assert set(vf.keys()) == {"x", "y", "z"}

    def test_partial_components(self, tmp_path):
        """2D TE mode: only ex and ey."""
        p = tmp_path / "te.h5"
        data = np.ones((8, 8))
        _write_vector_fields(p, {"ex": data, "ey": data * 2})
        vf = from_meep_hdf5(VectorField, p)
        assert isinstance(vf, VectorField)
        assert set(vf.keys()) == {"x", "y"}

    def test_complex_vector_field(self, tmp_path):
        p = tmp_path / "complex_e.h5"
        data = np.ones((4, 4, 4)) + 1j * np.ones((4, 4, 4))
        _write_vector_fields(p, {"ex": data, "ey": data})
        vf = from_meep_hdf5(VectorField, p)
        assert isinstance(vf, VectorField)
        for comp in vf.values():
            assert np.iscomplexobj(comp.value)

    def test_component_shapes_consistent(self, tmp_path):
        p = tmp_path / "e3d.h5"
        data = np.ones((4, 4, 4))
        _write_vector_fields(p, {"ex": data, "ey": data, "ez": data})
        vf = from_meep_hdf5(VectorField, p)
        shapes = {k: v.shape for k, v in vf.items()}
        assert len(set(shapes.values())) == 1, f"Shapes differ: {shapes}"
