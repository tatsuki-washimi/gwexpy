"""Tests for emg3d Field interoperability.

These tests use mock emg3d objects to avoid requiring the emg3d package in the
test environment.  The staggered-grid interpolation helpers are tested with
plain NumPy arrays.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gwexpy.fields import ScalarField, VectorField
from gwexpy.interop.emg3d_ import (
    _build_cell_center_coords,
    _interp_edge_to_cell,
    _interp_face_to_cell,
    from_emg3d_field,
)

# ---------------------------------------------------------------------------
# Minimal mock objects
# ---------------------------------------------------------------------------

NCX, NCY, NCZ = 4, 5, 6  # cell counts
NNX, NNY, NNZ = NCX + 1, NCY + 1, NCZ + 1  # node counts


def _make_mesh() -> MagicMock:
    """Fake emg3d TensorMesh with cell-centre and node coordinate arrays."""
    mesh = MagicMock()
    mesh.cell_centers_x = np.linspace(0.125, 0.875, NCX)
    mesh.cell_centers_y = np.linspace(0.1, 0.9, NCY)
    mesh.cell_centers_z = np.linspace(0.083, 0.916, NCZ)
    mesh.nodes_x = np.linspace(0, 1, NNX)
    mesh.nodes_y = np.linspace(0, 1, NNY)
    mesh.nodes_z = np.linspace(0, 1, NNZ)
    mesh.nCx = NCX
    mesh.nCy = NCY
    mesh.nCz = NCZ
    return mesh


def _make_efield(real: bool = True) -> MagicMock:
    """Fake emg3d E-field (edge-centred).

    E-field component shapes:
        fx: (nCx, nNy, nNz)
        fy: (nNx, nCy, nNz)
        fz: (nNx, nNy, nCz)
    """
    mesh = _make_mesh()
    field = MagicMock()
    field.grid = mesh
    field.electric = True
    field.frequency = 1.0

    dtype = np.float64 if real else np.complex128
    rng = np.random.default_rng(42)
    field.fx = rng.random((NCX, NNY, NNZ)).astype(dtype)
    field.fy = rng.random((NNX, NCY, NNZ)).astype(dtype)
    field.fz = rng.random((NNX, NNY, NCZ)).astype(dtype)
    return field


def _make_hfield() -> MagicMock:
    """Fake emg3d H-field (face-centred).

    H-field component shapes:
        fx: (nNx, nCy, nCz)
        fy: (nCx, nNy, nCz)
        fz: (nCx, nCy, nNz)
    """
    mesh = _make_mesh()
    field = MagicMock()
    field.grid = mesh
    field.electric = False
    field.frequency = 1.0

    rng = np.random.default_rng(7)
    field.fx = rng.random((NNX, NCY, NCZ))
    field.fy = rng.random((NCX, NNY, NCZ))
    field.fz = rng.random((NCX, NCY, NNZ))
    return field


def _make_complex_efield() -> MagicMock:
    """Fake complex E-field."""
    mesh = _make_mesh()
    field = MagicMock()
    field.grid = mesh
    field.electric = True
    field.frequency = 1e6

    rng = np.random.default_rng(13)
    def _cpx(shape):
        return rng.random(shape) + 1j * rng.random(shape)
    field.fx = _cpx((NCX, NNY, NNZ))
    field.fy = _cpx((NNX, NCY, NNZ))
    field.fz = _cpx((NNX, NNY, NCZ))
    return field


# ---------------------------------------------------------------------------
# _interp_edge_to_cell / _interp_face_to_cell
# ---------------------------------------------------------------------------


class TestInterpHelpers:
    def test_edge_interp_shape(self):
        arr = np.ones((NCX, NNY, NNZ))
        out = _interp_edge_to_cell(arr, axis=1)
        assert out.shape == (NCX, NCY, NNZ)

    def test_edge_interp_values_constant(self):
        arr = np.full((3, 5, 4), 2.0)
        out = _interp_edge_to_cell(arr, axis=2)
        assert out.shape == (3, 5, 3)
        np.testing.assert_allclose(out, 2.0)

    def test_face_interp_shape(self):
        arr = np.ones((NNX, NCY, NCZ))
        out = _interp_face_to_cell(arr, axis=0)
        assert out.shape == (NCX, NCY, NCZ)

    def test_face_interp_averaging(self):
        arr = np.zeros((4, 3, 3))
        arr[0, :, :] = 2.0
        arr[1, :, :] = 4.0
        out = _interp_face_to_cell(arr, axis=0)
        assert out.shape == (3, 3, 3)
        np.testing.assert_allclose(out[0, :, :], 3.0)


# ---------------------------------------------------------------------------
# _build_cell_center_coords
# ---------------------------------------------------------------------------


class TestBuildCellCenterCoords:
    def test_shapes(self):
        mesh = _make_mesh()
        cx, cy, cz = _build_cell_center_coords(mesh)
        assert cx.shape == (NCX,)
        assert cy.shape == (NCY,)
        assert cz.shape == (NCZ,)


# ---------------------------------------------------------------------------
# from_emg3d_field — E-field
# ---------------------------------------------------------------------------


class TestFromEmg3dEfield:
    def test_returns_vectorfield(self):
        field = _make_efield()
        vf = from_emg3d_field(VectorField, field)
        assert isinstance(vf, VectorField)
        assert set(vf.keys()) == {"x", "y", "z"}

    def test_component_shapes_equal(self):
        field = _make_efield()
        vf = from_emg3d_field(VectorField, field)
        shapes = {k: v.shape for k, v in vf.items()}
        assert len(set(shapes.values())) == 1, f"Shapes differ: {shapes}"
        # expected: (1, NCX, NCY, NCZ)
        for sf in vf.values():
            assert sf.shape == (1, NCX, NCY, NCZ)

    def test_axis0_domain_frequency(self):
        field = _make_efield()
        vf = from_emg3d_field(VectorField, field)
        for sf in vf.values():
            assert sf.axis0_domain == "frequency"

    def test_unit_efield(self):
        field = _make_efield()
        vf = from_emg3d_field(VectorField, field)
        for sf in vf.values():
            assert "V" in str(sf.unit)

    def test_metadata_interpolated_from(self):
        field = _make_efield()
        vf = from_emg3d_field(VectorField, field)
        for sf in vf.values():
            assert hasattr(sf, "metadata")
            assert sf.metadata.get("interpolated_from") == "edge"


# ---------------------------------------------------------------------------
# from_emg3d_field — H-field
# ---------------------------------------------------------------------------


class TestFromEmg3dHfield:
    def test_component_shapes_equal(self):
        field = _make_hfield()
        vf = from_emg3d_field(VectorField, field)
        for sf in vf.values():
            assert sf.shape == (1, NCX, NCY, NCZ)

    def test_unit_hfield(self):
        field = _make_hfield()
        vf = from_emg3d_field(VectorField, field)
        for sf in vf.values():
            assert "A" in str(sf.unit)

    def test_metadata_face(self):
        field = _make_hfield()
        vf = from_emg3d_field(VectorField, field)
        for sf in vf.values():
            assert sf.metadata.get("interpolated_from") == "face"


# ---------------------------------------------------------------------------
# from_emg3d_field — complex field
# ---------------------------------------------------------------------------


class TestFromEmg3dComplex:
    def test_complex_dtype(self):
        field = _make_complex_efield()
        vf = from_emg3d_field(VectorField, field)
        for sf in vf.values():
            assert np.iscomplexobj(sf.value)


# ---------------------------------------------------------------------------
# from_emg3d_field — single component
# ---------------------------------------------------------------------------


class TestFromEmg3dComponent:
    def test_component_x_returns_scalarfield(self):
        field = _make_efield()
        sf = from_emg3d_field(VectorField, field, component="x")
        assert isinstance(sf, ScalarField)
        assert sf.shape == (1, NCX, NCY, NCZ)

    def test_component_z(self):
        field = _make_efield()
        sf = from_emg3d_field(ScalarField, field, component="z")
        assert isinstance(sf, ScalarField)

    def test_invalid_component_raises(self):
        field = _make_efield()
        with pytest.raises(ValueError, match="Invalid component"):
            from_emg3d_field(VectorField, field, component="w")


# ---------------------------------------------------------------------------
# from_emg3d_field — no interpolation
# ---------------------------------------------------------------------------


class TestFromEmg3dNoInterp:
    def test_raises_when_shapes_differ(self):
        field = _make_efield()  # E-field has different component shapes
        with pytest.raises(ValueError, match="Component shapes differ"):
            from_emg3d_field(VectorField, field, interpolate_to_cell_center=False)

    def test_ok_when_shapes_equal(self):
        """Artificial field where all components share the same shape."""
        mesh = _make_mesh()
        field = MagicMock()
        field.grid = mesh
        field.electric = True
        field.frequency = 1.0
        rng = np.random.default_rng(0)
        shape = (NCX, NCY, NCZ)
        field.fx = rng.random(shape)
        field.fy = rng.random(shape)
        field.fz = rng.random(shape)

        vf = from_emg3d_field(VectorField, field, interpolate_to_cell_center=False)
        assert isinstance(vf, VectorField)
        for sf in vf.values():
            assert sf.shape == (1, NCX, NCY, NCZ)


# ---------------------------------------------------------------------------
# ScalarField cls fallback
# ---------------------------------------------------------------------------


class TestFromEmg3dScalarCls:
    def test_scalar_cls_returns_x_component(self):
        field = _make_efield()
        sf = from_emg3d_field(ScalarField, field)
        assert isinstance(sf, ScalarField)
        assert not isinstance(sf, VectorField)
