"""Tests for meshio field reader.

Uses synthetic meshio.Mesh objects (no dolfinx or real mesh files needed).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from gwexpy.fields import ScalarField, VectorField
from gwexpy.interop.meshio_ import (
    _build_regular_grid,
    _detect_vector_components,
    from_meshio,
)

# ---------------------------------------------------------------------------
# Helpers to create mock meshio.Mesh objects
# ---------------------------------------------------------------------------


def _make_2d_tri_mesh(
    nx: int = 10,
    ny: int = 10,
    field_name: str = "temperature",
    field_fn=None,
) -> MagicMock:
    """2D triangular mesh with scalar point data."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)])
    # z=0 everywhere → will be trimmed to 2D

    if field_fn is None:
        field_fn = lambda pts: pts[:, 0] ** 2 + pts[:, 1] ** 2  # noqa: E731
    values = field_fn(points)

    mesh = MagicMock()
    mesh.points = points
    mesh.point_data = {field_name: values}
    mesh.cell_data = {}
    mesh.cells = []
    return mesh


def _make_3d_tet_mesh(
    n: int = 5,
    field_name: str = "pressure",
) -> MagicMock:
    """3D tetrahedral mesh with scalar point data."""
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    z = np.linspace(0, 1, n)
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    values = np.sin(np.pi * points[:, 0]) * np.cos(np.pi * points[:, 1])

    mesh = MagicMock()
    mesh.points = points
    mesh.point_data = {field_name: values}
    mesh.cell_data = {}
    mesh.cells = []
    return mesh


def _make_vector_mesh(nx: int = 10, ny: int = 10) -> MagicMock:
    """2D mesh with named vector components (ex, ey, ez)."""
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(nx * ny)])

    npts = len(points)
    mesh = MagicMock()
    mesh.points = points
    mesh.point_data = {
        "ex": np.ones(npts) * 1.0,
        "ey": np.ones(npts) * 2.0,
        "ez": np.ones(npts) * 3.0,
    }
    mesh.cell_data = {}
    mesh.cells = []
    return mesh


# ---------------------------------------------------------------------------
# _detect_vector_components
# ---------------------------------------------------------------------------


class TestDetectVectorComponents:
    def test_ex_ey_ez(self):
        data = {"ex": np.zeros(10), "ey": np.zeros(10), "ez": np.zeros(10)}
        comps = _detect_vector_components(data)
        assert set(comps.keys()) == {"x", "y", "z"}

    def test_ux_uy(self):
        data = {"ux": np.ones(5), "uy": np.ones(5)}
        comps = _detect_vector_components(data)
        assert set(comps.keys()) == {"x", "y"}

    def test_no_vectors(self):
        data = {"temperature": np.zeros(10)}
        comps = _detect_vector_components(data)
        assert comps == {}


# ---------------------------------------------------------------------------
# _build_regular_grid
# ---------------------------------------------------------------------------


class TestBuildRegularGrid:
    def test_2d_shape(self):
        rng = np.random.default_rng(0)
        pts = rng.random((50, 2))
        vals = pts[:, 0] + pts[:, 1]
        data, axes = _build_regular_grid(pts, vals, resolution=0.1)
        assert data.ndim == 2
        assert len(axes) == 2

    def test_3d_shape(self):
        rng = np.random.default_rng(1)
        pts = rng.random((100, 3))
        vals = pts.sum(axis=1)
        data, axes = _build_regular_grid(pts, vals, resolution=0.2, method="nearest")
        assert data.ndim == 3
        assert len(axes) == 3


# ---------------------------------------------------------------------------
# from_meshio — 2D scalar
# ---------------------------------------------------------------------------


class TestFromMeshio2DScalar:
    def test_returns_scalarfield(self):
        mesh = _make_2d_tri_mesh()
        sf = from_meshio(ScalarField, mesh, grid_resolution=0.1)
        assert isinstance(sf, ScalarField)

    def test_4d_shape(self):
        mesh = _make_2d_tri_mesh()
        sf = from_meshio(ScalarField, mesh, grid_resolution=0.1)
        assert sf.ndim == 4
        # (1, nx, ny, 1)
        assert sf.shape[0] == 1
        assert sf.shape[3] == 1

    def test_field_name_selection(self):
        mesh = _make_2d_tri_mesh(field_name="temp")
        sf = from_meshio(ScalarField, mesh, field_name="temp", grid_resolution=0.1)
        assert isinstance(sf, ScalarField)

    def test_invalid_field_name_raises(self):
        mesh = _make_2d_tri_mesh(field_name="temp")
        with pytest.raises(ValueError, match="not found"):
            from_meshio(ScalarField, mesh, field_name="missing", grid_resolution=0.1)

    def test_interpolation_accuracy_linear(self):
        """Known parabolic function: f(x,y) = x^2 + y^2."""
        mesh = _make_2d_tri_mesh(
            nx=20, ny=20,
            field_fn=lambda p: p[:, 0] ** 2 + p[:, 1] ** 2,
        )
        sf = from_meshio(ScalarField, mesh, grid_resolution=0.05)
        # Centre of domain: (0.5, 0.5) → expected 0.5
        data = np.asarray(sf.value)
        centre_idx = tuple(s // 2 for s in data.shape)
        assert data[centre_idx] == pytest.approx(0.5, abs=0.05)


# ---------------------------------------------------------------------------
# from_meshio — 3D scalar
# ---------------------------------------------------------------------------


class TestFromMeshio3DScalar:
    def test_returns_scalarfield(self):
        mesh = _make_3d_tet_mesh(n=5)
        sf = from_meshio(ScalarField, mesh, grid_resolution=0.2, method="nearest")
        assert isinstance(sf, ScalarField)

    def test_4d_shape_3d(self):
        mesh = _make_3d_tet_mesh(n=5)
        sf = from_meshio(ScalarField, mesh, grid_resolution=0.2, method="nearest")
        assert sf.ndim == 4
        assert sf.shape[0] == 1  # singleton axis0


# ---------------------------------------------------------------------------
# from_meshio — VectorField
# ---------------------------------------------------------------------------


class TestFromMeshioVector:
    def test_vector_components_detected(self):
        mesh = _make_vector_mesh()
        vf = from_meshio(VectorField, mesh, grid_resolution=0.1)
        assert isinstance(vf, VectorField)
        assert set(vf.keys()) == {"x", "y", "z"}


# ---------------------------------------------------------------------------
# from_meshio — unit and axis0
# ---------------------------------------------------------------------------


class TestFromMeshioOptions:
    def test_custom_unit(self):
        mesh = _make_2d_tri_mesh()
        sf = from_meshio(ScalarField, mesh, grid_resolution=0.1, unit="Pa")
        assert sf.unit is not None

    def test_custom_axis0(self):
        mesh = _make_2d_tri_mesh()
        axis0 = np.array([0.0, 0.5, 1.0])
        sf = from_meshio(
            ScalarField, mesh, grid_resolution=0.1, axis0=axis0,
        )
        assert sf.shape[0] == 1  # single snapshot, axis0 stored

    def test_grid_resolution_required(self):
        """grid_resolution is mandatory — missing it should raise TypeError."""
        mesh = _make_2d_tri_mesh()
        with pytest.raises(TypeError):
            from_meshio(ScalarField, mesh)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# from_meshio — cell_data fallback
# ---------------------------------------------------------------------------


class TestFromMeshioCellData:
    def test_cell_data_only_raises(self):
        """cell_data without point_data must raise ValueError (unsupported interpolation)."""
        mesh = MagicMock()
        mesh.points = np.column_stack([
            np.linspace(0, 1, 100),
            np.linspace(0, 1, 100),
            np.zeros(100),
        ])
        mesh.point_data = {}
        # Realistic mesh: 50 cells != 100 points
        mesh.cell_data = {"sigma": [np.ones(30), np.ones(20)]}
        mesh.cells = []
        with pytest.raises(ValueError, match="cell_data"):
            from_meshio(ScalarField, mesh, grid_resolution=0.1)

    def test_no_data_raises(self):
        mesh = MagicMock()
        mesh.points = np.zeros((10, 3))
        mesh.point_data = {}
        mesh.cell_data = {}
        with pytest.raises(ValueError, match="neither point_data nor cell_data"):
            from_meshio(ScalarField, mesh, grid_resolution=0.1)
