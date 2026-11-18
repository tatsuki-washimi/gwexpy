"""Tests for provisional surface-defined physical fields (#902)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

import gwexpy
import gwexpy.fields as fields_pkg
from gwexpy.fields._surface import (
    OrientationReport,
    SurfaceField,
    SurfaceGeometry,
    SurfaceProvenance,
    check_surface_orientation,
    reorient_faces,
)
from gwexpy.fields.scalar import ScalarField
from gwexpy.fields.vector import VectorField


def _octahedron_geometry(*, unit: u.UnitBase = u.m) -> SurfaceGeometry:
    """Closed unit-radius octahedron with outward-oriented triangles."""
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    # Outward-oriented faces (positive signed volume = 4/3)
    tris = np.array(
        [
            [0, 2, 4],
            [2, 1, 4],
            [1, 3, 4],
            [3, 0, 4],
            [2, 0, 5],
            [1, 2, 5],
            [3, 1, 5],
            [0, 3, 5],
        ],
        dtype=np.int64,
    )
    return SurfaceGeometry.from_triangles(vertices, tris, unit=unit)


def test_provisional_module_not_reexported_publicly() -> None:
    assert not hasattr(gwexpy, "SurfaceField")
    assert not hasattr(fields_pkg, "SurfaceField")
    assert "SurfaceField" not in getattr(fields_pkg, "__all__", [])


def test_structured_and_unstructured_surface_scalar_and_vector_associations() -> None:
    # 1a. Structured planar surface with scalar height/temperature at vertices
    x = np.linspace(0.0, 2.0, 3) * u.m
    y = np.linspace(0.0, 1.0, 2) * u.m
    geom_struct = SurfaceGeometry.from_structured_grid(x, y, z=0.5 * u.m)
    assert geom_struct.is_planar_rectilinear
    assert geom_struct.topological_dimension == 2
    assert geom_struct.embedding_dimension == 3
    assert geom_struct.n_vertices == 6
    assert geom_struct.n_faces == 2

    vals_v = np.arange(12, dtype=np.float64).reshape(2, 6) * u.K
    sf_v = SurfaceField(
        vals_v,
        geometry=geom_struct,
        association="vertex",
        axis0=[0.0, 1.0] * u.s,
        name="T",
    )
    assert sf_v.shape == (2, 6)
    assert sf_v.tensor_rank == 0
    assert sf_v.unit == u.K
    assert sf_v.entity_coordinates.shape == (6, 3)

    # 1b. Unstructured triangle surface with 3-component displacement vector at faces
    geom_tri = _octahedron_geometry()
    disp_f = np.ones((2, geom_tri.n_faces, 3), dtype=np.float64) * 1e-3 * u.m
    vf_f = SurfaceField(
        disp_f,
        geometry=geom_tri,
        association="face",
        component_basis="cartesian",
        axis0=[0.0, 0.5] * u.s,
        name="displacement",
    )
    assert vf_f.shape == (2, 8, 3)
    assert vf_f.tensor_rank == 1
    assert vf_f.component_names == ("x", "y", "z")
    assert vf_f.unit == u.m
    assert vf_f.entity_coordinates.shape == (8, 3)


def test_selection_slicing_and_copy_preserve_correspondence_and_metadata() -> None:
    geom = _octahedron_geometry()
    face_centers = geom.face_centroids().value
    # Assign unique vector equal to face centroid at t=0 and 2*centroid at t=1
    data = np.stack([face_centers, 2.0 * face_centers], axis=0)
    mask = np.ones((2, geom.n_faces), dtype=bool)
    mask[0, 3] = False
    prov = SurfaceProvenance(source="synthetic", operation="init", backend="numpy")

    field = SurfaceField(
        data,
        geometry=geom,
        association="face",
        unit=u.m / u.s,
        axis0=[10.0, 11.0] * u.s,
        epoch=1400000000.0,
        component_basis="cartesian",
        valid_mask=mask,
        provenance=prov,
        name="vel",
    )

    sub = field.isel(axis0=[1], entity=[5, 2, 0])
    assert sub.shape == (1, 3, 3)
    assert sub.unit == u.m / u.s
    assert sub.time_reference == "absolute"
    assert float(getattr(sub.epoch, "gps", sub.epoch)) == pytest.approx(1400000000.0)
    np.testing.assert_allclose(sub.axis0.to_value(u.s), [11.0])
    np.testing.assert_allclose(
        sub.entity_coordinates.to_value(u.m), face_centers[[5, 2, 0]]
    )
    np.testing.assert_allclose(sub.value[0], 2.0 * face_centers[[5, 2, 0]])
    assert sub.geometry.n_faces == 3

    copied = field.copy()
    assert copied.geometry is not field.geometry
    np.testing.assert_array_equal(copied.value, field.value)
    np.testing.assert_array_equal(copied.valid_mask, field.valid_mask)
    assert copied.provenance.source == "synthetic"

    # Relative time reference and with_epoch (#725 Plan I)
    rel = SurfaceField(
        data,
        geometry=geom,
        association="face",
        unit=u.m,
        axis0=[0.0, 0.5] * u.s,
        epoch=None,
    )
    assert rel.time_reference == "relative"
    anchored = rel.with_epoch(1400000000.0)
    assert anchored.time_reference == "absolute"
    np.testing.assert_allclose(
        anchored.axis0.to_value(u.s), [1400000000.0, 1400000000.5]
    )


def test_orientation_normals_areas_and_gauss_divergence_flux() -> None:
    geom = _octahedron_geometry()
    report = check_surface_orientation(geom.vertices.value, geom.cell_blocks)
    assert isinstance(report, OrientationReport)
    assert report.is_consistent
    assert report.is_closed
    assert report.is_outward is True
    assert geom.signed_volume().to_value(u.m**3) == pytest.approx(4.0 / 3.0)
    assert geom.total_area().to_value(u.m**2) == pytest.approx(4.0 * np.sqrt(3.0))

    # Flip one triangle to create inconsistent orientation, then repair via BFS
    bad_tris = geom.triangles.copy()
    bad_tris[0] = bad_tris[0, [0, 2, 1]]
    bad_report = check_surface_orientation(
        geom.vertices.value, [("triangle", bad_tris)]
    )
    assert not bad_report.is_consistent
    repaired_blocks = reorient_faces([("triangle", bad_tris)])
    repaired_geom = SurfaceGeometry(
        geom.vertices, cell_blocks=repaired_blocks
    ).orient_outward()
    assert repaired_geom.check_orientation().is_consistent
    assert repaired_geom.check_orientation().is_outward is True

    # Gauss's divergence theorem on closed octahedron:
    # For F(r) = r = (x, y, z), div(F) = 3, so flux int_S (F . n) dS = 3 * V = 4.0 m^3.
    # For polyhedral faces, evaluating F at vertices and averaging per face matches
    # the exact linear integral over each triangle!
    verts = geom.vertices.value
    sf_vec = SurfaceField(
        verts[None, :, :],
        geometry=geom,
        association="vertex",
        unit=u.m,
        axis0=[0.0] * u.s,
        component_basis="cartesian",
    )
    flux = sf_vec.surface_integral(oriented=True)
    assert flux.unit == u.m**3
    assert flux.to_value(u.m**3)[0] == pytest.approx(4.0, rel=1e-12, abs=1e-12)

    # Flipping geometry reverses oriented flux sign
    flux_flipped = sf_vec.flip_orientation().surface_integral(oriented=True)
    assert flux_flipped.to_value(u.m**3)[0] == pytest.approx(-4.0, rel=1e-12, abs=1e-12)

    # Normal component on flat xy plane with z-normal
    plane = SurfaceGeometry.from_structured_grid(
        [0.0, 1.0] * u.m, [0.0, 1.0] * u.m, z=0.0 * u.m
    )
    vec_plane = SurfaceField(
        np.broadcast_to([1.0, 2.0, -5.0], (1, plane.n_faces, 3)).copy(),
        geometry=plane,
        association="face",
        unit=u.m / u.s,
        axis0=[0.0] * u.s,
        component_basis="cartesian",
    )
    nc = vec_plane.normal_component()
    np.testing.assert_allclose(nc.value[0], -5.0)


def test_rectilinear_vs_explicit_sampling_separation() -> None:
    # 1. Exact planar rectilinear -> lossless 1-to-1 conversion succeeds
    x = np.array([0.0, 1.0, 2.0]) * u.m
    y = np.array([0.0, 2.0]) * u.m
    planar = SurfaceGeometry.from_structured_grid(x, y, z=3.0 * u.m)
    X, Y = np.meshgrid(x.value, y.value, indexing="ij")
    vals = (2.0 * X + 5.0 * Y).reshape(1, -1)
    sf_planar = SurfaceField(
        vals,
        geometry=planar,
        association="vertex",
        unit=u.Pa,
        axis0=[0.0] * u.s,
    )
    rect = sf_planar.to_rectilinear()
    assert isinstance(rect, ScalarField)
    assert rect.shape == (1, 3, 2, 1)
    np.testing.assert_allclose(rect.value[0, :, :, 0], 2.0 * X + 5.0 * Y)
    assert rect._gwex_provenance["mapping_kind"] == "exact_one_to_one"

    # 2. Curved surface z = x^2 -> to_rectilinear MUST fail closed (no silent flattening)
    curved = SurfaceGeometry.from_structured_grid(x, y, z=(X**2) * u.m)
    sf_curved = SurfaceField(
        vals,
        geometry=curved,
        association="vertex",
        unit=u.Pa,
        axis0=[0.0] * u.s,
    )
    with pytest.raises(ValueError, match="sample_to_grid"):
        sf_curved.to_rectilinear()

    # Explicit sample_to_grid succeeds and records interpolated provenance + mask
    sampled = sf_curved.sample_to_grid(
        np.array([0.5, 1.5, 5.0]) * u.m,
        np.array([1.0]) * u.m,
        method="linear",
    )
    assert isinstance(sampled, ScalarField)
    assert sampled._gwex_provenance["mapping_kind"] == "interpolated"
    assert sampled._gwex_provenance["interpolation_method"] == "linear"
    # Points inside domain (0.5, 1.0) and (1.5, 1.0) are valid; (5.0, 1.0) is outside -> NaN
    assert sampled._gwex_valid_mask[0, 0, 0, 0]
    assert sampled._gwex_valid_mask[0, 1, 0, 0]
    assert not sampled._gwex_valid_mask[0, 2, 0, 0]
    assert sampled.value[0, 0, 0, 0] == pytest.approx(2.0 * 0.5 + 5.0 * 1.0)
    assert sampled.value[0, 1, 0, 0] == pytest.approx(2.0 * 1.5 + 5.0 * 1.0)
    assert np.isnan(sampled.value[0, 2, 0, 0])


def test_optional_meshio_and_pyvista_delegation() -> None:
    geom = _octahedron_geometry()
    sf = SurfaceField(
        np.arange(geom.n_vertices, dtype=np.float64)[None, :],
        geometry=geom,
        association="vertex",
        unit=u.K,
        axis0=[0.0] * u.s,
        name="temp",
    )

    try:
        import meshio  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="meshio"):
            sf.to_meshio("temp")
    else:
        mesh = sf.to_meshio("temp")
        rt = SurfaceField.from_meshio(mesh, "temp", unit=u.K, axis0=[0.0] * u.s)
        np.testing.assert_allclose(rt.value, sf.value)
        assert rt.provenance.backend == "meshio"

    try:
        import pyvista  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="pyvista"):
            sf.to_pyvista("temp")
    else:
        pd = sf.to_pyvista("temp")
        rt_pv = SurfaceField.from_pyvista(pd, "temp", unit=u.K, axis0=[0.0] * u.s)
        np.testing.assert_allclose(rt_pv.value, sf.value)
        assert rt_pv.provenance.backend == "pyvista"
        flux_pv = SurfaceField(
            geom.vertices.value[None, :, :],
            geometry=geom,
            association="vertex",
            unit=u.m,
            axis0=[0.0] * u.s,
            component_basis="cartesian",
        ).surface_integral(oriented=True, backend="pyvista")
        assert flux_pv.to_value(u.m**3)[0] == pytest.approx(4.0, rel=1e-12)


def test_quadrature_tensor_and_3d_surface_sampling() -> None:
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) * u.m
    weights = np.array([0.2, 0.3, 0.5]) * u.m**2
    geom_q = SurfaceGeometry.from_points(pts, weights=weights, as_quadrature=True)
    sf_q = SurfaceField(
        np.array([[2.0, 4.0, 6.0]]),
        geometry=geom_q,
        association="quadrature",
        unit=u.Pa,
        axis0=[0.0] * u.s,
    )
    assert sf_q.surface_integral().to_value(u.Pa * u.m**2)[0] == pytest.approx(4.6)

    geom_tri = _octahedron_geometry()
    stress = np.zeros((1, geom_tri.n_faces, 3, 3), dtype=np.float64)
    stress[:, :, 0, 0] = 10.0
    tf = SurfaceField(
        stress,
        geometry=geom_tri,
        association="face",
        unit=u.Pa,
        axis0=[0.0] * u.s,
        field_kind="tensor",
        component_basis="cartesian",
    )
    assert tf.tensor_rank == 2
    assert tf.shape == (1, 8, 3, 3)

    vals = np.ones((1, geom_tri.n_vertices), dtype=np.float64)
    sf_v = SurfaceField(
        vals,
        geometry=geom_tri,
        association="vertex",
        unit=u.K,
        axis0=[0.0] * u.s,
    )
    sampled_3d = sf_v.sample_to_grid(
        np.array([0.5, 5.0]) * u.m,
        np.array([0.5]) * u.m,
        np.array([0.0]) * u.m,
        method="barycentric",
        max_distance=0.5 * u.m,
    )
    assert isinstance(sampled_3d, ScalarField)
    assert sampled_3d._gwex_valid_mask[0, 0, 0, 0]
    assert not sampled_3d._gwex_valid_mask[0, 1, 0, 0]
    assert sampled_3d.value[0, 0, 0, 0] == pytest.approx(1.0)
    assert np.isnan(sampled_3d.value[0, 1, 0, 0])
