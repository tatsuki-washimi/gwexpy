"""Tests for OpenFOAM grid inference, explicit sampling, and moving-mesh rules."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField
from gwexpy.interop._openfoam.convert import read_scalarfield_openfoam
from gwexpy.interop._openfoam.grid import (
    RegularGridSpec,
    check_moving_mesh,
    infer_exact_cartesian_grid,
    sample_to_regular_grid,
)
from gwexpy.interop._openfoam.model import FoamFieldData, FoamSnapshot


def test_infer_exact_cartesian_grid_succeeds_on_shuffled_tensor_product():
    xs = np.array([0.5, 1.5, 2.5])
    ys = np.array([10.0, 20.0])
    zs = np.array([-1.0])
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    coords = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    # Shuffle point ordering to verify permutation recovery
    perm = np.array([5, 2, 0, 4, 1, 3])
    shuffled_coords = coords[perm]
    values = shuffled_coords[:, 0] + 10.0 * shuffled_coords[:, 1]

    x_1d, y_1d, z_1d, idx_grid = infer_exact_cartesian_grid(shuffled_coords)
    assert np.allclose(x_1d, xs)
    assert np.allclose(y_1d, ys)
    assert np.allclose(z_1d, zs)
    grid_vals = values[idx_grid]
    assert grid_vals.shape == (3, 2, 1)
    assert np.allclose(grid_vals[:, :, 0], gx[:, :, 0] + 10.0 * gy[:, :, 0])


def test_infer_exact_cartesian_grid_rejects_missing_or_unstructured_cell():
    # 2x2x1 missing 1 corner -> 3 cells != 2*2*1
    coords = np.array(
        [
            [0.5, 0.5, 0.5],
            [1.5, 0.5, 0.5],
            [0.5, 1.5, 0.5],
        ],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="exact Cartesian tensor-product"):
        infer_exact_cartesian_grid(coords)


def test_sample_to_regular_grid_marks_outside_domain_nan():
    # Scattered points covering [0, 2] x [0, 2] x [0, 2]
    rng = np.random.default_rng(7)
    pts = rng.uniform(0.0, 2.0, size=(200, 3))
    vals = 2.0 * pts[:, 0] - 3.0 * pts[:, 1] + 0.5 * pts[:, 2]

    spec = RegularGridSpec(
        x=np.array([-0.5, 0.5, 1.5, 2.5]),
        y=np.array([0.5, 1.5]),
        z=np.array([0.5, 1.5]),
    )
    sampled, mask, method_used = sample_to_regular_grid(
        pts, vals, spec, method="linear"
    )
    assert method_used == "scipy_linear"
    assert sampled.shape == (4, 2, 2)
    # x = -0.5 and x = 2.5 are strictly outside [0, 2]
    assert not np.any(mask[0, :, :])
    assert not np.any(mask[3, :, :])
    assert np.all(np.isnan(sampled[0, :, :]))
    assert np.all(np.isnan(sampled[3, :, :]))
    # Interior points are finite and reasonably accurate for linear field
    assert np.all(mask[1:3, :, :])
    assert np.allclose(sampled[1, 0, 0], 2.0 * 0.5 - 3.0 * 0.5 + 0.5 * 0.5, atol=0.1)


def test_unstructured_tetra_linear_sampling_matches_analytic_field():
    """An affine field must interpolate exactly inside an irregular tetrahedron."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.2, 1.5, 0.0], [0.1, 0.2, 1.3]]
    )

    def analytic(points: np.ndarray) -> np.ndarray:
        return (
            280.0 + 2.0 * points[..., 0] - 3.0 * points[..., 1] + 4.0 * points[..., 2]
        )

    snap = FoamSnapshot(
        time_s=0.25,
        time_name="0.25",
        region="region0",
        points_m=vertices,
        cell_centers_m=vertices.mean(axis=0, keepdims=True),
        cell_volumes_m3=np.array([0.65]),
        cell_blocks=(("tetra", np.array([[0, 1, 2, 3]])),),
        cell_ids=np.array([0]),
        fields={
            "T": FoamFieldData(
                name="T",
                values=analytic(vertices),
                association="point",
                foam_class="pointScalarField",
                dimensions=(0, 0, 0, 1, 0, 0, 0),
                unit=u.K,
            )
        },
        provenance={"backend": "synthetic-tetra"},
    )
    grid = RegularGridSpec(
        x=np.array([-0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.1]),
        y=np.array([0.2, 0.4]),
        z=np.array([0.2]),
    )
    field = read_scalarfield_openfoam(
        snap, "T", grid_mode="sample", grid=grid, sample_method="linear"
    )
    gx, gy, gz = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    query = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    barycentric = np.linalg.solve(
        (vertices[1:] - vertices[0]).T, (query - vertices[0]).T
    ).T
    inside = np.all(barycentric >= -1e-12, axis=1) & (
        barycentric.sum(axis=1) <= 1 + 1e-12
    )
    expected_mask = inside.reshape(grid.shape_xyz)
    actual_mask = field._gwex_openfoam.valid_mask[0]
    assert np.array_equal(actual_mask, expected_mask)
    assert np.all(np.isnan(field.value[0][~expected_mask]))
    # (1.5, 0.4, 0.2) is inside the mesh AABB but outside the tetrahedron.
    interior_bbox_exterior_cell = np.array([1.5, 0.4, 0.2])
    assert np.all(interior_bbox_exterior_cell >= vertices.min(axis=0))
    assert np.all(interior_bbox_exterior_cell <= vertices.max(axis=0))
    assert not actual_mask[5, 1, 0]
    assert np.isnan(field.value[0, 5, 1, 0])
    assert (
        np.max(np.abs(field.value[0][expected_mask] - analytic(query[inside]))) < 1e-12
    )
    assert field.unit == u.K
    assert field._gwex_openfoam.interpolation_method == "scipy_linear"
    assert field._gwex_openfoam.association == "point"


def _two_dimensional_snapshot() -> FoamSnapshot:
    x = np.array([0.5, 1.5])
    y = np.array([10.0, 20.0])
    gx, gy = np.meshgrid(x, y, indexing="ij")
    centers = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, -3.25)])
    values = 100.0 + 2.0 * centers[:, 0] - 0.5 * centers[:, 1]
    return FoamSnapshot(
        time_s=1.25,
        time_name="1.25",
        region="region0",
        points_m=centers,
        cell_centers_m=centers,
        cell_volumes_m3=None,
        cell_blocks=(),
        cell_ids=np.arange(len(centers)),
        fields={
            "p": FoamFieldData(
                name="p",
                values=values,
                association="cell",
                foam_class="volScalarField",
                dimensions=(1, -1, -2, 0, 0, 0, 0),
                unit=u.Pa,
            )
        },
        provenance={"backend": "synthetic-2d"},
    )


def test_two_dimensional_exact_and_sampled_openfoam_fields_fft():
    snap = _two_dimensional_snapshot()
    exact = read_scalarfield_openfoam(snap, "p", grid_mode="exact")
    grid = RegularGridSpec(
        x=np.array([0.5, 1.5]),
        y=np.array([10.0, 20.0]),
        z=np.array([-3.25]),
    )
    sampled = read_scalarfield_openfoam(
        snap, "p", grid_mode="sample", grid=grid, sample_method="linear"
    )
    assert exact.shape == sampled.shape == (1, 2, 2, 1)
    assert np.array_equal(exact._axis3_index.to_value(u.m), [-3.25])
    assert np.array_equal(sampled._axis3_index.to_value(u.m), [-3.25])
    expected = np.array([[96.0, 91.0], [98.0, 93.0]])
    assert np.allclose(exact.value[0, :, :, 0], expected, rtol=0.0, atol=1e-12)
    assert np.allclose(sampled.value[0, :, :, 0], expected, rtol=0.0, atol=1e-12)
    assert np.allclose(exact.value, sampled.value, rtol=0.0, atol=1e-12)
    for field in (exact, sampled):
        spectrum = field.fft_space(axes=["x", "y"])
        assert np.all(np.isfinite(spectrum.value))
        restored = spectrum.ifft_space(axes=["kx", "ky"])
        assert np.allclose(restored.value, field.value, rtol=0.0, atol=1e-12)
        assert restored.unit == u.Pa


@pytest.mark.skipif(
    not hasattr(ScalarField, "fill_missing"),
    reason="ScalarField.fill_missing() is on the separate Field integration branch",
)
def test_sampled_openfoam_missing_values_can_be_filled_explicitly():
    snap = _two_dimensional_snapshot()
    grid = RegularGridSpec(
        x=np.array([-0.5, 0.5, 1.5, 2.5]),
        y=np.array([10.0, 20.0]),
        z=np.array([-3.25]),
    )
    sampled = read_scalarfield_openfoam(
        snap, "p", grid_mode="sample", grid=grid, sample_method="nearest"
    )
    assert np.isnan(sampled.value[:, 0]).all()
    assert np.isnan(sampled.value[:, -1]).all()
    filled = sampled.fill_missing(method="nearest")
    assert np.all(np.isfinite(filled.value))
    assert np.all(np.isnan(sampled.value[:, 0]))
    assert np.array_equal(filled.value[:, 1:3], sampled.value[:, 1:3])
    assert filled._gwex_openfoam.to_dict() == sampled._gwex_openfoam.to_dict()


def test_openfoam_provenance_survives_copy_arithmetic_and_slice():
    snap = _two_dimensional_snapshot()
    grid = RegularGridSpec(
        x=np.array([-0.5, 0.5, 1.5, 2.5]),
        y=np.array([10.0, 20.0]),
        z=np.array([-3.25]),
    )
    exact = read_scalarfield_openfoam(snap, "p", grid_mode="exact")
    sampled = read_scalarfield_openfoam(
        snap, "p", grid_mode="sample", grid=grid, sample_method="nearest"
    )
    for source in (exact, sampled):
        source_prov = source._gwex_openfoam
        for derived in (source.copy(), source * 2, source + source):
            derived_prov = derived._gwex_openfoam
            assert derived_prov.to_dict() == source_prov.to_dict()
            if source_prov.valid_mask is None:
                assert derived_prov.valid_mask is None
            else:
                assert np.array_equal(derived_prov.valid_mask, source_prov.valid_mask)
            assert derived.unit == u.Pa
        sliced = source[:, 1:3, :, :]
        sliced_prov = sliced._gwex_openfoam
        assert sliced_prov.field_name == source_prov.field_name
        assert sliced_prov.grid_mode == source_prov.grid_mode
        assert sliced_prov.time_names == source_prov.time_names
        if source_prov.valid_mask is not None:
            expected_mask = source_prov.valid_mask[:, 1:3, :, :]
            assert sliced_prov.valid_mask.shape == sliced.shape
            assert np.array_equal(sliced_prov.valid_mask, expected_mask)
            assert sliced_prov.valid_fraction == pytest.approx(np.mean(expected_mask))
    assert np.isnan((sampled * 2).value[:, 0]).all()
    assert np.isnan((sampled + sampled).value[:, -1]).all()

    later = replace(snap, time_s=2.25, time_name="2.25")
    two_times = read_scalarfield_openfoam(
        [snap, later], "p", grid_mode="sample", grid=grid, sample_method="nearest"
    )
    selected = two_times[1, 1:3, :, :]
    selected_prov = selected._gwex_openfoam
    assert selected_prov.time_names == ("2.25",)
    assert selected_prov.valid_mask.shape == selected.shape
    assert np.array_equal(
        selected_prov.valid_mask, two_times._gwex_openfoam.valid_mask[1:2, 1:3]
    )


def test_check_moving_mesh_raises_on_motion_unless_resample():
    def make_snap(t: float, dx: float) -> FoamSnapshot:
        pts = np.array([[0 + dx, 0, 0], [1 + dx, 0, 0], [1 + dx, 1, 0], [0 + dx, 1, 0]])
        centers = np.array([[0.5 + dx, 0.5, 0.5]])
        fld = FoamFieldData(
            name="p",
            values=np.array([1.0]),
            association="cell",
            foam_class="volScalarField",
            dimensions=(1, -1, -2, 0, 0, 0, 0),
            unit=u.Pa,
        )
        return FoamSnapshot(
            time_s=t,
            time_name=str(t),
            region="region0",
            points_m=pts,
            cell_centers_m=centers,
            cell_volumes_m3=None,
            cell_blocks=(),
            cell_ids=None,
            fields={"p": fld},
            provenance={"backend": "test"},
        )

    s0 = make_snap(0.0, 0.0)
    s1_static = make_snap(0.5, 0.0)
    s1_moved = make_snap(0.5, 0.25)

    assert not check_moving_mesh([s0, s1_static], moving_mesh="error")
    with pytest.raises(ValueError, match="Moving or deforming mesh"):
        check_moving_mesh([s0, s1_moved], moving_mesh="error")
    assert check_moving_mesh([s0, s1_moved], moving_mesh="resample")
