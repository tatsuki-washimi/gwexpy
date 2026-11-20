"""Tests for #556 local coordinate frames, detector frames, and spatial rotations."""

from __future__ import annotations

import math

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import (
    DETECTOR_ARM_AZIMUTHS_DEG,
    DetectorFrame,
    GeoMetadata,
    GridGeometry,
    ScalarField,
    TensorField,
    VectorField,
)
from gwexpy.plot.geomap import DETECTORS


def _make_metric_scalar_field() -> ScalarField:
    t = np.array([0.0, 0.5]) * u.s
    x = np.linspace(-100.0, 100.0, 21) * u.m
    y = np.linspace(-100.0, 100.0, 21) * u.m
    z = np.array([0.0]) * u.m
    xx, yy = np.meshgrid(x.value, y.value, indexing="ij")
    # Linear planar gradient: phi(x, y) = 2*x + 3*y + 10
    plane_2d = 2.0 * xx + 3.0 * yy + 10.0
    data = np.broadcast_to(plane_2d[None, :, :, None], (2, 21, 21, 1)).copy()
    sf = ScalarField(
        data,
        unit=u.m,
        axis0=t,
        axis1=x,
        axis2=y,
        axis3=z,
        axis_names=["t", "x", "y", "z"],
        space_domain="real",
    )
    sf.geo = GeoMetadata(
        vertical_datum="T.P. (JGD2011 vertical)",
        active=False,
        local_origin=(DETECTORS["K1"]["lon"], DETECTORS["K1"]["lat"]),
    )
    return sf


def test_detector_frame_reuses_detector_registry_and_distinguishes_orthogonalize():
    for code in ("K1", "H1", "L1", "V1", "G1"):
        # 1. Local site ENU reference frame (built solely from geomap.DETECTORS lon/lat)
        df_enu = DetectorFrame.from_detector(code, frame_kind="local_enu")
        assert df_enu.detector == code
        assert df_enu.frame_kind == "local_enu"
        assert df_enu.arm_source == "geomap.DETECTORS"
        assert df_enu.local_origin_lonlat == (
            DETECTORS[code]["lon"],
            DETECTORS[code]["lat"],
        )
        assert df_enu.is_orthonormal
        assert df_enu.raw_arms is None
        np.testing.assert_array_equal(df_enu.basis_matrix, np.eye(3))

        # 2. Arm-aligned detector frame (nominal horizontal arm azimuths)
        df = DetectorFrame.from_detector(code)
        assert df.detector == code
        assert df.frame_kind == "arms"
        assert df.arm_source == "DETECTOR_ARM_AZIMUTHS_DEG"
        assert df.local_origin_lonlat == (
            DETECTORS[code]["lon"],
            DETECTORS[code]["lat"],
        )
        assert df.is_orthonormal
        mat = df.basis_matrix
        np.testing.assert_allclose(mat @ mat.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(mat) == pytest.approx(1.0)
        assert df.raw_arms is not None

    # 3. Arm-aligned detector frame from gwexpy.utils.lal.LAL_DETECTORS (3D arm azimuth + altitude)
    pytest.importorskip("lal")
    k1_lal = DetectorFrame.from_detector("K1", arm_source="lal", orthogonalize=False)
    assert k1_lal.frame_kind == "arms"
    assert k1_lal.arm_source == "gwexpy.utils.lal.LAL_DETECTORS"
    assert k1_lal.local_origin_lonlat == (
        DETECTORS["K1"]["lon"],
        DETECTORS["K1"]["lat"],
    )
    # KAGRA has non-zero arm tilt (xArmAltitudeRadians ~ +0.00314 rad, yArmAltitudeRadians ~ -0.00363 rad)
    assert k1_lal.raw_arms is not None
    assert abs(k1_lal.raw_arms[0][2]) > 1e-3
    assert abs(k1_lal.raw_arms[1][2]) > 1e-3
    k1_lal_ortho = DetectorFrame.from_detector(
        "K1", arm_source="lal", orthogonalize=True
    )
    assert k1_lal_ortho.is_orthonormal
    np.testing.assert_allclose(
        k1_lal_ortho.basis_matrix @ k1_lal_ortho.basis_matrix.T,
        np.eye(3),
        atol=1e-12,
    )

    # GEO600 (G1) has non-orthogonal arms (~94.33 deg opening angle)
    g1_raw = DetectorFrame.from_detector("G1", orthogonalize=False)
    assert not g1_raw.is_orthonormal
    u_x = np.asarray(g1_raw.basis[0])
    u_y = np.asarray(g1_raw.basis[1])
    dot_xy = float(np.dot(u_x, u_y))
    expected_angle_rad = math.radians(
        abs(DETECTOR_ARM_AZIMUTHS_DEG["G1"][0] - DETECTOR_ARM_AZIMUTHS_DEG["G1"][1])
    )
    assert dot_xy == pytest.approx(math.cos(expected_angle_rad), abs=1e-12)

    # orthogonalize=True (x_primary vs symmetric)
    arm_x = [1.0, 0.0, 0.0]
    arm_y = [math.cos(math.radians(80.0)), math.sin(math.radians(80.0)), 0.0]
    df_xprim = DetectorFrame.from_arms(
        arm_x, arm_y, orthogonalize=True, orthogonalize_mode="x_primary"
    )
    assert df_xprim.is_orthonormal
    np.testing.assert_allclose(df_xprim.basis_matrix[0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(df_xprim.basis_matrix[1], [0.0, 1.0, 0.0], atol=1e-12)

    df_sym = DetectorFrame.from_arms(
        arm_x, arm_y, orthogonalize=True, orthogonalize_mode="symmetric"
    )
    assert df_sym.is_orthonormal
    # Both orthogonalized axes are shifted by 5 deg from the 80-deg arms (to 90 deg)
    np.testing.assert_allclose(
        df_sym.basis_matrix[0],
        [math.cos(math.radians(-5.0)), math.sin(math.radians(-5.0)), 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        df_sym.basis_matrix[1],
        [math.cos(math.radians(85.0)), math.sin(math.radians(85.0)), 0.0],
        atol=1e-12,
    )

    df_nonortho = DetectorFrame.from_arms(arm_x, arm_y, orthogonalize=False)
    assert not df_nonortho.is_orthonormal
    np.testing.assert_allclose(df_nonortho.basis_matrix[1], arm_y, atol=1e-12)


def test_scalar_to_frame_passive_vs_regrid_to_frame():
    sf = _make_metric_scalar_field()
    k1_frame = DetectorFrame.from_detector("K1")

    # 1. to_frame() performs NO spatial interpolation: values & sample array are identical
    sf_passive = sf.to_frame(k1_frame)
    np.testing.assert_array_equal(sf_passive.value, sf.value)
    assert sf_passive.unit == sf.unit
    assert sf_passive.frame == replace_history(k1_frame, sf_passive.frame.history)
    assert "to_frame" in sf_passive.frame.history[-1]

    # Non-orthonormal target frame rejected by default when require_orthogonal=True
    g1_raw = DetectorFrame.from_detector("G1", orthogonalize=False)
    with pytest.raises(ValueError, match="not orthonormal"):
        sf.to_frame(g1_raw)
    # Allowed when require_orthogonal=False
    sf_raw = sf.to_frame(g1_raw, require_orthogonal=False)
    assert not sf_raw.frame.is_orthonormal

    # 2. Pure translation shifts 1D axes without interpolation
    shifted_frame = GridGeometry(
        name="shifted_enu",
        origin=(10.0, -25.0, 5.0),
        local_origin_lonlat=sf.geo.local_origin,
        vertical_datum=sf.geo.vertical_datum,
    )
    sf_shifted = sf.to_frame(shifted_frame)
    np.testing.assert_allclose(
        sf_shifted._axis1_index.to_value(u.m),
        sf._axis1_index.to_value(u.m) - 10.0,
    )
    np.testing.assert_allclose(
        sf_shifted._axis2_index.to_value(u.m),
        sf._axis2_index.to_value(u.m) + 25.0,
    )
    np.testing.assert_allclose(
        sf_shifted._axis3_index.to_value(u.m),
        sf._axis3_index.to_value(u.m) - 5.0,
    )

    # 3. regrid_to_frame() explicitly resamples onto the rotated frame axes
    rot_90 = sf.rotate_horizontal(90.0 * u.deg, resample=True, resolution=10.0 * u.m)
    # At origin (0, 0), phi(0, 0) = 10.0 is invariant
    mid_x = np.argmin(np.abs(rot_90._axis1_index.to_value(u.m)))
    mid_y = np.argmin(np.abs(rot_90._axis2_index.to_value(u.m)))
    assert rot_90.value[0, mid_x, mid_y, 0] == pytest.approx(10.0, abs=1e-9)
    # Along new x' (+y_old direction), gradient is d(phi)/dy_old = +3
    assert rot_90.value[0, mid_x + 1, mid_y, 0] - rot_90.value[
        0, mid_x, mid_y, 0
    ] == pytest.approx(3.0 * 10.0, abs=1e-6)
    # Along new y' (-x_old direction), gradient is -d(phi)/dx_old = -2
    assert rot_90.value[0, mid_x, mid_y + 1, 0] - rot_90.value[
        0, mid_x, mid_y, 0
    ] == pytest.approx(-2.0 * 10.0, abs=1e-6)


def replace_history(frame: GridGeometry, history: tuple[str, ...]) -> GridGeometry:
    from dataclasses import replace

    return replace(frame, history=history)


def test_vectorfield_component_rotation_v_prime_equals_R_v():
    sf = _make_metric_scalar_field()
    ones = ScalarField(
        np.ones_like(sf.value),
        unit=u.m / u.s,
        axis0=sf._axis0_index,
        axis1=sf._axis1_index,
        axis2=sf._axis2_index,
        axis3=sf._axis3_index,
    )
    vx_val, vy_val, vz_val = 3.0, 4.0, -2.0
    vf = VectorField(
        {
            "x": ones * vx_val,
            "y": ones * vy_val,
            "z": ones * vz_val,
        }
    )

    # Rotate frame counter-clockwise by 90 deg around +z:
    # e_x' = e_y, e_y' = -e_x, e_z' = e_z
    # => v_x' = v_y = 4.0, v_y' = -v_x = -3.0, v_z' = v_z = -2.0
    vf_rot90 = vf.rotate_horizontal(90.0 * u.deg, resample=False)
    np.testing.assert_allclose(vf_rot90["x"].to_value(u.m / u.s), 4.0, atol=1e-12)
    np.testing.assert_allclose(vf_rot90["y"].to_value(u.m / u.s), -3.0, atol=1e-12)
    np.testing.assert_allclose(vf_rot90["z"].to_value(u.m / u.s), -2.0, atol=1e-12)

    # L2 norm is strictly invariant under any 3D detector frame rotation
    k1 = DetectorFrame.from_detector("K1")
    vf_k1 = vf.to_frame(k1)
    np.testing.assert_allclose(
        vf_k1.norm().to_value(u.m / u.s),
        vf.norm().to_value(u.m / u.s),
        rtol=1e-12,
        atol=1e-12,
    )

    # Round-trip back to standard ENU restores exact original components
    vf_back = vf_k1.to_frame("local_enu")
    for comp in ("x", "y", "z"):
        np.testing.assert_allclose(
            vf_back[comp].to_value(u.m / u.s),
            vf[comp].to_value(u.m / u.s),
            rtol=1e-12,
            atol=1e-12,
        )


def test_tensorfield_component_rotation_T_prime_equals_R_T_R_transpose():
    sf = _make_metric_scalar_field()
    base = ScalarField(
        np.ones_like(sf.value),
        unit=u.Pa,
        axis0=sf._axis0_index,
        axis1=sf._axis1_index,
        axis2=sf._axis2_index,
        axis3=sf._axis3_index,
    )
    mat_t = np.array(
        [
            [10.0, 2.0, -1.0],
            [2.0, 5.0, 3.0],
            [-1.0, 3.0, 8.0],
        ],
        dtype=np.float64,
    )
    tf = TensorField(
        {(i, j): base * mat_t[i, j] for i in range(3) for j in range(3)},
        rank=2,
    )

    k1 = DetectorFrame.from_detector("K1")
    rot = k1.basis_matrix  # since source is standard ENU (identity)
    expected_t_prime = rot @ mat_t @ rot.T

    tf_k1 = tf.to_frame(k1)
    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(
                tf_k1[(i, j)].to_value(u.Pa),
                expected_t_prime[i, j],
                rtol=1e-12,
                atol=1e-12,
            )

    # Trace is invariant under orthogonal similarity transform T' = R T R^T
    np.testing.assert_allclose(
        tf_k1.trace().to_value(u.Pa),
        tf.trace().to_value(u.Pa),
        rtol=1e-12,
        atol=1e-12,
    )

    # Round-trip back to local_enu
    tf_back = tf_k1.to_frame("local_enu")
    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(
                tf_back[(i, j)].to_value(u.Pa),
                mat_t[i, j],
                rtol=1e-12,
                atol=1e-12,
            )


def test_angle_domain_and_vertical_datum_guards():
    # Angle-domain field must convert to local Cartesian first
    dem = ScalarField(
        np.ones((1, 4, 4, 1)),
        unit=u.m,
        axis0=np.array([0.0]) * u.s,
        axis1=np.linspace(137.30, 137.33, 4) * u.deg,
        axis2=np.linspace(36.40, 36.43, 4) * u.deg,
        axis3=np.array([0.0]) * u.m,
        axis_names=["t", "lon", "lat", "z"],
        space_domain={"lon": "angle", "lat": "angle", "z": "real"},
    )
    with pytest.raises(ValueError, match="to_local_cartesian"):
        dem.to_frame("K1")

    # Vertical datum mismatch between geoid and ellipsoid fails closed by default
    sf = _make_metric_scalar_field()
    ellip_frame = DetectorFrame.from_detector("K1", vertical_datum="WGS84 ellipsoid")
    with pytest.raises(ValueError, match="Vertical datum mismatch"):
        sf.to_frame(ellip_frame)
    with pytest.warns(UserWarning, match="Vertical datum mismatch"):
        sf.to_frame(ellip_frame, strict_datum=False)
