"""Tests for #550 ScalarField.to_local_cartesian."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import GeoMetadata, ScalarField
from gwexpy.time._reference import TIME_REFERENCE_ABSOLUTE


def _ellipsoid_radii_grs80(lat0_rad: float) -> tuple[float, float]:
    a = 6378137.0
    f = 1.0 / 298.257222101
    e2 = f * (2.0 - f)
    w = np.sqrt(1.0 - e2 * np.sin(lat0_rad) ** 2)
    n_phi = a / w
    m_phi = a * (1.0 - e2) / (w**3)
    return float(n_phi), float(m_phi)


def test_to_local_cartesian_uniform_and_fft_ready():
    n_lon, n_lat = 16, 12
    lon = (139.5 + np.linspace(-0.005, 0.005, n_lon)) * u.deg
    lat = (35.5 + np.linspace(0.004, -0.004, n_lat)) * u.deg
    data = np.ones((2, n_lon, n_lat, 1))
    sf = ScalarField(
        data,
        unit=u.m,
        epoch=1_400_000_000.0,
        axis0=np.array([0.0, 1.0]) * u.s,
        axis1=lon,
        axis2=lat,
        axis3=np.array([0.0]) * u.m,
        axis_names=("t", "lon", "lat", "z"),
        space_domain={"lon": "angle", "lat": "angle", "z": "real"},
    )
    sf.geo = GeoMetadata(crs="EPSG:6668", active=True)

    cart = sf.to_local_cartesian()
    assert cart.axis_names == ("t", "x", "y", "z")
    assert cart.space_domains == {"x": "real", "y": "real", "z": "real"}
    assert cart._axis1_index.unit == u.m
    assert cart._axis2_index.unit == u.m
    assert cart.time_reference == TIME_REFERENCE_ABSOLUTE
    assert cart.geo is not None
    assert cart.geo.active is False
    assert cart.geo.local_origin == pytest.approx((139.5, 35.5))
    assert "to_local_cartesian" in cart.geo.history[-1]

    # Resynthesized axes are strictly regular and FFT-ready
    assert cart.is_axis_regular(1)
    assert cart.is_axis_regular(2)
    kf = cart.fft_space("x")
    assert kf.space_domains["kx"] == "k"


def test_to_local_cartesian_analytic_grs80_and_nonuniform():
    lon0, lat0 = 139.0, 36.0
    dlon = 0.01
    dlat = 0.01
    lon = np.array([lon0 - dlon, lon0, lon0 + 3.0 * dlon]) * u.deg
    lat = np.array([lat0 - dlat, lat0, lat0 + 2.0 * dlat]) * u.deg
    sf = ScalarField(
        np.arange(9, dtype=float).reshape(1, 3, 3, 1),
        unit=u.m,
        axis0=np.array([0.0]) * u.s,
        axis1=lon,
        axis2=lat,
        axis3=np.array([0.0]) * u.m,
        axis_names=("t", "lon", "lat", "z"),
        space_domain={"lon": "angle", "lat": "angle", "z": "real"},
    )

    cart = sf.to_local_cartesian(
        origin=(lon0, lat0), names=("east", "north"), ellipsoid="GRS80"
    )
    assert cart.axis_names == ("t", "east", "north", "z")
    n_phi, m_phi = _ellipsoid_radii_grs80(np.deg2rad(lat0))
    expected_x = n_phi * np.cos(np.deg2rad(lat0)) * np.deg2rad(lon.value - lon0)
    expected_y = m_phi * np.deg2rad(lat.value - lat0)
    np.testing.assert_allclose(cart._axis1_index.to_value(u.m), expected_x, atol=1e-6)
    np.testing.assert_allclose(cart._axis2_index.to_value(u.m), expected_y, atol=1e-6)
    # Preserves nonuniform ratios
    assert not cart.is_axis_regular(1)


def test_to_local_cartesian_rejects_non_angle_field():
    real_sf = ScalarField(np.ones((1, 4, 4, 1)), unit=u.m)
    with pytest.raises(ValueError, match="angle"):
        real_sf.to_local_cartesian()
