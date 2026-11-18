"""Tests for #545 'angle' space_domain validation, propagation, and FFT guards."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField
from gwexpy.fields import signal as field_signal


def _make_geo_scalar_field(*, nt: int = 8) -> ScalarField:
    lon = np.linspace(139.0, 139.5, 6) * u.deg
    lat = np.linspace(35.0, 35.5, 5) * u.deg
    t = np.arange(nt, dtype=float) * 0.1 * u.s
    data = np.ones((nt, 6, 5, 1), dtype=float)
    return ScalarField(
        data,
        unit=u.m,
        axis0=t,
        axis1=lon,
        axis2=lat,
        axis3=np.array([0.0]) * u.m,
        axis_names=("t", "lon", "lat", "z"),
        space_domain={"lon": "angle", "lat": "angle", "z": "real"},
    )


def test_angle_domain_accepts_angular_units_and_rejects_length():
    sf_deg = _make_geo_scalar_field()
    assert sf_deg.space_domains == {"lon": "angle", "lat": "angle", "z": "real"}

    sf_rad = ScalarField(
        np.ones((1, 4, 3, 1)),
        unit=u.m,
        axis0=np.array([0.0]) * u.s,
        axis1=np.linspace(0.0, 0.1, 4) * u.rad,
        axis2=np.linspace(0.0, 0.1, 3) * u.rad,
        axis3=np.array([0.0]) * u.m,
        axis_names=("t", "phi", "theta", "r"),
        space_domain={"phi": "angle", "theta": "angle", "r": "real"},
    )
    assert sf_rad.space_domains["phi"] == "angle"

    with pytest.raises(ValueError, match="angle"):
        ScalarField(
            np.ones((1, 4, 3, 1)),
            axis1=np.arange(4) * u.m,
            axis2=np.arange(3) * u.deg,
            axis_names=("t", "lon", "lat", "z"),
            space_domain={"lon": "angle", "lat": "angle", "z": "real"},
        )

    with pytest.raises(ValueError, match="angle"):
        ScalarField(
            np.ones((1, 4, 3, 1)),
            axis1=np.arange(4) * u.m,
            axis2=np.arange(3) * u.m,
            space_domain="angle",
        )


def test_angle_domain_propagation_and_fft_guards():
    sf = _make_geo_scalar_field(nt=16)
    sliced = sf[:, 1:4, 1:3, :]
    assert sliced.space_domains == {"lon": "angle", "lat": "angle", "z": "real"}

    copied = sf.copy()
    assert copied.space_domains == sf.space_domains

    scaled = sf * 2.0
    assert scaled.space_domains == sf.space_domains

    freq_sf = sf.fft_time()
    assert freq_sf.space_domains == sf.space_domains
    assert freq_sf.ifft_time().space_domains == sf.space_domains

    with pytest.raises(ValueError, match="to_local_cartesian"):
        sf.fft_space("lon")

    with pytest.raises(ValueError, match="to_local_cartesian"):
        sf.fft_space(["lon", "lat"])

    with pytest.raises(ValueError, match="to_local_cartesian"):
        sf.ifft_space("lon")

    with pytest.raises(ValueError, match="to_local_cartesian"):
        field_signal.spectral_density(sf, axis="lon")
