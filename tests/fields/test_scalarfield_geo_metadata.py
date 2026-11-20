"""Tests for #546 GeoMetadata provenance, geo_bounds, FFT deactivation, and #725 interaction."""

from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import GeoMetadata, ScalarField, geo_bounds
from gwexpy.time._reference import TIME_REFERENCE_ABSOLUTE, TIME_REFERENCE_RELATIVE


def _make_geo_field(*, epoch=None) -> ScalarField:
    lon = (139.0 + (np.arange(4) + 0.5) * 0.1) * u.deg
    lat = (35.4 - (np.arange(3) + 0.5) * 0.1) * u.deg
    sf = ScalarField(
        np.arange(12, dtype=float).reshape(1, 4, 3, 1),
        unit=u.m,
        epoch=epoch,
        axis0=np.array([0.0]) * u.s,
        axis1=lon,
        axis2=lat,
        axis3=np.array([0.0]) * u.m,
        axis_names=("t", "lon", "lat", "z"),
        space_domain={"lon": "angle", "lat": "angle", "z": "real"},
    )
    sf.geo = GeoMetadata(
        crs="EPSG:6668",
        transform=(139.0, 0.1, 0.0, 35.4, 0.0, -0.1),
        vertical_datum="T.P.",
        nodata=-9999.0,
        band=1,
        source=("dem.xml",),
        source_format="gsi-dem",
        active=True,
    )
    return sf


def test_geometadata_frozen_and_helpers():
    gm = GeoMetadata(crs="EPSG:6668", transform=(0.0, 1.0, 0.0, 1.0, 0.0, -1.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        gm.active = False  # type: ignore[misc]

    gm2 = gm.with_history("fill_missing(nearest)")
    assert gm2.history == ("fill_missing(nearest)",)
    assert gm.history == ()

    gm3 = gm2.deactivated("to_local_cartesian", local_origin=(139.5, 35.3))
    assert gm3.active is False
    assert gm3.local_origin == (139.5, 35.3)
    assert gm3.history == ("fill_missing(nearest)", "to_local_cartesian")


def test_geo_bounds_and_singleton_fallback():
    sf = _make_geo_field()
    xmin, ymin, xmax, ymax = geo_bounds(sf)
    assert xmin == pytest.approx(139.0)
    assert xmax == pytest.approx(139.4)
    assert ymin == pytest.approx(35.1)
    assert ymax == pytest.approx(35.4)

    # Slicing dynamically updates derived geo_bounds while leaving frozen transform intact
    sub = sf[:, 1:3, 0:2, :]
    sxmin, symin, sxmax, symax = geo_bounds(sub)
    assert sxmin == pytest.approx(139.1)
    assert sxmax == pytest.approx(139.3)
    assert symin == pytest.approx(35.2)
    assert symax == pytest.approx(35.4)
    assert sub.geo.transform == sf.geo.transform

    # Singleton cell falls back to GeoMetadata.transform step
    single = sf[:, 0:1, 0:1, :]
    oxmin, oymin, oxmax, oymax = geo_bounds(single)
    assert oxmin == pytest.approx(139.0)
    assert oxmax == pytest.approx(139.1)
    assert oymin == pytest.approx(35.3)
    assert oymax == pytest.approx(35.4)


def test_geo_propagation_pickle_and_fft_deactivation():
    sf_rel = _make_geo_field()
    assert sf_rel.time_reference == TIME_REFERENCE_RELATIVE

    sf_abs = _make_geo_field(epoch=1_400_000_000.0)
    assert sf_abs.time_reference == TIME_REFERENCE_ABSOLUTE

    assert (sf_rel + 1.0 * u.m).geo == sf_rel.geo
    assert (sf_rel * 2.0).time_reference == TIME_REFERENCE_RELATIVE
    assert sf_abs.copy().geo == sf_abs.geo
    assert sf_abs.copy().time_reference == TIME_REFERENCE_ABSOLUTE

    unpickled = pickle.loads(pickle.dumps(sf_abs))
    assert unpickled.geo == sf_abs.geo
    assert unpickled.time_reference == TIME_REFERENCE_ABSOLUTE

    # Real cartesian field carrying active GeoMetadata (e.g. UTM projected)
    proj = ScalarField(
        np.ones((4, 8, 8, 1)),
        unit=u.m,
        epoch=1_400_000_000.0,
        axis0=np.arange(4, dtype=float) * u.s,
        axis1=np.arange(8, dtype=float) * 10.0 * u.m,
        axis2=np.arange(8, dtype=float) * 10.0 * u.m,
        axis3=np.array([0.0]) * u.m,
    )
    proj.geo = GeoMetadata(crs="EPSG:32654", active=True)

    # fft_time keeps geo active
    f_time = proj.fft_time()
    assert f_time.geo is not None and f_time.geo.active is True
    assert f_time.time_reference == TIME_REFERENCE_ABSOLUTE

    # fft_space deactivates geo and ifft_space does NOT reactivate it
    k_space = proj.fft_space("x")
    assert k_space.geo is not None
    assert k_space.geo.active is False
    assert k_space.geo.history[-1] == "fft_space"
    rt = k_space.ifft_space("kx")
    assert rt.geo is not None
    assert rt.geo.active is False
    assert rt.geo.history == ("fft_space", "ifft_space")
    assert rt.time_reference == TIME_REFERENCE_ABSOLUTE
