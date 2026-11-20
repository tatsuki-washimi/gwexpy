"""Tests for #552 ScalarField.to_projected (requires pyproj)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

pyproj = pytest.importorskip("pyproj")

from gwexpy.fields import GeoMetadata, ScalarField


def _make_geographic_dem() -> ScalarField:
    lon = np.linspace(139.70, 139.75, 11) * u.deg
    lat = np.linspace(35.70, 35.65, 11) * u.deg
    llon, llat = np.meshgrid(lon.value, lat.value, indexing="ij")
    elev = 100.0 * (llon - 139.70) / 0.05 + 50.0 * (llat - 35.65) / 0.05
    # Leave a corner NaN to exercise NaN-safe resampling
    elev[0, 0] = np.nan
    sf = ScalarField(
        elev[None, :, :, None],
        unit=u.m,
        name="elevation",
        axis0=np.array([0.0]) * u.s,
        axis1=lon,
        axis2=lat,
        axis3=np.array([0.0]) * u.m,
        axis_names=("t", "lon", "lat", "z"),
        space_domain={"lon": "angle", "lat": "angle", "z": "real"},
    )
    dlon = lon.value[1] - lon.value[0]
    dlat = lat.value[1] - lat.value[0]
    sf.geo = GeoMetadata(
        crs="EPSG:6668",
        transform=(
            lon.value[0] - 0.5 * dlon,
            dlon,
            0.0,
            lat.value[0] - 0.5 * dlat,
            0.0,
            dlat,
        ),
        vertical_datum="T.P.",
        source_format="gsi-dem",
        active=True,
    )
    return sf


@pytest.mark.parametrize("method", ["nearest", "bilinear", "cubic"])
def test_to_projected_roundtrip_and_methods(method):
    sf = _make_geographic_dem()
    utm = sf.to_projected("EPSG:32654", resolution=400.0 * u.m, method=method)

    assert utm.axis_names == ("t", "x", "y", "z")
    assert utm.space_domains == {"x": "real", "y": "real", "z": "real"}
    assert utm._axis1_index.unit == u.m
    assert utm._axis2_index.unit == u.m
    assert utm.geo is not None and utm.geo.active is True
    assert "32654" in (utm.geo.crs or "")
    assert any("to_projected(" in h and "32654" in h for h in utm.geo.history)
    # Interior remains finite even with an edge NaN pixel
    assert np.any(np.isfinite(utm.value))
    cx, cy = utm.shape[1] // 2, utm.shape[2] // 2
    assert np.isfinite(utm.value[0, cx, cy, 0])

    # Projected -> geographic round-trip
    back = utm.to_projected("EPSG:6668", resolution=0.005)
    assert back.axis_names == ("t", "lon", "lat", "z")
    assert back.space_domains["lon"] == "angle"
    assert back._axis1_index.unit == u.deg


def test_to_projected_requires_active_geo():
    sf = _make_geographic_dem()
    sf.geo = sf.geo.deactivated("test")  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="active GeoMetadata"):
        sf.to_projected("EPSG:32654")
