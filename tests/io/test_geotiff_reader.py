"""Tests for #548 GeoTIFF reader and writer (requires rasterio)."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import Affine

from gwexpy.fields import ScalarField, geo_bounds
from gwexpy.fields.io import read_scalarfield_geotiff, write_scalarfield_geotiff


def _write_sample_tiff(
    path,
    *,
    crs="EPSG:6668",
    nodata=-9999.0,
    unit_tag="m",
    transform=None,
    multiband=False,
):
    if transform is None:
        transform = Affine.from_gdal(139.0, 0.1, 0.0, 35.4, 0.0, -0.1)
    band1 = np.array([[10.0, 20.0, -9999.0], [30.0, 40.0, 50.0]], dtype=np.float64)
    count = 2 if multiband else 1
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=3,
        count=count,
        dtype="float64",
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(band1, 1)
        if multiband:
            dst.write(band1 * 2.0, 2)
        if unit_tag:
            dst.units = tuple([unit_tag] * count)


def test_geotiff_read_geographic_projected_and_bounds(tmp_path):
    geo_tif = tmp_path / "dem_geo.tif"
    _write_sample_tiff(geo_tif, crs="EPSG:6668", multiband=True)

    sf = read_scalarfield_geotiff(geo_tif, band=1)
    assert sf.shape == (1, 3, 2, 1)
    assert sf.axis_names == ("t", "lon", "lat", "z")
    assert sf.space_domains == {"lon": "angle", "lat": "angle", "z": "real"}
    assert sf.unit == u.m
    assert np.isnan(sf.value[0, 2, 0, 0])
    assert sf.value[0, 1, 1, 0] == pytest.approx(40.0)
    assert geo_bounds(sf) == pytest.approx((139.0, 35.2, 139.3, 35.4))

    sf_b2 = read_scalarfield_geotiff(geo_tif, band=2)
    assert sf_b2.name == "band_2"
    assert sf_b2.value[0, 1, 1, 0] == pytest.approx(80.0)
    assert sf_b2.geo is not None and sf_b2.geo.band == 2

    cropped = read_scalarfield_geotiff(
        geo_tif, band=1, bounds=(139.0, 35.2, 139.2, 35.4), axis_order="ascending"
    )
    assert cropped.shape == (1, 2, 2, 1)
    assert cropped._axis2_index.value[1] > cropped._axis2_index.value[0]

    proj_tif = tmp_path / "dem_utm.tif"
    _write_sample_tiff(
        proj_tif,
        crs="EPSG:32654",
        transform=Affine.from_gdal(400000.0, 10.0, 0.0, 3950020.0, 0.0, -10.0),
    )
    sf_utm = read_scalarfield_geotiff(proj_tif)
    assert sf_utm.axis_names == ("t", "x", "y", "z")
    assert sf_utm.space_domains == {"x": "real", "y": "real", "z": "real"}
    assert sf_utm._axis1_index.unit == u.m


def test_multiband_geotiff_requires_selector(tmp_path):
    path = tmp_path / "multiband.tif"
    _write_sample_tiff(path, multiband=True)
    with pytest.raises(ValueError, match="multiband.*requires a band selector"):
        ScalarField.read(path, format="geotiff")
    assert ScalarField.read(path, "band_2", format="geotiff").name == "band_2"


def test_unknown_geotiff_unit_requires_override(tmp_path):
    path = tmp_path / "unknown_unit.tif"
    _write_sample_tiff(path, unit_tag="invalid_terrain_unit")
    with pytest.raises(ValueError, match="invalid_terrain_unit.*unit="):
        ScalarField.read(path, format="geotiff")
    assert ScalarField.read(path, format="geotiff", unit="m").unit == u.m


def test_geotiff_validation_errors(tmp_path):
    no_crs = tmp_path / "no_crs.tif"
    _write_sample_tiff(no_crs, crs=None)
    with pytest.raises(ValueError, match="no CRS"):
        read_scalarfield_geotiff(no_crs)

    rotated = tmp_path / "rot.tif"
    _write_sample_tiff(rotated, transform=Affine(0.1, 0.02, 139.0, 0.0, -0.1, 35.4))
    with pytest.raises(ValueError, match="Rotated"):
        read_scalarfield_geotiff(rotated)

    south_up = tmp_path / "south_up.tif"
    _write_sample_tiff(
        south_up, transform=Affine.from_gdal(139.0, 0.1, 0.0, 35.0, 0.0, 0.1)
    )
    with pytest.raises(ValueError, match="north-up"):
        read_scalarfield_geotiff(south_up)


def test_geotiff_write_roundtrip_including_ascending_flip(tmp_path):
    src_tif = tmp_path / "src.tif"
    _write_sample_tiff(src_tif)
    sf_asc = read_scalarfield_geotiff(src_tif, axis_order="ascending")

    out_tif = tmp_path / "out.tif"
    write_scalarfield_geotiff(sf_asc, out_tif)

    sf_re = read_scalarfield_geotiff(out_tif, axis_order="ascending")
    np.testing.assert_allclose(sf_re.value, sf_asc.value, equal_nan=True)
    np.testing.assert_allclose(sf_re._axis1_index.value, sf_asc._axis1_index.value)
    np.testing.assert_allclose(sf_re._axis2_index.value, sf_asc._axis2_index.value)
    assert sf_re.unit == sf_asc.unit
    assert sf_re.geo is not None and "6668" in (sf_re.geo.crs or "")

    # Deactivated geo rejects write unless explicit crs= supplied
    sf_deact = sf_asc.copy()
    sf_deact.geo = sf_asc.geo.deactivated("fft_space")  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="active GeoMetadata"):
        write_scalarfield_geotiff(sf_deact, tmp_path / "fail.tif")
