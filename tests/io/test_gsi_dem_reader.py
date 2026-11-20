"""Tests for #547 GSI FGD DEM XML/ZIP reader and mosaic assembly."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField, geo_bounds
from gwexpy.fields.io import read_scalarfield_gsi_dem
from tests.io_conformance.generators.gsi_dem import write_gsi_dem_xml, write_gsi_dem_zip


def test_gsi_dem_single_xml_coords_nodata_and_startpoint(tmp_path):
    # 3 rows (lat desc), 4 cols (lon asc); startPoint=(2, 0) skips first 2 ocean cells -> NaN
    grid_yx = np.array(
        [
            [-9999.0, -9999.0, 10.5, 11.5],
            [20.5, 21.5, 22.5, -9999.0],
            [30.5, 31.5, 32.5, 33.5],
        ],
        dtype=float,
    )
    xml_path = write_gsi_dem_xml(
        tmp_path / "FG-GML-5339-DEM5A.xml",
        grid_yx,
        lon_min=139.0,
        lon_max=139.4,
        lat_min=35.0,
        lat_max=35.3,
        start_point=(2, 0),
        skip_leading_cells=2,
    )

    sf = read_scalarfield_gsi_dem(xml_path)
    assert isinstance(sf, ScalarField)
    assert sf.shape == (1, 4, 3, 1)
    assert sf.unit == u.m
    assert sf.axis_names == ("t", "lon", "lat", "z")
    assert sf.space_domains == {"lon": "angle", "lat": "angle", "z": "real"}

    # Pixel center coordinates
    np.testing.assert_allclose(
        sf._axis1_index.to_value(u.deg), [139.05, 139.15, 139.25, 139.35]
    )
    np.testing.assert_allclose(sf._axis2_index.to_value(u.deg), [35.25, 35.15, 35.05])
    assert geo_bounds(sf) == pytest.approx((139.0, 35.0, 139.4, 35.3))

    # startPoint padded (0,0) and (1,0), plus (3,1) is -9999 -> NaN
    assert np.isnan(sf.value[0, 0, 0, 0])
    assert np.isnan(sf.value[0, 1, 0, 0])
    assert sf.value[0, 2, 0, 0] == pytest.approx(10.5)
    assert np.isnan(sf.value[0, 3, 1, 0])
    assert sf.value[0, 3, 2, 0] == pytest.approx(33.5)

    assert sf.geo is not None and sf.geo.active is True
    assert sf.geo.crs == "EPSG:6668"
    assert sf.geo.source_format == "gsi-dem"
    assert sf.geo.nodata == -9999.0

    # ascending axis_order flips lat strictly increasing
    sf_asc = read_scalarfield_gsi_dem(xml_path, axis_order="ascending")
    np.testing.assert_allclose(
        sf_asc._axis2_index.to_value(u.deg), [35.05, 35.15, 35.25]
    )
    assert sf_asc.value[0, 3, 0, 0] == pytest.approx(33.5)


def test_gsi_dem_zip_mosaic_gaps_overlap_and_bounds(tmp_path):
    west = np.array([[1.0, 2.0], [3.0, 4.0]])
    east = np.array([[5.0, 6.0], [7.0, 8.0]])
    far_gap_east = np.array([[9.0, 10.0], [11.0, 12.0]])

    # Two tiles inside one ZIP with a 2-col gap in the middle -> NaN gap filled
    zpath = write_gsi_dem_zip(
        tmp_path / "FG-GML-MOSAIC-DEM5A.zip",
        [
            (
                "west.xml",
                west,
                dict(lon_min=139.0, lon_max=139.2, lat_min=35.0, lat_max=35.2),
            ),
            (
                "gap_east.xml",
                far_gap_east,
                dict(lon_min=139.4, lon_max=139.6, lat_min=35.0, lat_max=35.2),
            ),
        ],
    )
    mosaic = read_scalarfield_gsi_dem(zpath)
    assert mosaic.shape == (1, 6, 2, 1)
    assert mosaic.value[0, 0, 0, 0] == pytest.approx(1.0)
    assert np.all(np.isnan(mosaic.value[0, 2:4, :, 0]))
    assert mosaic.value[0, 5, 1, 0] == pytest.approx(12.0)

    # bounds= skips disjoint west tile and crops only east tile
    cropped = read_scalarfield_gsi_dem(zpath, bounds=(139.4, 35.0, 139.6, 35.2))
    assert cropped.shape == (1, 2, 2, 1)
    assert cropped.value[0, 0, 0, 0] == pytest.approx(9.0)

    # Disjoint bounds raises ValueError
    with pytest.raises(ValueError, match="No GSI DEM tiles intersect bounds"):
        read_scalarfield_gsi_dem(zpath, bounds=(140.0, 36.0, 141.0, 37.0))

    # Duplicate / overlapping tile sequence emits UserWarning and later tile wins
    xml1 = write_gsi_dem_xml(
        tmp_path / "t1.xml",
        west,
        lon_min=139.0,
        lon_max=139.2,
        lat_min=35.0,
        lat_max=35.2,
    )
    xml2 = write_gsi_dem_xml(
        tmp_path / "t2.xml",
        east,
        lon_min=139.0,
        lon_max=139.2,
        lat_min=35.0,
        lat_max=35.2,
    )
    with pytest.warns(UserWarning, match="Overlapping"):
        over = read_scalarfield_gsi_dem([xml1, xml2])
    assert over.value[0, 0, 0, 0] == pytest.approx(5.0)


def test_gsi_dem_error_conditions(tmp_path):
    g = np.ones((2, 2))
    # Bad sequenceRule
    bad_seq = write_gsi_dem_xml(tmp_path / "bad_seq.xml", g, axis_order="+y+x")
    with pytest.raises(ValueError, match="axisOrder"):
        read_scalarfield_gsi_dem(bad_seq)

    # All-NaN DEM
    all_nan = write_gsi_dem_xml(tmp_path / "all_nan.xml", np.full((2, 2), np.nan))
    with pytest.raises(ValueError, match="only missing"):
        read_scalarfield_gsi_dem(all_nan)

    # Mosaic CRS mismatch
    a = write_gsi_dem_xml(tmp_path / "a.xml", g, srs_name="fguuid:jgd2011.bl")
    b = write_gsi_dem_xml(tmp_path / "b.xml", g, srs_name="fguuid:jgd2000.bl")
    with pytest.raises(ValueError, match="mismatched CRS"):
        read_scalarfield_gsi_dem([a, b])

    # Mosaic lattice misalignment > 0.01 px
    c = write_gsi_dem_xml(
        tmp_path / "c.xml",
        g,
        lon_min=139.05,
        lon_max=139.25,
        lat_min=35.0,
        lat_max=35.2,
    )
    with pytest.raises(ValueError, match="lattice misalignment"):
        read_scalarfield_gsi_dem([a, c])

    # Mosaic resolution mismatch
    d = write_gsi_dem_xml(
        tmp_path / "d.xml",
        np.ones((4, 4)),
        lon_min=139.2,
        lon_max=139.4,
        lat_min=35.0,
        lat_max=35.2,
    )
    with pytest.raises(ValueError, match="mismatched resolution"):
        read_scalarfield_gsi_dem([a, d])
