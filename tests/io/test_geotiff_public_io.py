"""Tests for #548 + #520 GeoTIFF public I/O and optional-dependency gating."""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import FieldDict, ScalarField, list_fields


def test_geotiff_missing_rasterio_hints_without_side_effect(tmp_path):
    if importlib.util.find_spec("rasterio") is not None:
        pytest.skip("rasterio is installed in this environment")

    assert "rasterio" not in sys.modules
    dummy = tmp_path / "sample.tif"
    dummy.write_bytes(b"II*\x00")

    with pytest.raises(ImportError, match="geospatial"):
        ScalarField.read(dummy)
    with pytest.raises(ImportError, match="geospatial"):
        ScalarField.read(tmp_path / "any.dat", format="geotiff")
    for alias in ("tif", "tiff"):
        with pytest.raises(ImportError, match="geospatial"):
            ScalarField.read(tmp_path / "any.dat", format=alias)
        with pytest.raises(ImportError, match="geospatial"):
            FieldDict.read(tmp_path / "any.dat", ["band_1"], format=alias)

    sf = ScalarField(np.ones((1, 2, 2, 1)), unit=u.m)
    with pytest.raises(ImportError, match="geospatial"):
        sf.write(tmp_path / "out.tif")
    for alias in ("tif", "tiff"):
        with pytest.raises(ImportError, match="geospatial"):
            sf.write(tmp_path / "out.dat", format=alias)


def test_geotiff_public_io_roundtrip_and_aliases(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import Affine

    tif_path = tmp_path / "multi.tif"
    b1 = np.array([[10.0, -9999.0], [30.0, 40.0]], dtype=np.float64)
    b2 = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float64)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=2,
        dtype="float64",
        crs="EPSG:6668",
        transform=Affine.from_gdal(139.7, 0.01, 0.0, 35.7, 0.0, -0.01),
        nodata=-9999.0,
    ) as dst:
        dst.write(b1, 1)
        dst.write(b2, 2)
        dst.units = ("m", "m")

    assert list_fields(tif_path) == ["band_1", "band_2"]

    sf = ScalarField.read(tif_path)
    assert sf.name == "band_1"
    assert np.isnan(sf.value[0, 1, 0, 0])

    with pytest.raises(KeyError):
        ScalarField.read(tif_path, "band_99")

    fd = FieldDict.read(tif_path, ["band_2", "band_1"])
    assert list(fd.keys()) == ["band_2", "band_1"]
    assert fd["band_2"].value[0, 0, 0, 0] == pytest.approx(100.0)

    from gwpy.io.registry import default_registry as io_registry

    registry_fd = io_registry.get_reader("geotiff", FieldDict)(
        tif_path, names=["band_2", "band_1"]
    )
    assert list(registry_fd) == ["band_2", "band_1"]

    out_path = tmp_path / "written.tif"
    fd["band_2"].write(out_path)
    re_read = ScalarField.read(out_path)
    np.testing.assert_allclose(re_read.value, fd["band_2"].value)

    if importlib.util.find_spec("pyproj") is not None:
        proj_unfilled = ScalarField.read(
            tif_path,
            projected={"target_crs": "EPSG:32654", "resolution": 500.0},
        )
        proj_filled = ScalarField.read(
            tif_path,
            fill_missing="nearest",
            projected={"target_crs": "EPSG:32654", "resolution": 500.0},
        )
        assert proj_filled.axis_names == ("t", "x", "y", "z")
        assert np.all(np.isfinite(proj_filled.value[0, 1:3, 1:3, 0]))
        assert np.count_nonzero(np.isfinite(proj_filled.value)) > np.count_nonzero(
            np.isfinite(proj_unfilled.value)
        )
