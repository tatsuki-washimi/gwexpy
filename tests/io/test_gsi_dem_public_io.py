"""Tests for #547 + #520 public Field I/O integration with GSI DEM."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from gwpy.io.registry import default_registry as io_registry

from gwexpy.fields import FieldDict, ScalarField, list_fields
from tests.io_conformance.generators.gsi_dem import write_gsi_dem_xml, write_gsi_dem_zip


def test_gsi_dem_auto_identify_and_non_gsi_rejection(tmp_path):
    xml_path = write_gsi_dem_xml(
        tmp_path / "plain_name.xml", np.array([[1.0, 2.0], [3.0, 4.0]])
    )
    zip_path = write_gsi_dem_zip(
        tmp_path / "plain_archive.zip",
        [("inner.xml", np.array([[5.0, 6.0], [7.0, 8.0]]), {})],
    )

    assert list_fields(xml_path) == ["elevation"]
    assert list_fields(zip_path) == ["elevation"]

    sf = ScalarField.read(xml_path)
    assert sf.name == "elevation"
    assert sf.value[0, 1, 0, 0] == pytest.approx(2.0)

    sf_zip = ScalarField.read(zip_path, "elevation")
    assert sf_zip.value[0, 0, 1, 0] == pytest.approx(7.0)

    fd = FieldDict.read(xml_path, ["elevation"])
    assert list(fd.keys()) == ["elevation"]
    assert isinstance(fd["elevation"], ScalarField)

    registry_fd = io_registry.get_reader("gsi-dem", FieldDict)(
        xml_path, names=["elevation"]
    )
    assert isinstance(registry_fd, FieldDict)
    assert list(registry_fd) == ["elevation"]
    assert registry_fd["elevation"].value[0, 1, 0, 0] == pytest.approx(2.0)
    with pytest.raises(ValueError, match="Available fields"):
        io_registry.get_reader("gsi-dem", FieldDict)(xml_path)
    with pytest.raises(ValueError, match="No axis-0 samples"):
        FieldDict.read(xml_path, ["elevation"], start=1.0)

    with pytest.raises(KeyError):
        ScalarField.read(xml_path, "bogus_field")

    # Non-GSI XML must NOT be identified as gsi-dem
    other_xml = tmp_path / "other.xml"
    other_xml.write_text("<root><node>123</node></root>", encoding="utf-8")
    with pytest.raises(ValueError, match="Could not identify format"):
        ScalarField.read(other_xml)


def test_gsi_dem_read_aliases_fill_missing_and_projected(tmp_path):
    grid = np.array([[10.0, np.nan], [20.0, 30.0]])
    xml_path = write_gsi_dem_xml(
        tmp_path / "FG-GML-ALIAS-DEM.xml",
        grid,
        lon_min=139.70,
        lon_max=139.72,
        lat_min=35.68,
        lat_max=35.70,
    )
    filled = ScalarField.read(xml_path, fill_missing="nearest")
    assert not np.any(np.isnan(filled.value))

    if importlib.util.find_spec("pyproj") is not None:
        proj = ScalarField.read(
            xml_path, fill_missing="nearest", projected="EPSG:32654"
        )
        assert proj.axis_names == ("t", "x", "y", "z")
        assert proj.geo is not None and proj.geo.active is True
