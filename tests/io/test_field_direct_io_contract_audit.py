"""Keep the #553 Field I/O audit aligned with the live terrain registry."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from astropy.io.registry.base import IORegistryError
from gwpy.io.registry import default_registry as io_registry

from gwexpy.fields import FieldDict, ScalarField

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/developers/contracts/public_io_contract.json"
AUDIT = ROOT / "docs/developers/contracts/field_direct_io_audit.md"


def _has_handler(operation: str, format_name: str, cls: type) -> bool:
    getter = io_registry.get_reader if operation == "read" else io_registry.get_writer
    try:
        getter(format_name, cls)
    except IORegistryError:
        return False
    return True


def test_field_contract_exclusion_has_explicit_audit() -> None:
    formats = {
        entry["canonical"] for entry in json.loads(CONTRACT.read_text())["formats"]
    }
    assert {"gsi-dem", "geotiff"}.isdisjoint(formats)

    audit = AUDIT.read_text(encoding="utf-8")
    for required in (
        "#553",
        "gsi-dem",
        "geotiff",
        "FieldDict",
        "rasterio",
        "io-geospatial",
        "conformance",
    ):
        assert required in audit


@pytest.mark.parametrize("format_name", ["gsi-dem", "fgd-dem"])
def test_gsi_registry_matches_audit(format_name: str) -> None:
    assert _has_handler("read", format_name, ScalarField)
    assert _has_handler("read", format_name, FieldDict)
    for cls in (ScalarField, FieldDict):
        assert not _has_handler("write", format_name, cls)


@pytest.mark.parametrize("format_name", ["geotiff", "tif", "tiff"])
def test_geotiff_conditional_registry_matches_audit(format_name: str) -> None:
    available = importlib.util.find_spec("rasterio") is not None
    assert _has_handler("read", format_name, ScalarField) is available
    assert _has_handler("read", format_name, FieldDict) is available
    assert _has_handler("write", format_name, ScalarField) is available
    assert not _has_handler("write", format_name, FieldDict)
