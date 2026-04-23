"""Contract tests for public interop documentation and namespace exposure."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import gwexpy.interop as interop

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs/developers/contracts/public_interop_contract.json"
EN_REF_INDEX = ROOT / "docs/web/en/reference/api/interop.rst"
JA_REF_INDEX = ROOT / "docs/web/ja/reference/api/interop.rst"

VALID_STATUSES = {
    "public",
    "implemented",
    "implemented_partial",
    "in_progress",
    "planned",
}


def _load_contract() -> list[dict]:
    data = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return data["targets"]


TARGETS = _load_contract()


def test_interop_contract_schema_is_well_formed():
    data = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert data["schema_version"] >= 1

    names: set[str] = set()
    for entry in data["targets"]:
        assert isinstance(entry["name"], str) and entry["name"]
        assert entry["name"] not in names
        names.add(entry["name"])

        assert isinstance(entry["module"], str) and entry["module"]
        assert entry["status"] in VALID_STATUSES
        assert isinstance(entry["guide_api"], list)
        assert all(isinstance(name, str) and name for name in entry["guide_api"])
        assert isinstance(entry["row_match_en"], str) and entry["row_match_en"]
        assert isinstance(entry["row_match_ja"], str) and entry["row_match_ja"]
        assert isinstance(entry["reference_indexed"], bool)
        assert isinstance(entry.get("notes", []), list)

        reference_page = entry.get("reference_page")
        details_link = entry.get("details_link")
        assert reference_page is None or isinstance(reference_page, str)
        assert details_link is None or isinstance(details_link, str)

        if entry["status"] == "public":
            assert entry["reference_indexed"] is True
            assert reference_page


def test_documented_interop_api_is_available_from_public_namespace():
    exported = set(interop.__all__)

    for entry in TARGETS:
        module = importlib.import_module(f"gwexpy.interop.{entry['module']}")
        for func_name in entry["guide_api"]:
            assert hasattr(module, func_name)
            assert hasattr(interop, func_name)
            assert func_name in exported


def test_reference_index_matches_contract():
    en_index = EN_REF_INDEX.read_text(encoding="utf-8")
    ja_index = JA_REF_INDEX.read_text(encoding="utf-8")

    for entry in TARGETS:
        module_name = entry["module"]
        reference_page = entry.get("reference_page")

        if entry["reference_indexed"]:
            assert f"\n   {module_name}\n" in en_index
            assert f"\n   {module_name}\n" in ja_index
            assert reference_page
            assert (ROOT / "docs/web/en/reference/api" / reference_page).exists()
            assert (ROOT / "docs/web/ja/reference/api" / reference_page).exists()


def test_mth5_is_importable_from_public_namespace():
    from gwexpy.interop import from_mth5, to_mth5

    assert callable(from_mth5)
    assert callable(to_mth5)
