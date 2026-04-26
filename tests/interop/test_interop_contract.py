"""Contract tests for public interop documentation and namespace exposure."""

from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path

import gwexpy.interop as interop

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs/developers/contracts/public_interop_contract.json"
PYPROJECT_PATH = ROOT / "pyproject.toml"
EN_REF_INDEX = ROOT / "docs/web/en/reference/api/interop.rst"
JA_REF_INDEX = ROOT / "docs/web/ja/reference/api/interop.rst"

VALID_STATUSES = {
    "public",
    "implemented",
    "implemented_partial",
    "in_progress",
    "planned",
}

VALID_EXTRAS = {
    "audio",
    "control",
    "gw",
    "netcdf4",
    "seismic",
    "zarr",
}

VALID_UNAVAILABLE_BEHAVIORS = {
    "available_in_base_install",
    "not_applicable",
    "raises_import_error",
}


def _load_contract() -> list[dict]:
    data = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return data["targets"]


def _package_name(requirement: str) -> str:
    return (
        requirement.split(";", maxsplit=1)[0]
        .split("[", maxsplit=1)[0]
        .split(">", maxsplit=1)[0]
        .split("<", maxsplit=1)[0]
        .split("=", maxsplit=1)[0]
        .split("!", maxsplit=1)[0]
        .split("~", maxsplit=1)[0]
        .strip()
        .lower()
    )


TARGETS = _load_contract()


def test_interop_contract_schema_is_well_formed():
    data = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    assert data["schema_version"] >= 1

    names: set[str] = set()
    for entry in data["targets"]:
        assert isinstance(entry["name"], str) and entry["name"]
        assert entry["name"] not in names
        names.add(entry["name"])

        module = entry["module"]
        assert module is None or isinstance(module, str)
        assert entry["status"] in VALID_STATUSES
        assert isinstance(entry["guide_api"], list)
        assert all(isinstance(name, str) and name for name in entry["guide_api"])
        assert isinstance(entry["optional_dependencies"], list)
        assert all(
            isinstance(name, str) and name for name in entry["optional_dependencies"]
        )
        assert len(entry["optional_dependencies"]) == len(
            set(entry["optional_dependencies"])
        )
        source_dependencies = entry.get("source_dependencies", [])
        assert isinstance(source_dependencies, list)
        assert all(isinstance(name, str) and name for name in source_dependencies)
        assert len(source_dependencies) == len(set(source_dependencies))
        assert isinstance(entry["extras"], list)
        assert all(extra in VALID_EXTRAS for extra in entry["extras"])
        assert len(entry["extras"]) == len(set(entry["extras"]))
        assert entry["unavailable_behavior"] in VALID_UNAVAILABLE_BEHAVIORS
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

        if entry["guide_api"] or entry["reference_indexed"]:
            assert isinstance(module, str) and module
        else:
            assert module is None
            assert entry["unavailable_behavior"] == "not_applicable"


def test_public_namespace_is_fully_covered_by_contract():
    """Reverse-direction check: every symbol in interop.__all__ must appear in the contract.

    Prevents __all__ from growing beyond what the contract explicitly tracks.
    """
    contract_symbols: set[str] = set()
    for entry in TARGETS:
        contract_symbols.update(entry.get("guide_api", []))

    exported = set(interop.__all__)
    unregistered = sorted(exported - contract_symbols)

    assert unregistered == [], (
        f"Symbols in interop.__all__ not registered in any contract entry's guide_api: "
        f"{unregistered}\n"
        f"Either add them to the contract or remove from __all__."
    )


def test_documented_interop_api_is_available_from_public_namespace():
    exported = set(interop.__all__)

    for entry in TARGETS:
        if not entry["guide_api"]:
            continue

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
        elif module_name:
            assert f"\n   {module_name}\n" not in en_index
            assert f"\n   {module_name}\n" not in ja_index


def test_mth5_is_importable_from_public_namespace():
    from gwexpy.interop import from_mth5, to_mth5

    assert callable(from_mth5)
    assert callable(to_mth5)


def test_no_helper_backed_implemented_only_targets_remain():
    helper_backed_implemented = [
        entry
        for entry in TARGETS
        if entry["status"] == "implemented" and entry["guide_api"]
    ]
    assert helper_backed_implemented == []


def test_non_public_module_backed_targets_are_explicit_exceptions():
    non_public_module_backed = [
        entry["name"]
        for entry in TARGETS
        if entry["module"] is not None and entry["status"] != "public"
    ]
    assert non_public_module_backed == ["root"]


def test_import_only_source_packages_are_not_runtime_dependencies():
    entries = {entry["name"]: entry for entry in TARGETS}

    expected_source_dependencies = {
        "pyoma": ["pyOMA"],
        "multitaper": ["multitaper"],
        "mtspec": ["mtspec"],
        "sdypy": ["pyuff"],
        "sdynpy": ["sdynpy"],
    }

    for name, source_dependencies in expected_source_dependencies.items():
        entry = entries[name]
        assert entry["source_dependencies"] == source_dependencies
        assert entry["optional_dependencies"] == []
        assert entry["extras"] == []
        assert entry["unavailable_behavior"] == "available_in_base_install"


def test_xarray_source_adapters_distinguish_runtime_from_producer_packages():
    entries = {entry["name"]: entry for entry in TARGETS}

    expected_source_dependencies = {
        "metpy": ["metpy"],
        "wrf": ["wrf-python"],
        "harmonica": ["harmonica"],
    }

    for name, source_dependencies in expected_source_dependencies.items():
        entry = entries[name]
        assert entry["source_dependencies"] == source_dependencies
        assert entry["optional_dependencies"] == ["xarray"]
        assert entry["extras"] == ["netcdf4"]
        assert entry["unavailable_behavior"] == "raises_import_error"


def test_declared_extras_contain_runtime_dependencies():
    """Contract extras must install the runtime packages they advertise."""
    pyproject = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    optional_dependency_groups = pyproject["project"]["optional-dependencies"]
    normalized_extras = {
        extra: {_package_name(requirement) for requirement in requirements}
        for extra, requirements in optional_dependency_groups.items()
    }

    for entry in TARGETS:
        for extra in entry["extras"]:
            installed_packages = normalized_extras[extra]
            missing_dependencies = [
                dependency
                for dependency in entry["optional_dependencies"]
                if dependency.lower() not in installed_packages
            ]
            assert missing_dependencies == [], (
                f"{entry['name']} declares extra {extra!r}, but that extra does not "
                f"install runtime dependencies: {missing_dependencies}"
            )
