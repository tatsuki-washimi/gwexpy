from __future__ import annotations

import json

import pytest

from tests.io_conformance.contract import (
    CONTRACT_PATH,
    LOGICAL_CI_JOB_IDS,
    MissingDependencyPolicy,
    ScenarioName,
    ScenarioStatus,
    load_public_io_contract,
    validate_logical_ci_job_ids,
)
from tests.io_conformance.scenarios import expand_contract_scenarios


def _formats_by_canonical(contract: dict[str, object]) -> dict[str, dict[str, object]]:
    return {entry["canonical"]: entry for entry in contract["formats"]}


def test_v2_contract_loads_with_v3_defaults():
    contract = load_public_io_contract(CONTRACT_PATH)

    assert contract["schema_version"] == 3
    assert len(contract["formats"]) == 25

    formats = _formats_by_canonical(contract)
    assert formats["hdf5"]["registry_auto_identify"] is False
    assert formats["xml.diaggui"]["registry_auto_identify"] is True
    assert "read_auto_identify" not in formats["hdf5"]
    assert "read_auto_identify" not in formats["xml.diaggui"]


def test_v2_legacy_auto_identify_keys_do_not_leak(tmp_path):
    contract_path = tmp_path / "public_io_contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "formats": [
                    {
                        "canonical": "legacy.synthetic",
                        "read_auto_identify": True,
                        "write_auto_identify": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    contract = load_public_io_contract(contract_path)
    [entry] = contract["formats"]

    assert "read_auto_identify" not in entry
    assert "write_auto_identify" not in entry
    assert entry["public_auto_identify"] is False
    assert entry["registry_auto_identify"] is False


def test_contract_expands_all_formats_and_core_blocking_rows():
    contract = load_public_io_contract(CONTRACT_PATH)
    rows = expand_contract_scenarios(contract)
    formats = _formats_by_canonical(contract)

    policy_rows = [row for row in rows if row["scenario"] == ScenarioName.POLICY.value]
    check_rows = [row for row in rows if row["scenario"] == ScenarioName.CHECK.value]
    blocking_rows = [
        row for row in rows if row["scenario"] == ScenarioName.BLOCKING.value
    ]

    assert {row["format"] for row in policy_rows} == set(formats)
    assert {row["format"] for row in check_rows} == set(formats)
    assert {row["format"] for row in blocking_rows} == {
        canonical
        for canonical, entry in formats.items()
        if entry["public_api"]["read"] or entry["public_api"]["write"]
        if not entry["optional_dependencies"]
    }

    assert all(row["status"] == ScenarioStatus.PASS.value for row in policy_rows)
    assert all(row["status"] == ScenarioStatus.PASS.value for row in check_rows)
    assert all(
        row["status"] == ScenarioStatus.BLOCKED.value for row in blocking_rows
    )


def test_logical_ci_ids_are_validated():
    contract = load_public_io_contract(CONTRACT_PATH)
    rows = expand_contract_scenarios(contract)

    validate_logical_ci_job_ids(rows)

    bad_row = dict(rows[0], logical_ci_job_id="io-conformance-unknown")

    with pytest.raises(ValueError, match="logical_ci_job_id"):
        validate_logical_ci_job_ids([bad_row])


def test_public_optional_dependency_formats_do_not_use_skip_policy():
    contract = load_public_io_contract(CONTRACT_PATH)
    rows = expand_contract_scenarios(contract)
    formats = _formats_by_canonical(contract)
    optional_public = {
        canonical
        for canonical, entry in formats.items()
        if entry["public_api"]["read"] or entry["public_api"]["write"]
        if entry["optional_dependencies"]
    }

    assert optional_public
    assert all(
        row["missing_dependency_policy"] != MissingDependencyPolicy.SKIP.value
        for row in rows
        if row["format"] in optional_public
    )


def test_registry_auto_identify_defaults_are_preserved():
    contract = load_public_io_contract(CONTRACT_PATH)
    formats = _formats_by_canonical(contract)

    assert formats["hdf5"]["registry_auto_identify"] is False
    assert formats["hdf5"]["public_auto_identify"] is False
    assert formats["xml.diaggui"]["registry_auto_identify"] is True


def test_enums_only_expose_the_supported_values():
    assert tuple(ScenarioName) == (
        ScenarioName.POLICY,
        ScenarioName.CHECK,
        ScenarioName.BLOCKING,
    )
    assert tuple(ScenarioStatus) == (
        ScenarioStatus.PASS,
        ScenarioStatus.BLOCKED,
        ScenarioStatus.SKIP,
    )
    assert tuple(MissingDependencyPolicy) == (
        MissingDependencyPolicy.FAIL,
        MissingDependencyPolicy.SKIP,
        MissingDependencyPolicy.NOT_PUBLIC,
        MissingDependencyPolicy.CONDITIONAL_REGISTRATION,
    )
    assert LOGICAL_CI_JOB_IDS == {
        ScenarioName.POLICY.value: "io-conformance-policy",
        ScenarioName.CHECK.value: "io-conformance-check",
        ScenarioName.BLOCKING.value: "io-conformance-blocking",
    }
