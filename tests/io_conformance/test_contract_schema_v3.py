from __future__ import annotations

import json

import pytest

from tests.io_conformance.contract import (
    CONTRACT_PATH,
    LOGICAL_CI_JOB_IDS,
    ContractCIJobId,
    CoverageStatus,
    MissingDependencyPolicy,
    ScenarioName,
    ScenarioStatus,
    load_public_io_contract,
    validate_contract_ci_jobs,
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
    assert contract["io_conformance_v3"]["base_gate"] == "io-conformance"


def test_contract_v3_policy_fields_are_normalized_for_all_formats():
    contract = load_public_io_contract(CONTRACT_PATH)

    for entry in contract["formats"]:
        v3 = entry["v3"]
        assert v3["coverage_status"] in {status.value for status in CoverageStatus}
        assert set(v3["missing_dependency_policy"]) == {"read", "write"}
        assert isinstance(v3["ci_jobs"], list)
        assert all(job in {ci_job.value for ci_job in ContractCIJobId} for job in v3["ci_jobs"])
        assert isinstance(v3["scenario_matrix"], list)
        assert {row["scenario"] for row in v3["scenario_matrix"]} == {
            ScenarioName.POLICY.value,
            ScenarioName.CHECK.value,
            ScenarioName.BLOCKING.value,
        }

    validate_contract_ci_jobs(contract["formats"])


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
        if entry["v3"]["coverage_status"] == CoverageStatus.BLOCKING.value
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
        row["missing_dependency_policy"]["read"] != MissingDependencyPolicy.SKIP.value
        for row in rows
        if row["format"] in optional_public
    )


def test_wav_write_missing_dependency_policy_stays_base_install() -> None:
    contract = load_public_io_contract(CONTRACT_PATH)
    wav = _formats_by_canonical(contract)["wav"]

    assert (
        wav["v3"]["missing_dependency_policy"]["read"]
        == MissingDependencyPolicy.WARNS_AND_SKIPS_OPTIONAL_METADATA.value
    )
    assert (
        wav["v3"]["missing_dependency_policy"]["write"]
        == MissingDependencyPolicy.SKIP.value
    )


def test_core_blocking_formats_have_base_ci_job_and_fixture_generators():
    contract = load_public_io_contract(CONTRACT_PATH)
    formats = _formats_by_canonical(contract)

    expected_generators = {
        "gwf": "gwf",
        "hdf.ndscope": "hdf_ndscope",
        "hdf5": "hdf5",
        "csv": "csv_txt",
        "txt": "csv_txt",
        "wav": "audio",
    }

    for canonical, generator in expected_generators.items():
        entry = formats[canonical]
        assert entry["v3"]["coverage_status"] == CoverageStatus.BLOCKING.value
        assert entry["v3"]["ci_jobs"] == [ContractCIJobId.BASE.value]
        assert entry["v3"]["fixture_generator"] == generator


def test_contract_ci_jobs_do_not_use_internal_stage_ids():
    contract = load_public_io_contract(CONTRACT_PATH)
    internal_stage_ids = set(LOGICAL_CI_JOB_IDS.values())

    for entry in contract["formats"]:
        assert not (set(entry["v3"]["ci_jobs"]) & internal_stage_ids)


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
        MissingDependencyPolicy.MUST_RAISE_IMPORT_ERROR,
        MissingDependencyPolicy.CONDITIONAL_REGISTRATION,
        MissingDependencyPolicy.WARNS_AND_SKIPS_OPTIONAL_METADATA,
    )
    assert LOGICAL_CI_JOB_IDS == {
        ScenarioName.POLICY.value: "io-conformance-policy",
        ScenarioName.CHECK.value: "io-conformance-check",
        ScenarioName.BLOCKING.value: "io-conformance-blocking",
    }
