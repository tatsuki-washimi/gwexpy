from __future__ import annotations

from typing import Any

from .contract import (
    LOGICAL_CI_JOB_IDS,
    CoverageStatus,
    ScenarioName,
    ScenarioStatus,
    validate_logical_ci_job_ids,
)


def _is_core_public_format(entry: dict[str, Any]) -> bool:
    return entry["v3"]["coverage_status"] == CoverageStatus.BLOCKING.value


def _scenario_row(
    entry: dict[str, Any],
    scenario: ScenarioName,
    status: ScenarioStatus,
) -> dict[str, Any]:
    return {
        "format": entry["canonical"],
        "scenario": scenario.value,
        "status": status.value,
        "coverage_status": entry["v3"]["coverage_status"],
        "logical_ci_job_id": LOGICAL_CI_JOB_IDS[scenario.value],
        "ci_jobs": entry["v3"]["ci_jobs"],
        "fixture_generator": entry["v3"]["fixture_generator"],
        "missing_dependency_policy": entry["v3"]["missing_dependency_policy"],
        "public_auto_identify": entry["public_auto_identify"],
        "registry_auto_identify": entry["registry_auto_identify"],
        "trusted_only": entry["trusted_only"],
    }


def expand_contract_scenarios(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in contract["formats"]:
        rows.append(_scenario_row(entry, ScenarioName.POLICY, ScenarioStatus.PASS))
        rows.append(_scenario_row(entry, ScenarioName.CHECK, ScenarioStatus.PASS))
        if _is_core_public_format(entry):
            rows.append(
                _scenario_row(entry, ScenarioName.BLOCKING, ScenarioStatus.BLOCKED)
            )

    validate_logical_ci_job_ids(rows)
    return rows
