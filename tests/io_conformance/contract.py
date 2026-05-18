from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION_V2 = 2
SCHEMA_VERSION_V3 = 3

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs/developers/contracts/public_io_contract.json"
)


class ScenarioName(str, Enum):
    POLICY = "policy"
    CHECK = "check"
    BLOCKING = "blocking"


class ScenarioStatus(str, Enum):
    PASS = "pass"
    BLOCKED = "blocked"
    SKIP = "skip"


class LogicalCIJobId(str, Enum):
    POLICY = "io-conformance-policy"
    CHECK = "io-conformance-check"
    BLOCKING = "io-conformance-blocking"


class MissingDependencyPolicy(str, Enum):
    FAIL = "fail"
    SKIP = "skip"
    NOT_PUBLIC = "not_public"
    CONDITIONAL_REGISTRATION = "conditional_registration"


LOGICAL_CI_JOB_IDS = {
    ScenarioName.POLICY.value: LogicalCIJobId.POLICY.value,
    ScenarioName.CHECK.value: LogicalCIJobId.CHECK.value,
    ScenarioName.BLOCKING.value: LogicalCIJobId.BLOCKING.value,
}

_ALLOWED_SCHEMA_VERSIONS = {SCHEMA_VERSION_V2, SCHEMA_VERSION_V3}
_ALLOWED_LOGICAL_CI_JOB_IDS = frozenset(LOGICAL_CI_JOB_IDS.values())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _public_api_is_present(entry: dict[str, Any]) -> bool:
    public_api = entry.get("public_api", {})
    return bool(public_api.get("read") or public_api.get("write"))


def _missing_dependency_policy(entry: dict[str, Any]) -> str:
    if not _public_api_is_present(entry):
        return MissingDependencyPolicy.NOT_PUBLIC.value

    unavailable = entry.get("unavailable_behavior", {})
    if (
        unavailable.get("read") == MissingDependencyPolicy.CONDITIONAL_REGISTRATION.value
        or unavailable.get("write")
        == MissingDependencyPolicy.CONDITIONAL_REGISTRATION.value
    ):
        return MissingDependencyPolicy.CONDITIONAL_REGISTRATION.value

    if entry.get("optional_dependencies"):
        return MissingDependencyPolicy.FAIL.value

    return MissingDependencyPolicy.FAIL.value


def _normalize_entry(entry: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(entry)
    normalized.pop("read_auto_identify", None)
    normalized.pop("write_auto_identify", None)
    normalized.setdefault("aliases", [])
    normalized.setdefault("metadata_requirements", [])
    normalized.setdefault("notes", [])
    normalized.setdefault("optional_dependencies", [])
    normalized.setdefault("extras", [])
    normalized.setdefault("required_args", {"read": [], "write": []})
    normalized.setdefault("direct_api", {"read": [], "write": []})
    normalized.setdefault("registry_api", {"read": [], "write": []})
    normalized.setdefault("public_api", {"read": [], "write": []})
    normalized["public_auto_identify"] = bool(
        normalized.get("public_auto_identify", False)
    )
    normalized["registry_auto_identify"] = bool(
        normalized.get("registry_auto_identify", False)
    )
    normalized["trusted_only"] = bool(normalized.get("trusted_only", False))
    normalized["v3"] = {
        "logical_ci_job_ids": {
            "policy": LogicalCIJobId.POLICY.value,
            "check": LogicalCIJobId.CHECK.value,
            "blocking": LogicalCIJobId.BLOCKING.value,
        },
        "scenario_statuses": {
            "policy": ScenarioStatus.PASS.value,
            "check": ScenarioStatus.PASS.value,
            "blocking": ScenarioStatus.BLOCKED.value,
        },
        "missing_dependency_policy": {
            "read": _missing_dependency_policy(normalized),
            "write": _missing_dependency_policy(normalized),
        },
    }
    return normalized


def load_public_io_contract(path: Path | str = CONTRACT_PATH) -> dict[str, Any]:
    contract_path = Path(path)
    data = _read_json(contract_path)
    schema_version = data.get("schema_version")
    if schema_version not in _ALLOWED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported public I/O contract schema_version: {schema_version!r}"
        )

    formats = data.get("formats", [])
    if not isinstance(formats, list):
        raise TypeError("public I/O contract formats must be a list")

    return {
        "schema_version": SCHEMA_VERSION_V3,
        "formats": [_normalize_entry(entry) for entry in formats],
    }


def validate_logical_ci_job_ids(rows: Iterable[dict[str, Any]]) -> None:
    bad_ids = {
        row.get("logical_ci_job_id")
        for row in rows
        if row.get("logical_ci_job_id") not in _ALLOWED_LOGICAL_CI_JOB_IDS
    }
    if bad_ids:
        raise ValueError(f"invalid logical_ci_job_id values: {sorted(bad_ids)!r}")
