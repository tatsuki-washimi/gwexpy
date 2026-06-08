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


class CoverageStatus(str, Enum):
    BLOCKING = "blocking"
    OPTIONAL_BLOCKING = "optional_blocking"
    NIGHTLY = "nightly"
    PLANNED = "planned"
    NOT_PUBLIC = "not_public"


class ContractCIJobId(str, Enum):
    BASE = "base"
    IO_AUDIO = "io-audio"
    IO_SEISMIC = "io-seismic"
    IO_NETCDF4 = "io-netcdf4"
    IO_ZARR = "io-zarr"
    IO_TDMS = "io-tdms"
    IO_ROOT = "io-root"
    NIGHTLY = "nightly"


class LogicalCIJobId(str, Enum):
    POLICY = "io-conformance-policy"
    CHECK = "io-conformance-check"
    BLOCKING = "io-conformance-blocking"


class MissingDependencyPolicy(str, Enum):
    FAIL = "fail"
    SKIP = "skip"
    NOT_PUBLIC = "not_public"
    MUST_RAISE_IMPORT_ERROR = "must_raise_import_error"
    CONDITIONAL_REGISTRATION = "conditional_registration"
    WARNS_AND_SKIPS_OPTIONAL_METADATA = "warns_and_skips_optional_metadata"


LOGICAL_CI_JOB_IDS = {
    ScenarioName.POLICY.value: LogicalCIJobId.POLICY.value,
    ScenarioName.CHECK.value: LogicalCIJobId.CHECK.value,
    ScenarioName.BLOCKING.value: LogicalCIJobId.BLOCKING.value,
}

_ALLOWED_SCHEMA_VERSIONS = {SCHEMA_VERSION_V2, SCHEMA_VERSION_V3}
_ALLOWED_LOGICAL_CI_JOB_IDS = frozenset(LOGICAL_CI_JOB_IDS.values())
_ALLOWED_CONTRACT_CI_JOB_IDS = frozenset(job.value for job in ContractCIJobId)

_OPTIONAL_CI_JOB_BY_FORMAT = {
    "root": ContractCIJobId.IO_ROOT.value,
    "wav": ContractCIJobId.IO_AUDIO.value,
    "flac": ContractCIJobId.IO_AUDIO.value,
    "ogg": ContractCIJobId.IO_AUDIO.value,
    "mp3": ContractCIJobId.IO_AUDIO.value,
    "m4a": ContractCIJobId.IO_AUDIO.value,
    "tdms": ContractCIJobId.IO_TDMS.value,
    "mseed": ContractCIJobId.IO_SEISMIC.value,
    "sac": ContractCIJobId.IO_SEISMIC.value,
    "gse2": ContractCIJobId.IO_SEISMIC.value,
    "knet": ContractCIJobId.IO_SEISMIC.value,
    "win": ContractCIJobId.IO_SEISMIC.value,
    "ats.mth5": ContractCIJobId.IO_SEISMIC.value,
    "nc": ContractCIJobId.IO_NETCDF4.value,
    "zarr": ContractCIJobId.IO_ZARR.value,
}

_DEFAULT_BLOCKING_FORMATS = frozenset({"gwf", "hdf.ndscope", "hdf5", "csv", "txt", "wav"})
_DEFAULT_FIXTURE_GENERATORS = {
    "gwf": "gwf",
    "hdf.ndscope": "hdf_ndscope",
    "hdf5": "hdf5",
    "csv": "csv_txt",
    "txt": "csv_txt",
    "wav": "audio",
}

_MISSING_DEPENDENCY_POLICY_BY_BEHAVIOR = {
    "available_in_base_install": MissingDependencyPolicy.SKIP.value,
    "raises_import_error": MissingDependencyPolicy.MUST_RAISE_IMPORT_ERROR.value,
    "warns_and_skips_optional_metadata": (
        MissingDependencyPolicy.WARNS_AND_SKIPS_OPTIONAL_METADATA.value
    ),
    "conditional_registration": (
        MissingDependencyPolicy.CONDITIONAL_REGISTRATION.value
    ),
    "not_public": MissingDependencyPolicy.NOT_PUBLIC.value,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _public_api_is_present(entry: dict[str, Any]) -> bool:
    public_api = entry.get("public_api", {})
    return bool(public_api.get("read") or public_api.get("write"))


def _missing_dependency_policy(entry: dict[str, Any], operation: str) -> str:
    public_api = entry.get("public_api", {})
    if not public_api.get(operation):
        return MissingDependencyPolicy.NOT_PUBLIC.value

    unavailable = entry.get("unavailable_behavior", {})
    behavior = unavailable.get(operation)
    if behavior in _MISSING_DEPENDENCY_POLICY_BY_BEHAVIOR:
        return _MISSING_DEPENDENCY_POLICY_BY_BEHAVIOR[behavior]

    if entry.get("optional_dependencies"):
        return MissingDependencyPolicy.MUST_RAISE_IMPORT_ERROR.value

    return MissingDependencyPolicy.SKIP.value


def _io_conformance_policy(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("io_conformance_v3", {})


def _blocking_formats(policy: dict[str, Any]) -> frozenset[str]:
    values = policy.get("blocking_formats", _DEFAULT_BLOCKING_FORMATS)
    return frozenset(values)


def _fixture_generator(entry: dict[str, Any], policy: dict[str, Any]) -> str | None:
    mapping = {
        **_DEFAULT_FIXTURE_GENERATORS,
        **policy.get("fixture_generator_by_format", {}),
    }
    return entry.get("fixture_generator") or mapping.get(entry["canonical"])


def _coverage_status(entry: dict[str, Any], policy: dict[str, Any]) -> str:
    canonical = entry["canonical"]
    if not _public_api_is_present(entry):
        return CoverageStatus.NOT_PUBLIC.value
    if canonical in _blocking_formats(policy):
        return CoverageStatus.BLOCKING.value
    if canonical in _OPTIONAL_CI_JOB_BY_FORMAT and entry.get("optional_dependencies"):
        return CoverageStatus.OPTIONAL_BLOCKING.value
    return CoverageStatus.PLANNED.value


def _ci_jobs(entry: dict[str, Any], coverage_status: str) -> list[str]:
    canonical = entry["canonical"]
    if coverage_status == CoverageStatus.NOT_PUBLIC.value:
        return []
    if coverage_status == CoverageStatus.BLOCKING.value:
        return [ContractCIJobId.BASE.value]
    if canonical in _OPTIONAL_CI_JOB_BY_FORMAT:
        return [_OPTIONAL_CI_JOB_BY_FORMAT[canonical]]
    return [ContractCIJobId.NIGHTLY.value]


def _scenario_matrix(coverage_status: str) -> list[dict[str, str]]:
    blocking_status = (
        ScenarioStatus.BLOCKED.value
        if coverage_status == CoverageStatus.BLOCKING.value
        else ScenarioStatus.SKIP.value
    )
    return [
        {
            "scenario": ScenarioName.POLICY.value,
            "status": ScenarioStatus.PASS.value,
            "logical_ci_job_id": LogicalCIJobId.POLICY.value,
        },
        {
            "scenario": ScenarioName.CHECK.value,
            "status": ScenarioStatus.PASS.value,
            "logical_ci_job_id": LogicalCIJobId.CHECK.value,
        },
        {
            "scenario": ScenarioName.BLOCKING.value,
            "status": blocking_status,
            "logical_ci_job_id": LogicalCIJobId.BLOCKING.value,
        },
    ]


def _normalize_entry(entry: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
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
    coverage_status = _coverage_status(normalized, policy)
    ci_jobs = _ci_jobs(normalized, coverage_status)
    fixture_generator = _fixture_generator(normalized, policy)
    normalized["v3"] = {
        "coverage_status": coverage_status,
        "ci_jobs": ci_jobs,
        "fixture_generator": fixture_generator,
        "logical_ci_job_ids": {
            "policy": LogicalCIJobId.POLICY.value,
            "check": LogicalCIJobId.CHECK.value,
            "blocking": LogicalCIJobId.BLOCKING.value,
        },
        "scenario_matrix": _scenario_matrix(coverage_status),
        "scenario_statuses": {
            "policy": ScenarioStatus.PASS.value,
            "check": ScenarioStatus.PASS.value,
            "blocking": (
                ScenarioStatus.BLOCKED.value
                if coverage_status == CoverageStatus.BLOCKING.value
                else ScenarioStatus.SKIP.value
            ),
        },
        "missing_dependency_policy": {
            "read": _missing_dependency_policy(normalized, "read"),
            "write": _missing_dependency_policy(normalized, "write"),
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
    policy = _io_conformance_policy(data)

    normalized = {
        "schema_version": SCHEMA_VERSION_V3,
        "io_conformance_v3": policy,
        "formats": [_normalize_entry(entry, policy) for entry in formats],
    }
    validate_contract_ci_jobs(normalized["formats"])
    return normalized


def validate_contract_ci_jobs(entries: Iterable[dict[str, Any]]) -> None:
    bad_ids = {
        ci_job
        for entry in entries
        for ci_job in entry.get("v3", {}).get("ci_jobs", [])
        if ci_job not in _ALLOWED_CONTRACT_CI_JOB_IDS
    }
    if bad_ids:
        raise ValueError(f"invalid contract ci_jobs values: {sorted(bad_ids)!r}")


def validate_logical_ci_job_ids(rows: Iterable[dict[str, Any]]) -> None:
    bad_ids = {
        row.get("logical_ci_job_id")
        for row in rows
        if row.get("logical_ci_job_id") not in _ALLOWED_LOGICAL_CI_JOB_IDS
    }
    if bad_ids:
        raise ValueError(f"invalid logical_ci_job_id values: {sorted(bad_ids)!r}")
