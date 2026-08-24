"""Load strict, versioned release-control contracts."""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

CONTRACT_PATH = Path(__file__).with_name("release_contracts.json")
CONTRACT_SCHEMA = "gwexpy-release-contracts-v1"
RELEASE_TAG = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
ARTIFACT_PREFIX = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
PROTECTED_REF = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._/-]*[A-Za-z0-9])?$")
CONTRACT_KEYS = {
    "plan_path",
    "review_evidence_path",
    "review_evidence_schema",
    "payload_schema",
    "integration_evidence_schema",
    "review_lanes",
    "s_to_r_allowed_paths",
    "artifact_prefix",
    "protected_refs",
}


class ReleaseContractError(ValueError):
    """Raised when release-control configuration is invalid or unsupported."""


class _DuplicateJSONKey(ValueError):
    """Raised when a release contract contains an ambiguous JSON object."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJSONKey(key)
        result[key] = value
    return result


def _safe_path(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not value.startswith("/")
        and ".." not in Path(value).parts
    )


def _sorted_unique(values: object) -> bool:
    return (
        isinstance(values, list)
        and bool(values)
        and all(_safe_path(value) for value in values)
        and values == sorted(set(values), key=lambda item: item.encode("utf-8"))
    )


def _protected_refs(values: object) -> bool:
    if (
        not isinstance(values, list)
        or len(values) != 2
        or not all(isinstance(value, str) for value in values)
    ):
        return False
    return (
        values == sorted(set(values), key=lambda item: item.encode("utf-8"))
        and "main" in values
        and all(
            PROTECTED_REF.fullmatch(value) is not None
            and not value.startswith("refs/")
            and ".." not in value
            and "@{" not in value
            for value in values
        )
    )


def _validate_contract(tag: str, contract: object) -> dict[str, Any]:
    if not isinstance(contract, dict) or set(contract) != CONTRACT_KEYS:
        raise ReleaseContractError(f"invalid release contract for {tag}")
    if not _safe_path(contract["plan_path"]) or not _safe_path(
        contract["review_evidence_path"]
    ):
        raise ReleaseContractError(f"invalid release path for {tag}")
    for field in (
        "review_evidence_schema",
        "payload_schema",
        "integration_evidence_schema",
    ):
        if not isinstance(contract[field], str) or not contract[field]:
            raise ReleaseContractError(f"invalid {field} for {tag}")
    artifact_prefix = contract["artifact_prefix"]
    if (
        not isinstance(artifact_prefix, str)
        or ARTIFACT_PREFIX.fullmatch(artifact_prefix) is None
    ):
        raise ReleaseContractError(f"invalid artifact_prefix for {tag}")
    lanes = contract["review_lanes"]
    if (
        not isinstance(lanes, dict)
        or not lanes
        or any(not isinstance(lane, str) or not lane for lane in lanes)
        or any(not _sorted_unique(paths) for paths in lanes.values())
    ):
        raise ReleaseContractError(f"invalid review lanes for {tag}")
    if not _sorted_unique(contract["s_to_r_allowed_paths"]):
        raise ReleaseContractError(f"invalid S-to-R paths for {tag}")
    if contract["plan_path"] not in contract["s_to_r_allowed_paths"]:
        raise ReleaseContractError(f"S-to-R paths omit the plan for {tag}")
    if contract["review_evidence_path"] not in contract["s_to_r_allowed_paths"]:
        raise ReleaseContractError(f"S-to-R paths omit review evidence for {tag}")
    if not _protected_refs(contract["protected_refs"]):
        raise ReleaseContractError(f"invalid protected refs for {tag}")
    return contract


def _load_contracts() -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(
            CONTRACT_PATH.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError, _DuplicateJSONKey) as exc:
        raise ReleaseContractError("invalid release contracts document") from exc
    if (
        not isinstance(data, dict)
        or set(data) != {"schema", "releases"}
        or data["schema"] != CONTRACT_SCHEMA
        or not isinstance(data["releases"], dict)
        or not data["releases"]
    ):
        raise ReleaseContractError("invalid release contracts document")
    releases: dict[str, dict[str, Any]] = {}
    for tag, contract in data["releases"].items():
        if not isinstance(tag, str) or RELEASE_TAG.fullmatch(tag) is None:
            raise ReleaseContractError("release contracts contain an invalid tag")
        releases[tag] = _validate_contract(tag, contract)
    return releases


def release_contract(tag: str) -> dict[str, Any]:
    """Return a defensive copy of the exact contract for *tag*.

    SemVer syntax alone is not authorization: unlisted releases fail closed.
    """
    try:
        contract = _load_contracts()[tag]
    except KeyError as exc:
        raise ReleaseContractError(f"unsupported release tag: {tag}") from exc
    return copy.deepcopy(contract)


def protected_refs(tag: str) -> list[str]:
    """Return the validated protected branch names for an exact release tag."""
    return list(release_contract(tag)["protected_refs"])


def main(argv: list[str] | None = None) -> int:
    """Print exact-tag protected refs for workflow consumption."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protected-ref", metavar="EXPECTED_TAG", required=True)
    args = parser.parse_args(argv)
    try:
        print(*protected_refs(args.protected_ref), sep="\n")
    except ReleaseContractError as exc:
        print(f"release contract failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
