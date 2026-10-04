"""Create and validate immutable candidate promotion manifests."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "gwexpy-release-promotion-manifest-v1"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FIELDS = {
    "schema",
    "repository",
    "version",
    "tag",
    "workflow_id",
    "workflow_path",
    "event",
    "dispatch_ref",
    "workflow_ref",
    "workflow_sha",
    "source_sha",
    "release_sha",
    "run_id",
    "run_attempt",
    "promotion_contract",
    "review_evidence",
    "release_notes",
    "gates",
    "artifacts",
}
EXPECTED_FIELDS = {
    "repository",
    "version",
    "tag",
    "workflow_id",
    "workflow_path",
    "event",
    "dispatch_ref",
    "workflow_ref",
    "workflow_sha",
    "source_sha",
    "release_sha",
    "run_id",
    "review_evidence_sha256",
    "release_notes_sha256",
}


class PromotionManifestError(ValueError):
    """Raised when manifest or Actions artifact data violates the contract."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PromotionManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def serialize_manifest(value: Mapping[str, Any]) -> bytes:
    """Serialize canonical UTF-8 JSON with sorted keys and one final LF."""
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PromotionManifestError(
            f"manifest is not canonical JSON data: {exc}"
        ) from exc


def load_manifest(raw: bytes | str) -> dict[str, Any]:
    """Load JSON while rejecting ambiguous duplicate object keys."""
    try:
        encoded = raw.encode("utf-8") if isinstance(raw, str) else raw
        value = json.loads(encoded, object_pairs_hook=_pairs)
    except (json.JSONDecodeError, UnicodeDecodeError, UnicodeEncodeError) as exc:
        raise PromotionManifestError(f"invalid manifest JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PromotionManifestError("manifest root must be an object")
    if encoded != serialize_manifest(value):
        raise PromotionManifestError("manifest bytes are not canonical")
    return value


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _parse_fixed_record(
    body: str | bytes, header: str, fields: Sequence[str]
) -> dict[str, str]:
    """Parse a fixed UTF-8/LF record, rejecting all noncanonical forms."""
    if isinstance(body, bytes):
        try:
            body = body.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise PromotionManifestError("record body is not valid UTF-8") from exc
    if not isinstance(body, str):
        raise PromotionManifestError("record body must be UTF-8 text")
    try:
        body.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise PromotionManifestError("record body is not valid UTF-8") from exc
    lines = body.split("\n")
    if "\r" in body or len(lines) != len(fields) + 1 or lines[0] != header:
        raise PromotionManifestError("record body is not canonical")
    result: dict[str, str] = {}
    for line, expected_key in zip(lines[1:], fields, strict=True):
        if "=" not in line:
            raise PromotionManifestError("record field is malformed")
        key, value = line.split("=", 1)
        if key != expected_key or key in result or not value:
            raise PromotionManifestError("record field order or set is invalid")
        result[key] = value
    return result


def _record_value(
    record: Mapping[str, str], key: str, pattern: re.Pattern[str]
) -> None:
    if pattern.fullmatch(record[key]) is None:
        raise PromotionManifestError(f"invalid {key} in record")


def parse_release_go(body: str) -> dict[str, str]:
    """Parse the canonical release-owner GO body."""
    fields = (
        "version",
        "source_sha",
        "candidate_run_id",
        "promotion_manifest_sha256",
        "sdist_sha256",
        "wheel_sha256",
        "decision",
    )
    record = _parse_fixed_record(body, "GWEXPY-RELEASE-GO-v1", fields)
    if re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", record["version"]) is None:
        raise PromotionManifestError("invalid GO version")
    _record_value(record, "source_sha", SHA40)
    _record_value(record, "candidate_run_id", re.compile(r"[1-9][0-9]*"))
    for key in ("promotion_manifest_sha256", "sdist_sha256", "wheel_sha256"):
        _record_value(record, key, SHA256)
    if record["decision"] != "GO":
        raise PromotionManifestError("release decision must be GO")
    return record


def parse_promotion_tag(body: str) -> dict[str, str]:
    """Parse the canonical annotated promotion tag body."""
    fields = (
        "repository",
        "tag",
        "source_sha",
        "candidate_run_id",
        "promotion_manifest_sha256",
        "release_go_comment_id",
    )
    record = _parse_fixed_record(body, "GWEXPY-PROMOTION-v1", fields)
    if record["repository"] != "tatsuki-washimi/gwexpy":
        raise PromotionManifestError("promotion tag repository mismatch")
    if re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", record["tag"]) is None:
        raise PromotionManifestError("invalid promotion tag name")
    _record_value(record, "source_sha", SHA40)
    _record_value(record, "candidate_run_id", re.compile(r"[1-9][0-9]*"))
    _record_value(record, "promotion_manifest_sha256", SHA256)
    _record_value(record, "release_go_comment_id", re.compile(r"[1-9][0-9]*"))
    return record


def validate_promotion_tag(
    record: Mapping[str, str],
    *,
    annotated: bool,
    tag_name: str,
    target_sha: str,
    repository: str,
    candidate_run_id: int,
    manifest_sha256: str,
    release_go_comment_id: int,
) -> None:
    """Bind parsed tag metadata to an annotated object and its peeled commit."""
    if not annotated:
        raise PromotionManifestError("promotion requires an annotated tag")
    if (
        not _positive_int(candidate_run_id)
        or not _positive_int(release_go_comment_id)
        or not isinstance(manifest_sha256, str)
        or SHA256.fullmatch(manifest_sha256) is None
    ):
        raise PromotionManifestError("invalid expected tag binding")
    if (
        record.get("tag") != tag_name
        or record.get("repository") != repository
        or record.get("source_sha") != target_sha
        or record.get("candidate_run_id") != str(candidate_run_id)
        or record.get("promotion_manifest_sha256") != manifest_sha256
        or record.get("release_go_comment_id") != str(release_go_comment_id)
    ):
        raise PromotionManifestError("annotated tag binding or target mismatch")
    _record_value(record, "source_sha", SHA40)


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _safe_filename(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and path.name == value
        and all(part not in {"", ".", ".."} for part in path.parts)
        and str(path) == value
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PromotionManifestError(message)


def validate_manifest(
    manifest: Mapping[str, Any],
    release_contract: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    run_attempt: int | None = None,
    jobs: Sequence[Mapping[str, Any]] | None = None,
    overall_run: Mapping[str, Any] | None = None,
    artifact_api: Sequence[Mapping[str, Any]] | None = None,
    observed_files: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
    declared_package_hashes: Mapping[str, str] | None = None,
) -> None:
    """Validate exact schema, source binding, contract, and completed gate jobs."""
    _require(
        isinstance(manifest, Mapping) and set(manifest) == FIELDS,
        "manifest has unknown or missing field",
    )
    _require(run_attempt is not None, "actual run attempt observation is required")
    _require(
        _positive_int(run_attempt),
        "actual run attempt must be a positive integer",
    )
    _require(jobs is not None, "actual job observations are required")
    _require(artifact_api is not None, "actual artifact API observations are required")
    _require(
        observed_files is not None, "actual artifact file observations are required"
    )
    _require(
        declared_package_hashes is not None, "distribution sidecar hashes are required"
    )
    _require(
        set(expected) == EXPECTED_FIELDS,
        "expected context has unknown or missing expected context fields",
    )
    _require(manifest["schema"] == SCHEMA, "unsupported promotion manifest schema")
    _require(
        manifest.get("tag") == f"v{manifest.get('version')}",
        "manifest tag/version mismatch",
    )
    for key, value in expected.items():
        if key in {"review_evidence_sha256", "release_notes_sha256"}:
            continue
        _require(
            manifest.get(key) == value,
            f"manifest {key} does not match expected metadata",
        )
    for key in ("workflow_sha", "source_sha", "release_sha", "dispatch_ref"):
        _require(
            isinstance(manifest[key], str)
            and SHA40.fullmatch(manifest[key]) is not None,
            f"invalid full SHA in {key}",
        )
    for key in ("run_id", "workflow_id", "run_attempt"):
        _require(_positive_int(manifest[key]), f"invalid positive {key}")
    _require(manifest["run_attempt"] == 1, "promotion manifest requires run attempt 1")
    promotion = release_contract.get("promotion")
    _require(
        isinstance(promotion, Mapping), "release contract has no promotion contract"
    )
    _require(
        manifest["workflow_path"] == promotion.get("workflow_path"),
        "workflow path differs from promotion contract",
    )
    expected_promotion = {
        "id": promotion.get("schema"),
        "sha256": sha256(serialize_manifest(promotion)),
    }
    _require(
        manifest["promotion_contract"] == expected_promotion,
        "promotion contract binding mismatch",
    )
    review = manifest["review_evidence"]
    _require(
        isinstance(review, Mapping) and set(review) == {"path", "sha256"},
        "invalid review evidence descriptor",
    )
    _require(
        review.get("path") == release_contract.get("review_evidence_path"),
        "review evidence path mismatch",
    )
    _require(
        isinstance(review.get("sha256"), str)
        and SHA256.fullmatch(review["sha256"]) is not None,
        "invalid review evidence hash",
    )
    notes = manifest["release_notes"]
    _require(
        isinstance(notes, Mapping) and set(notes) == {"path", "sha256"},
        "invalid release notes descriptor",
    )
    _require(
        notes.get("path") == f"release_notes/v{manifest['version']}.md",
        "release notes path mismatch",
    )
    _require(
        isinstance(notes.get("sha256"), str)
        and SHA256.fullmatch(notes["sha256"]) is not None,
        "invalid release notes hash",
    )
    for expected_key, descriptor, hash_key in (
        ("review_evidence_sha256", review, "sha256"),
        ("release_notes_sha256", notes, "sha256"),
    ):
        _require(
            isinstance(expected[expected_key], str)
            and SHA256.fullmatch(expected[expected_key]) is not None,
            f"invalid expected {expected_key}",
        )
        _require(
            descriptor[hash_key] == expected[expected_key],
            f"{expected_key} mismatch",
        )
    required_jobs = release_contract.get("promotion", {}).get("required_jobs", [])
    gates = manifest["gates"]
    _require(
        isinstance(gates, list)
        and all(
            isinstance(g, Mapping) and set(g) == {"job", "conclusion"} for g in gates
        ),
        "invalid gate evidence",
    )
    _require(
        [g["job"] for g in gates] == required_jobs
        and all(g["conclusion"] == "success" for g in gates),
        "missing, extra, or unsuccessful required gate evidence",
    )
    _require(run_attempt == 1, "promotion manifest requires run attempt 1")
    _require(
        manifest["run_attempt"] == run_attempt,
        "manifest run attempt differs from observation",
    )
    _require(
        all(
            isinstance(job, Mapping)
            and isinstance(job.get("name"), str)
            and isinstance(job.get("status"), str)
            and (
                job.get("conclusion") is None or isinstance(job.get("conclusion"), str)
            )
            for job in jobs
        ),
        "invalid actual job observation",
    )
    job_names = [job["name"] for job in jobs]
    _require(len(job_names) == len(set(job_names)), "duplicate job observation")
    _require(
        set(job_names) == set(required_jobs),
        "actual job observations have missing or extra jobs",
    )
    by_name = {job.get("name"): job for job in jobs}
    for name in required_jobs:
        job = by_name[name]
        _require(
            job.get("status") == "completed" and job.get("conclusion") == "success",
            f"required gate {name} did not complete successfully",
        )
    # The enclosing workflow run is intentionally not required to be complete.
    del overall_run
    required_roles = {
        "payload",
        "sidecar:distribution-sha256.json",
        "sidecar:LICENSE.sha256",
    }
    required_roles.update(
        f"evidence:{schema}" for schema in promotion.get("evidence_schema_ids", [])
    )
    _require(isinstance(manifest["artifacts"], list), "artifacts must be a list")
    roles = [
        artifact.get("role")
        for artifact in manifest["artifacts"]
        if isinstance(artifact, Mapping)
    ]
    _require(len(roles) == len(manifest["artifacts"]), "invalid artifact record")
    _require(len(roles) == len(set(roles)), "duplicate artifact roles")
    _require(
        set(roles) == required_roles and len(roles) == len(required_roles),
        "missing or extra required artifacts/roles",
    )
    validate_artifact_records(
        manifest["artifacts"],
        artifact_api,
        observed_files,
        manifest["run_id"],
    )
    payload_record = next(a for a in manifest["artifacts"] if a["role"] == "payload")
    validate_package_hashes(payload_record["files"], declared_package_hashes)


def validate_artifact_records(
    records: Sequence[Mapping[str, Any]],
    api_records: Sequence[Mapping[str, Any]],
    observed_files: Mapping[int, Sequence[Mapping[str, Any]]],
    run_id: int,
) -> None:
    """Compare manifest records with independent API and extracted-file observations."""
    _require(len(records) == len(api_records), "artifact API observation set mismatch")
    _require(
        all(isinstance(record, Mapping) for record in records),
        "invalid artifact record",
    )
    ids = [record.get("artifact_id") for record in records]
    names = [record.get("name") for record in records]
    _require(
        all(_positive_int(artifact_id) for artifact_id in ids)
        and all(isinstance(name, str) for name in names),
        "invalid artifact identity",
    )
    _require(
        len(ids) == len(set(ids)) and len(names) == len(set(names)),
        "duplicate artifact ID or name",
    )
    _require(
        all(isinstance(record, Mapping) for record in api_records),
        "invalid artifact API observation",
    )
    _require(
        all(_positive_int(record.get("artifact_id")) for record in api_records),
        "invalid API artifact ID",
    )
    api_by_id = {record.get("artifact_id"): record for record in api_records}
    _require(len(api_by_id) == len(api_records), "duplicate API artifact IDs")
    _require(set(observed_files) == set(ids), "artifact file observation set mismatch")
    for record in records:
        fields = {
            "role",
            "artifact_id",
            "name",
            "digest",
            "size_in_bytes",
            "run_id",
            "expired",
            "files",
        }
        _require(
            isinstance(record, Mapping) and set(record) == fields,
            "invalid artifact metadata fields",
        )
        for key in ("artifact_id", "size_in_bytes", "run_id"):
            _require(_positive_int(record[key]), f"invalid artifact {key}")
        _require(record["run_id"] == run_id, "artifact run ID mismatch")
        _require(record["expired"] is False, "artifact is expired")
        _require(
            isinstance(record["digest"], str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", record["digest"]) is not None,
            "invalid artifact digest",
        )
        api = api_by_id.get(record["artifact_id"])
        _require(api is not None, "manifest artifact ID absent from API observations")
        _require(
            all(
                record.get(k) == api.get(k)
                for k in (
                    "artifact_id",
                    "name",
                    "digest",
                    "size_in_bytes",
                    "run_id",
                    "expired",
                )
            ),
            "artifact API metadata mismatch",
        )
        files = record["files"]
        _require(isinstance(files, list) and files, "artifact file list is empty")
        names: set[str] = set()
        for item in files:
            _require(
                isinstance(item, Mapping)
                and set(item) == {"name", "size_in_bytes", "sha256"},
                "invalid artifact file record",
            )
            _require(_safe_filename(item["name"]), "unsafe artifact filename")
            _require(item["name"] not in names, "duplicate artifact filename")
            names.add(item["name"])
            _require(_positive_int(item["size_in_bytes"]), "invalid artifact file size")
            _require(
                isinstance(item["sha256"], str)
                and SHA256.fullmatch(item["sha256"]) is not None,
                "invalid artifact file hash",
            )
        observed = observed_files[record["artifact_id"]]
        _require(
            isinstance(observed, Sequence)
            and not isinstance(observed, (str, bytes))
            and all(isinstance(item, Mapping) for item in observed),
            "invalid artifact file observation",
        )
        _require(
            all(
                set(item) == {"name", "size_in_bytes", "sha256"}
                and _safe_filename(item.get("name"))
                and _positive_int(item.get("size_in_bytes"))
                and isinstance(item.get("sha256"), str)
                and SHA256.fullmatch(item["sha256"]) is not None
                for item in observed
            ),
            "invalid artifact file observation",
        )
        _require(
            sorted(files, key=lambda item: item["name"].encode("utf-8"))
            == sorted(observed, key=lambda item: item.get("name", "").encode("utf-8")),
            "artifact file observation mismatch",
        )


def validate_package_hashes(
    payload_files: Sequence[Mapping[str, Any]], declared_hashes: Mapping[str, str]
) -> None:
    """Require sidecar package hashes to equal the re-hashed payload files."""
    actual = {item.get("name"): item.get("sha256") for item in payload_files}
    _require(
        bool(declared_hashes) and dict(declared_hashes) == actual,
        "package hashes mismatch",
    )


def validate_manifest_artifact(
    api: Mapping[str, Any],
    archive_bytes: bytes,
    manifest_bytes: bytes,
    release_sha: str,
    run_id: int,
    expected_manifest_sha256: str,
) -> None:
    """Validate archive API metadata and contained manifest bytes independently."""
    _require(
        api.get("name") == f"release-promotion-manifest-{release_sha}",
        "promotion manifest artifact name mismatch",
    )
    _require(
        _positive_int(api.get("artifact_id")), "invalid promotion manifest artifact ID"
    )
    _require(
        api.get("run_id") == run_id and _positive_int(run_id),
        "promotion manifest artifact run ID mismatch",
    )
    _require(api.get("expired") is False, "promotion manifest artifact is expired")
    _require(
        _positive_int(api.get("size_in_bytes")),
        "promotion manifest artifact size must be a positive integer",
    )
    _require(
        api.get("size_in_bytes") == len(archive_bytes),
        "promotion manifest artifact size mismatch",
    )
    _require(
        api.get("digest") == f"sha256:{sha256(archive_bytes)}",
        "promotion manifest artifact digest mismatch",
    )
    _require(
        isinstance(expected_manifest_sha256, str)
        and SHA256.fullmatch(expected_manifest_sha256) is not None,
        "invalid expected promotion manifest hash",
    )
    _require(
        sha256(manifest_bytes) == expected_manifest_sha256,
        "promotion manifest content hash mismatch",
    )
    load_manifest(manifest_bytes)


def select_manifest_artifact(
    artifacts: Sequence[Mapping[str, Any]], release_sha: str, run_id: int
) -> Mapping[str, Any]:
    """Select exactly one fixed-name promotion artifact from this candidate run."""
    name = f"release-promotion-manifest-{release_sha}"
    matches = [
        artifact
        for artifact in artifacts
        if artifact.get("name") == name and artifact.get("run_id") == run_id
    ]
    _require(len(matches) == 1, "missing or ambiguous promotion manifest artifact")
    return matches[0]


def describe_existing_artifact(
    api: Mapping[str, Any], files: Mapping[str, Path]
) -> dict[str, Any]:
    """Re-hash existing files while retaining their Actions artifact identity."""
    entries = []
    for name, path in sorted(files.items(), key=lambda item: item[0].encode("utf-8")):
        _require(_safe_filename(name), "unsafe artifact filename")
        content = path.read_bytes()
        entries.append(
            {"name": name, "size_in_bytes": len(content), "sha256": sha256(content)}
        )
    return {
        "role": api.get("role"),
        "artifact_id": api.get("artifact_id"),
        "name": api.get("name"),
        "digest": api.get("digest"),
        "size_in_bytes": api.get("size_in_bytes"),
        "run_id": api.get("run_id"),
        "expired": api.get("expired"),
        "files": entries,
    }


def build_manifest(
    metadata: Mapping[str, Any],
    release_contract: Mapping[str, Any],
    artifact_inputs: Sequence[tuple[str, Mapping[str, Any], Mapping[str, Path]]],
    review_evidence_path: Path,
    release_notes_path: Path,
) -> dict[str, Any]:
    """Build manifest data from existing files and API metadata, without writes.

    Artifact inputs refer to already uploaded payload, sidecar, and evidence
    artifacts. This function only reads those files; the returned manifest is
    the sole artifact the caller should upload.
    """
    promotion = release_contract.get("promotion")
    _require(
        isinstance(promotion, Mapping), "release contract has no promotion contract"
    )
    artifacts = [
        describe_existing_artifact({**api, "role": role}, files)
        for role, api, files in artifact_inputs
    ]
    result = {
        **metadata,
        "schema": SCHEMA,
        "promotion_contract": {
            "id": promotion["schema"],
            "sha256": sha256(serialize_manifest(promotion)),
        },
        "review_evidence": {
            "path": release_contract["review_evidence_path"],
            "sha256": sha256(review_evidence_path.read_bytes()),
        },
        "release_notes": {
            "path": f"release_notes/v{metadata['version']}.md",
            "sha256": sha256(release_notes_path.read_bytes()),
        },
        "artifacts": artifacts,
    }
    return result
