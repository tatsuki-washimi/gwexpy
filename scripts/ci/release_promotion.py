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
    if body.endswith("\n"):
        body = body[:-1]
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


def parse_promotion_tag(body: str | bytes) -> dict[str, str]:
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


def validate_candidate_run(
    manifest: Mapping[str, Any],
    release_contract: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
    *,
    repository: str,
    tag: str,
    source_sha: str,
) -> None:
    """Bind a completed dispatch run to its manifest and annotated release tag."""
    promotion = release_contract.get("promotion")
    _require(isinstance(promotion, Mapping), "release contract has no promotion block")
    workflow_path = promotion.get("workflow_path")
    _require(
        isinstance(workflow_path, str)
        and workflow_path.startswith(".github/workflows/"),
        "promotion contract workflow path is invalid",
    )
    _require(isinstance(candidate_run, Mapping), "candidate run API record is invalid")
    _require(
        manifest.get("repository") == repository,
        "manifest repository does not match tag repository",
    )
    _require(
        isinstance(tag, str)
        and tag == f"v{manifest.get('version')}"
        and manifest.get("tag") == tag,
        "manifest version does not match release tag",
    )
    _require(
        isinstance(source_sha, str)
        and SHA40.fullmatch(source_sha) is not None
        and manifest.get("source_sha") == source_sha
        and manifest.get("release_sha") == source_sha
        and manifest.get("dispatch_ref") == source_sha,
        "manifest source SHA does not match annotated tag target",
    )
    _require(
        manifest.get("workflow_path") == workflow_path,
        "manifest workflow path differs from promotion contract",
    )
    _require(
        candidate_run.get("id") == manifest.get("run_id")
        and _positive_int(candidate_run.get("id")),
        "candidate run ID differs from manifest",
    )
    _require(
        candidate_run.get("workflow_id") == manifest.get("workflow_id")
        and _positive_int(candidate_run.get("workflow_id")),
        "candidate workflow ID differs from manifest",
    )
    branch = candidate_run.get("head_branch")
    _require(branch == "main", "candidate workflow ref must be main")
    expected_workflow_ref = f"{repository}/{workflow_path}@refs/heads/{branch}"
    _require(
        manifest.get("workflow_ref") == expected_workflow_ref,
        "manifest workflow ref differs from candidate run",
    )
    _require(
        candidate_run.get("path") == f"{workflow_path}@{branch}"
        and candidate_run.get("event") == "workflow_dispatch"
        and manifest.get("event") == candidate_run.get("event"),
        "candidate workflow path or event differs from contract",
    )
    _require(
        candidate_run.get("head_sha") == manifest.get("workflow_sha")
        and isinstance(candidate_run.get("head_sha"), str)
        and SHA40.fullmatch(candidate_run["head_sha"]) is not None,
        "candidate workflow SHA differs from manifest",
    )
    _require(
        candidate_run.get("status") == "completed"
        and candidate_run.get("conclusion") == "success",
        "candidate run must be completed successfully",
    )
    _require(
        type(candidate_run.get("run_attempt")) is int
        and candidate_run["run_attempt"] == 1
        and manifest.get("run_attempt") == 1,
        "candidate run must use attempt 1",
    )


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
        validate_artifact_api_metadata(record, api, run_id)
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


def validate_artifact_api_metadata(
    record: Mapping[str, Any], api: Mapping[str, Any], run_id: int
) -> None:
    """Compare a manifest-selected artifact with its API record before download."""
    for key in ("artifact_id", "size_in_bytes", "run_id"):
        _require(_positive_int(api.get(key)), f"invalid API artifact {key}")
    _require(api.get("run_id") == run_id, "artifact API run ID mismatch")
    _require(api.get("expired") is False, "artifact API record is expired")
    _require(
        isinstance(api.get("name"), str)
        and isinstance(api.get("digest"), str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", api["digest"]) is not None,
        "invalid artifact API name or digest",
    )
    _require(
        all(
            record.get(key) == api.get(key)
            for key in (
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
    """Validate API archive metadata and bind the sole member to manifest bytes."""
    validate_manifest_artifact_metadata(api, release_sha, run_id)
    _require(
        api.get("size_in_bytes") == len(archive_bytes),
        "promotion manifest artifact size mismatch",
    )
    _require(
        api.get("digest") == f"sha256:{sha256(archive_bytes)}",
        "promotion manifest artifact digest mismatch",
    )
    members = observe_archive(api, archive_bytes, run_id)
    _require(
        set(members) == {"promotion-manifest.json"},
        "promotion manifest archive must contain exactly promotion-manifest.json",
    )
    _require(
        members["promotion-manifest.json"] == manifest_bytes,
        "promotion manifest bytes differ from archive member",
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


def validate_manifest_artifact_metadata(
    api: Mapping[str, Any], release_sha: str, run_id: int
) -> None:
    """Check fixed manifest artifact API identity before downloading its archive."""
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
        isinstance(api.get("digest"), str)
        and re.fullmatch(r"sha256:[0-9a-f]{64}", api["digest"]) is not None,
        "promotion manifest artifact digest is malformed",
    )


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


def _local_module(name: str) -> Any:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(name + ".py")
    )
    _require(spec is not None and spec.loader is not None, "release helper unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_candidate_aggregate(
    evidence: Mapping[str, Any],
    schema: str,
    contract: Mapping[str, Any],
    metadata: Mapping[str, Any],
    payload: Mapping[str, Any],
    license_digest: str,
) -> None:
    """Re-run existing cell aggregators and require their exact allowlisted output."""
    import tempfile
    from types import SimpleNamespace

    profile = candidate_profile(contract)
    version = metadata["version"]
    source = metadata["source_sha"]
    try:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            sidecars = directory / "sidecars"
            sidecars.mkdir()
            manifest = sidecars / "distribution-sha256.json"
            manifest.write_bytes(serialize_manifest(payload))
            reports = directory / "reports"
            reports.mkdir()
            output = directory / "aggregate.json"
            if schema == "gwexpy-integration-evidence-v1":
                module = _local_module("assemble_release_evidence")
                module._release_contract = lambda tag: dict(contract)
                (sidecars / "LICENSE.sha256").write_text(
                    license_digest + "\n", encoding="ascii"
                )
                for name, report in evidence["smoke"].items():
                    _require(_safe_filename(name), "unsafe smoke evidence cell")
                    (reports / (name + ".json")).write_bytes(serialize_manifest(report))
                rebuilt = module.assemble_evidence(
                    manifest,
                    sidecars,
                    reports,
                    source,
                    metadata["repository"],
                    str(metadata["run_id"]),
                    metadata["workflow_sha"],
                    metadata["workflow_ref"],
                    metadata["tag"],
                )
                rebuilt["schema"] = profile["evidence_schemas"][schema]
                rebuilt["review_evidence"] = evidence["review_evidence"]
            else:
                names = {
                    "gwexpy-qualification-evidence-v1": (
                        "qualification_evidence",
                        "qualification.json",
                    ),
                    "gwexpy-diaggui-qualification-evidence-v1": (
                        "diaggui_qualification_evidence",
                        "diaggui-qualification.json",
                    ),
                    "gwexpy-cross-format-io-evidence-v1": (
                        "v025_cross_format_io_evidence",
                        "cross-format-io.json",
                    ),
                }
                module_name, filename = names[schema]
                module = _local_module(module_name)
                module._promotion_module = lambda: SimpleNamespace(
                    configured_contract=lambda value: contract,
                    candidate_profile=lambda value: profile,
                )
                if schema != "gwexpy-qualification-evidence-v1":
                    module._select_version(version)
                for index, cell in enumerate(evidence["cells"]):
                    _require(isinstance(cell, Mapping), "invalid aggregate cell")
                    if schema == "gwexpy-qualification-evidence-v1":
                        report = {
                            **cell,
                            "baseline_sha256": evidence["baseline_sha256"],
                            "files": payload["files"],
                            "source_sha": source,
                            "status": "passed",
                            "version": version,
                        }
                    elif schema == "gwexpy-diaggui-qualification-evidence-v1":
                        report = {
                            **cell,
                            "schema": module.CELL_SCHEMA,
                            "source_sha": source,
                            "version": version,
                        }
                    else:
                        report = cell
                    cell_dir = reports / str(index)
                    cell_dir.mkdir()
                    if schema == "gwexpy-qualification-evidence-v1":
                        raw = module._canonical_json_bytes(report)
                    elif schema == "gwexpy-diaggui-qualification-evidence-v1":
                        raw = module._canonical_json(report)
                    else:
                        raw = serialize_manifest(report)
                    (cell_dir / filename).write_bytes(raw)
                if schema == "gwexpy-qualification-evidence-v1":
                    rebuilt = module.aggregate_reports(
                        version=version,
                        source_sha=source,
                        payload_manifest=manifest,
                        reports_dir=reports,
                        output_path=output,
                    )
                elif schema == "gwexpy-diaggui-qualification-evidence-v1":
                    rebuilt = module.aggregate_reports(
                        source_sha=source,
                        payload_manifest=manifest,
                        reports_dir=reports,
                        output_path=output,
                    )
                else:
                    rebuilt = module.aggregate(source, manifest, reports, output)
            _require(
                rebuilt == evidence,
                "aggregate differs from validated evidence contract",
            )
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise PromotionManifestError(
            f"invalid aggregate evidence {schema}: {exc}"
        ) from exc


def configured_contract(version: str) -> dict[str, Any]:
    """Load the frozen control registry, including its strict profile validation."""
    try:
        return _local_module("release_contract").release_contract("v" + version)
    except ValueError as exc:
        raise PromotionManifestError(str(exc)) from exc


def candidate_profile(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve the complete supported future profile; unknown lanes fail closed."""
    registry = _local_module("release_contract")
    promotion = contract.get("promotion")
    try:
        registry._validate_promotion("synthetic", promotion)
    except ValueError as exc:
        raise PromotionManifestError(str(exc)) from exc
    return {
        **promotion["qualification_profiles"][promotion["qualification_profile"]],
        "evidence_schemas": dict(promotion["evidence_schemas"]),
    }


def observe_archive(
    api: Mapping[str, Any], raw: bytes, run_id: int
) -> dict[str, bytes]:
    """Verify the exact API archive before reading its flat regular members.

    The returned bytes all come from this archive. Callers must use this same
    operation for a later manifest download, avoiding unrelated member bytes.
    """
    import io
    import stat
    import zipfile

    _require(_positive_int(api.get("artifact_id")), "invalid artifact ID")
    _require(api.get("run_id") == run_id, "artifact run ID mismatch")
    _require(api.get("expired") is False, "artifact is expired")
    _require(api.get("size_in_bytes") == len(raw), "artifact archive size mismatch")
    _require(
        api.get("digest") == "sha256:" + sha256(raw), "artifact archive digest mismatch"
    )
    result = {}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for member in archive.infolist():
                mode = member.external_attr >> 16
                _require(
                    _safe_filename(member.filename)
                    and not member.is_dir()
                    and not stat.S_ISLNK(mode),
                    "unsafe archive member",
                )
                _require(member.filename not in result, "duplicate archive member")
                _require(
                    0 < member.file_size <= 100 * 1024 * 1024,
                    "invalid archive member size",
                )
                result[member.filename] = archive.read(member)
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise PromotionManifestError("invalid artifact archive") from exc
    _require(bool(result), "empty artifact archive")
    return result


def validate_downloaded_artifact(
    manifest: Mapping[str, Any],
    role: str,
    directory: Path,
    *,
    manifest_sha256: str | None = None,
) -> None:
    """Re-hash a separately downloaded artifact against its manifest record."""
    import stat

    if role == "promotion-manifest":
        _require(
            isinstance(manifest_sha256, str)
            and SHA256.fullmatch(manifest_sha256) is not None,
            "expected promotion manifest hash is missing",
        )
        expected = {"promotion-manifest.json": manifest_sha256}
    else:
        artifacts = manifest.get("artifacts")
        _require(isinstance(artifacts, list), "manifest artifacts are missing")
        matches = [
            item
            for item in artifacts
            if isinstance(item, Mapping) and item.get("role") == role
        ]
        _require(len(matches) == 1, "manifest artifact role is missing or ambiguous")
        files = matches[0].get("files")
        _require(isinstance(files, list), "manifest artifact files are invalid")
        expected = {item["name"]: item["sha256"] for item in files}
    _require(
        directory.is_dir() and not directory.is_symlink(),
        "downloaded artifact path is not a directory",
    )
    actual = {}
    for path in directory.iterdir():
        _require(
            stat.S_ISREG(path.lstat().st_mode),
            "downloaded artifact contains a non-file",
        )
        actual[path.name] = sha256(path.read_bytes())
    _require(
        actual == expected,
        f"downloaded artifact {role} has unexpected members or file hashes",
    )


def _api_json(repository: str, token: str, path: str) -> Any:
    import urllib.request

    url = "https://api.github.com/repos/" + repository + "/" + path
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def _api_pages(
    repository: str, token: str, path: str, key: str
) -> list[dict[str, Any]]:
    result = []
    page = 1
    while True:
        data = _api_json(repository, token, f"{path}?per_page=100&page={page}")
        entries = data[key]
        result.extend(entries)
        if len(entries) < 100:
            return result
        page += 1


def _artifact_observation(record: Mapping[str, Any]) -> dict[str, Any]:
    workflow_run = record.get("workflow_run")
    _require(isinstance(workflow_run, Mapping), "artifact has no workflow run")
    return {
        "artifact_id": record.get("id"),
        "name": record.get("name"),
        "digest": record.get("digest"),
        "size_in_bytes": record.get("size_in_bytes"),
        "run_id": workflow_run.get("id"),
        "expired": record.get("expired"),
    }


def _download_artifact_archive(repository: str, token: str, artifact_id: int) -> bytes:
    import urllib.request

    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/actions/artifacts/{artifact_id}/zip",
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _candidate_gate_jobs(
    repository: str,
    token: str,
    run_id: int,
    required_jobs: Sequence[str],
) -> list[dict[str, str]]:
    actual = _api_pages(
        repository, token, f"actions/runs/{run_id}/attempts/1/jobs", "jobs"
    )
    prefixes = {
        "verify": "Verify immutable release source",
        "build": "Build and check release artifacts",
        "smoke": "Smoke-test ",
        "qualify": "Qualify ",
        "qualification_evidence": "Aggregate nineteen qualification cells",
        "diaggui_qualification": "Qualify installed DiagGUI ",
        "diaggui_qualification_evidence": "Aggregate four DiagGUI qualification cells",
        "cross_format_io": "Qualify installed cross-format I/O ",
        "cross_format_io_evidence": "Aggregate eight cross-format I/O cells",
        "evidence": "Collect same-run integration evidence",
    }
    counts = {
        "smoke": 4,
        "qualify": 19,
        "diaggui_qualification": 4,
        "cross_format_io": 8,
    }
    results = []
    for name in required_jobs:
        _require(name in prefixes, f"unknown required candidate job {name}")
        matches = [
            job
            for job in actual
            if job.get("name", "").startswith(prefixes[name])
            and (
                name != "qualify"
                or not job.get("name", "").startswith("Qualify installed ")
            )
        ]
        _require(
            len(matches) == counts.get(name, 1)
            and all(
                job.get("status") == "completed" and job.get("conclusion") == "success"
                for job in matches
            ),
            f"required candidate job {name} did not complete successfully",
        )
        results.append({"name": name, "status": "completed", "conclusion": "success"})
    return results


def verify_tag_candidate(
    *,
    source_root: Path,
    tag: str,
    source_sha: str,
    tag_body: str | bytes,
    repository: str,
    token: str,
    output: Path,
) -> dict[str, Any]:
    """Verify tag, GO, run, manifest and every manifest-selected artifact ID."""
    import os

    record = parse_promotion_tag(tag_body)
    validate_promotion_tag(
        record,
        annotated=True,
        tag_name=tag,
        target_sha=source_sha,
        repository=repository,
        candidate_run_id=int(record["candidate_run_id"]),
        manifest_sha256=record["promotion_manifest_sha256"],
        release_go_comment_id=int(record["release_go_comment_id"]),
    )
    version = tag.removeprefix("v")
    _require(tag == "v" + version, "release tag is not canonical")
    contract = configured_contract(version)
    profile = candidate_profile(contract)
    candidate_id = int(record["candidate_run_id"])
    candidate_run = _api_json(repository, token, f"actions/runs/{candidate_id}")

    listed = _api_pages(
        repository, token, f"actions/runs/{candidate_id}/artifacts", "artifacts"
    )
    manifest_name = f"release-promotion-manifest-{source_sha}"
    manifest_candidates = [
        {
            **_artifact_observation(item),
            "created_at": item.get("created_at"),
        }
        for item in listed
        if item.get("name") == manifest_name
    ]
    manifest_api = select_manifest_artifact(
        manifest_candidates, source_sha, candidate_id
    )
    validate_manifest_artifact_metadata(manifest_api, source_sha, candidate_id)
    manifest_archive = _download_artifact_archive(
        repository, token, manifest_api["artifact_id"]
    )
    manifest_members = observe_archive(manifest_api, manifest_archive, candidate_id)
    _require(
        set(manifest_members) == {"promotion-manifest.json"},
        "promotion manifest archive must contain exactly promotion-manifest.json",
    )
    manifest_raw = manifest_members["promotion-manifest.json"]
    manifest = load_manifest(manifest_raw)
    manifest_sha256 = sha256(manifest_raw)
    validate_manifest_artifact(
        manifest_api,
        manifest_archive,
        manifest_raw,
        source_sha,
        candidate_id,
        record["promotion_manifest_sha256"],
    )
    validate_candidate_run(
        manifest,
        contract,
        candidate_run,
        repository=repository,
        tag=tag,
        source_sha=source_sha,
    )
    _require(
        manifest_sha256 == record["promotion_manifest_sha256"],
        "tag promotion manifest SHA-256 mismatch",
    )

    artifact_records = manifest["artifacts"]
    api_records = []
    observed_files: dict[int, list[dict[str, Any]]] = {}
    member_bytes: dict[str, dict[str, bytes]] = {}
    for manifest_record in artifact_records:
        artifact_id = manifest_record["artifact_id"]
        _require(_positive_int(artifact_id), "manifest artifact ID is invalid")
        actual = _api_json(repository, token, f"actions/artifacts/{artifact_id}")
        api = _artifact_observation(actual)
        validate_artifact_api_metadata(manifest_record, api, candidate_id)
        raw = _download_artifact_archive(repository, token, artifact_id)
        members = observe_archive(api, raw, candidate_id)
        member_bytes[manifest_record["role"]] = members
        actual_files = [
            {"name": name, "size_in_bytes": len(content), "sha256": sha256(content)}
            for name, content in sorted(
                members.items(), key=lambda item: item[0].encode("utf-8")
            )
        ]
        observed_files[artifact_id] = actual_files
        api_records.append(api)

    sidecar_members = member_bytes.get("sidecar:distribution-sha256.json", {})
    license_members = member_bytes.get("sidecar:LICENSE.sha256", {})
    _require(
        set(sidecar_members) == {"distribution-sha256.json"}
        and set(license_members) == {"LICENSE.sha256"},
        "candidate sidecar artifact members are invalid",
    )
    sidecar_raw = sidecar_members["distribution-sha256.json"]
    sidecar = json.loads(sidecar_raw, object_pairs_hook=_pairs)
    _require(
        sidecar.get("schema") == contract.get("payload_schema")
        and sidecar.get("version") == version
        and sidecar.get("source_sha") == source_sha,
        "candidate distribution sidecar binding mismatch",
    )
    declared_package_hashes = {
        item["name"]: item["sha256"] for item in sidecar["files"].values()
    }
    license_raw = license_members["LICENSE.sha256"]
    license_match = re.fullmatch(rb"([0-9a-f]{64})\n", license_raw)
    _require(license_match is not None, "candidate LICENSE.sha256 is malformed")
    license_digest = license_match.group(1).decode("ascii")
    _require(
        license_digest == sha256((source_root / "LICENSE.txt").read_bytes()),
        "candidate LICENSE.sha256 differs from source license",
    )

    review_path = source_root / contract["review_evidence_path"]
    review_raw = review_path.read_bytes()
    review = json.loads(review_raw, object_pairs_hook=_pairs)
    release_notes = source_root / "release_notes" / f"v{version}.md"
    expected = {
        "repository": repository,
        "version": version,
        "tag": tag,
        "workflow_id": candidate_run.get("workflow_id"),
        "workflow_path": contract["promotion"]["workflow_path"],
        "event": "workflow_dispatch",
        "dispatch_ref": source_sha,
        "workflow_ref": (
            f"{repository}/{contract['promotion']['workflow_path']}@refs/heads/main"
        ),
        "workflow_sha": candidate_run.get("head_sha"),
        "source_sha": source_sha,
        "release_sha": source_sha,
        "run_id": candidate_id,
        "review_evidence_sha256": sha256(review_raw),
        "release_notes_sha256": sha256(release_notes.read_bytes()),
    }
    jobs = _candidate_gate_jobs(
        repository, token, candidate_id, profile["required_jobs"]
    )
    validate_manifest(
        manifest,
        contract,
        expected,
        run_attempt=candidate_run.get("run_attempt"),
        jobs=jobs,
        artifact_api=api_records,
        observed_files=observed_files,
        declared_package_hashes=declared_package_hashes,
    )

    metadata = {
        **{
            key: expected[key]
            for key in (
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
            )
        },
    }
    for role in profile["evidence_schema_ids"]:
        evidence_schema = profile["evidence_schemas"][role]
        evidence_role = "evidence:" + role
        members = member_bytes[evidence_role]
        _require(len(members) == 1, "evidence artifact must contain one aggregate")
        evidence = json.loads(next(iter(members.values())), object_pairs_hook=_pairs)
        _require(
            evidence.get("schema") == evidence_schema
            and evidence.get("version") == version
            and evidence.get("source_sha") == source_sha,
            "candidate evidence aggregate is not bound to source and version",
        )
        _require(
            evidence.get("payload", evidence.get("files")) == sidecar["files"],
            "candidate evidence aggregate payload binding mismatch",
        )
        if role == "gwexpy-integration-evidence-v1":
            expected_approval = review.get("human_approval")
            _require(
                isinstance(expected_approval, Mapping),
                "source review has no configured human approval",
            )
            _require(
                evidence.get("review_evidence")
                == {
                    "path": contract["review_evidence_path"],
                    "sha256": sha256(review_raw),
                    "human_approval": expected_approval,
                },
                "candidate evidence S-approval binding mismatch",
            )
        validate_candidate_aggregate(
            evidence,
            role,
            contract,
            metadata,
            sidecar,
            license_digest,
        )

    go_module = _local_module("verify_release_go")
    go_module.verify_release_go(
        contract=contract,
        manifest=manifest,
        manifest_raw=manifest_raw,
        manifest_sha256=manifest_sha256,
        comment_id=int(record["release_go_comment_id"]),
        candidate_run=candidate_run,
        manifest_artifact={**manifest_api, "created_at": manifest_api["created_at"]},
        manifest_archive=manifest_archive,
        repository=repository,
        token=token,
    )

    output.mkdir(parents=True, exist_ok=True)
    (output / "promotion-manifest.json").write_bytes(manifest_raw)
    artifact_by_role = {item["role"]: item for item in artifact_records}
    result = {
        "candidate_run_id": candidate_id,
        "manifest_artifact_id": manifest_api["artifact_id"],
        "manifest_sha256": manifest_sha256,
        "payload_artifact_id": artifact_by_role["payload"]["artifact_id"],
        "distribution_sidecar_artifact_id": artifact_by_role[
            "sidecar:distribution-sha256.json"
        ]["artifact_id"],
        "license_sidecar_artifact_id": artifact_by_role["sidecar:LICENSE.sha256"][
            "artifact_id"
        ],
        "sdist_sha256": next(
            item["sha256"]
            for item in artifact_by_role["payload"]["files"]
            if item["name"].endswith(".tar.gz")
        ),
        "wheel_sha256": next(
            item["sha256"]
            for item in artifact_by_role["payload"]["files"]
            if item["name"].endswith(".whl")
        ),
    }
    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with Path(output_file).open("a", encoding="utf-8") as stream:
            for key, value in result.items():
                stream.write(f"{key}={value}\n")
    return result


def finalize_candidate(root: Path, version: str, output: Path) -> None:
    """Read existing current-run artifacts by ID and emit one validated manifest."""
    import datetime
    import os
    import tempfile
    import urllib.request

    contract = configured_contract(version)
    profile = candidate_profile(contract)
    repository = os.environ["GITHUB_REPOSITORY"]
    token = os.environ["GITHUB_TOKEN"]
    run_id = int(os.environ["GITHUB_RUN_ID"])
    attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
    _require(attempt == 1, "promotion manifest requires run attempt 1")
    source = os.environ["SOURCE_SHA"]
    run = _api_json(repository, token, f"actions/runs/{run_id}")
    _require(
        run["event"] == "workflow_dispatch" and run["run_attempt"] == 1,
        "candidate must be a first-attempt dispatch",
    )
    _require(
        run["head_sha"] == os.environ["GITHUB_WORKFLOW_SHA"], "workflow SHA mismatch"
    )
    needs = json.loads(os.environ["GATE_RESULTS"])
    _require(
        all(
            needs.get(job, {}).get("result") == "success"
            for job in profile["required_jobs"]
        ),
        "required lane failed or skipped",
    )
    actual_jobs = _api_pages(
        repository, token, f"actions/runs/{run_id}/attempts/1/jobs", "jobs"
    )
    prefixes = {
        "verify": "Verify immutable release source",
        "build": "Build and check release artifacts",
        "smoke": "Smoke-test ",
        "qualify": "Qualify ",
        "qualification_evidence": "Aggregate nineteen qualification cells",
        "diaggui_qualification": "Qualify installed DiagGUI ",
        "diaggui_qualification_evidence": "Aggregate four DiagGUI qualification cells",
        "cross_format_io": "Qualify installed cross-format I/O ",
        "cross_format_io_evidence": "Aggregate eight cross-format I/O cells",
        "evidence": "Collect same-run integration evidence",
    }
    jobs = []
    for job in profile["required_jobs"]:
        matches = [
            j
            for j in actual_jobs
            if j["name"].startswith(prefixes[job])
            and (job != "qualify" or not j["name"].startswith("Qualify installed "))
        ]
        counts = {
            "smoke": 4,
            "qualify": 19,
            "diaggui_qualification": 4,
            "cross_format_io": 8,
        }
        _require(
            len(matches) == counts.get(job, 1)
            and all(
                j["status"] == "completed" and j["conclusion"] == "success"
                for j in matches
            ),
            f"required gate {job} did not succeed",
        )
        jobs.append({"name": job, "status": "completed", "conclusion": "success"})
    available = _api_pages(
        repository, token, f"actions/runs/{run_id}/artifacts", "artifacts"
    )
    naming = contract["promotion"]["artifact_naming"]
    required = {"payload": naming["payload_prefix"] + source}
    required.update(
        {
            f"sidecar:{name}": f"release-sidecar-{name}-{source}"
            for name in naming["sidecar_names"]
        }
    )
    required.update(
        {
            f"evidence:{schema}": schema + "-" + source
            for schema in profile["evidence_schema_ids"]
        }
    )
    observations = []
    inputs = []
    observed = {}
    declared = None
    aggregates = []
    sidecar = None
    license_digest = None
    with tempfile.TemporaryDirectory() as temporary:
        for role, name in required.items():
            matches = [a for a in available if a["name"] == name]
            _require(
                len(matches) == 1, "missing or ambiguous candidate artifact: " + name
            )
            record = matches[0]
            api = {
                key: record[key]
                for key in ("name", "digest", "size_in_bytes", "expired")
            }
            api.update(artifact_id=record["id"], run_id=record["workflow_run"]["id"])
            _require(api["run_id"] == run_id, "artifact belongs to another run")
            if role.startswith("evidence:"):
                created = datetime.datetime.fromisoformat(
                    record["created_at"].replace("Z", "+00:00")
                )
                expires = datetime.datetime.fromisoformat(
                    record["expires_at"].replace("Z", "+00:00")
                )
                _require(
                    expires - created >= datetime.timedelta(days=90),
                    "evidence retention must be 90 days",
                )
            request = urllib.request.Request(
                f"https://api.github.com/repos/{repository}/actions/artifacts/{record['id']}/zip",
                headers={"Authorization": "Bearer " + token},
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
            members = observe_archive(api, raw, run_id)
            if role.startswith("sidecar:"):
                _require(
                    set(members) == {role.split(":", 1)[1]},
                    "unexpected sidecar members",
                )
            if role == "sidecar:distribution-sha256.json":
                sidecar = json.loads(
                    members["distribution-sha256.json"], object_pairs_hook=_pairs
                )
                _require(
                    sidecar["version"] == version
                    and sidecar["source_sha"] == source
                    and sidecar["schema"] == contract["payload_schema"],
                    "payload sidecar binding mismatch",
                )
                declared = {
                    item["name"]: item["sha256"] for item in sidecar["files"].values()
                }
            if role == "sidecar:LICENSE.sha256":
                raw_license = members["LICENSE.sha256"]
                _require(
                    re.fullmatch(rb"[0-9a-f]{64}\n", raw_license) is not None,
                    "LICENSE.sha256 must be one lowercase SHA-256 and LF",
                )
                license_digest = raw_license[:-1].decode("ascii")
                _require(
                    license_digest == sha256((root / "LICENSE.txt").read_bytes()),
                    "LICENSE.sha256 differs from repository license",
                )
            if role.startswith("evidence:"):
                _require(len(members) == 1, "evidence must have one aggregate member")
                evidence = json.loads(
                    next(iter(members.values())), object_pairs_hook=_pairs
                )
                schema = role.split(":", 1)[1]
                _require(
                    evidence.get("schema") == profile["evidence_schemas"][schema]
                    and evidence.get("source_sha") == source
                    and evidence.get("version") == version,
                    "aggregate evidence binding mismatch",
                )
                aggregates.append((schema, evidence))
            files = {}
            directory = Path(temporary) / str(record["id"])
            directory.mkdir()
            for filename, content in members.items():
                path = directory / filename
                path.write_bytes(content)
                files[filename] = path
            inputs.append((role, api, files))
            observations.append(api)
            observed[api["artifact_id"]] = describe_existing_artifact(api, files)[
                "files"
            ]
        metadata = {
            "repository": repository,
            "version": version,
            "tag": "v" + version,
            "workflow_id": run["workflow_id"],
            "workflow_path": contract["promotion"]["workflow_path"],
            "event": "workflow_dispatch",
            "dispatch_ref": source,
            "workflow_ref": os.environ["GITHUB_WORKFLOW_REF"],
            "workflow_sha": os.environ["GITHUB_WORKFLOW_SHA"],
            "source_sha": source,
            "release_sha": source,
            "run_id": run_id,
            "run_attempt": attempt,
            "gates": [
                {"job": job, "conclusion": "success"}
                for job in profile["required_jobs"]
            ],
        }
        review = root / contract["review_evidence_path"]
        notes = root / f"release_notes/v{version}.md"
        review_raw = review.read_bytes()
        review_data = json.loads(review_raw, object_pairs_hook=_pairs)
        for schema, evidence in aggregates:
            _require(
                evidence.get("payload", evidence.get("files")) == sidecar["files"],
                "aggregate payload hash binding mismatch",
            )
            if schema == "gwexpy-integration-evidence-v1":
                _require(
                    evidence.get("review_evidence")
                    == {
                        "path": contract["review_evidence_path"],
                        "sha256": sha256(review_raw),
                        "human_approval": review_data["human_approval"],
                    },
                    "aggregate source approval identity mismatch",
                )
            validate_candidate_aggregate(
                evidence, schema, contract, metadata, sidecar, license_digest
            )
        manifest = build_manifest(metadata, contract, inputs, review, notes)
        expected = {key: metadata[key] for key in EXPECTED_FIELDS if key in metadata}
        expected.update(
            review_evidence_sha256=sha256(review.read_bytes()),
            release_notes_sha256=sha256(notes.read_bytes()),
        )
        validate_manifest(
            manifest,
            contract,
            expected,
            run_attempt=attempt,
            jobs=jobs,
            artifact_api=observations,
            observed_files=observed,
            declared_package_hashes=declared,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(serialize_manifest(manifest))


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("profile", "finalize", "verify-tag", "validate-download")
    )
    parser.add_argument("--version")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--source-sha")
    parser.add_argument("--tag-body", type=Path)
    parser.add_argument("--repository")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--role")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.command == "profile":
        if not args.version:
            parser.error("profile requires --version")
        contract = configured_contract(args.version)
        enabled = "promotion" in contract
        print("promotion_enabled=" + str(enabled).lower())
        if enabled:
            profile = candidate_profile(contract)
            print(
                "qualification_profile="
                + contract["promotion"]["qualification_profile"]
            )
            for job in profile["required_jobs"]:
                print(job + "=true")
            for schema, value in profile["evidence_schemas"].items():
                print(schema.replace("-", "_") + "=" + value)
    elif args.command == "finalize":
        if not args.version or args.repo_root is None or args.output is None:
            parser.error("finalize requires --version, --repo-root, and --output")
        finalize_candidate(args.repo_root, args.version, args.output)
    elif args.command == "verify-tag":
        if (
            args.repo_root is None
            or args.tag is None
            or args.source_sha is None
            or args.tag_body is None
            or args.repository is None
            or args.output is None
        ):
            parser.error(
                "verify-tag requires --repo-root, --tag, --source-sha, --tag-body, "
                "--repository, and --output"
            )
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            parser.error("verify-tag requires GITHUB_TOKEN in the environment")
        try:
            verify_tag_candidate(
                source_root=args.repo_root,
                tag=args.tag,
                source_sha=args.source_sha,
                tag_body=args.tag_body.read_bytes(),
                repository=args.repository,
                token=token,
                output=args.output,
            )
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"tag promotion verification failed: {exc}") from exc
    else:
        if args.manifest is None or args.role is None or args.directory is None:
            parser.error(
                "validate-download requires --manifest, --role, and --directory"
            )
        try:
            manifest = load_manifest(args.manifest.read_bytes())
            validate_downloaded_artifact(
                manifest,
                args.role,
                args.directory,
                manifest_sha256=args.manifest_sha256,
            )
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise SystemExit(f"downloaded artifact rejected: {exc}") from exc


if __name__ == "__main__":
    main()
