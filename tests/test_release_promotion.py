"""Strict validation for candidate release promotion manifests."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "release_promotion.py"
FIXTURE = ROOT / "tests" / "fixtures" / "release" / "promotion-manifest-v1.json"
CONTRACTS = ROOT / "scripts" / "ci" / "release_contract.py"
PROMOTION_CONTRACT = (
    ROOT / "tests" / "fixtures" / "release" / "promotion-contract-v1.json"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def promotion():
    module = load_module(SCRIPT, "release_promotion_test")
    manifest = json.loads(FIXTURE.read_text(encoding="utf-8"))
    contract_module = load_module(CONTRACTS, "release_contract_for_promotion_test")
    contract = copy.deepcopy(contract_module.release_contract("v0.2.5"))
    contract["promotion"] = json.loads(PROMOTION_CONTRACT.read_text(encoding="utf-8"))[
        "promotion"
    ]
    return module, manifest, contract


def expected() -> dict[str, object]:
    sha = "3439fba41dcc644af12870d91bee794524098fb4"
    return {
        "repository": "example/gwexpy",
        "version": "0.2.5",
        "tag": "v0.2.5",
        "workflow_id": 123456,
        "workflow_path": ".github/workflows/publish-release.yml",
        "event": "workflow_dispatch",
        "dispatch_ref": sha,
        "workflow_ref": "example/gwexpy/.github/workflows/publish-release.yml@refs/heads/main",
        "workflow_sha": sha,
        "source_sha": sha,
        "release_sha": sha,
        "run_id": 987654,
        "review_evidence_sha256": "b" * 64,
        "release_notes_sha256": "c" * 64,
    }


def declared_package_hashes() -> dict[str, str]:
    return {
        "gwexpy-0.2.5-py3-none-any.whl": "d" * 64,
        "gwexpy-0.2.5.tar.gz": "e" * 64,
    }


def observed_inputs(manifest: dict[str, object]) -> dict[str, object]:
    artifacts = manifest["artifacts"]
    return {
        "run_attempt": 1,
        "jobs": [
            {"name": gate["job"], "status": "completed", "conclusion": "success"}
            for gate in manifest["gates"]
        ],
        "artifact_api": [
            {
                key: artifact[key]
                for key in (
                    "artifact_id",
                    "name",
                    "digest",
                    "size_in_bytes",
                    "run_id",
                    "expired",
                )
            }
            for artifact in artifacts
        ],
        "observed_files": {
            artifact["artifact_id"]: artifact["files"] for artifact in artifacts
        },
        "declared_package_hashes": declared_package_hashes(),
    }


def test_canonical_json_is_sorted_utf8_and_has_one_final_newline(promotion) -> None:
    module, manifest, _ = promotion
    raw = module.serialize_manifest({"z": "é", "a": 1})
    assert raw == '{\n  "a": 1,\n  "z": "é"\n}\n'.encode()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert module.load_manifest(raw) == {"a": 1, "z": "é"}


def test_release_go_and_tag_records_are_canonical(promotion) -> None:
    module, _, _ = promotion
    go = "\n".join(
        [
            "GWEXPY-RELEASE-GO-v1",
            "version=v0.2.6",
            f"source_sha={'a' * 40}",
            "candidate_run_id=987654",
            f"promotion_manifest_sha256={'b' * 64}",
            f"sdist_sha256={'c' * 64}",
            f"wheel_sha256={'d' * 64}",
            "decision=GO",
        ]
    )
    tag = "\n".join(
        [
            "GWEXPY-PROMOTION-v1",
            "repository=tatsuki-washimi/gwexpy",
            "tag=v0.2.6",
            f"source_sha={'a' * 40}",
            "candidate_run_id=987654",
            f"promotion_manifest_sha256={'b' * 64}",
            "release_go_comment_id=12345",
        ]
    )
    assert module.parse_release_go(go)["source_sha"] == "a" * 40
    parsed = module.parse_promotion_tag(tag)
    module.validate_promotion_tag(
        parsed,
        annotated=True,
        tag_name="v0.2.6",
        target_sha="a" * 40,
        repository="tatsuki-washimi/gwexpy",
        candidate_run_id=987654,
        manifest_sha256="b" * 64,
        release_go_comment_id=12345,
    )


@pytest.mark.parametrize(
    "suffix", ["\nextra=x", "\nversion=v0.2.6", "\nunknown=x", "\r\nlast"]
)
def test_release_go_rejects_noncanonical_bodies(promotion, suffix: str) -> None:
    module, _, _ = promotion
    body = "\n".join(
        [
            "GWEXPY-RELEASE-GO-v1",
            "version=v0.2.6",
            f"source_sha={'a' * 40}",
            "candidate_run_id=987654",
            f"promotion_manifest_sha256={'b' * 64}",
            f"sdist_sha256={'c' * 64}",
            f"wheel_sha256={'d' * 64}",
            "decision=GO",
        ]
    )
    with pytest.raises(module.PromotionManifestError):
        module.parse_release_go(body + suffix)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "duplicate", "unknown", "wrong_sha", "bad_id"],
)
def test_promotion_tag_parser_rejects_malformed_records(
    promotion, mutation: str
) -> None:
    module, _, _ = promotion
    lines = [
        "GWEXPY-PROMOTION-v1",
        "repository=tatsuki-washimi/gwexpy",
        "tag=v0.2.6",
        f"source_sha={'a' * 40}",
        "candidate_run_id=987654",
        f"promotion_manifest_sha256={'b' * 64}",
        "release_go_comment_id=12345",
    ]
    if mutation == "missing":
        lines.pop()
    elif mutation == "extra":
        lines.append("unexpected=x")
    elif mutation == "duplicate":
        lines.append("tag=v0.2.6")
    elif mutation == "unknown":
        lines[1] = "repo=tatsuki-washimi/gwexpy"
    elif mutation == "wrong_sha":
        lines[3] = f"source_sha={'A' * 40}"
    else:
        lines[4] = "candidate_run_id=01"
    with pytest.raises(module.PromotionManifestError):
        module.parse_promotion_tag("\n".join(lines))


@pytest.mark.parametrize(
    "observation",
    [
        {
            "annotated": False,
            "tag_name": "v0.2.6",
            "target_sha": "a" * 40,
            "repository": "tatsuki-washimi/gwexpy",
            "candidate_run_id": 987654,
            "manifest_sha256": "b" * 64,
            "release_go_comment_id": 12345,
        },
        {
            "annotated": True,
            "tag_name": "v0.2.7",
            "target_sha": "a" * 40,
            "repository": "tatsuki-washimi/gwexpy",
            "candidate_run_id": 987654,
            "manifest_sha256": "b" * 64,
            "release_go_comment_id": 12345,
        },
        {
            "annotated": True,
            "tag_name": "v0.2.6",
            "target_sha": "b" * 40,
            "repository": "tatsuki-washimi/gwexpy",
            "candidate_run_id": 987654,
            "manifest_sha256": "b" * 64,
            "release_go_comment_id": 12345,
        },
        {
            "annotated": True,
            "tag_name": "v0.2.6",
            "target_sha": "a" * 40,
            "repository": "example/gwexpy",
            "candidate_run_id": 987654,
            "manifest_sha256": "b" * 64,
            "release_go_comment_id": 12345,
        },
        {
            "annotated": True,
            "tag_name": "v0.2.6",
            "target_sha": "a" * 40,
            "repository": "tatsuki-washimi/gwexpy",
            "candidate_run_id": 1,
            "manifest_sha256": "b" * 64,
            "release_go_comment_id": 12345,
        },
        {
            "annotated": True,
            "tag_name": "v0.2.6",
            "target_sha": "a" * 40,
            "repository": "tatsuki-washimi/gwexpy",
            "candidate_run_id": 987654,
            "manifest_sha256": "e" * 64,
            "release_go_comment_id": 12345,
        },
        {
            "annotated": True,
            "tag_name": "v0.2.6",
            "target_sha": "a" * 40,
            "repository": "tatsuki-washimi/gwexpy",
            "candidate_run_id": 987654,
            "manifest_sha256": "b" * 64,
            "release_go_comment_id": 1,
        },
    ],
)
def test_promotion_tag_requires_annotated_correct_identity_and_release_sha(
    promotion, observation: dict[str, object]
) -> None:
    module, _, _ = promotion
    body = "\n".join(
        [
            "GWEXPY-PROMOTION-v1",
            "repository=tatsuki-washimi/gwexpy",
            "tag=v0.2.6",
            f"source_sha={'a' * 40}",
            "candidate_run_id=987654",
            f"promotion_manifest_sha256={'b' * 64}",
            "release_go_comment_id=12345",
        ]
    )
    record = module.parse_promotion_tag(body)
    with pytest.raises(module.PromotionManifestError):
        module.validate_promotion_tag(record, **observation)


def test_load_manifest_enforces_canonicality_for_string_input(promotion) -> None:
    module, _, _ = promotion
    with pytest.raises(module.PromotionManifestError, match="canonical"):
        module.load_manifest('{"z": 1, "a": 2}\n')
    with pytest.raises(module.PromotionManifestError, match="canonical"):
        module.load_manifest('{\n  "a": 1\n}')


def test_manifest_fixture_validates_against_contract_and_expected_context(
    promotion,
) -> None:
    module, manifest, contract = promotion
    module.validate_manifest(
        manifest,
        contract,
        expected(),
        **observed_inputs(manifest),
    )


@pytest.mark.parametrize(
    "missing", ["run_attempt", "jobs", "artifact_api", "observed_files"]
)
def test_manifest_rejects_missing_trusted_observations(promotion, missing: str) -> None:
    module, manifest, contract = promotion
    observations = observed_inputs(manifest)
    observations.pop(missing)
    with pytest.raises(module.PromotionManifestError):
        module.validate_manifest(manifest, contract, expected(), **observations)


@pytest.mark.parametrize("attempt", [True, 1.0])
def test_manifest_rejects_non_integer_actual_run_attempt(promotion, attempt) -> None:
    module, manifest, contract = promotion
    observations = observed_inputs(manifest)
    observations["run_attempt"] = attempt
    with pytest.raises(module.PromotionManifestError, match="run attempt"):
        module.validate_manifest(manifest, contract, expected(), **observations)


@pytest.mark.parametrize(
    "missing",
    [
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
    ],
)
def test_manifest_rejects_incomplete_expected_context(promotion, missing: str) -> None:
    module, manifest, contract = promotion
    context = expected()
    context.pop(missing)
    with pytest.raises(module.PromotionManifestError, match="expected context"):
        module.validate_manifest(
            manifest, contract, context, **observed_inputs(manifest)
        )


@pytest.mark.parametrize("missing", ["review_evidence_sha256", "release_notes_sha256"])
def test_manifest_rejects_missing_expected_descriptor_hash(
    promotion, missing: str
) -> None:
    module, manifest, contract = promotion
    context = {
        **expected(),
        "review_evidence_sha256": manifest["review_evidence"]["sha256"],
        "release_notes_sha256": manifest["release_notes"]["sha256"],
    }
    context.pop(missing)
    with pytest.raises(module.PromotionManifestError, match="expected context"):
        module.validate_manifest(
            manifest, contract, context, **observed_inputs(manifest)
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "hash", "size"])
def test_manifest_rejects_artifact_file_observation_mismatch(
    promotion, mutation: str
) -> None:
    module, manifest, contract = promotion
    observations = observed_inputs(manifest)
    files = observations["observed_files"]
    artifact_id = manifest["artifacts"][0]["artifact_id"]
    changed = [dict(row) for row in files[artifact_id]]
    if mutation == "missing":
        changed.pop()
    elif mutation == "extra":
        changed.append({"name": "extra.txt", "size_in_bytes": 1, "sha256": "f" * 64})
    elif mutation == "hash":
        changed[0]["sha256"] = "f" * 64
    else:
        changed[0]["size_in_bytes"] += 1
    files[artifact_id] = changed
    with pytest.raises(module.PromotionManifestError, match="file observation"):
        module.validate_manifest(manifest, contract, expected(), **observations)


def test_expected_descriptor_hashes_bind_nested_evidence_and_notes(promotion) -> None:
    module, manifest, contract = promotion
    inputs = observed_inputs(manifest)
    supplied = {
        **expected(),
        "review_evidence_sha256": manifest["review_evidence"]["sha256"],
        "release_notes_sha256": manifest["release_notes"]["sha256"],
    }
    module.validate_manifest(manifest, contract, supplied, **inputs)
    supplied["review_evidence_sha256"] = "f" * 64
    with pytest.raises(module.PromotionManifestError, match="review_evidence_sha256"):
        module.validate_manifest(manifest, contract, supplied, **inputs)
    supplied["review_evidence_sha256"] = manifest["review_evidence"]["sha256"]
    supplied["release_notes_sha256"] = "f" * 64
    with pytest.raises(module.PromotionManifestError, match="release_notes_sha256"):
        module.validate_manifest(manifest, contract, supplied, **inputs)


def test_duplicate_artifact_roles_and_extra_or_duplicate_jobs_reject(promotion) -> None:
    module, manifest, contract = promotion
    observations = observed_inputs(manifest)
    manifest["artifacts"][-1] = {
        **manifest["artifacts"][-1],
        "role": manifest["artifacts"][0]["role"],
    }
    with pytest.raises(module.PromotionManifestError, match="roles"):
        module.validate_manifest(manifest, contract, expected(), **observations)
    manifest["artifacts"][-1]["role"] = "evidence:gwexpy-qualification-evidence-v1"
    for mutation in ("extra", "duplicate"):
        module, manifest, contract = promotion
        observations = observed_inputs(manifest)
        observations["jobs"].append(
            observations["jobs"][0]
            if mutation == "duplicate"
            else {
                "name": "unexpected_job",
                "status": "completed",
                "conclusion": "success",
            }
        )
        with pytest.raises(module.PromotionManifestError, match="job"):
            module.validate_manifest(manifest, contract, expected(), **observations)


def test_manifest_rejects_duplicate_json_keys(promotion) -> None:
    module, _, _ = promotion
    with pytest.raises(module.PromotionManifestError, match="duplicate"):
        module.load_manifest(b'{"schema":"one","schema":"two"}\n')


@pytest.mark.parametrize("field", ["workflow_sha", "source_sha", "release_sha"])
def test_manifest_rejects_non_full_commit_shas(promotion, field: str) -> None:
    module, manifest, contract = promotion
    manifest[field] = "a" * 39
    with pytest.raises(module.PromotionManifestError):
        module.validate_manifest(
            manifest, contract, expected(), **observed_inputs(manifest)
        )


def test_manifest_requires_exact_schema_and_contract_binding(promotion) -> None:
    module, manifest, contract = promotion
    manifest["unexpected"] = True
    with pytest.raises(module.PromotionManifestError, match="field"):
        module.validate_manifest(
            manifest, contract, expected(), **observed_inputs(manifest)
        )


def test_manifest_job_checks_attempt_and_gate_conclusions_not_run_conclusion(
    promotion,
) -> None:
    module, manifest, contract = promotion
    module.validate_manifest(
        manifest,
        contract,
        expected(),
        run_attempt=1,
        jobs=[
            {"name": gate["job"], "status": "completed", "conclusion": "success"}
            for gate in manifest["gates"]
        ],
        overall_run={"status": "in_progress", "conclusion": None},
        declared_package_hashes=declared_package_hashes(),
        artifact_api=observed_inputs(manifest)["artifact_api"],
        observed_files=observed_inputs(manifest)["observed_files"],
    )
    with pytest.raises(module.PromotionManifestError, match="attempt"):
        module.validate_manifest(
            manifest,
            contract,
            expected(),
            **{**observed_inputs(manifest), "run_attempt": 2},
        )


def test_artifact_api_metadata_mismatches_are_checked_independently(promotion) -> None:
    module, manifest, _ = promotion
    record = {
        "role": "payload",
        "artifact_id": 101,
        "name": "release-payload-" + manifest["release_sha"],
        "digest": "sha256:" + "d" * 64,
        "size_in_bytes": 12,
        "run_id": manifest["run_id"],
        "expired": False,
        "files": [{"name": "a.tar.gz", "size_in_bytes": 12, "sha256": "d" * 64}],
    }
    manifest["artifacts"] = [record]
    api_record = {
        key: record[key]
        for key in (
            "artifact_id",
            "name",
            "digest",
            "size_in_bytes",
            "run_id",
            "expired",
        )
    }
    module.validate_artifact_records(
        manifest["artifacts"], [api_record], {101: record["files"]}, manifest["run_id"]
    )
    for field, wrong in (
        ("artifact_id", 102),
        ("name", "wrong"),
        ("digest", "sha256:" + "e" * 64),
        ("size_in_bytes", 13),
        ("run_id", 987655),
        ("expired", True),
    ):
        changed = {**record, field: wrong}
        with pytest.raises(module.PromotionManifestError):
            changed_api = {key: changed[key] for key in api_record}
            module.validate_artifact_records(
                [record], [changed_api], {101: record["files"]}, manifest["run_id"]
            )


@pytest.mark.parametrize("filename", ["../evil", "/etc/passwd", "a\\b", ""])
def test_manifest_rejects_unsafe_artifact_filenames(promotion, filename: str) -> None:
    module, _, _ = promotion
    record = {
        "role": "payload",
        "artifact_id": 1,
        "name": "artifact",
        "digest": "sha256:" + "a" * 64,
        "size_in_bytes": 1,
        "run_id": 2,
        "expired": False,
        "files": [{"name": filename, "size_in_bytes": 1, "sha256": "a" * 64}],
    }
    with pytest.raises(module.PromotionManifestError, match="unsafe artifact filename"):
        module.validate_artifact_records(
            [record],
            [
                {
                    key: record[key]
                    for key in (
                        "artifact_id",
                        "name",
                        "digest",
                        "size_in_bytes",
                        "run_id",
                        "expired",
                    )
                }
            ],
            {1: record["files"]},
            2,
        )


def test_manifest_artifact_uses_archive_metadata_and_separate_manifest_hash(
    promotion,
) -> None:
    module, manifest, _ = promotion
    manifest_raw = module.serialize_manifest(manifest)
    archive_raw = b"independent zip archive bytes are longer than the JSON file"
    assert len(archive_raw) != len(manifest_raw)
    assert module.sha256(archive_raw) != module.sha256(manifest_raw)
    api = {
        "name": f"release-promotion-manifest-{manifest['release_sha']}",
        "artifact_id": 501,
        "digest": "sha256:" + module.sha256(archive_raw),
        "size_in_bytes": len(archive_raw),
        "run_id": manifest["run_id"],
        "expired": False,
    }
    module.validate_manifest_artifact(
        api,
        archive_raw,
        manifest_raw,
        manifest["release_sha"],
        manifest["run_id"],
        module.sha256(manifest_raw),
    )
    assert module.select_manifest_artifact(
        [{**api, "run_id": manifest["run_id"]}],
        manifest["release_sha"],
        manifest["run_id"],
    ) == {**api, "run_id": manifest["run_id"]}
    with pytest.raises(module.PromotionManifestError, match="ambiguous"):
        module.select_manifest_artifact(
            [{**api, "run_id": manifest["run_id"]}] * 2,
            manifest["release_sha"],
            manifest["run_id"],
        )
    with pytest.raises(module.PromotionManifestError):
        module.validate_manifest_artifact(
            {**api, "artifact_id": 0},
            archive_raw,
            manifest_raw,
            manifest["release_sha"],
            manifest["run_id"],
            module.sha256(manifest_raw),
        )
    for changed_api, changed_raw, expected_hash in (
        (
            {**api, "size_in_bytes": len(archive_raw) + 1},
            archive_raw,
            module.sha256(manifest_raw),
        ),
        (
            {**api, "digest": "sha256:" + "f" * 64},
            archive_raw,
            module.sha256(manifest_raw),
        ),
        (api, manifest_raw + b"x", module.sha256(manifest_raw)),
        (api, manifest_raw, "f" * 64),
    ):
        with pytest.raises(module.PromotionManifestError):
            module.validate_manifest_artifact(
                changed_api,
                archive_raw,
                changed_raw,
                manifest["release_sha"],
                manifest["run_id"],
                expected_hash,
            )


@pytest.mark.parametrize(("archive_bytes", "api_size"), [(b"", 0), (b"x", True)])
def test_manifest_artifact_rejects_nonpositive_or_boolean_api_size(
    promotion, archive_bytes: bytes, api_size: object
) -> None:
    module, manifest, _ = promotion
    manifest_bytes = module.serialize_manifest(manifest)
    api = {
        "name": f"release-promotion-manifest-{manifest['release_sha']}",
        "artifact_id": 502,
        "digest": "sha256:" + module.sha256(archive_bytes),
        "size_in_bytes": api_size,
        "run_id": manifest["run_id"],
        "expired": False,
    }
    with pytest.raises(module.PromotionManifestError, match="size"):
        module.validate_manifest_artifact(
            api,
            archive_bytes,
            manifest_bytes,
            manifest["release_sha"],
            manifest["run_id"],
            module.sha256(manifest_bytes),
        )


def test_package_hash_mismatch_is_rejected(promotion) -> None:
    module, _, _ = promotion
    files = [{"name": "gwexpy-0.2.5.tar.gz", "size_in_bytes": 3, "sha256": "a" * 64}]
    module.validate_package_hashes(files, {"gwexpy-0.2.5.tar.gz": "a" * 64})
    with pytest.raises(module.PromotionManifestError, match="package hashes"):
        module.validate_package_hashes(files, {"gwexpy-0.2.5.tar.gz": "b" * 64})


def test_missing_required_evidence_artifact_is_rejected(promotion) -> None:
    module, manifest, contract = promotion
    manifest["artifacts"].pop()
    with pytest.raises(module.PromotionManifestError, match="required artifacts"):
        module.validate_manifest(
            manifest, contract, expected(), **observed_inputs(manifest)
        )


def test_builder_rehashes_existing_files_without_mutating_them(
    promotion, tmp_path: Path
) -> None:
    module, _, _ = promotion
    source = tmp_path / "wheel.whl"
    source.write_bytes(b"existing artifact")
    record = module.describe_existing_artifact(
        {
            "artifact_id": 7,
            "name": "release-payload-x",
            "digest": "sha256:" + "a" * 64,
            "size_in_bytes": 17,
            "run_id": 9,
            "expired": False,
        },
        {"wheel.whl": source},
    )
    assert record["files"] == [
        {
            "name": "wheel.whl",
            "size_in_bytes": 17,
            "sha256": module.sha256(b"existing artifact"),
        }
    ]
    assert source.read_bytes() == b"existing artifact"


def test_manifest_builder_only_reads_existing_artifacts(
    promotion, tmp_path: Path
) -> None:
    module, _, contract = promotion
    payload = tmp_path / "gwexpy-0.2.5.tar.gz"
    evidence = tmp_path / "review.yaml"
    notes = tmp_path / "notes.md"
    payload.write_bytes(b"payload bytes")
    evidence.write_bytes(b"review evidence")
    notes.write_bytes(b"release notes")
    api = {
        "artifact_id": 41,
        "name": "release-payload-x",
        "digest": "sha256:" + "f" * 64,
        "size_in_bytes": 13,
        "run_id": 99,
        "expired": False,
    }
    result = module.build_manifest(
        expected(),
        contract,
        [("payload", api, {payload.name: payload})],
        evidence,
        notes,
    )
    assert result["review_evidence"]["sha256"] == module.sha256(b"review evidence")
    assert result["release_notes"]["sha256"] == module.sha256(b"release notes")
    assert result["artifacts"][0]["files"][0]["sha256"] == module.sha256(
        b"payload bytes"
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "gwexpy-0.2.5.tar.gz",
        "notes.md",
        "review.yaml",
    ]
