"""Static security contract for the release workflow."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "publish-release.yml"
)


def read_workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_release_workflow_replaces_legacy_identity_and_is_manual_dry_run_only():
    workflow = read_workflow()
    assert not (WORKFLOW.parent / "release.yml").exists()
    assert "workflow_dispatch:" in workflow
    assert "release_ref:" in workflow
    assert "expected_tag:" in workflow
    assert "      publish:" not in workflow
    assert "workflow_dispatch must run with --ref main" in workflow


def test_all_actions_are_full_sha_pinned_and_publish_job_is_minimal():
    workflow = read_workflow()
    uses = re.findall(r"^\s*uses:\s*([^\s]+)$", workflow, flags=re.MULTILINE)
    assert uses
    assert all(re.search(r"@[0-9a-f]{40}$", action) for action in uses)

    publish = workflow.split("\n  publish:\n", maxsplit=1)[1].split(
        "\n  github_release_legacy:\n", maxsplit=1
    )[0]
    publish_uses = re.findall(r"^\s*uses:\s*([^\s]+)$", publish, flags=re.MULTILINE)
    assert publish_uses == [
        "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
        "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
        "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
        "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
        "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
        "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
    ]
    assert "id-token: write" in publish
    assert workflow.count("id-token: write") == 2


def test_verify_separates_validator_and_source_trees_and_publish_requires_tag_push():
    """The verify job keeps validator code and validated source apart.

    This asserts separation and SHA pinning only. It deliberately does *not*
    assert that the `control` checkout is an independent trust boundary: on a
    tag push `github.workflow_sha` is the revision the tag points at, not a
    protected `main`, so a tag carrying a rewritten workflow would supply its
    own validator. The controls that bound that risk are the tag rulesets,
    the `pypi` environment, and the PyPI Trusted Publisher binding -- all
    configured outside this repository and documented in RELEASING.md.
    """
    workflow = read_workflow()
    assert "ref: ${{ github.workflow_sha }}" in workflow
    assert "path: control" in workflow
    assert "path: source" in workflow
    assert "--repo-root source" in workflow
    assert "source_sha" in workflow
    assert "scripts/validate_release.py" in workflow
    assert "github.event_name == 'push'" in workflow
    assert "startsWith(github.ref, 'refs/tags/v')" in workflow
    assert "twine check --strict" in workflow
    assert "sys.prefix" in workflow


def test_releasing_doc_separates_enforced_controls_from_operational_rules():
    """Immutability is a maintainer rule until the ruleset enforces it.

    A security-boundary document that lists an unenforced convention beside
    configured controls overstates the guarantee, so the two must stay in
    separate sections and the ruleset rules that would enforce tag
    immutability must be named explicitly.
    """
    # Collapse wrapping so the prose assertions below do not depend on where
    # the source lines happen to break.
    releasing = re.sub(
        r"\s+", " ", (WORKFLOW.parents[2] / "RELEASING.md").read_text(encoding="utf-8")
    )
    enforced = releasing.index("### Enforced by configuration")
    operational = releasing.index("### Operational rules, not enforced")
    assert enforced < operational
    for rule in ("`update`", "`deletion`", "`non_fast_forward`"):
        assert rule in releasing[enforced:operational]
    assert "not a guarantee the platform provides" in releasing[operational:]


def test_releasing_documents_future_build_once_promotion_contract():
    path = WORKFLOW.parents[2] / "RELEASING.md"
    releasing = path.read_text(encoding="utf-8")
    normalized = re.sub(r"\s+", " ", releasing)
    assert (
        "A promotion release applies only when a future release contract opts into "
        "the promotion schema"
    ) in normalized
    assert "v0.2.5 is published and immutable" in normalized
    assert "No future release version has been selected or contracted" in normalized
    assert "The workflow does not add a speculative v0.2.6 contract" in normalized
    assert "gh workflow run publish-release.yml --ref main" in releasing
    assert "R='<40-character-SHA>'" in releasing
    assert 'release_ref="$R"' in releasing
    assert (
        "The configured human approval must bind that exact `S`; it is separate "
        "from the later release-owner GO"
    ) in normalized
    assert (
        "`R` may differ from reviewed `S` only for the contract-allowed approval "
        "and evidence updates and existing release-plan checkbox transitions"
    ) in normalized
    assert (
        "freeze package source at exact `R` and the committed "
        "`release_notes/vX.Y.Z.md` before candidate dispatch"
    ) in normalized
    assert "candidate run is completed with conclusion success" in normalized
    assert "`run_attempt: 1`" in normalized
    assert "promotion manifest artifact has been uploaded" in normalized
    assert normalized.index("promotion manifest artifact has been uploaded") < (
        normalized.index("issue the release-owner GO")
    )
    assert "All three conditions must hold" in normalized
    assert "canonical `promotion-manifest.json` bytes" in normalized
    assert (
        "GitHub artifact ZIP's API digest is a separate artifact-metadata check"
        in normalized
    )

    go_record = "\n".join(
        [
            "GWEXPY-RELEASE-GO-v1",
            "version=<vX.Y.Z>",
            "source_sha=<R-full-SHA>",
            "candidate_run_id=<run-id>",
            "promotion_manifest_sha256=<promotion-manifest-SHA-256>",
            "sdist_sha256=<sdist-SHA-256>",
            "wheel_sha256=<wheel-SHA-256>",
            "decision=GO",
        ]
    )
    tag_record = "\n".join(
        [
            "GWEXPY-PROMOTION-v1",
            "repository=tatsuki-washimi/gwexpy",
            "tag=<vX.Y.Z>",
            "source_sha=<R-full-SHA>",
            "candidate_run_id=<run-id>",
            "promotion_manifest_sha256=<promotion-manifest-SHA-256>",
            "release_go_comment_id=<comment-id>",
        ]
    )
    assert go_record in releasing
    assert tag_record in releasing
    assert 'git tag -a "$TAG" "$R" -F /tmp/promotion-tag.txt' in releasing
    assert 'git push origin "refs/tags/$TAG"' in releasing
    assert (
        "promotion manifest by its artifact ID from the original candidate run "
        "(attempt one)"
    ) in normalized
    assert (
        "API metadata for every manifest-bound payload, sidecar, and gate-evidence "
        "artifact"
    ) in normalized
    assert (
        "candidate workflow uploads the payload, sidecars, aggregate gate evidence, "
        "and promotion manifest with `retention-days: 90`"
    ) in normalized
    assert (
        "each required aggregate evidence artifact's measured `expires_at - "
        "created_at` to be at least 90 days"
    ) in normalized
    assert (
        "any required candidate artifact that is expired, unavailable, or not "
        "downloadable by its recorded ID fails promotion"
    ) in normalized
    assert "an artifact from another run cannot replace it" in normalized
    assert (
        "complete a new candidate qualification and obtain a new exact-candidate "
        "GO bound to that candidate run, manifest, and distribution hashes"
    ) in normalized
    assert (
        "For legacy contract runs, the measured `90 days - 5 minutes` threshold "
        "above applies to the legacy integration aggregate"
    ) in normalized
    assert (
        "These historical legacy checks do not change the separate 90-day "
        "retention and tag-time availability requirements for future promotion "
        "contracts"
    ) in normalized

    for required in (
        "manifest-bound payload and sidecars by their exact artifact IDs from that "
        "same candidate run",
        "without rebuilding",
        "check the exact tag and target `R`, committed release notes, exact five "
        "assets, and downloaded bytes",
        "the manifest's sdist filename, wheel filename",
        "`distribution-sha256.json`",
        "`LICENSE.sha256`",
        "`promotion-manifest.json`",
        "Release is idempotent only when its target, notes, exact assets, and bytes "
        "all match",
        "exactly two PyPI files",
        "without a publishing credential",
        "bounded retry window",
        "identity or hash mismatch fails immediately",
    ):
        assert required in normalized


def test_tag_push_creates_verified_github_release_before_pypi_but_dispatch_stays_dry_run():
    workflow = read_workflow()
    jobs = workflow.split("\njobs:\n", maxsplit=1)[1]
    release = jobs.split("\n  github_release:\n", maxsplit=1)[1].split(
        "\n  publish:\n", maxsplit=1
    )[0]
    legacy = jobs.split("\n  github_release_legacy:\n", maxsplit=1)[1].split(
        "\n  publish_legacy:\n", maxsplit=1
    )[0]
    publish = jobs.split("\n  publish:\n", maxsplit=1)[1].split(
        "\n  github_release_legacy:\n", maxsplit=1
    )[0]
    legacy_publish = jobs.split("\n  publish_legacy:\n", maxsplit=1)[1]

    required_gates = {
        "verify",
        "build",
        "smoke",
        "qualify",
        "qualification_evidence",
        "diaggui_qualification_evidence",
        "cross_format_io_evidence",
        "historical_74_gate",
        "evidence",
    }
    assert "needs: [verify, promotion_verify]" in release
    legacy_needs = legacy.split("    needs: ", maxsplit=1)[1].splitlines()[0]
    assert all(name in legacy_needs for name in required_gates)
    assert (
        "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
        in release
    )
    assert "contents: write" in release
    assert "EXPECTED_SOURCE_SHA" in release
    assert "refs/tags/$EXPECTED_TAG^{}" in release
    assert "git cat-file -t" in legacy
    assert "gh release create" in release
    assert "--verify-tag" in release
    assert '--target "$EXPECTED_SOURCE_SHA"' in release
    assert "gh api --paginate --slurp" in legacy
    assert "gh release download" in release
    assert "validate_github_release_readback" in release
    assert "validate_tag_identity" in release
    assert "release-readback.json" in release
    assert "distribution-sha256.json" in release
    assert "LICENSE.sha256" in release
    assert 'cp "release_notes/v$EXPECTED_VERSION.md"' in release
    assert "tools/gen_release_notes.py --version" not in release

    assert "needs: [verify, promotion_verify, github_release]" in publish
    legacy_publish_needs = legacy_publish.split("    needs: ", maxsplit=1)[
        1
    ].splitlines()[0]
    assert "github_release_legacy" in legacy_publish_needs
    assert (
        "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v')"
        in publish
    )
    assert "id-token: write" in publish
    assert "contents: read" in publish
    assert "contents: write" not in publish
    assert "workflow_dispatch" not in publish

    import yaml

    parsed = yaml.safe_load(workflow)
    exact_needs = {
        "verify",
        "build",
        "smoke",
        "qualify",
        "qualification_evidence",
        "diaggui_qualification_evidence",
        "cross_format_io_evidence",
        "historical_74_gate",
        "evidence",
    }
    assert set(parsed["jobs"]["github_release"]["needs"]) == {
        "verify",
        "promotion_verify",
    }
    assert set(parsed["jobs"]["publish"]["needs"]) == {
        "verify",
        "promotion_verify",
        "github_release",
    }
    assert set(parsed["jobs"]["github_release_legacy"]["needs"]) == exact_needs
    assert set(parsed["jobs"]["publish_legacy"]["needs"]) == exact_needs | {
        "github_release_legacy"
    }


def test_release_and_pypi_jobs_have_separate_least_permissions():
    workflow = read_workflow()
    jobs = workflow.split("\njobs:\n", maxsplit=1)[1]
    release = jobs.split("\n  github_release:\n", maxsplit=1)[1].split(
        "\n  publish:\n", maxsplit=1
    )[0]
    publish = jobs.split("\n  publish:\n", maxsplit=1)[1].split(
        "\n  github_release_legacy:\n", maxsplit=1
    )[0]
    assert "    permissions:\n      contents: write" in release
    assert "id-token: write" not in release
    assert (
        "    permissions:\n      contents: read\n      actions: read\n      id-token: write"
        in publish
    )
    assert "contents: write" not in publish


def test_pypi_readback_runs_after_publish_with_read_only_access_and_exact_inputs():
    import yaml

    jobs = yaml.safe_load(read_workflow())["jobs"]
    readback = jobs["pypi_readback"]

    assert set(readback["needs"]) == {
        "verify",
        "promotion_verify",
        "github_release",
        "publish",
    }
    assert "needs.publish.result == 'success'" in readback["if"]
    assert "needs.verify.outputs.promotion_enabled == 'true'" in readback["if"]
    assert readback["permissions"] == {"contents": "read", "actions": "read"}
    assert readback.get("environment") is None
    assert readback.get("continue-on-error") is not True

    steps = readback["steps"]
    downloads = [
        step
        for step in steps
        if step.get("uses", "").startswith("actions/download-artifact@")
    ]
    assert {step["with"]["artifact-ids"] for step in downloads} >= {
        "${{ needs.promotion_verify.outputs.payload_artifact_id }}",
        "${{ needs.promotion_verify.outputs.manifest_artifact_id }}",
    }
    release_download = next(
        step for step in steps if "gh release download" in step.get("run", "")
    )
    assert "$EXPECTED_TAG" in release_download["run"]
    validator_step = next(
        step for step in steps if "validate_pypi_readback.py" in step.get("run", "")
    )
    assert "--manifest" in validator_step["run"]
    assert "--github-release-dir" in validator_step["run"]
    assert "--payload-dir" in validator_step["run"]
    assert "--output-dir" in validator_step["run"]
    assert validator_step.get("continue-on-error") is not True
    assert "--role promotion-manifest" in validator_step["run"]
    assert '--manifest-sha256 "$MANIFEST_SHA256"' in validator_step["run"]


def test_pypi_readback_workflow_keeps_missing_or_extra_file_a_closure_failure():
    import yaml

    validator_path = WORKFLOW.parents[2] / "scripts/ci/validate_pypi_readback.py"
    spec = importlib.util.spec_from_file_location(
        "workflow_pypi_readback", validator_path
    )
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)

    jobs = yaml.safe_load(read_workflow())["jobs"]
    readback = jobs["pypi_readback"]
    step = next(
        item
        for item in readback["steps"]
        if "validate_pypi_readback.py" in item.get("run", "")
    )
    assert "validate_pypi_readback.py" in step["run"]
    assert step.get("continue-on-error") is not True
    assert "needs.publish.result == 'success'" in readback["if"]

    version = "99.88.77"
    payload = {
        f"gwexpy-{version}-py3-none-any.whl": b"synthetic wheel",
        f"gwexpy-{version}.tar.gz": b"synthetic sdist",
    }
    manifest = {
        "schema": "gwexpy-release-promotion-manifest-v1",
        "version": version,
        "artifacts": [
            {
                "role": "payload",
                "files": [
                    {
                        "name": name,
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "size_in_bytes": len(data),
                    }
                    for name, data in payload.items()
                ],
            }
        ],
    }
    metadata = {
        "info": {"name": "gwexpy", "version": version},
        "urls": [
            {
                "filename": name,
                "digests": {"sha256": hashlib.sha256(data).hexdigest()},
                "url": f"https://files.pythonhosted.org/packages/test/{name}",
            }
            for name, data in payload.items()
        ],
    }
    for invalid_files in (
        {name: data for name, data in payload.items() if name.endswith(".whl")},
        {**payload, "gwexpy-99.88.77-extra.tar.gz": b"extra"},
    ):
        with pytest.raises(validator.PypiReadbackError):
            validator.validate_pypi_readback(
                metadata=metadata,
                manifest=manifest,
                payload_files=payload,
                downloaded_files=invalid_files,
                github_release_files=payload,
            )


def _load_release_readback_validator():
    path = (
        WORKFLOW.parents[2] / "scripts" / "ci" / "validate_github_release_readback.py"
    )
    spec = importlib.util.spec_from_file_location("release_readback_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_existing_release_lookup_requires_one_well_formed_exact_tag():
    validator = _load_release_readback_validator()
    release = {"id": 12, "tag_name": "v0.2.6"}
    assert validator.find_release([[release]], "v0.2.6") == release
    assert validator.find_release([[{"tag_name": "v0.2.5"}]], "v0.2.6") is None
    with pytest.raises(validator.ReleaseReadbackError, match="multiple"):
        validator.find_release([[release], [release]], "v0.2.6")
    with pytest.raises(validator.ReleaseReadbackError, match="malformed"):
        validator.find_release([[{"name": "missing tag name"}]], "v0.2.6")


def test_release_readback_validator_accepts_only_exact_uploaded_same_run_assets(
    tmp_path,
):
    validator = _load_release_readback_validator()
    payload = tmp_path / "payload"
    sidecars = tmp_path / "sidecars"
    downloaded = tmp_path / "downloaded"
    for directory in (payload, sidecars, downloaded):
        directory.mkdir()
    source_files = {
        payload / "gwexpy-0.2.5-py3-none-any.whl": b"wheel bytes",
        payload / "gwexpy-0.2.5.tar.gz": b"sdist bytes",
        sidecars / "distribution-sha256.json": b'{"source_sha":"abc"}\n',
        sidecars / "LICENSE.sha256": b"license digest\n",
        sidecars / "promotion-manifest.json": b'{"schema":"synthetic"}\n',
    }
    for source, data in source_files.items():
        source.write_bytes(data)
        (downloaded / source.name).write_bytes(data)
    assets = [
        {
            "id": index,
            "name": path.name,
            "state": "uploaded",
            "size": path.stat().st_size,
            "digest": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
        }
        for index, path in enumerate(source_files, start=1)
    ]
    release = {
        "id": 17,
        "tag_name": "v0.2.5",
        "name": "v0.2.5",
        "target_commitish": "a" * 40,
        "draft": False,
        "prerelease": False,
        "body": "Generated CHANGELOG notes\n",
        "assets": assets,
    }
    kwargs = dict(
        release=release,
        expected_tag="v0.2.5",
        expected_source_sha="a" * 40,
        notes="Generated CHANGELOG notes\n",
        payload_dir=payload,
        sidecars_dir=sidecars,
        downloaded_dir=downloaded,
    )
    validator.validate_release_readback(**kwargs)

    wrong_notes = dict(kwargs, release={**release, "body": "different notes"})
    wrong_tag = dict(kwargs, release={**release, "tag_name": "v0.2.4"})
    wrong_target = dict(kwargs, release={**release, "target_commitish": "b" * 40})
    wrong_state = dict(
        kwargs,
        release={**release, "assets": [{**assets[0], "state": "starter"}, *assets[1:]]},
    )
    wrong_size = dict(
        kwargs, release={**release, "assets": [{**assets[0], "size": 0}, *assets[1:]]}
    )
    wrong_digest = dict(
        kwargs,
        release={
            **release,
            "assets": [{**assets[0], "digest": "sha256:" + "0" * 64}, *assets[1:]],
        },
    )
    wrong_bytes = dict(kwargs)
    for invalid in (
        wrong_notes,
        wrong_tag,
        wrong_target,
        wrong_state,
        wrong_size,
        wrong_digest,
    ):
        with pytest.raises(validator.ReleaseReadbackError):
            validator.validate_release_readback(**invalid)

    (downloaded / "gwexpy-0.2.5-py3-none-any.whl").write_bytes(b"wrong wheel")
    with pytest.raises(validator.ReleaseReadbackError):
        validator.validate_release_readback(**wrong_bytes)

    (downloaded / "gwexpy-0.2.5-py3-none-any.whl").write_bytes(b"wheel bytes")
    duplicate = dict(kwargs, release={**release, "assets": [*assets, assets[0]]})
    missing = dict(kwargs, release={**release, "assets": assets[:-1]})
    extra = dict(
        kwargs,
        release={
            **release,
            "assets": [*assets, {**assets[0], "id": 99, "name": "unexpected.txt"}],
        },
    )
    with pytest.raises(validator.ReleaseReadbackError):
        validator.validate_release_readback(**duplicate)
    with pytest.raises(validator.ReleaseReadbackError):
        validator.validate_release_readback(**missing)
    with pytest.raises(validator.ReleaseReadbackError):
        validator.validate_release_readback(**extra)
    validator.validate_release_readback(**kwargs, existing_release=True)
    for conflict in (wrong_notes, wrong_target, duplicate):
        with pytest.raises(validator.ReleaseReadbackError):
            validator.validate_release_readback(**conflict, existing_release=True)

    (sidecars / "promotion-manifest.json").unlink()
    (downloaded / "promotion-manifest.json").unlink()
    legacy_assets = [
        asset for asset in assets if asset["name"] != "promotion-manifest.json"
    ]
    legacy_release = {**release, "assets": legacy_assets}
    validator.validate_release_readback(
        **{**kwargs, "release": legacy_release}, existing_release=True
    )


def test_release_readback_validator_rejects_conflict_api_errors_and_changed_annotated_tag():
    validator = _load_release_readback_validator()
    with pytest.raises(validator.ReleaseReadbackError):
        validator.validate_no_conflicting_release([[{"tag_name": "v0.2.5"}]], "v0.2.5")
    with pytest.raises(validator.ReleaseReadbackError):
        validator.validate_no_conflicting_release(
            {"message": "API unavailable"}, "v0.2.5"
        )
    validator.validate_no_conflicting_release([[{"tag_name": "v0.2.4"}]], "v0.2.5")
    validator.validate_tag_identity(
        tag_object_sha="b" * 40,
        peeled_sha="a" * 40,
        expected_tag_object_sha="b" * 40,
        expected_source_sha="a" * 40,
    )
    for tag_object_sha, peeled_sha in (
        ("", "a" * 40),
        ("c" * 40, "a" * 40),
        ("b" * 40, "d" * 40),
    ):
        with pytest.raises(validator.ReleaseReadbackError):
            validator.validate_tag_identity(
                tag_object_sha=tag_object_sha,
                peeled_sha=peeled_sha,
                expected_tag_object_sha="b" * 40,
                expected_source_sha="a" * 40,
            )


def test_promotion_tag_graph_verifies_candidate_before_release_and_uses_exact_ids():
    import yaml

    jobs = yaml.safe_load(read_workflow())["jobs"]
    verification = jobs["promotion_verify"]
    assert verification["if"] == (
        "${{ github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v') "
        "&& needs.verify.outputs.promotion_enabled == 'true' }}"
    )
    assert verification["permissions"] == {
        "contents": "read",
        "actions": "read",
        "issues": "read",
    }
    assert "verify" in verification["needs"]
    verify_run = "\n".join(step.get("run", "") for step in verification["steps"])
    assert "release_promotion.py verify-tag" in verify_run
    assert "verify_release_human_approval.py" in verify_run
    promotion_source = (
        WORKFLOW.parents[2] / "scripts/ci/release_promotion.py"
    ).read_text()
    assert "parse_promotion_tag" in promotion_source
    assert "validate_candidate_run" in promotion_source
    assert "verify_release_go" in promotion_source

    release = jobs["github_release"]
    assert "promotion_verify" in release["needs"]
    assert not {
        "build",
        "smoke",
        "qualify",
        "qualification_evidence",
        "diaggui_qualification_evidence",
        "cross_format_io_evidence",
        "historical_74_gate",
        "evidence",
    } & set(release["needs"])
    release_downloads = [
        step
        for step in release["steps"]
        if step.get("uses", "").startswith("actions/download-artifact@")
    ]
    assert len(release_downloads) == 4
    assert all("artifact-ids" in step.get("with", {}) for step in release_downloads)
    assert all("run-id" in step.get("with", {}) for step in release_downloads)

    publish = jobs["publish"]
    assert "promotion_verify" in publish["needs"]
    publish_downloads = [
        step
        for step in publish["steps"]
        if step.get("uses", "").startswith("actions/download-artifact@")
    ]
    assert len(publish_downloads) == 4
    assert all("artifact-ids" in step.get("with", {}) for step in publish_downloads)
    assert all("run-id" in step.get("with", {}) for step in publish_downloads)


def test_promoted_release_uses_manifest_bound_committed_notes():
    import yaml

    release = yaml.safe_load(read_workflow())["jobs"]["github_release"]
    notes_step = next(
        step
        for step in release["steps"]
        if step.get("name")
        == "Use committed notes and check for an existing exact Release"
    )
    run = notes_step["run"]
    assert 'cp "release_notes/v$EXPECTED_VERSION.md"' in run
    assert "gen_release_notes.py" not in run


def test_tag_promotion_path_retains_legacy_historical_release_behavior():
    import yaml

    jobs = yaml.safe_load(read_workflow())["jobs"]
    legacy = jobs["github_release_legacy"]
    assert "promotion_enabled != 'true'" in legacy["if"]
    assert {
        "verify",
        "build",
        "smoke",
        "qualify",
        "qualification_evidence",
        "diaggui_qualification_evidence",
        "cross_format_io_evidence",
        "historical_74_gate",
        "evidence",
    } <= set(legacy["needs"])
    assert "github_release_legacy" in jobs["publish_legacy"]["needs"]


def test_releasing_documents_strict_v023_source_to_evidence_transition():
    releasing = re.sub(
        r"\s+", " ", (WORKFLOW.parents[2] / "RELEASING.md").read_text(encoding="utf-8")
    )

    assert "For v0.2.3, `S` and `R` must be distinct commits." in releasing
    assert "byte-exact empty placeholder" in releasing
    assert "absent, already populated, malformed, or contains extra YAML" in releasing


def releasing_readback(*, collapse: bool) -> str:
    """Return the readback section, optionally with wrapping collapsed.

    Prose assertions use the collapsed form so they do not depend on line
    breaks; table rows are read from the raw form, because a Markdown row is
    a single line and collapsing would merge the two rulesets' rows together.
    """
    text = (WORKFLOW.parents[2] / "RELEASING.md").read_text(encoding="utf-8")
    if collapse:
        text = re.sub(r"\s+", " ", text)
    return text[text.index("### Readback commands") :]


def test_releasing_doc_requires_full_ruleset_readback_fields():
    """`rules[].type` alone does not prove a ruleset constrains anything.

    An `evaluate` ruleset, one targeting branches, or one whose conditions
    miss `refs/tags/v*` can carry exactly the right rules and still permit the
    tag operations they name. The readback checklist must therefore pin every
    field an auditor has to look at.
    """
    readback = releasing_readback(collapse=True)
    for field in (
        "`enforcement`",
        "`active`",
        "`target`",
        "`conditions.ref_name.include`",
        "`refs/tags/v*`",
        "`bypass_actors`",
    ):
        assert field in readback, field
    # The Trusted Publisher binding has no GitHub API readback, so the doc
    # must say where to confirm it instead of implying the commands cover it.
    assert "not readable through the GitHub API" in readback


def test_readback_requires_opposite_bypass_actor_policies_per_ruleset():
    """A single `bypass_actors` rule for both rulesets is wrong either way.

    `creation`, `update`, and `deletion` restrict an operation *to* the
    bypass actors rather than forbidding it, so the two rulesets need
    opposite states: `release-tags-create-admin-only` must enumerate the
    permitted creators (empty locks out the release), while
    `release-tags-integrity` must be empty (any entry can move or delete a
    published tag). This asserts each ruleset's own row, so swapping the two
    policies fails rather than passing on a shared substring.
    """
    rows = {
        name: next(
            line
            for line in releasing_readback(collapse=False).splitlines()
            if line.startswith(f"| `{name}`")
        )
        for name in ("release-tags-create-admin-only", "release-tags-integrity")
    }
    creation, integrity = rows.values()

    # Each ruleset names only its own rules; a row listing the other's rules
    # would mean the responsibilities have been merged or transposed.
    assert "`creation`" in creation
    for rule in ("`update`", "`deletion`", "`non_fast_forward`"):
        assert rule not in creation, rule
    for rule in ("`update`", "`deletion`", "`non_fast_forward`"):
        assert rule in integrity, rule
    assert "`creation`" not in integrity
    # `tag_name_pattern` belongs to neither row: the API refuses the rule type
    # on this repository, so requiring it here would fail every readback on a
    # check that can never pass. See the dedicated test below.
    assert "`tag_name_pattern`" not in integrity

    # Opposite bypass requirements, each stated with its failure mode.
    assert "enumerated" in creation
    assert "An empty list here means" in creation
    assert "Empty." in integrity
    assert "may move or delete a published release tag" in integrity

    prose = releasing_readback(collapse=True)
    # Enumerated actors are only auditable with the fields that identify them
    # and say whether the bypass is conditional.
    for field in ("`actor_type`", "`actor_id`", "`bypass_mode`"):
        assert field in prose, field
    assert "break-glass" in prose


def test_readback_explains_the_unavailable_tag_name_rule_and_its_substitute():
    """An absent control must be explained, not silently dropped.

    `tag_name_pattern` cannot be configured on this repository -- the API
    rejects the rule type regardless of parameters -- so the readback table
    omits it. Omitting it without a reason would read as an oversight to the
    next auditor, who would then either re-attempt the impossible change or
    record a false finding. The doc must therefore say it is unavailable and
    name what enforces the tag name in its place, so the absence is auditable
    as a deliberate state rather than a gap.
    """
    prose = releasing_readback(collapse=True)
    assert "`tag_name_pattern` is not available on this repository" in prose
    # The evidence, so a future reader need not rediscover it by trying again.
    assert "Invalid rule 'tag_name_pattern'" in prose
    # Each substitute control, named where an auditor can verify it.
    assert "release-tags-create-admin-only" in prose
    assert "RELEASE_TAG_PATTERN" in prose
    assert "`github.ref_name`" in prose
    # The residual gap is stated rather than implied to be closed.
    assert "residual gap" in prose.lower()


def test_verify_pins_python_before_running_the_validator():
    """The validator needs Python 3.11+ (`datetime.UTC`), so verify pins it.

    Without an explicit `setup-python`, the validator would run on whatever
    interpreter the runner image ships, letting an image update silently
    break release verification while build/smoke stay pinned.
    """
    workflow = read_workflow()
    verify = workflow.split("\n  verify:\n", maxsplit=1)[1].split("\n  build:\n")[0]
    setup_python = verify.index("actions/setup-python@")
    # The invocation path, not the bare script name: the latter also appears
    # in the explanatory comment above the setup-python step.
    validator = verify.index("python control/scripts/validate_release.py")
    assert setup_python < validator
    assert verify.count('python-version: "3.11"') == 1


def test_release_smoke_covers_both_artifacts_on_python_311_and_312():
    workflow = read_workflow()
    smoke = workflow.split("\n  smoke:\n", maxsplit=1)[1].split("\n  publish:\n")[0]

    for token in (
        'python-version: ["3.11", "3.12"]',
        "distribution: [wheel, sdist]",
        "${{ matrix.python-version }}",
        "${{ matrix.distribution }}",
        "gwexpy.register_all()",
        "LICENSE.sha256",
        "distribution-sha256.json",
        "retention-days: 90",
    ):
        assert token in workflow if token == "retention-days: 90" else token in smoke


def test_v022_through_v024_release_qualification_share_exact_nineteen_cells():
    import yaml

    workflow = yaml.safe_load(read_workflow())
    matrix = workflow["jobs"]["qualify"]["strategy"]["matrix"]["include"]

    assert len(matrix) == 19
    allowlist = "${{ (github.event_name == 'workflow_dispatch' || (github.event_name == 'push' && needs.verify.outputs.promotion_enabled != 'true')) && (needs.verify.outputs.version == '0.2.2' || needs.verify.outputs.version == '0.2.3' || needs.verify.outputs.version == '0.2.4' || needs.verify.outputs.version == '0.2.5' || needs.verify.outputs.qualify == 'true') }}"
    assert workflow["jobs"]["qualify"]["if"] == allowlist
    assert workflow["jobs"]["qualification_evidence"]["if"] == allowlist
    assert matrix == [
        {
            "cell": "install-ubuntu-3.11-wheel",
            "os": "ubuntu-latest",
            "python": "3.11",
            "distribution": "wheel",
            "runtime": "pip",
            "extras": "pytest",
        },
        {
            "cell": "install-ubuntu-3.11-sdist",
            "os": "ubuntu-latest",
            "python": "3.11",
            "distribution": "sdist",
            "runtime": "pip",
            "extras": "pytest",
        },
        {
            "cell": "install-ubuntu-3.12-wheel",
            "os": "ubuntu-latest",
            "python": "3.12",
            "distribution": "wheel",
            "runtime": "pip",
            "extras": "pytest",
        },
        {
            "cell": "install-ubuntu-3.12-sdist",
            "os": "ubuntu-latest",
            "python": "3.12",
            "distribution": "sdist",
            "runtime": "pip",
            "extras": "pytest",
        },
        {
            "cell": "install-ubuntu-3.13-wheel",
            "os": "ubuntu-latest",
            "python": "3.13",
            "distribution": "wheel",
            "runtime": "pip",
            "extras": "pytest",
        },
        {
            "cell": "install-ubuntu-3.13-sdist",
            "os": "ubuntu-latest",
            "python": "3.13",
            "distribution": "sdist",
            "runtime": "pip",
            "extras": "pytest",
        },
        {
            "cell": "install-ubuntu-3.14-wheel",
            "os": "ubuntu-latest",
            "python": "3.14",
            "distribution": "wheel",
            "runtime": "pip",
            "extras": "pytest",
        },
        {
            "cell": "install-ubuntu-3.14-sdist",
            "os": "ubuntu-latest",
            "python": "3.14",
            "distribution": "sdist",
            "runtime": "pip",
            "extras": "pytest",
        },
        {
            "cell": "install-macos-3.11-wheel",
            "os": "macos-latest",
            "python": "3.11",
            "distribution": "wheel",
            "runtime": "pip",
            "extras": "pytest",
        },
        {
            "cell": "install-macos-3.14-wheel",
            "os": "macos-latest",
            "python": "3.14",
            "distribution": "wheel",
            "runtime": "pip",
            "extras": "pytest",
        },
        {
            "cell": "install-windows-3.11-wheel",
            "os": "windows-latest",
            "python": "3.11",
            "distribution": "wheel",
            "runtime": "pip",
            "extras": "pytest",
        },
        {
            "cell": "install-windows-3.14-wheel",
            "os": "windows-latest",
            "python": "3.14",
            "distribution": "wheel",
            "runtime": "pip",
            "extras": "pytest",
        },
        {
            "cell": "gwpy-4.0.1-wheel",
            "os": "ubuntu-latest",
            "python": "3.11",
            "distribution": "wheel",
            "runtime": "pip",
            "extras": "pytest gwpy==4.0.1",
        },
        {
            "cell": "gwpy-4.0.2-wheel",
            "os": "ubuntu-latest",
            "python": "3.11",
            "distribution": "wheel",
            "runtime": "pip",
            "extras": "pytest gwpy==4.0.2",
        },
        {
            "cell": "sdist-3.12-claims",
            "os": "ubuntu-latest",
            "python": "3.12",
            "distribution": "sdist",
            "runtime": "pip",
            "extras": "pytest",
        },
        {
            "cell": "conda-3.11",
            "os": "ubuntu-latest",
            "python": "3.11",
            "distribution": "wheel",
            "runtime": "conda",
            "extras": "pytest",
        },
        {
            "cell": "conda-3.14",
            "os": "ubuntu-latest",
            "python": "3.14",
            "distribution": "wheel",
            "runtime": "conda",
            "extras": "pytest",
        },
        {
            "cell": "scientific-3.11-wheel",
            "os": "ubuntu-latest",
            "python": "3.11",
            "distribution": "wheel",
            "runtime": "pip",
            "extras": "pytest mne lalsuite",
        },
        {
            "cell": "docs-en-ja-3.11-wheel",
            "os": "ubuntu-latest",
            "python": "3.11",
            "distribution": "wheel",
            "runtime": "pip",
            "extras": "pytest matplotlib",
        },
    ]
    text = read_workflow()
    qualify = text.split("\n  qualify:\n", maxsplit=1)[1].split(
        "\n  qualification_evidence:\n", maxsplit=1
    )[0]
    assert "release-payload-${{ needs.verify.outputs.source_sha }}" in qualify
    assert "distribution-sha256.json" in qualify
    assert "Verify qualification payload digest" in qualify
    assert "tests/timeseries/test_gwpy_behavioral_compatibility.py" in qualify
    assert '--junitxml="$JUNIT"' in qualify
    assert "v023_qualification_expected_skips.json" in qualify
    assert "v024_qualification_expected_skips.json" in qualify
    assert "qualification_evidence.py record" in qualify
    assert "needs.verify.outputs.version == '0.2.3'" in qualify
    publish = text.split("\n  publish_legacy:\n", maxsplit=1)[1]
    assert "qualification_evidence" in publish.split("\n    if:", maxsplit=1)[0]


def test_qualification_evidence_switch_is_fail_closed_and_versioned():
    import yaml

    text = read_workflow()
    workflow = yaml.safe_load(text)
    qualify_job = workflow["jobs"]["qualify"]
    run_step = next(
        step
        for step in qualify_job["steps"]
        if step["name"] == "Run installed candidate compatibility contracts"
    )
    assert "if" not in run_step
    assert "0.2.3)" in run_step["run"]
    assert "0.2.4|0.2.5)" in run_step["run"]
    assert '--junitxml="$JUNIT"' in run_step["run"]

    aggregate = text.split("\n  qualification_evidence:\n", maxsplit=1)[1].split(
        "\n  evidence:\n", maxsplit=1
    )[0]

    assert "qualification_evidence.py contract" in aggregate
    assert "qualification_evidence.py aggregate" in aggregate
    assert "v023_qualification_expected_skips.json" in aggregate
    assert "v024_qualification_expected_skips.json" in aggregate
    assert "steps.qualification_contract.outputs.artifact_prefix" in aggregate
    assert "continue-on-error" not in aggregate
    assert "always()" not in aggregate

    publish = text.split("\n  publish_legacy:\n", maxsplit=1)[1]
    needs = publish.split("\n    if:", maxsplit=1)[0]
    assert (
        "needs: [verify, build, smoke, qualify, qualification_evidence, diaggui_qualification_evidence, cross_format_io_evidence, historical_74_gate, evidence, github_release_legacy]"
        in needs
    )


def test_v024_diaggui_lane_is_four_cell_digest_bound_and_required_for_publish():
    import yaml

    workflow = yaml.safe_load(read_workflow())
    jobs = workflow["jobs"]
    cells = jobs["diaggui_qualification"]["strategy"]["matrix"]["include"]

    assert cells == [
        {"cell": "base-wheel", "mode": "base", "distribution": "wheel"},
        {"cell": "base-sdist", "mode": "base", "distribution": "sdist"},
        {"cell": "dttxml-wheel", "mode": "dttxml", "distribution": "wheel"},
        {"cell": "dttxml-sdist", "mode": "dttxml", "distribution": "sdist"},
    ]
    assert jobs["diaggui_qualification"]["needs"] == ["verify", "build"]
    assert jobs["diaggui_qualification_evidence"]["needs"] == [
        "verify",
        "build",
        "diaggui_qualification",
    ]
    assert "diaggui_qualification_evidence" in jobs["publish_legacy"]["needs"]
    assert (
        "needs.diaggui_qualification.result == 'success'"
        in jobs["diaggui_qualification_evidence"]["if"]
    )
    diag_run = "\n".join(
        step.get("run", "") for step in jobs["diaggui_qualification"]["steps"]
    )
    assert "validate_release_payload.py" in diag_run
    assert "dttxml==1.1.8" in diag_run
    assert "diaggui_qualification_evidence.py test-nodes" in diag_run
    assert "diaggui_qualification_evidence.py record" in diag_run
    assert "diaggui_qualification_evidence.py aggregate" in "\n".join(
        step.get("run", "") for step in jobs["diaggui_qualification_evidence"]["steps"]
    )
    assert "always()" not in read_workflow()
    assert "continue-on-error" not in read_workflow()


def test_v024_diaggui_copies_every_file_referenced_by_node_lists():
    import yaml

    workflow = yaml.safe_load(read_workflow())
    diag_run = "\n".join(
        step.get("run", "")
        for step in workflow["jobs"]["diaggui_qualification"]["steps"]
    )
    copied_files = set(
        re.findall(
            r'^[ \t]*cp source/tests/io/([^ \t]+) "\$tests_dir/io/"$',
            diag_run,
            flags=re.MULTILINE,
        )
    )
    script = (
        WORKFLOW.parents[2] / "scripts" / "ci" / "diaggui_qualification_evidence.py"
    )
    node_paths = set()
    for mode in ("base", "dttxml"):
        nodes = subprocess.check_output(
            [sys.executable, str(script), "test-nodes", "--mode", mode],
            text=True,
        ).splitlines()
        node_paths.update(node.split("::", maxsplit=1)[0] for node in nodes)

    assert node_paths
    assert all(path.startswith("io/") for path in node_paths)
    node_files = {path.removeprefix("io/") for path in node_paths}
    assert copied_files == node_files


def test_v024_human_approval_verifier_uses_github_read_permission_and_canonical_path():
    import yaml

    workflow = yaml.safe_load(read_workflow())
    verify = workflow["jobs"]["verify"]
    assert verify["permissions"] == {"contents": "read", "issues": "read"}
    validate = next(step for step in verify["steps"] if step.get("id") == "validate")
    verifier = next(
        step
        for step in verify["steps"]
        if step["name"] == "Verify GitHub human approval for the reviewed source"
    )
    assert "--review-evidence-path" in validate["run"]
    assert '--review-evidence "$canonical_review_evidence"' in validate["run"]
    assert (
        verifier["if"]
        == "steps.validate.outputs.version == '0.2.4' || steps.validate.outputs.version == '0.2.5' || steps.profile.outputs.promotion_enabled == 'true'"
    )
    assert "verify_release_human_approval.py" in verifier["run"]
    assert verifier["env"]["GITHUB_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"


def test_v023_expected_skip_baseline_declares_lf_checkout_contract():
    repository = WORKFLOW.parents[2]
    attributes = repository / ".gitattributes"
    declarations = attributes.read_text(encoding="utf-8").splitlines()

    assert (
        "scripts/ci/v023_qualification_expected_skips.json text eol=lf" in declarations
    )

    result = subprocess.run(
        [
            "git",
            "check-attr",
            "text",
            "eol",
            "--",
            "scripts/ci/v023_qualification_expected_skips.json",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    assert "scripts/ci/v023_qualification_expected_skips.json: text: set" in lines
    assert "scripts/ci/v023_qualification_expected_skips.json: eol: lf" in lines


def test_v024_expected_skip_baseline_declares_lf_checkout_contract():
    repository = WORKFLOW.parents[2]
    attributes = repository / ".gitattributes"
    declarations = attributes.read_text(encoding="utf-8").splitlines()

    assert (
        "scripts/ci/v024_qualification_expected_skips.json text eol=lf" in declarations
    )

    result = subprocess.run(
        [
            "git",
            "check-attr",
            "text",
            "eol",
            "--",
            "scripts/ci/v024_qualification_expected_skips.json",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    assert "scripts/ci/v024_qualification_expected_skips.json: text: set" in lines
    assert "scripts/ci/v024_qualification_expected_skips.json: eol: lf" in lines


def test_v025_expected_skip_baseline_declares_lf_checkout_contract():
    repository = WORKFLOW.parents[2]
    attributes = repository / ".gitattributes"
    declarations = attributes.read_text(encoding="utf-8").splitlines()

    assert (
        "scripts/ci/v025_qualification_expected_skips.json text eol=lf" in declarations
    )

    result = subprocess.run(
        [
            "git",
            "check-attr",
            "text",
            "eol",
            "--",
            "scripts/ci/v025_qualification_expected_skips.json",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    assert "scripts/ci/v025_qualification_expected_skips.json: text: set" in lines
    assert "scripts/ci/v025_qualification_expected_skips.json: eol: lf" in lines


def test_v022_historical_evidence_does_not_require_v023_candidate_files():
    import yaml

    text = read_workflow()
    workflow = yaml.safe_load(text)
    historical_record = next(
        step
        for step in workflow["jobs"]["qualify"]["steps"]
        if step["name"]
        == "Record qualification cell digest evidence (v0.2.2 historical)"
    )
    assert historical_record["if"] == "needs.verify.outputs.version == '0.2.2'"
    assert "python - <<'PY'" in historical_record["run"]
    assert "qualification_evidence.py" not in historical_record["run"]

    historical_aggregate = next(
        step
        for step in workflow["jobs"]["qualification_evidence"]["steps"]
        if step["name"] == "Assert historical v0.2.2 same-payload qualification ledger"
    )
    assert historical_aggregate["if"] == ("needs.verify.outputs.version == '0.2.2'")
    assert (
        '"schema": "gwexpy-v022-qualification-evidence-v1"'
        in historical_aggregate["run"]
    )
    assert "qualification_evidence.py" not in historical_aggregate["run"]

    aggregate_steps = workflow["jobs"]["qualification_evidence"]["steps"]
    checkout = next(
        step
        for step in aggregate_steps
        if step["name"] == "Check out verified qualification source"
    )
    setup_python = next(
        step for step in aggregate_steps if step["name"] == "Set up Python"
    )
    assert checkout["if"] == (
        "needs.verify.outputs.version == '0.2.3' || "
        "needs.verify.outputs.version == '0.2.4' || "
        "needs.verify.outputs.version == '0.2.5' || needs.verify.outputs.promotion_enabled == 'true'"
    )
    assert setup_python["if"] == checkout["if"]


def test_release_smoke_executes_with_license_sidecar_path(tmp_path):
    """The shell/Python boundary passes the sidecar path, not its hash value."""
    workflow = read_workflow()
    smoke = workflow.split("\n  smoke:\n", maxsplit=1)[1].split(
        "\n  publish:\n", maxsplit=1
    )[0]

    invocation = re.search(
        r'"\$smoke_dir/venv/bin/python" - "\$artifact" "\$(?P<name>[a-z_]+)" "\$REPORT" <<\'PY\'',
        smoke,
    )
    assert invocation is not None
    argument_name = invocation.group("name")

    license_bytes = b"release-license\n"
    license_digest = hashlib.sha256(license_bytes).hexdigest()
    license_sidecar = tmp_path / "LICENSE.sha256"
    license_sidecar.write_text(license_digest + "\n", encoding="ascii")
    artifact = tmp_path / "gwexpy-0.1.13-py3-none-any.whl"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("gwexpy-0.1.13.dist-info/licenses/LICENSE.txt", license_bytes)

    if f'{argument_name}="$(cat ' in smoke:
        license_argument = license_digest
    else:
        assert (
            f'{argument_name}="${{{{ runner.temp }}}}/release-sidecars/LICENSE.sha256"'
            in smoke
        )
        license_argument = str(license_sidecar)

    embedded = re.search(
        r"<<'PY'\n(?P<body>.*?)\n          PY",
        smoke,
        flags=re.DOTALL,
    )
    assert embedded is not None
    script = textwrap.dedent(embedded.group("body"))
    script = script.split(
        'assert gwexpy.__version__ == os.environ["EXPECTED_VERSION"]', maxsplit=1
    )[0]
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "mpl")
    result = subprocess.run(
        [
            sys.executable,
            "-",
            str(artifact),
            license_argument,
            str(tmp_path / "report.json"),
        ],
        input=script,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert argument_name == "expected_license_hash_file"


def test_releasing_manual_dispatch_supplies_v023_review_evidence():
    releasing = (WORKFLOW.parents[2] / "RELEASING.md").read_text(encoding="utf-8")
    assert (
        "-f review_evidence="
        "docs/developers/plans/manifests/"
        "audit-manifest-v0.2.3-release-readiness.yaml"
    ) in releasing


def test_partial_pypi_upload_recovery_requires_same_run_bytes_and_review():
    releasing = (WORKFLOW.parents[2] / "RELEASING.md").read_text(encoding="utf-8")
    procedure = releasing.split("## Partial PyPI upload recovery\n", maxsplit=1)[
        1
    ].split("\n## Frozen source, payload, and evidence", maxsplit=1)[0]
    normalized = re.sub(r"\s+", " ", procedure)

    for required in (
        "stop release acceptance",
        "decision on HOLD",
        "failed PyPI publisher run ID separately from the original `workflow_dispatch` candidate run ID",
        "source `R`",
        "final tag and peeled SHA",
        "`release-payload-<R>`",
        "`release-sidecar-distribution-sha256.json-<R>`",
        "`release-sidecar-LICENSE.sha256-<R>`",
        "promotion manifest SHA-256 and artifact ID",
        "manifest-bound payload and sidecar artifact IDs, names, GitHub digests, and sizes",
        "failed publisher run does not create payload or sidecar artifacts",
        "all gate reports and aggregate evidence",
        "`urls[].filename` and `urls[].digests.sha256`",
        "`files.wheel` and `files.sdist`",
        "manifest's source SHA is `R`",
        "preserved original candidate payload",
        "explicit reviewed release-owner decision",
        "only the missing file from that verified original candidate payload",
        "both expected files and hashes are present",
        "Normal successful closure requires exactly two PyPI files",
        "Do not blindly rerun the strict publish job",
        "the tag publisher reuses the same manifest-bound candidate payload bytes",
        "does not build a new payload",
        "A new candidate dispatch builds a different payload",
        "Do not rebuild the missing file",
        "`skip-existing`",
    ):
        assert required in normalized
    assert "a new run builds a new payload" not in normalized
    assert normalized.index("Compare it with both") < normalized.index(
        "explicit reviewed release-owner decision"
    )


def test_workflow_is_payload_only_locked_and_collects_same_run_evidence():
    workflow = read_workflow()
    assert "--require-hashes -r requirements/release-build.txt" in workflow
    assert "python -m build --no-isolation" in workflow
    assert "pip install --upgrade pip build twine" not in workflow
    assert "release-payload-${{ needs.verify.outputs.source_sha }}" in workflow
    assert (
        "release-sidecar-distribution-sha256.json-${{ needs.verify.outputs.source_sha }}"
        in workflow
    )
    assert (
        "release-sidecar-LICENSE.sha256-${{ needs.verify.outputs.source_sha }}"
        in workflow
    )
    publish = workflow.split("\n  publish_legacy:\n", maxsplit=1)[1]
    assert "release-payload-${{ needs.verify.outputs.source_sha }}" in publish
    assert "release-sidecars-${{ needs.verify.outputs.source_sha }}" not in publish
    assert 'find "$artifact_dir"' not in workflow
    assert "--frozen-tip" in workflow
    assert "--review-evidence" in workflow
    assert "review_evidence:" in workflow
    assert (
        "github.event_name == 'workflow_dispatch' && inputs.review_evidence" in workflow
    )
    assert "artifact_prefix: ${{ steps.validate.outputs.artifact_prefix }}" in workflow
    assert 'print "artifact_prefix=" $2' in workflow
    assert "assemble_release_evidence.py" in workflow
    assert (
        "|| needs.verify.outputs.artifact_prefix }}-"
        "${{ needs.verify.outputs.source_sha }}"
    ) in workflow
    assert "audit-manifest-v0.1.13-sol-followup.yaml" not in workflow
    assert "v0113-integration-evidence-" not in workflow


def test_legacy_candidates_keep_combined_sidecars_while_promotion_splits_them():
    import yaml

    jobs = yaml.safe_load(read_workflow())["jobs"]
    build_uploads = [
        step
        for step in jobs["build"]["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    legacy = next(
        step for step in build_uploads if step.get("name") == "Upload legacy sidecars"
    )
    assert legacy["if"] == "needs.verify.outputs.promotion_enabled != 'true'"
    assert (
        legacy["with"]["name"]
        == "release-sidecars-${{ needs.verify.outputs.source_sha }}"
    )
    assert legacy["with"]["path"] == "source/release-sidecars"

    promotion_uploads = [
        step
        for step in build_uploads
        if step.get("name")
        in {
            "Upload promotion distribution sidecar",
            "Upload promotion license sidecar",
        }
    ]
    assert len(promotion_uploads) == 2
    assert all(
        step["if"] == "needs.verify.outputs.promotion_enabled == 'true'"
        for step in promotion_uploads
    )
    assert {step["with"]["name"] for step in promotion_uploads} == {
        "release-sidecar-distribution-sha256.json-${{ needs.verify.outputs.source_sha }}",
        "release-sidecar-LICENSE.sha256-${{ needs.verify.outputs.source_sha }}",
    }

    sidecar_download_patterns = [
        step["with"]["pattern"]
        for job in jobs.values()
        for step in job.get("steps", [])
        if step.get("uses", "").startswith("actions/download-artifact@")
        and "release-sidecar" in step.get("with", {}).get("pattern", "")
    ]
    assert sidecar_download_patterns
    assert set(sidecar_download_patterns) == {
        "release-sidecar*-${{ needs.verify.outputs.source_sha }}"
    }


def test_workflow_dispatch_inputs_are_preserved_for_manual_candidates():
    workflow = read_workflow()
    dispatch = workflow.split("  workflow_dispatch:\n", maxsplit=1)[1].split(
        "\npermissions:", maxsplit=1
    )[0]

    for name in ("release_ref", "expected_tag", "review_evidence"):
        assert f"      {name}:" in dispatch
    assert dispatch.count("required: true") == 3


def test_workflow_fetches_the_exact_tags_contract_protected_refs():
    workflow = read_workflow()
    fetch = workflow.split(
        "      - name: Fetch frozen protected branch tips\n", maxsplit=1
    )[1].split("\n      - name: Validate metadata", maxsplit=1)[0]

    assert "python control/scripts/ci/release_contract.py --protected-ref" in fetch
    assert "$EXPECTED_TAG" in fetch
    assert "release contract lookup failed" in fetch
    assert "+refs/heads/${protected_ref}:refs/remotes/origin/${protected_ref}" in fetch
    assert "maint/0.1" not in fetch


def test_workflow_rejects_empty_contract_ref_output(tmp_path: Path):
    workflow = read_workflow()
    fetch = workflow.split(
        "      - name: Fetch frozen protected branch tips\n", maxsplit=1
    )[1].split("\n      - name: Validate metadata", maxsplit=1)[0]
    script = textwrap.dedent(fetch.split("        run: |\n", maxsplit=1)[1])
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)

    result = subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=tmp_path,
        env=os.environ
        | {
            "EXPECTED_TAG": "v0.2.0",
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stderr == "release contract has no protected refs\n"


def test_releasing_documents_tag_specific_integration_artifact_prefixes():
    releasing = (WORKFLOW.parents[2] / "RELEASING.md").read_text(encoding="utf-8")

    assert "selected from the exact release contract" in releasing
    assert "`v020-integration-evidence-<40-character-source-sha>`" in releasing
    assert "`v022-integration-evidence-<40-character-source-sha>`" in releasing
    assert "`v023-integration-evidence-<40-character-source-sha>`" in releasing
    assert "currently `v0114-integration-evidence" not in releasing


def test_releasing_preserves_v022_history_and_documents_v023_qualification():
    releasing = (WORKFLOW.parents[2] / "RELEASING.md").read_text(encoding="utf-8")

    assert "For v0.2.2" in releasing
    assert "`v022-qualification-evidence-<40-character-source-sha>`" in releasing
    assert "For v0.2.3" in releasing
    assert "`v023-qualification-evidence-<40-character-source-sha>`" in releasing
    assert "`scripts/ci/v023_qualification_expected_skips.json`" in releasing


def test_workflow_contract_revision_disagreement_fails_closed(tmp_path: Path):
    """The workflow producer and validator consumer must use one revision.

    A hypothetical mixed revision fetches v0.2.0's `maint/0.2` from one
    control tree while a validator from another requires `maint/0.3`.  The
    validator must reject the missing second revision's ref rather than
    accepting the successfully fetched first revision's refs.
    """
    root = WORKFLOW.parents[2]

    def write_control_revision(name: str, maintenance_ref: str) -> Path:
        control = tmp_path / name / "scripts"
        ci = control / "ci"
        ci.mkdir(parents=True)
        for filename in ("release_contract.py", "release_contracts.json"):
            shutil.copy2(root / "scripts" / "ci" / filename, ci / filename)
        shutil.copy2(root / "scripts" / "validate_release.py", control)
        contracts_path = ci / "release_contracts.json"
        contracts = json.loads(contracts_path.read_text(encoding="utf-8"))
        contracts["releases"]["v0.2.0"]["protected_refs"] = [
            "main",
            maintenance_ref,
        ]
        contracts_path.write_text(json.dumps(contracts), encoding="utf-8")
        return control

    producer = write_control_revision("producer", "maint/0.2")
    consumer = write_control_revision("consumer", "maint/0.3")
    emitted = subprocess.run(
        [
            sys.executable,
            str(producer / "ci" / "release_contract.py"),
            "--protected-ref",
            "v0.2.0",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    assert emitted == ["main", "maint/0.2"]

    repo = tmp_path / "source"
    repo.mkdir()
    for args in (
        ("init", "-b", "main"),
        ("config", "user.name", "Release Test"),
        ("config", "user.email", "release-test@example.invalid"),
        ("commit", "--allow-empty", "-m", "source"),
    ):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    source_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    for ref in emitted:
        subprocess.run(
            ["git", "update-ref", f"refs/remotes/origin/{ref}", source_sha],
            cwd=repo,
            check=True,
            capture_output=True,
        )

    spec = importlib.util.spec_from_file_location(
        "consumer_validate_release", consumer / "validate_release.py"
    )
    assert spec and spec.loader
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)

    with pytest.raises(validator.ReleaseValidationError, match="origin/maint/0.3"):
        validator.validate_frozen_tip(repo, source_sha, expected_tag="v0.2.0")


def test_v025_cross_format_io_gate_is_candidate_bound_and_required_for_publish():
    import yaml

    workflow = yaml.safe_load(read_workflow())
    jobs = workflow["jobs"]
    matrix = jobs["cross_format_io"]["strategy"]["matrix"]["include"]
    assert len(matrix) == 8
    assert {(row["mode"], row["python"], row["distribution"]) for row in matrix} == {
        (mode, python, distribution)
        for mode in ("base", "optional")
        for python in ("3.11", "3.12")
        for distribution in ("wheel", "sdist")
    }
    assert (
        jobs["cross_format_io"]["if"]
        == "${{ (github.event_name == 'workflow_dispatch' || (github.event_name == 'push' && needs.verify.outputs.promotion_enabled != 'true')) && ((needs.verify.outputs.version == '0.2.5' || needs.verify.outputs.cross_format_io == 'true')) }}"
    )
    install = next(
        step
        for step in jobs["cross_format_io"]["steps"]
        if step["name"]
        == "Install digest-checked candidate and run selected public regressions"
    )
    assert "validate_release_payload.py" in install["run"]
    assert '--source-sha "$SOURCE_SHA"' in install["run"]
    assert '"${artifact}[io,netcdf4,zarr]"' in install["run"]
    assert "test_tdms_invalid_increment_contract.py" in install["run"]
    assert 'v025_cross_format_io_evidence.py" --version' in install["run"]
    assert " record" in install["run"]
    assert (
        "v025_cross_format_io_evidence.py --version"
        in jobs["cross_format_io_evidence"]["steps"][-2]["run"]
    )
    assert "cross_format_io_evidence" in jobs["publish_legacy"]["needs"]
    assert {
        "smoke",
        "qualification_evidence",
        "diaggui_qualification_evidence",
        "cross_format_io_evidence",
    } <= set(jobs["publish_legacy"]["needs"])


def test_candidate_graph_is_dispatch_only_and_finalizer_is_read_only():
    import yaml

    jobs = yaml.safe_load(read_workflow())["jobs"]
    for name in (
        "build",
        "smoke",
        "qualify",
        "qualification_evidence",
        "diaggui_qualification",
        "diaggui_qualification_evidence",
        "cross_format_io",
        "cross_format_io_evidence",
        "historical_74_gate",
        "evidence",
    ):
        assert "github.event_name == 'workflow_dispatch'" in jobs[name]["if"]
        assert "github.event_name == 'push'" in jobs[name]["if"]
        assert "needs.verify.outputs.promotion_enabled != 'true'" in jobs[name]["if"]
    finalizer = jobs["promotion_manifest"]
    assert finalizer["permissions"] == {"contents": "read", "actions": "read"}
    assert set(finalizer["needs"]) == {
        "verify",
        "build",
        "smoke",
        "qualify",
        "qualification_evidence",
        "diaggui_qualification",
        "diaggui_qualification_evidence",
        "cross_format_io",
        "cross_format_io_evidence",
        "evidence",
    }
    uploads = [
        step
        for step in finalizer["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    ]
    assert len(uploads) == 1
    assert (
        uploads[0]["with"]["name"]
        == "release-promotion-manifest-${{ needs.verify.outputs.source_sha }}"
    )
    verify = jobs["verify"]
    approval = next(
        s
        for s in verify["steps"]
        if s["name"] == "Verify GitHub human approval for the reviewed source"
    )
    assert "promotion_enabled" in approval["if"]
    assert jobs["build"]["needs"] == "verify"


@pytest.mark.parametrize(
    ("version", "promotion_enabled"),
    [("0.2.4", "false"), ("0.2.5", "false"), ("99.88.77", "true")],
)
def test_tag_graph_preserves_historical_jobs_and_excludes_future_candidate(
    version, promotion_enabled
):
    import yaml

    jobs = yaml.safe_load(read_workflow())["jobs"]
    for name in (
        "build",
        "smoke",
        "qualify",
        "qualification_evidence",
        "diaggui_qualification",
        "diaggui_qualification_evidence",
        "cross_format_io",
        "cross_format_io_evidence",
        "historical_74_gate",
        "evidence",
    ):
        expression = jobs[name]["if"].removeprefix("${{").removesuffix("}}").strip()
        values = {
            "github.event_name": "push",
            "needs.verify.outputs.version": version,
            "needs.verify.outputs.promotion_enabled": promotion_enabled,
            "needs.verify.outputs.qualify": "true"
            if promotion_enabled == "true"
            else "",
            "needs.verify.outputs.diaggui_qualification": "true"
            if promotion_enabled == "true"
            else "",
            "needs.verify.outputs.cross_format_io": "true"
            if promotion_enabled == "true"
            else "",
            "needs.verify.result": "success",
            "needs.build.result": "success",
            "needs.diaggui_qualification.result": "success",
            "needs.cross_format_io.result": "success",
        }
        for key, value in sorted(values.items(), key=lambda item: -len(item[0])):
            expression = expression.replace(key, repr(value))
        expression = (
            expression.replace("!cancelled()", "True")
            .replace("&&", " and ")
            .replace("||", " or ")
        )
        applies = eval(expression, {"__builtins__": {}}, {})
        expected = promotion_enabled != "true" and not (
            version == "0.2.4" and name == "cross_format_io"
        )
        assert applies is expected, (name, expression)


def test_synthetic_future_compatibility_path_runs_junit_and_records_evidence(tmp_path):
    import yaml

    version = "99.88.77"
    scripts = tmp_path / "source/scripts/ci"
    scripts.mkdir(parents=True)
    root = WORKFLOW.parents[2]
    for name in (
        "release_promotion.py",
        "release_contract.py",
        "qualification_evidence.py",
    ):
        shutil.copy2(root / "scripts/ci" / name, scripts / name)
    registry = json.loads((root / "scripts/ci/release_contracts.json").read_text())
    contract = dict(registry["releases"]["v0.2.5"])
    contract.pop("review_base_sha", None)
    contract["promotion"] = json.loads(
        (root / "tests/fixtures/release/promotion-contract-v1.json").read_text()
    )["promotion"]
    registry["releases"]["v" + version] = contract
    (scripts / "release_contracts.json").write_text(json.dumps(registry))
    testcase = tmp_path / "test_synthetic.py"
    testcase.write_text("def test_required_contract():\n    assert True\n")
    junit = tmp_path / "report/pytest.xml"
    jobs = yaml.safe_load(read_workflow())["jobs"]
    compatibility = next(
        step
        for step in jobs["qualify"]["steps"]
        if step["name"] == "Run installed candidate compatibility contracts"
    )
    case = (
        compatibility["run"]
        .split('case "$EXPECTED_VERSION" in', 1)[1]
        .split("esac", 1)[0]
    )
    program = (
        'runner=("'
        + sys.executable
        + '")\ntests=("'
        + str(testcase)
        + '")\ncase "$EXPECTED_VERSION" in'
        + case
        + "esac\n"
    )
    result = subprocess.run(
        ["bash", "-c", program],
        cwd=tmp_path,
        env={**os.environ, "EXPECTED_VERSION": version, "JUNIT": str(junit)},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert junit.is_file()
    payload = tmp_path / "distribution-sha256.json"
    payload.write_text(
        json.dumps(
            {
                "schema": contract["payload_schema"],
                "source_sha": "a" * 40,
                "version": version,
                "files": {
                    "wheel": {
                        "name": f"gwexpy-{version}-py3-none-any.whl",
                        "sha256": "b" * 64,
                    },
                    "sdist": {"name": f"gwexpy-{version}.tar.gz", "sha256": "c" * 64},
                },
            }
        )
    )
    record = tmp_path / "record.json"
    result = subprocess.run(
        [
            sys.executable,
            str(scripts / "qualification_evidence.py"),
            "record",
            "--version",
            version,
            "--cell",
            "install-ubuntu-3.11-wheel",
            "--source-sha",
            "a" * 40,
            "--payload-manifest",
            str(payload),
            "--junit",
            str(junit),
            "--report",
            str(record),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(record.read_bytes())["testcase_count"] == 1
    finalizer = jobs["promotion_manifest"]
    upload = next(
        step
        for step in finalizer["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["with"]["path"].endswith("/promotion-manifest.json")
