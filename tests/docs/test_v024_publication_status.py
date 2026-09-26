"""Contracts for the partial post-PyPI v0.2.4 publication record."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    ROOT
    / "docs/developers/plans/manifests/audit-manifest-v0.2.4-publication-status.yaml"
)
REPORT_PATH = (
    ROOT / "docs/developers/reports/report_v0.2.4_publication_status_20260926.md"
)


def test_v024_publication_status_binds_verified_source_and_channels() -> None:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert manifest["schema"] == "gwexpy-v024-publication-status-v1"
    assert manifest["version"] == "0.2.4"
    assert manifest["status"] == "publication-partial"
    assert manifest["closure_complete"] is False
    assert manifest["release"] == {
        "source_sha": "522e52a082925da4dd37966d82a7616bdd2a5248",
        "runtime_tree": "27165f31eb4a60d129ef4e2eb7f159b023b56d9b",
        "tag": "v0.2.4",
        "tag_object_sha": "2bc0789918fbc0327114df0d1ac0840faf5ed2ba",
        "tagger_utc": "2026-09-26T12:53:57Z",
        "release_notes": "release_notes/v0.2.4.md",
    }

    publication = manifest["publication"]
    assert publication["workflow_run_id"] == 36243706918
    assert publication["conclusion"] == "success"
    assert publication["strict_checks"] == "33/33"
    assert publication["pypi"]["sdist_and_wheel_byte_match"] == "pass"
    assert publication["github_release"] == {
        "id": 397242347,
        "url": "https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.4",
        "published_at": "2026-09-26T13:16:44Z",
        "draft": False,
        "prerelease": False,
        "release_body_content_corresponds_to_release_notes": True,
    }
    assert publication["zenodo"] == {
        "state": "webhook_accepted_public_record_not_yet_visible",
        "latest_available_version": "0.2.3",
        "latest_doi": "10.5281/zenodo.22344992",
        "release_event_http_status": 202,
        "duplicate_event_http_status": 409,
        "duplicate_event_message": "release already received",
        "redelivery": "not_attempted",
        "v024_record": "not_yet_publicly_visible_as_of_readback",
    }
    assert publication["conda_forge"] == {
        "state": "pending",
        "latest_available_version": "0.2.3",
        "feedstock_pull_request": "none_as_of_readback",
    }


def test_v024_publication_record_preserves_example_and_docs_readback_state() -> None:
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert manifest["public_docs"]["latest_release"] == "0.2.4"
    assert manifest["public_docs"]["intro_examples_release"] == "0.2.4"
    assert manifest["public_docs"]["intro_examples_024_verification"] == "pass"
    assert manifest["public_docs"]["intro_examples_verification"]["sources"] == [
        "candidate_docs",
        "clean_r2_git_archive",
    ]
    assert manifest["public_docs"]["intro_examples_verification"]["helper_sha256"] == (
        "772b6d979b760c14796852482f0fb4428c6808ab4f82a81e1cf1b36b8f45cad3"
    )
    assert manifest["public_docs"]["intro_examples_verification"][
        "pypi_wheel_sha256"
    ] == "34afe8188c753cd9da0b182a5d2a88cce2f7ec36633730fe0bee15e2506df56d"
    assert manifest["public_docs"]["docs_deployment_readback"] == "pending"
    assert manifest["verification"]["publication_closure_claimed"] is False
    assert "does not" in report
    assert "mark publication closure complete" in report
    assert "dttxml==1.1.8" in report


def test_public_release_sources_distinguish_pypi_from_pending_channels() -> None:
    status = json.loads(
        (ROOT / "docs_redesign/release_status.json").read_text(encoding="utf-8")
    )
    assert status == {"latest_release": "0.2.4", "intro_examples_release": "0.2.4"}

    public_sources = (
        ROOT / "docs_redesign/tutorials/installation.md",
        ROOT / "docs_redesign/explanation/roadmap.md",
        ROOT / "docs_redesign/about/changelog.md",
        ROOT / "docs/web/en/user_guide/roadmap.md",
        ROOT / "docs/web/ja/user_guide/roadmap.md",
        ROOT / "docs/web/en/user_guide/changelog.md",
        ROOT / "docs/web/ja/user_guide/changelog.md",
    )
    joined = "\n".join(path.read_text(encoding="utf-8") for path in public_sources)
    assert "v0.2.4" in joined
    assert "latest conda-forge package is v0.2.3" in joined
    assert "The latest conda-forge and Zenodo versions remain v0.2.3" in joined
    assert "conda-forge と Zenodo の最新バージョンは v0.2.3" in joined
