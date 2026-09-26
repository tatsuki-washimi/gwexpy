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
        "state": "published",
        "readback_at_utc": "2026-09-26T14:09:01Z",
        "latest_available_version": "0.2.4",
        "record_id": 22978439,
        "doi": "10.5281/zenodo.22978439",
        "record_url": "https://zenodo.org/records/22978439",
        "api_url": "https://zenodo.org/api/records/22978439",
        "api_record_state": "done",
        "api_publication_status": "published",
        "direct_record_http_status": 200,
        "doi_resolver_url": "https://doi.org/10.5281/zenodo.22978439",
        "doi_resolver_http_status": 404,
        "doi_resolver_readback_at_utc": "2026-09-26T14:13:34Z",
        "created_at_utc": "2026-09-26T13:52:18.326969Z",
        "updated_at_utc": "2026-09-26T13:52:18.508100Z",
        "version": "0.2.4",
        "publication_date": "2026-09-26",
        "resource_type": "Software",
        "title": "GWexpy: Extending GWpy with metadata-preserving multidimensional abstractions for detector commissioning",
        "previous_release": {
            "version": "0.2.3",
            "doi": "10.5281/zenodo.22344992",
        },
        "archive": {
            "key": "tatsuki-washimi/gwexpy-v0.2.4.zip",
            "bytes": 14595789,
            "api_md5": "c18f22ce6a63cc176a8cfb7cb81a5e4e",
            "downloaded_sha256": "67c224393870a0f6103e324fb4469e0c0064e3004e6efb9c8b64f3455b193515",
            "source_tree_commit": "522e52a082925da4dd37966d82a7616bdd2a5248",
            "path_and_content_match": True,
            "file_count": 2590,
        },
        "webhook_delivery_history": {
            "released_event_http_status": 202,
            "duplicate_event_http_status": 409,
            "duplicate_event_message": "release already received",
            "redelivery": "not_attempted",
        },
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
    assert (
        manifest["public_docs"]["intro_examples_verification"]["pypi_wheel_sha256"]
        == "34afe8188c753cd9da0b182a5d2a88cce2f7ec36633730fe0bee15e2506df56d"
    )
    assert manifest["public_docs"]["docs_deployment_readback"] == "pending"
    assert manifest["verification"]["publication_closure_claimed"] is False
    assert manifest["verification"]["zenodo_publication_verified"] is True
    assert manifest["verification"]["zenodo_doi_resolver_verified"] is False
    assert manifest["verification"]["conda_forge_publication_verified"] is False
    assert "does not" in report
    assert "mark publication closure complete" in report
    assert "dttxml==1.1.8" in report
    assert "The public Zenodo API reports record `22978439` as published" in report
    assert "DOI resolver returned HTTP 404" in report
    assert "file contents match `git archive`" in report


def test_public_release_sources_distinguish_published_and_pending_channels() -> None:
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
    assert "Zenodo DOI 10.5281/zenodo.22978439" in joined
    assert "https://zenodo.org/records/22978439" in joined
    assert "https://doi.org/10.5281/zenodo.22978439" not in joined
    assert "latest conda-forge package remains v0.2.3" in joined
    assert "conda-forge の最新パッケージは v0.2.3" in joined
