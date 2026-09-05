"""Contracts for the post-publication v0.2.3 closure records."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    REPO_ROOT
    / "docs/developers/plans/manifests/audit-manifest-v0.2.3-release-closure.yaml"
)
REPORT_PATH = (
    REPO_ROOT / "docs/developers/reports/report_v0.2.3_release_closure_20260906.md"
)
RELEASE_SOURCE = "75d3d1a89ebc8942af1f3228152fea99d2d3420e"
TAG_OBJECT = "b79ad05ca51527a048dd18e5c3cf84bc9e57487a"


def _manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_v023_closure_manifest_binds_immutable_release_and_distributions() -> None:
    """The closure record must remain tied to the published bytes."""
    manifest = _manifest()

    assert manifest["schema"] == "gwexpy-v023-release-closure-v1"
    assert manifest["status"] == "closure-complete"
    assert manifest["release"] == {
        "version": "0.2.3",
        "source_sha": RELEASE_SOURCE,
        "reviewed_source_sha": "235163a336752ced188a3fcde7190d0da3bedac3",
        "runtime_tree": "88c9de982f4b284afbb5845c13cecb2d90d938dc",
        "tag": "v0.2.3",
        "tag_object_sha": TAG_OBJECT,
        "tagger_utc": "2026-09-05T12:59:24Z",
        "github_release_id": 383250425,
        "github_release_url": (
            "https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.3"
        ),
        "github_release_published_at": "2026-09-05T13:11:33Z",
        "github_release_draft": False,
        "github_release_prerelease": False,
        "publication_date_utc": "2026-09-05",
    }
    assert manifest["publication"]["workflow_run_id"] == 33967606952
    assert manifest["publication"]["qualification_cells"] == "19/19"
    assert manifest["publication"]["smoke_cells"] == "4/4"
    assert manifest["publication"]["unexpected_skips"] == 0
    assert manifest["publication"]["same_payload_verification"] == "pass"
    assert manifest["publication"]["pypi"]["sdist_sha256"] == (
        "a5e752c6a53b5c6cabf41de309ca3077534f068384425deff3b1bccc461377b2"
    )
    assert manifest["publication"]["pypi"]["wheel_sha256"] == (
        "e73da65cff769615fc78e264f6f17730573472a8587031444ccef237294e1e9c"
    )
    assert manifest["publication"]["conda_forge"]["channel_package"] == {
        "basename": "noarch/gwexpy-0.2.3-pyhc364b38_0.conda",
        "version": "0.2.3",
        "build": "pyhc364b38_0",
        "build_number": 0,
        "sha256": "4723b718fb80fd9676ba085bd74ea4ca08db864df89e9539ee1da5877d67bbf6",
        "md5": "671b1fe41659c23e234cf871cf77ee6f",
        "uploaded_at": "2026-09-05T13:20:58.538000Z",
        "labels": ["main"],
    }
    assert manifest["publication"]["zenodo"]["doi"] == "10.5281/zenodo.22344992"
    assert manifest["publication"]["zenodo"]["record_id"] == 22344992
    assert manifest["verification"]["authoritative_readback"]["current_main"] == (
        "1ff2cb99cc5a6013417cd5e3d147e1d21631e9e7 (PR #715 Docs Pages readback fix; current readback)"
    )


def test_v023_closure_manifest_records_issue_and_scope_boundary() -> None:
    """Closure is bounded and does not silently absorb deferred work."""
    manifest = _manifest()
    issues = manifest["issues"]

    assert issues["closed"] == [
        639,
        698,
        699,
        700,
        701,
        702,
        703,
        704,
        705,
        706,
        707,
        709,
        710,
        711,
    ]
    assert issues["deferred"] == [634, 688]
    assert issues["separate"] == [611, 713]
    assert issues["milestone"] == {"exists": False, "action": "none"}
    assert issues["compatibility_audit"]["scope_classification"]["unfinished"] == (
        "none within the frozen inventory"
    )
    assert manifest["scope"] == {
        "runtime_code_changed": False,
        "public_api_changed": False,
        "dependencies_changed": False,
        "scientific_semantics_changed": False,
        "released_source_changed": False,
        "release_tag_changed": False,
        "release_payload_rebuilt": False,
        "historical_evidence_changed": False,
        "docs_and_bookkeeping_only": True,
    }


def test_v023_closure_report_states_the_runtime_boundary() -> None:
    """The human-readable record preserves the same closure boundary."""
    report = REPORT_PATH.read_text(encoding="utf-8")

    assert RELEASE_SOURCE in report
    assert "Date: 2026-09-06 (JST; 2026-09-05 UTC)" in report
    assert "19/19 qualification" in report
    assert "4/4 smoke" in report
    assert "#634" in report and "#688" in report
    assert (
        "No runtime, public API, dependency, or scientific-semantic changes." in report
    )
    assert "No new runtime or correctness defect was found." in report
    assert "PR #715" in report
