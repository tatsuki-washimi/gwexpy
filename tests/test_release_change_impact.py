from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "release_change_impact.py"
sys.path.insert(0, str(SCRIPT.parent))

import release_change_impact as impact


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repository(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Impact Test")
    git(repo, "config", "user.email", "impact@example.invalid")
    files = {
        "gwexpy/signal.py": "def value(): return 1\n",
        "tests/test_signal.py": "def test_value(): assert True\n",
        "tests/test_release_promotion.py": "def test_contract(): assert True\n",
        "tests/docs/test_guide.py": "def test_guide(): assert True\n",
        ".github/workflows/publish-release.yml": "name: Release\n",
        "docs/developers/plans/manifests/audit.yaml": "status: pending\n",
        ".zenodo.json": '{"version": "0.2.6"}\n',
        "README.md": "GWexpy\n",
    }
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    return repo, git(repo, "rev-parse", "HEAD")


def commit_change(repo: Path, path: str, content: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    git(repo, "add", "--", path)
    git(repo, "commit", "-qm", "change")
    return git(repo, "rev-parse", "HEAD")


def test_path_classifier_separates_runtime_tests_release_and_metadata():
    assert impact.classify_path("gwexpy/signal.py") == ["scientific_product"]
    assert impact.classify_path("tests/test_signal.py") == ["scientific_tests"]
    assert impact.classify_path("tests/test_release_promotion.py") == [
        "release_control"
    ]
    assert impact.classify_path("tests/docs/test_guide.py") == ["public_documentation"]
    assert impact.classify_path("docs/developers/plans/manifests/audit.yaml") == [
        "administrative_evidence",
    ]
    assert impact.classify_path("requirements/runtime.txt") == ["build_input"]
    assert impact.classify_path("requirements.txt") == ["build_input"]
    assert impact.classify_path("README.md") == ["public_documentation"]
    assert impact.classify_path("new_policy.txt") == []


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../outside", "docs/../secret.md", "docs\\secret.md", ""],
)
def test_path_classifier_rejects_non_repository_paths(path: str):
    with pytest.raises(impact.ChangeImpactError):
        impact.classify_path(path)


def test_metadata_only_change_preserves_scientific_scope_candidate_but_not_approval(
    tmp_path: Path,
):
    repo, base = repository(tmp_path)
    candidate = commit_change(
        repo,
        "docs/developers/plans/manifests/audit.yaml",
        "status: corrected\n",
    )

    report = impact.analyze_change_impact(
        repo, tag="v0.2.6", base=base, candidate=candidate
    )

    assert report["current_contract_affected_lanes"] == ["documentation"]
    assert report["proposed_impact_review_lanes"] == []
    assert report["proposed_local_validation_scopes"] == [
        "structured_record_syntax_check"
    ]
    assert report["scientific_inputs"]["unchanged"] is True
    assert (
        report["scientific_inputs"]["reuse_candidate_only_after_contract_adoption"]
        is True
    )
    assert report["lane_scope_digests"]["scientific-data-model"]["unchanged"] is True
    assert report["lane_scope_digests"]["release-security"]["unchanged"] is True
    assert report["lane_scope_digests"]["documentation"]["unchanged"] is False
    assert (
        report["active_contract_effect"]["source_approval"]
        == "invalidated_for_new_source_sha"
    )
    assert report["active_contract_effect"]["review_evidence_transfer"] == (
        "not_authorized_by_this_report"
    )
    assert report["active_contract_effect"]["review_lanes"] == (
        "all_configured_lanes_remain_required_for_exact_source"
    )
    assert report["publication_authorized"] is False
    assert report["changed_structured_records"] == [
        {
            "path": "docs/developers/plans/manifests/audit.yaml",
            "status": "syntactically_valid",
        }
    ]


def test_product_change_invalidates_scientific_scope_and_maps_to_science_lane(
    tmp_path: Path,
):
    repo, base = repository(tmp_path)
    candidate = commit_change(repo, "gwexpy/signal.py", "def value(): return 2\n")

    report = impact.analyze_change_impact(
        repo, tag="v0.2.6", base=base, candidate=candidate
    )

    assert report["current_contract_affected_lanes"] == ["scientific-data-model"]
    assert report["proposed_impact_review_lanes"] == ["scientific-data-model"]
    assert report["scientific_inputs"]["unchanged"] is False
    assert (
        report["scientific_inputs"]["reuse_candidate_only_after_contract_adoption"]
        is False
    )


def test_workflow_change_maps_to_security_without_changing_science_fingerprint(
    tmp_path: Path,
):
    repo, base = repository(tmp_path)
    candidate = commit_change(
        repo,
        ".github/workflows/publish-release.yml",
        "name: Release\npermissions: read-all\n",
    )

    report = impact.analyze_change_impact(
        repo, tag="v0.2.6", base=base, candidate=candidate
    )

    assert report["current_contract_affected_lanes"] == ["release-security"]
    assert report["proposed_impact_review_lanes"] == ["release-security"]
    assert report["scientific_inputs"]["unchanged"] is True
    assert report["active_contract_effect"]["release_owner_go"] == (
        "must_bind_new_candidate_run_manifest_and_hashes"
    )


def test_unmapped_path_fails_closed_and_disables_reuse_candidate(tmp_path: Path):
    repo, base = repository(tmp_path)
    candidate = commit_change(repo, "new_policy.txt", "policy\n")

    report = impact.analyze_change_impact(
        repo, tag="v0.2.6", base=base, candidate=candidate
    )

    assert report["unmapped_paths"] == ["new_policy.txt"]
    assert (
        report["current_contract_affected_lanes"]
        == report["all_configured_review_lanes"]
    )
    assert (
        report["proposed_impact_review_lanes"] == report["all_configured_review_lanes"]
    )
    assert (
        report["scientific_inputs"]["reuse_candidate_only_after_contract_adoption"]
        is False
    )


def test_release_test_change_is_not_misclassified_as_scientific_test():
    categories = impact.classify_path("tests/test_release_promotion.py")
    assert "release_control" in categories
    assert "scientific_tests" not in categories


def test_build_input_change_impacts_science_and_release_security(tmp_path: Path):
    repo, base = repository(tmp_path)
    candidate = commit_change(repo, "pyproject.toml", "[build-system]\nrequires=[]\n")

    report = impact.analyze_change_impact(
        repo, tag="v0.2.6", base=base, candidate=candidate
    )

    assert report["current_contract_affected_lanes"] == ["release-security"]
    assert report["proposed_impact_review_lanes"] == [
        "release-security",
        "scientific-data-model",
    ]
    assert report["scientific_inputs"]["unchanged"] is False


def test_uncovered_public_document_path_is_currently_fail_closed(tmp_path: Path):
    repo, base = repository(tmp_path)
    candidate = commit_change(repo, "README.md", "GWexpy updated\n")

    report = impact.analyze_change_impact(
        repo, tag="v0.2.6", base=base, candidate=candidate
    )

    assert (
        report["current_contract_affected_lanes"]
        == report["all_configured_review_lanes"]
    )
    assert report["proposed_impact_review_lanes"] == ["documentation"]
    assert report["unmapped_paths"] == ["README.md"]


def test_invalid_structured_admin_record_is_rejected(tmp_path: Path):
    repo, base = repository(tmp_path)
    candidate = commit_change(
        repo,
        "docs/developers/plans/manifests/audit.yaml",
        "status: [invalid\n",
    )

    with pytest.raises(
        impact.ChangeImpactError, match="structured evidence is invalid"
    ):
        impact.analyze_change_impact(repo, tag="v0.2.6", base=base, candidate=candidate)


def test_deleted_structured_admin_record_needs_manual_review_and_disables_reuse(
    tmp_path: Path,
):
    repo, base = repository(tmp_path)
    git(repo, "rm", "--", "docs/developers/plans/manifests/audit.yaml")
    git(repo, "commit", "-qm", "delete evidence")
    candidate = git(repo, "rev-parse", "HEAD")

    report = impact.analyze_change_impact(
        repo, tag="v0.2.6", base=base, candidate=candidate
    )

    assert report["changed_structured_records"] == [
        {
            "path": "docs/developers/plans/manifests/audit.yaml",
            "status": "deleted_requires_manual_review",
        }
    ]
    assert report["scientific_inputs"]["unchanged"] is True
    assert (
        report["scientific_inputs"]["reuse_candidate_only_after_contract_adoption"]
        is False
    )
