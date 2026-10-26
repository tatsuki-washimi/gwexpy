"""Regression checks for integrity-bound, ordered Docs Pages publishing."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.ci.docs_publish_guard import (
    DOCS_BUILD_GIT_PATHS,
    DOCS_BUILD_INPUTS,
    DocsPublishError,
    check_publication_order,
    create_artifact_manifest,
    verify_artifact,
)

AUDIT_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "docs/developers/plans/manifests/audit-manifest-issue-822-docs-publication-isolation.yaml"
)


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def _init_history(repo: Path) -> tuple[str, str, str]:
    repo.mkdir()
    _run_git(repo, "init", "-b", "main")
    _run_git(repo, "config", "user.name", "Docs test")
    _run_git(repo, "config", "user.email", "docs@example.invalid")

    source = repo / "docs_redesign" / "published-source.txt"
    source.parent.mkdir()
    source.write_text("version A\n", encoding="utf-8")
    _run_git(repo, "add", "docs_redesign/published-source.txt")
    _run_git(repo, "commit", "-m", "publish A")
    revision_a = _run_git(repo, "rev-parse", "HEAD")

    source.write_text("version B\n", encoding="utf-8")
    _run_git(repo, "commit", "-am", "publish B")
    revision_b = _run_git(repo, "rev-parse", "HEAD")

    source.write_text("version A\n", encoding="utf-8")
    _run_git(repo, "commit", "-am", "revert docs content in a new commit")
    revision_rollback = _run_git(repo, "rev-parse", "HEAD")
    return revision_a, revision_b, revision_rollback


def _write_site(site: Path, source_revision: str, *, dirty: bool = False) -> None:
    site.mkdir(parents=True, exist_ok=True)
    for language in ("", "ja/"):
        root = site / language
        root.mkdir(parents=True, exist_ok=True)
        (root / "index.html").write_text("<!doctype html>docs\n", encoding="utf-8")
        (root / "build-info.json").write_text(
            json.dumps({"source_revision": source_revision, "dirty": dirty}) + "\n",
            encoding="utf-8",
        )


def test_audit_manifest_records_issue_and_owner_boundary() -> None:
    manifest = yaml.safe_load(AUDIT_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["issue"] == 822
    assert manifest["status"] == "draft_pending_owner_adoption"
    assert manifest["source_base_sha"] == "fd567f3cfacfdc5beaa1589c20588add832b7eb8"
    assert any(
        "owner adoption" in item.lower()
        for item in manifest["omitted"]
        if isinstance(item, str)
    )


def test_artifact_manifest_binds_both_languages_and_every_file(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    revision_a, _, _ = _init_history(repo)
    artifact = tmp_path / "artifact"
    site = artifact / "site"
    _write_site(site, revision_a)
    manifest = artifact / "manifest.json"

    created = create_artifact_manifest(site, manifest, revision_a)

    assert created["source_revision"] == revision_a
    assert verify_artifact(artifact, revision_a) == created
    (site / "index.html").write_text("modified after manifest\n", encoding="utf-8")
    with pytest.raises(DocsPublishError, match="hash"):
        verify_artifact(artifact, revision_a)


def test_artifact_manifest_rejects_wrong_ja_source_and_symlinks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    revision_a, revision_b, _ = _init_history(repo)
    artifact = tmp_path / "artifact"
    site = artifact / "site"
    _write_site(site, revision_a)
    (site / "ja" / "build-info.json").write_text(
        json.dumps({"source_revision": revision_b, "dirty": False}), encoding="utf-8"
    )

    with pytest.raises(DocsPublishError, match="same source revision"):
        create_artifact_manifest(site, artifact / "manifest.json", revision_a)

    (site / "ja" / "build-info.json").write_text(
        json.dumps({"source_revision": revision_a, "dirty": False}), encoding="utf-8"
    )
    (site / "linked.html").symlink_to(site / "index.html")
    with pytest.raises(DocsPublishError, match="symlink"):
        create_artifact_manifest(site, artifact / "manifest.json", revision_a)


def test_stale_a_cannot_replace_newer_b_and_revert_commit_is_ordered(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    revision_a, revision_b, revision_rollback = _init_history(repo)
    deployed_site = tmp_path / "gh-pages" / "docs"

    _run_git(repo, "checkout", "--detach", revision_b)
    _write_site(deployed_site, revision_b)
    with pytest.raises(DocsPublishError, match="not the current main commit"):
        check_publication_order(repo, revision_a, deployed_site)

    # A normal B deployment can advance from A or safely republish B.
    _run_git(repo, "checkout", "--detach", revision_b)
    _write_site(deployed_site, revision_a)
    assert check_publication_order(repo, revision_b, deployed_site) == revision_a
    _write_site(deployed_site, revision_b)
    assert check_publication_order(repo, revision_b, deployed_site) == revision_b

    # A deliberate content rollback is a new main commit descending from B.
    _run_git(repo, "checkout", "main")
    _write_site(deployed_site, revision_b)
    assert check_publication_order(repo, revision_rollback, deployed_site) == revision_b


def test_legacy_preview_uses_gh_pages_commit_source_binding(tmp_path: Path) -> None:
    main = tmp_path / "main"
    revision_a, _, revision_rollback = _init_history(main)

    gh_pages = tmp_path / "gh-pages"
    gh_pages.mkdir()
    _run_git(gh_pages, "init", "-b", "gh-pages")
    _run_git(gh_pages, "config", "user.name", "Docs test")
    _run_git(gh_pages, "config", "user.email", "docs@example.invalid")
    preview = gh_pages / "preview-docs-redesign"
    preview.mkdir()
    (preview / "index.html").write_text("legacy preview\n", encoding="utf-8")
    _run_git(gh_pages, "add", "preview-docs-redesign/index.html")
    _run_git(
        gh_pages,
        "commit",
        "-m",
        f"deploy(preview): docs_redesign {revision_a}",
    )

    assert check_publication_order(main, revision_rollback, preview) == revision_a


def test_legacy_preview_without_trusted_commit_message_fails_closed(
    tmp_path: Path,
) -> None:
    main = tmp_path / "main"
    _, _, revision_rollback = _init_history(main)

    gh_pages = tmp_path / "gh-pages"
    gh_pages.mkdir()
    _run_git(gh_pages, "init", "-b", "gh-pages")
    _run_git(gh_pages, "config", "user.name", "Docs test")
    _run_git(gh_pages, "config", "user.email", "docs@example.invalid")
    preview = gh_pages / "preview-docs-redesign"
    preview.mkdir()
    (preview / "index.html").write_text("unbound preview\n", encoding="utf-8")
    _run_git(gh_pages, "add", "preview-docs-redesign/index.html")
    _run_git(gh_pages, "commit", "-m", "unknown legacy publisher")

    with pytest.raises(DocsPublishError, match="legacy preview source binding"):
        check_publication_order(main, revision_rollback, preview)


def test_publication_order_uses_refreshed_main_ref(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    revision_a, revision_b, revision_rollback = _init_history(repo)
    _run_git(repo, "update-ref", "refs/remotes/origin/main", revision_b)
    deployed_site = tmp_path / "gh-pages" / "docs"
    _write_site(deployed_site, revision_a)

    with pytest.raises(DocsPublishError, match="not the current main commit"):
        check_publication_order(
            repo,
            revision_rollback,
            deployed_site,
            main_ref="refs/remotes/origin/main",
        )


def test_docs_candidate_survives_an_unrelated_newer_main_commit(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    revision_a, _, _ = _init_history(repo)
    _run_git(repo, "checkout", "main")
    unrelated = repo / "README.md"
    unrelated.write_text("not a Docs build input\n", encoding="utf-8")
    _run_git(repo, "add", "README.md")
    _run_git(repo, "commit", "-m", "unrelated main update")
    revision_b = _run_git(repo, "rev-parse", "HEAD")
    deployed_site = tmp_path / "gh-pages" / "docs"

    assert revision_a != revision_b
    assert check_publication_order(repo, revision_a, deployed_site) is None


def test_docs_input_patterns_match_the_publisher_guard_pathspecs() -> None:
    assert DOCS_BUILD_GIT_PATHS == tuple(
        pattern[:-3] if pattern.endswith("/**") else pattern
        for pattern in DOCS_BUILD_INPUTS
    )


def test_publication_fails_closed_for_unbound_site_state(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    revision_a, _, revision_rollback = _init_history(repo)
    deployed_site = tmp_path / "gh-pages" / "docs"
    deployed_site.mkdir(parents=True)
    (deployed_site / "index.html").write_text("legacy site\n", encoding="utf-8")

    with pytest.raises(DocsPublishError, match="missing build-info"):
        check_publication_order(repo, revision_rollback, deployed_site)

    _write_site(deployed_site, revision_a, dirty=True)
    with pytest.raises(DocsPublishError, match="dirty"):
        check_publication_order(repo, revision_rollback, deployed_site)
