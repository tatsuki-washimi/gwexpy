"""Contract tests for the release workflow validator."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_release.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_release", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )
    return result.stdout.strip()


def make_repo(
    tmp_path: Path, *, version: str = "0.1.12", date: str = "2026-07-30"
) -> Path:
    repo = tmp_path / "release-repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Release Test")
    git(repo, "config", "user.email", "release-test@example.invalid")
    (repo / "gwexpy").mkdir()
    (repo / "gwexpy" / "_version.py").write_text(
        f'__version__ = "{version}"\n', encoding="utf-8"
    )
    (repo / "CITATION.cff").write_text(
        f"version: {version}\ndate-released: {date}\n", encoding="utf-8"
    )
    (repo / ".zenodo.json").write_text(
        json.dumps({"version": version, "publication_date": date}), encoding="utf-8"
    )
    (repo / "CHANGELOG.md").write_text(f"## [{version}] - {date}\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "release candidate")
    return repo


def tag_annotated(
    repo: Path, tag: str, date: str = "2026-07-30T12:00:00+00:00"
) -> None:
    env = os.environ | {"GIT_COMMITTER_DATE": date, "GIT_AUTHOR_DATE": date}
    git(repo, "tag", "-a", tag, "-m", tag, env=env)


def test_release_tag_regex_rejects_leading_zeroes():
    validator = load_validator()
    assert validator.is_release_tag("v0.1.12")
    assert not validator.is_release_tag("v01.1.12")
    assert not validator.is_release_tag("v0.1.12-rc1")
    assert not validator.is_release_tag("v0.1")


def test_candidate_mode_resolves_only_full_sha_and_requires_absent_tag(tmp_path: Path):
    validator = load_validator()
    repo = make_repo(tmp_path)
    source_sha = git(repo, "rev-parse", "HEAD")

    result = validator.validate_release(repo, source_sha, "v0.1.12")

    assert result.mode == "candidate"
    assert result.source_sha == source_sha
    with pytest.raises(validator.ReleaseValidationError, match="full 40-character"):
        validator.validate_release(repo, "main", "v0.1.12")
    tag_annotated(repo, "v0.1.12")
    with pytest.raises(
        validator.ReleaseValidationError, match="must not already exist"
    ):
        validator.validate_release(repo, source_sha, "v0.1.12")


def test_historical_mode_requires_annotated_tag_peel_and_utc_date(tmp_path: Path):
    validator = load_validator()
    repo = make_repo(tmp_path)
    tag_annotated(repo, "v0.1.12")

    result = validator.validate_release(repo, "v0.1.12", "v0.1.12")

    assert result.mode == "strict"
    assert result.source_sha == git(repo, "rev-parse", "HEAD")

    light_repo = make_repo(tmp_path / "light")
    git(light_repo, "tag", "v0.1.12")
    with pytest.raises(validator.ReleaseValidationError, match="annotated"):
        validator.validate_release(light_repo, "v0.1.12", "v0.1.12")


def test_strict_mode_rejects_tagger_date_mismatch_and_missing_maintenance_branch(
    tmp_path: Path,
):
    validator = load_validator()
    repo = make_repo(tmp_path, version="0.1.13", date="2026-07-30")
    tag_annotated(repo, "v0.1.13", "2026-07-29T23:59:59+00:00")
    with pytest.raises(validator.ReleaseValidationError, match="tagger date"):
        validator.validate_release(repo, "v0.1.13", "v0.1.13")

    repo = make_repo(tmp_path / "maintenance", version="0.1.13")
    tag_annotated(repo, "v0.1.13")
    with pytest.raises(validator.ReleaseValidationError, match="maint/0.1"):
        validator.validate_release(repo, "v0.1.13", "v0.1.13")


def test_duplicate_changelog_release_heading_is_rejected(tmp_path: Path):
    """Two headings for one version must fail, not resolve to the first.

    ``tools/gen_release_notes.py`` refuses to generate from a duplicated
    heading. A first-match read here would let the validator pass on a tree
    whose release notes cannot be produced, splitting the two contracts.
    """
    validator = load_validator()
    repo = make_repo(tmp_path)
    changelog = repo / "CHANGELOG.md"
    changelog.write_text(
        "## [0.1.12] - 2026-07-30\n\nfirst\n\n## [0.1.12] - 2026-07-31\n\nsecond\n",
        encoding="utf-8",
    )
    git(repo, "commit", "-am", "duplicate heading")
    source_sha = git(repo, "rev-parse", "HEAD")

    with pytest.raises(validator.ReleaseValidationError, match="2 release headings"):
        validator.validate_release(repo, source_sha, "v0.1.12")


@pytest.mark.parametrize(
    ("field", "duplicate_line"),
    [("version", "version: 0.1.99"), ("date-released", "date-released: 2026-01-01")],
)
def test_duplicate_citation_cff_field_is_rejected(
    tmp_path: Path, field: str, duplicate_line: str
):
    """A duplicated top-level CFF field is ambiguous release metadata.

    Reading only the first occurrence would let a stale second value ship in
    the published citation record while validation reports success.
    """
    validator = load_validator()
    repo = make_repo(tmp_path / field)
    citation = repo / "CITATION.cff"
    citation.write_text(
        citation.read_text(encoding="utf-8") + f"{duplicate_line}\n", encoding="utf-8"
    )
    git(repo, "commit", "-am", f"duplicate {field}")
    source_sha = git(repo, "rev-parse", "HEAD")

    with pytest.raises(
        validator.ReleaseValidationError, match=f"2 top-level '{field}' fields"
    ):
        validator.validate_release(repo, source_sha, "v0.1.12")
