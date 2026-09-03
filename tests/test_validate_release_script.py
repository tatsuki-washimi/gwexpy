"""Contract tests for the release workflow validator."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_release.py"
CONTRACTS_PATH = SCRIPT_PATH.parent / "ci" / "release_contracts.json"
V023_PLAN_PATH = (
    "docs/developers/plans/20260902_v0.2.3_gwpy_behavioral_compatibility.md"
)
V023_REVIEW_EVIDENCE_PATH = (
    "docs/developers/plans/manifests/audit-manifest-v0.2.3-release-readiness.yaml"
)
V023_EMPTY_REVIEW_EVIDENCE = """\
review_evidence_json: |
  {
    "schema": "gwexpy-v023-review-evidence-v1",
    "entries": []
  }
"""
V023_POPULATED_REVIEW_EVIDENCE = """\
review_evidence_json: |
  {
    "schema": "gwexpy-v023-review-evidence-v1",
    "entries": [
      {
        "lane": "release-security"
      }
    ]
  }
"""


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


def v023_review_lanes() -> dict[str, list[str]]:
    contracts = json.loads(CONTRACTS_PATH.read_text(encoding="utf-8"))
    return contracts["releases"]["v0.2.3"]["review_lanes"]


def materialize_v023_review_scope(repo: Path) -> None:
    scope_paths = {path for paths in v023_review_lanes().values() for path in paths}
    for path in sorted(scope_paths):
        if path == V023_REVIEW_EVIDENCE_PATH:
            continue
        candidate = repo / path
        if candidate.exists():
            continue
        if any(other.startswith(f"{path}/") for other in scope_paths):
            candidate.mkdir(parents=True)
        else:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.write_text(f"review scope fixture: {path}\n", encoding="utf-8")


def make_repo(
    tmp_path: Path,
    *,
    version: str = "0.1.13",
    date: str = "2026-07-30",
    maintenance_branch: str | None = "maint/0.1",
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
    if maintenance_branch is not None:
        git(repo, "branch", maintenance_branch)
    return repo


def make_v023_s_to_r_repo(
    tmp_path: Path,
    *,
    source_manifest: str | None = V023_EMPTY_REVIEW_EVIDENCE,
) -> tuple[Path, Path, Path, str]:
    repo = tmp_path / "review-repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Release Test")
    git(repo, "config", "user.email", "release-test@example.invalid")
    plan = repo / V023_PLAN_PATH
    manifest = repo / V023_REVIEW_EVIDENCE_PATH
    plan.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    plan.write_text("- [ ] coordinator evidence\n", encoding="utf-8")
    if source_manifest is not None:
        manifest.write_text(source_manifest, encoding="utf-8")
    materialize_v023_review_scope(repo)
    git(repo, "add", ".")
    git(repo, "commit", "-m", "S")
    return repo, plan, manifest, git(repo, "rev-parse", "HEAD")


def commit_v023_r(
    repo: Path,
    plan: Path,
    manifest: Path,
    *,
    update_plan: bool = False,
) -> str:
    reviewed_commit = git(repo, "rev-parse", "HEAD")
    if update_plan:
        plan.write_text("- [x] coordinator evidence\n", encoding="utf-8")
    entries = []
    for lane, configured_paths in sorted(v023_review_lanes().items()):
        paths = sorted(set(configured_paths), key=lambda item: item.encode("utf-8"))
        tree = subprocess.run(
            [
                "git",
                "ls-tree",
                "-r",
                "-z",
                "--full-tree",
                reviewed_commit,
                "--",
                *paths,
            ],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
        entries.append(
            {
                "lane": lane,
                "role": "reviewer",
                "model": "gpt-5.6-terra",
                "effort": "high",
                "reviewed_commit": reviewed_commit,
                "scope_paths": paths,
                "scope_digest": hashlib.sha256(tree).hexdigest(),
                "verdict": "APPROVED",
                "timestamp_utc": "2026-09-03T00:00:00Z",
                "raw_report_sha256": hashlib.sha256(
                    f"{lane}-report".encode()
                ).hexdigest(),
                "finding_ids": [],
            }
        )
    payload = json.dumps(
        {
            "schema": "gwexpy-v023-review-evidence-v1",
            "entries": entries,
        },
        indent=2,
    )
    manifest.write_text(
        "review_evidence_json: |\n"
        + "".join(f"  {line}\n" for line in payload.splitlines()),
        encoding="utf-8",
    )
    git(repo, "add", ".")
    git(repo, "commit", "-m", "R")
    return git(repo, "rev-parse", "HEAD")


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

    result = validator.validate_release(repo, source_sha, "v0.1.13")

    assert result.mode == "candidate"
    assert result.source_sha == source_sha
    assert result.artifact_prefix == "v0113-integration-evidence"
    with pytest.raises(validator.ReleaseValidationError, match="full 40-character"):
        validator.validate_release(repo, "main", "v0.1.13")
    tag_annotated(repo, "v0.1.13")
    with pytest.raises(
        validator.ReleaseValidationError, match="must not already exist"
    ):
        validator.validate_release(repo, source_sha, "v0.1.13")


def test_historical_mode_requires_annotated_tag_peel_and_utc_date(tmp_path: Path):
    validator = load_validator()
    repo = make_repo(tmp_path)
    tag_annotated(repo, "v0.1.13")

    result = validator.validate_release(repo, "v0.1.13", "v0.1.13")

    assert result.mode == "strict"
    assert result.source_sha == git(repo, "rev-parse", "HEAD")

    light_repo = make_repo(tmp_path / "light")
    git(light_repo, "tag", "v0.1.13")
    with pytest.raises(validator.ReleaseValidationError, match="annotated"):
        validator.validate_release(light_repo, "v0.1.13", "v0.1.13")


def test_strict_mode_rejects_tagger_date_mismatch_and_missing_maintenance_branch(
    tmp_path: Path,
):
    validator = load_validator()
    repo = make_repo(tmp_path, version="0.1.13", date="2026-07-30")
    tag_annotated(repo, "v0.1.13", "2026-07-29T23:59:59+00:00")
    with pytest.raises(validator.ReleaseValidationError, match="tagger date"):
        validator.validate_release(repo, "v0.1.13", "v0.1.13")

    repo = make_repo(
        tmp_path / "maintenance", version="0.1.13", maintenance_branch=None
    )
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
        "## [0.1.13] - 2026-07-30\n\nfirst\n\n## [0.1.13] - 2026-07-31\n\nsecond\n",
        encoding="utf-8",
    )
    git(repo, "commit", "-am", "duplicate heading")
    source_sha = git(repo, "rev-parse", "HEAD")

    with pytest.raises(validator.ReleaseValidationError, match="2 release headings"):
        validator.validate_release(repo, source_sha, "v0.1.13")


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
        validator.validate_release(repo, source_sha, "v0.1.13")


def test_frozen_tip_requires_both_fetched_remote_branches_at_source_sha(tmp_path: Path):
    validator = load_validator()
    repo = make_repo(tmp_path, version="0.1.13")
    source_sha = git(repo, "rev-parse", "HEAD")

    with pytest.raises(validator.ReleaseValidationError, match="origin/main"):
        validator.validate_frozen_tip(repo, source_sha)

    git(repo, "update-ref", "refs/remotes/origin/main", source_sha)
    git(repo, "update-ref", "refs/remotes/origin/maint/0.1", source_sha)
    validator.validate_frozen_tip(repo, source_sha)

    git(repo, "commit", "--allow-empty", "-m", "branch moved")
    moved_sha = git(repo, "rev-parse", "HEAD")
    git(repo, "update-ref", "refs/remotes/origin/main", moved_sha)
    with pytest.raises(validator.ReleaseValidationError, match="origin/main"):
        validator.validate_frozen_tip(repo, source_sha)


def test_v020_requires_its_exact_contract_protected_refs(tmp_path: Path):
    validator = load_validator()
    repo = make_repo(
        tmp_path,
        version="0.2.0",
        maintenance_branch="maint/0.2",
    )
    source_sha = git(repo, "rev-parse", "HEAD")
    git(repo, "update-ref", "refs/remotes/origin/main", source_sha)
    git(repo, "update-ref", "refs/remotes/origin/maint/0.2", source_sha)

    result = validator.validate_release(
        repo,
        source_sha,
        "v0.2.0",
        frozen_tip=True,
        review_evidence=None,
    )

    assert result.version == "0.2.0"
    git(repo, "update-ref", "-d", "refs/remotes/origin/maint/0.2")
    with pytest.raises(validator.ReleaseValidationError, match="origin/maint/0.2"):
        validator.validate_frozen_tip(repo, source_sha, expected_tag="v0.2.0")


def test_release_validation_rejects_missing_review_evidence(tmp_path: Path):
    validator = load_validator()
    repo = make_repo(tmp_path)
    source_sha = git(repo, "rev-parse", "HEAD")

    with pytest.raises(validator.ReleaseValidationError, match="review evidence"):
        validator.validate_release(
            repo,
            source_sha,
            "v0.1.13",
            review_evidence=repo / "missing-review-evidence.json",
        )


def test_release_validation_rejects_noncontract_review_evidence_path(
    tmp_path: Path,
):
    validator = load_validator()
    repo = make_repo(tmp_path)
    source_sha = git(repo, "rev-parse", "HEAD")
    alternate = repo / "alternate-review.json"
    alternate.write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        validator.ReleaseValidationError,
        match="configured review evidence path",
    ):
        validator.validate_release(
            repo,
            source_sha,
            "v0.1.13",
            review_evidence=alternate,
        )


def test_release_validation_rejects_unconfigured_semver_tag(tmp_path: Path):
    validator = load_validator()
    repo = make_repo(tmp_path, version="0.1.15")
    source_sha = git(repo, "rev-parse", "HEAD")

    with pytest.raises(
        validator.ReleaseValidationError, match="unsupported release tag"
    ):
        validator.validate_release(repo, source_sha, "v0.1.15")


def test_s_to_r_rejects_any_plan_delta_beyond_checkbox_transition(tmp_path: Path):
    validator = load_validator()
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Release Test")
    git(repo, "config", "user.email", "release-test@example.invalid")
    plan = repo / "docs" / "plans" / "2026-08-06-v0.1.13-sol-no-go-followup-plan.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("- [ ] original task\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "S")
    reviewed_commit = git(repo, "rev-parse", "HEAD")
    plan.write_text("- [x] rewritten task\n", encoding="utf-8")
    git(repo, "commit", "-am", "invalid R")
    source_sha = git(repo, "rev-parse", "HEAD")

    with pytest.raises(validator.ReleaseValidationError, match="plan delta"):
        validator.validate_s_to_r(repo, reviewed_commit, source_sha)


def test_s_to_r_rejects_non_task_list_checkbox_change(tmp_path: Path):
    validator = load_validator()
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Release Test")
    git(repo, "config", "user.email", "release-test@example.invalid")
    plan = repo / "docs" / "plans" / "2026-08-06-v0.1.13-sol-no-go-followup-plan.md"
    plan.parent.mkdir(parents=True)
    plan.write_text("Narrative [ ] marker\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "S")
    reviewed_commit = git(repo, "rev-parse", "HEAD")
    plan.write_text("Narrative [x] marker\n", encoding="utf-8")
    git(repo, "commit", "-am", "invalid R")

    with pytest.raises(validator.ReleaseValidationError, match="plan delta"):
        validator.validate_s_to_r(repo, reviewed_commit, git(repo, "rev-parse", "HEAD"))


def test_s_to_r_uses_v0114_plan_and_allowed_paths(tmp_path: Path):
    validator = load_validator()
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "config", "user.name", "Release Test")
    git(repo, "config", "user.email", "release-test@example.invalid")
    plan = repo / "docs" / "plans" / "2026-08-08-v0114-release-plan.md"
    manifest = (
        repo
        / "docs"
        / "developers"
        / "plans"
        / "manifests"
        / "audit-manifest-v0.1.14-release-readiness.yaml"
    )
    plan.parent.mkdir(parents=True)
    manifest.parent.mkdir(parents=True)
    plan.write_text("- [ ] evidence commit\n", encoding="utf-8")
    manifest.write_text("review_evidence_json: |\n  {}\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-m", "S")
    reviewed_commit = git(repo, "rev-parse", "HEAD")

    plan.write_text("- [x] evidence commit\n", encoding="utf-8")
    manifest.write_text(
        'review_evidence_json: |\n  {"entries": []}\n', encoding="utf-8"
    )
    git(repo, "commit", "-am", "R")
    source_sha = git(repo, "rev-parse", "HEAD")

    validator.validate_s_to_r(repo, reviewed_commit, source_sha, expected_tag="v0.1.14")


@pytest.mark.parametrize(
    "expected_tag",
    ["v0.1.13", "v0.1.14", "v0.2.0", "v0.2.2"],
)
def test_historical_s_to_r_contracts_retain_same_commit_behavior(
    tmp_path: Path,
    expected_tag: str,
) -> None:
    validator = load_validator()
    repo = make_repo(tmp_path)
    commit = git(repo, "rev-parse", "HEAD")

    validator.validate_s_to_r(
        repo,
        commit,
        commit,
        expected_tag=expected_tag,
    )


def test_v023_s_to_r_rejects_the_same_reviewed_and_source_commit(
    tmp_path: Path,
) -> None:
    validator = load_validator()
    repo, _plan, _manifest, reviewed_commit = make_v023_s_to_r_repo(tmp_path)

    with pytest.raises(validator.ReleaseValidationError, match="distinct commits"):
        validator.validate_s_to_r(
            repo,
            reviewed_commit,
            reviewed_commit,
            expected_tag="v0.2.3",
        )


@pytest.mark.parametrize(
    "source_manifest",
    [
        pytest.param(None, id="absent"),
        pytest.param(V023_POPULATED_REVIEW_EVIDENCE, id="already-populated"),
        pytest.param(
            f"unreviewed: true\n{V023_EMPTY_REVIEW_EVIDENCE}",
            id="extra-yaml",
        ),
        pytest.param(
            "review_evidence_json: |\n"
            '  {"schema": "gwexpy-v023-review-evidence-v1", "entries": []}\n',
            id="alternate-json-format",
        ),
        pytest.param(
            "review_evidence_json: >\n"
            '  {"schema":"gwexpy-v023-review-evidence-v1","entries":[]}\n',
            id="malformed-block",
        ),
    ],
)
def test_v023_s_to_r_rejects_noncanonical_source_placeholder(
    tmp_path: Path,
    source_manifest: str | None,
) -> None:
    validator = load_validator()
    repo, plan, manifest, reviewed_commit = make_v023_s_to_r_repo(
        tmp_path,
        source_manifest=source_manifest,
    )
    source_sha = commit_v023_r(repo, plan, manifest, update_plan=True)

    with pytest.raises(
        validator.ReleaseValidationError,
        match="exact empty review evidence placeholder",
    ):
        validator.validate_s_to_r(
            repo,
            reviewed_commit,
            source_sha,
            expected_tag="v0.2.3",
        )


@pytest.mark.parametrize("update_plan", [False, True])
def test_v023_s_to_r_accepts_distinct_source_with_exact_empty_placeholder(
    tmp_path: Path,
    update_plan: bool,
) -> None:
    validator = load_validator()
    repo, plan, manifest, reviewed_commit = make_v023_s_to_r_repo(tmp_path)
    source_sha = commit_v023_r(repo, plan, manifest, update_plan=update_plan)

    validated_commit = validator.validate_review_evidence(
        repo,
        manifest,
        source_sha,
        expected_tag="v0.2.3",
    )

    assert validated_commit == reviewed_commit
