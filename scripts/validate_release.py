#!/usr/bin/env python3
"""Fail-closed validation for GWexpy release sources and release tags."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

RELEASE_TAG_PATTERN = re.compile(
    r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
)
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ReleaseValidationError(ValueError):
    """Raised when a release candidate violates a required invariant."""


@dataclass(frozen=True)
class ValidationResult:
    mode: str
    source_sha: str
    version: str
    release_date: str


def is_release_tag(value: str) -> bool:
    """Return whether *value* is a supported final-release SemVer tag."""
    return RELEASE_TAG_PATTERN.fullmatch(value) is not None


def _git(repo_root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode:
        raise ReleaseValidationError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _read_cff_field(repo_root: Path, field: str) -> str:
    """Return the single top-level CITATION.cff *field* value.

    Duplicates are rejected rather than resolved by first-match: with two
    ``version:`` lines the release metadata is ambiguous, and reading only the
    first one lets a stale value ship while validation reports success.
    """
    path = repo_root / "CITATION.cff"
    matches = re.findall(
        rf"^{re.escape(field)}\s*:\s*['\"]?([^'\"#\n]+)['\"]?\s*(?:#.*)?$",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if not matches:
        raise ReleaseValidationError(f"CITATION.cff has no valid {field!r} field")
    if len(matches) > 1:
        raise ReleaseValidationError(
            f"CITATION.cff has {len(matches)} top-level {field!r} fields; "
            "release metadata must be unambiguous"
        )
    return matches[0].strip()


def _read_changelog_date(repo_root: Path, version: str) -> str:
    """Return the date of the single CHANGELOG heading for *version*.

    A duplicate heading is an error, not a first-match lookup: the release-note
    generator rejects it outright, so accepting it here would let the validator
    pass on a tree whose release notes cannot be generated.
    """
    content = (repo_root / "CHANGELOG.md").read_text(encoding="utf-8")
    matches = re.findall(
        rf"^## \[{re.escape(version)}\]\s*-\s*(\d{{4}}-\d{{2}}-\d{{2}})\s*$",
        content,
        flags=re.MULTILINE,
    )
    if not matches:
        raise ReleaseValidationError(
            f"CHANGELOG.md has no dated release entry for {version}"
        )
    if len(matches) > 1:
        raise ReleaseValidationError(
            f"CHANGELOG.md has {len(matches)} release headings for {version}; "
            "release metadata must be unambiguous"
        )
    return matches[0]


def _read_metadata(repo_root: Path, version: str) -> str:
    py_match = re.search(
        r'__version__\s*=\s*["\']([^"\']+)["\']',
        (repo_root / "gwexpy" / "_version.py").read_text(encoding="utf-8"),
    )
    if not py_match:
        raise ReleaseValidationError("gwexpy/_version.py has no __version__")
    versions = {
        "package": py_match.group(1),
        "CITATION.cff": _read_cff_field(repo_root, "version"),
        ".zenodo.json": str(
            json.loads((repo_root / ".zenodo.json").read_text(encoding="utf-8"))[
                "version"
            ]
        ),
    }
    expected_version = version.removeprefix("v")
    mismatches = [
        f"{name}={value}"
        for name, value in versions.items()
        if value != expected_version
    ]
    if mismatches:
        raise ReleaseValidationError(
            f"release version must be {expected_version}; " + ", ".join(mismatches)
        )

    dates = {
        "CITATION.cff": _read_cff_field(repo_root, "date-released"),
        ".zenodo.json": str(
            json.loads((repo_root / ".zenodo.json").read_text(encoding="utf-8"))[
                "publication_date"
            ]
        ),
        "CHANGELOG.md": _read_changelog_date(repo_root, expected_version),
    }
    for name, value in dates.items():
        if not DATE_PATTERN.fullmatch(value):
            raise ReleaseValidationError(
                f"{name} release date is not YYYY-MM-DD: {value}"
            )
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except ValueError as exc:
            raise ReleaseValidationError(
                f"{name} release date is invalid: {value}"
            ) from exc
    if len(set(dates.values())) != 1:
        raise ReleaseValidationError(
            "release dates differ: " + ", ".join(f"{k}={v}" for k, v in dates.items())
        )
    return next(iter(dates.values()))


def _ref_exists(repo_root: Path, ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            cwd=repo_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def _branch_ref(repo_root: Path, branch: str) -> str:
    for ref in (f"refs/heads/{branch}", f"refs/remotes/origin/{branch}"):
        if _ref_exists(repo_root, ref):
            return ref
    raise ReleaseValidationError(f"required ancestry branch {branch!r} is unavailable")


def _require_ancestry(repo_root: Path, source_sha: str, tag: str) -> None:
    match = RELEASE_TAG_PATTERN.fullmatch(tag)
    assert match is not None
    version = tuple(int(part) for part in match.groups())
    branch = (
        "maint/0.1"
        if version[0] == 0 and version[1] == 1 and version[2] > 12
        else "main"
    )
    ref = _branch_ref(repo_root, branch)
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_sha, ref],
        cwd=repo_root,
        check=False,
    )
    if result.returncode:
        raise ReleaseValidationError(f"{source_sha} is not an ancestor of {branch}")


def _tagger_utc_date(repo_root: Path, tag: str) -> str:
    content = _git(repo_root, "cat-file", "-p", f"refs/tags/{tag}")
    match = re.search(r"^tagger .+ (\d+) [+-]\d{4}$", content, flags=re.MULTILINE)
    if not match:
        raise ReleaseValidationError(
            f"annotated tag {tag} has no parseable tagger date"
        )
    return datetime.fromtimestamp(int(match.group(1)), UTC).date().isoformat()


def validate_release(
    repo_root: Path | str,
    release_ref: str,
    expected_tag: str,
) -> ValidationResult:
    """Validate a strict historical tag or a candidate commit SHA."""
    root = Path(repo_root).resolve()
    if not is_release_tag(expected_tag):
        raise ReleaseValidationError(
            f"expected_tag is not a final SemVer tag: {expected_tag}"
        )

    if is_release_tag(release_ref):
        if release_ref != expected_tag:
            raise ReleaseValidationError(
                "historical release_ref must equal expected_tag"
            )
        tag_ref = f"refs/tags/{release_ref}"
        if not _ref_exists(root, tag_ref):
            raise ReleaseValidationError(f"release tag does not exist: {release_ref}")
        if _git(root, "cat-file", "-t", tag_ref) != "tag":
            raise ReleaseValidationError(
                f"release tag must be annotated: {release_ref}"
            )
        source_sha = _git(root, "rev-parse", f"{tag_ref}^{{commit}}")
        if _git(root, "rev-parse", "HEAD") != source_sha:
            raise ReleaseValidationError(
                "checked-out source SHA does not match the tag peel"
            )
        release_date = _read_metadata(root, expected_tag)
        if _tagger_utc_date(root, release_ref) != release_date:
            raise ReleaseValidationError(
                "annotated tagger date does not match release metadata"
            )
        _require_ancestry(root, source_sha, expected_tag)
        return ValidationResult(
            "strict", source_sha, expected_tag.removeprefix("v"), release_date
        )

    if not SHA_PATTERN.fullmatch(release_ref):
        raise ReleaseValidationError(
            "release_ref must be a final version tag or full 40-character lowercase SHA"
        )
    if _ref_exists(root, f"refs/tags/{expected_tag}"):
        raise ReleaseValidationError("candidate expected_tag must not already exist")
    source_sha = _git(root, "rev-parse", f"{release_ref}^{{commit}}")
    if source_sha != release_ref:
        raise ReleaseValidationError(
            "release_ref must resolve to the supplied full commit SHA"
        )
    if _git(root, "rev-parse", "HEAD") != source_sha:
        raise ReleaseValidationError(
            "checked-out source SHA does not match release_ref"
        )
    release_date = _read_metadata(root, expected_tag)
    _require_ancestry(root, source_sha, expected_tag)
    return ValidationResult(
        "candidate", source_sha, expected_tag.removeprefix("v"), release_date
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release-ref", required=True)
    parser.add_argument("--expected-tag", required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_release(args.repo_root, args.release_ref, args.expected_tag)
    except ReleaseValidationError as exc:
        print(f"release validation failed: {exc}")
        return 1
    print(f"mode={result.mode}")
    print(f"source_sha={result.source_sha}")
    print(f"version={result.version}")
    print(f"release_date={result.release_date}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
