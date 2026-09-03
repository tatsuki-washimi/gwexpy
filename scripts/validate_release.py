#!/usr/bin/env python3
"""Fail-closed validation for GWexpy release sources and release tags."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

RELEASE_TAG_PATTERN = re.compile(
    r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
)
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
V023_EMPTY_REVIEW_EVIDENCE_PLACEHOLDER = b"""\
review_evidence_json: |
  {
    "schema": "gwexpy-v023-review-evidence-v1",
    "entries": []
  }
"""


class ReleaseValidationError(ValueError):
    """Raised when a release candidate violates a required invariant."""


@dataclass(frozen=True)
class ValidationResult:
    mode: str
    source_sha: str
    version: str
    release_date: str
    artifact_prefix: str


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


def _require_ancestry(repo_root: Path, source_sha: str, expected_tag: str) -> None:
    protected_refs = cast(list[str], _release_contract(expected_tag)["protected_refs"])
    for branch in protected_refs:
        ref = _branch_ref(repo_root, str(branch))
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_sha, ref],
            cwd=repo_root,
            check=False,
        )
        if result.returncode:
            raise ReleaseValidationError(f"{source_sha} is not an ancestor of {branch}")


def validate_frozen_tip(
    repo_root: Path | str, source_sha: str, *, expected_tag: str = "v0.1.13"
) -> None:
    """Require every fetched contract protected tip to equal *source_sha*."""
    root = Path(repo_root).resolve()
    if not SHA_PATTERN.fullmatch(source_sha):
        raise ReleaseValidationError(
            "frozen-tip source SHA must be a full lowercase SHA"
        )
    protected_refs = cast(list[str], _release_contract(expected_tag)["protected_refs"])
    for branch in protected_refs:
        ref = f"refs/remotes/origin/{branch}"
        if not _ref_exists(root, ref):
            raise ReleaseValidationError(
                f"frozen-tip requires fetched {ref.removeprefix('refs/remotes/')}"
            )
        if _git(root, "rev-parse", ref) != source_sha:
            raise ReleaseValidationError(
                f"frozen-tip requires {ref.removeprefix('refs/remotes/')} == {source_sha}"
            )


def _review_validator_module():
    path = Path(__file__).with_name("ci") / "validate_release_review_evidence.py"
    spec = importlib.util.spec_from_file_location("release_review_evidence", path)
    if spec is None or spec.loader is None:
        raise ReleaseValidationError("review evidence validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _release_contract(expected_tag: str) -> dict[str, object]:
    path = Path(__file__).with_name("ci") / "release_contract.py"
    spec = importlib.util.spec_from_file_location("release_contract", path)
    if spec is None or spec.loader is None:
        raise ReleaseValidationError("release contract loader is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        return module.release_contract(expected_tag)
    except module.ReleaseContractError as exc:
        raise ReleaseValidationError(str(exc)) from exc


def validate_review_evidence(
    repo_root: Path | str,
    evidence_path: Path | str,
    source_sha: str | None = None,
    *,
    expected_tag: str = "v0.1.13",
) -> str:
    """Validate Terra evidence at S and, when supplied, bind S safely to R."""
    root = Path(repo_root).resolve()
    candidate = Path(evidence_path)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(root)
    except ValueError as exc:
        raise ReleaseValidationError(
            "review evidence must remain inside repo root"
        ) from exc
    contract = _release_contract(expected_tag)
    configured_evidence = (root / str(contract["review_evidence_path"])).resolve()
    if resolved_candidate != configured_evidence:
        raise ReleaseValidationError(
            "review evidence must use the configured review evidence path"
        )
    module = _review_validator_module()
    try:
        evidence = module.validate_review_evidence(
            candidate,
            None,
            None,
            root,
            expected_tag=expected_tag,
        )
    except module.ReleaseReviewEvidenceError as exc:
        raise ReleaseValidationError(
            f"review evidence validation failed: {exc}"
        ) from exc
    reviewed_commit = evidence["entries"][0]["reviewed_commit"]
    if source_sha is not None:
        validate_s_to_r(root, reviewed_commit, source_sha, expected_tag=expected_tag)
    return reviewed_commit


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, check=False
    )
    if result.returncode:
        raise ReleaseValidationError(f"git {' '.join(args)} failed")
    return result.stdout


def _validate_plan_delta(
    repo_root: Path,
    reviewed_commit: str,
    source_sha: str,
    plan: str,
) -> None:
    before = _git_bytes(repo_root, "show", f"{reviewed_commit}:{plan}").splitlines(
        keepends=True
    )
    after = _git_bytes(repo_root, "show", f"{source_sha}:{plan}").splitlines(
        keepends=True
    )
    if len(before) != len(after):
        raise ReleaseValidationError("S-to-R plan delta changes its line structure")
    for old, new in zip(before, after, strict=True):
        if old == new:
            continue
        if (
            not old.startswith(b"- [ ]")
            or old.count(b"[ ]") != 1
            or new != old.replace(b"[ ]", b"[x]", 1)
        ):
            raise ReleaseValidationError(
                "S-to-R plan delta must only transition existing checkbox [ ] to [x]"
            )


def _validate_v023_review_source_placeholder(
    repo_root: Path,
    reviewed_commit: str,
    evidence_path: str,
) -> None:
    try:
        source = _git_bytes(
            repo_root,
            "show",
            f"{reviewed_commit}:{evidence_path}",
        )
    except ReleaseValidationError as exc:
        raise ReleaseValidationError(
            "v0.2.3 reviewed source must contain the exact empty review "
            "evidence placeholder"
        ) from exc
    if source != V023_EMPTY_REVIEW_EVIDENCE_PLACEHOLDER:
        raise ReleaseValidationError(
            "v0.2.3 reviewed source must contain the exact empty review "
            "evidence placeholder"
        )


def validate_s_to_r(
    repo_root: Path | str,
    reviewed_commit: str,
    source_sha: str,
    *,
    expected_tag: str = "v0.1.13",
) -> None:
    """Bind Terra's reviewed S to R through only coordinator-owned deltas."""
    root = Path(repo_root).resolve()
    if expected_tag == "v0.2.3" and reviewed_commit == source_sha:
        raise ReleaseValidationError("v0.2.3 S-to-R binding requires distinct commits")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", reviewed_commit, source_sha],
        cwd=root,
        check=False,
    )
    if ancestor.returncode:
        raise ReleaseValidationError("reviewed commit is not an ancestor of source SHA")
    changed = _git(
        root, "diff", "--name-only", "-z", f"{reviewed_commit}..{source_sha}"
    )
    paths = {path for path in changed.split("\0") if path}
    contract = _release_contract(expected_tag)
    if expected_tag == "v0.2.3":
        _validate_v023_review_source_placeholder(
            root,
            reviewed_commit,
            str(contract["review_evidence_path"]),
        )
    allowed_paths = set(cast(list[str], contract["s_to_r_allowed_paths"]))
    if not paths <= allowed_paths:
        raise ReleaseValidationError(
            "S-to-R changes invalidate review evidence outside coordinator audit paths"
        )
    plan = str(contract["plan_path"])
    if plan in paths:
        _validate_plan_delta(root, reviewed_commit, source_sha, plan)


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
    *,
    frozen_tip: bool = False,
    review_evidence: Path | str | None = None,
) -> ValidationResult:
    """Validate a strict historical tag or a candidate commit SHA."""
    root = Path(repo_root).resolve()
    if not is_release_tag(expected_tag):
        raise ReleaseValidationError(
            f"expected_tag is not a final SemVer tag: {expected_tag}"
        )
    contract = _release_contract(expected_tag)
    artifact_prefix = str(contract["artifact_prefix"])

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
        if review_evidence is not None:
            validate_review_evidence(
                root,
                review_evidence,
                source_sha,
                expected_tag=expected_tag,
            )
        if frozen_tip:
            validate_frozen_tip(root, source_sha, expected_tag=expected_tag)
        return ValidationResult(
            "strict",
            source_sha,
            expected_tag.removeprefix("v"),
            release_date,
            artifact_prefix,
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
    if review_evidence is not None:
        validate_review_evidence(
            root,
            review_evidence,
            source_sha,
            expected_tag=expected_tag,
        )
    if frozen_tip:
        validate_frozen_tip(root, source_sha, expected_tag=expected_tag)
    return ValidationResult(
        "candidate",
        source_sha,
        expected_tag.removeprefix("v"),
        release_date,
        artifact_prefix,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release-ref")
    parser.add_argument("--expected-tag")
    parser.add_argument("--frozen-tip", action="store_true")
    parser.add_argument("--review-evidence", type=Path)
    parser.add_argument("--review-evidence-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.review_evidence_only:
            if args.review_evidence is None:
                raise ReleaseValidationError(
                    "--review-evidence-only requires --review-evidence"
                )
            if args.expected_tag is None:
                raise ReleaseValidationError(
                    "--review-evidence-only requires --expected-tag"
                )
            reviewed_commit = validate_review_evidence(
                args.repo_root,
                args.review_evidence,
                expected_tag=args.expected_tag,
            )
            print(f"reviewed_commit={reviewed_commit}")
            return 0
        if args.release_ref is None or args.expected_tag is None:
            raise ReleaseValidationError(
                "--release-ref and --expected-tag are required"
            )
        review_evidence = args.review_evidence
        if review_evidence is None:
            review_evidence = Path(
                str(_release_contract(args.expected_tag)["review_evidence_path"])
            )
        result = validate_release(
            args.repo_root,
            args.release_ref,
            args.expected_tag,
            frozen_tip=args.frozen_tip,
            review_evidence=review_evidence,
        )
    except ReleaseValidationError as exc:
        print(f"release validation failed: {exc}")
        return 1
    print(f"mode={result.mode}")
    print(f"source_sha={result.source_sha}")
    print(f"version={result.version}")
    print(f"release_date={result.release_date}")
    print(f"artifact_prefix={result.artifact_prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
