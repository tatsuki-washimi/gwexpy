#!/usr/bin/env python3
# ruff: noqa: D103
"""Warn about release metadata and public posture drift.

This script is intentionally read-only and warning-only. It is used by the
project Stop hook, and can also be tested with a synthetic changed-file list.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

METADATA_FILES = {
    "pyproject.toml",
    "gwexpy/_version.py",
    "CITATION.cff",
    ".zenodo.json",
    "CHANGELOG.md",
}
PUBLIC_POSTURE_DOCS = {
    "README.md",
    "docs/web/en/user_guide/installation.md",
    "docs/web/en/user_guide/quickstart.md",
    "docs/web/en/user_guide/changelog.md",
    "docs/web/ja/user_guide/installation.md",
    "docs/web/ja/user_guide/quickstart.md",
    "docs/web/ja/user_guide/changelog.md",
}
TEXT_SUFFIXES = {".md", ".rst", ".txt"}
POSTURE_RE = re.compile(
    r"\b(pending|unpublished|published|released|available on PyPI|conda-forge)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check gwexpy release metadata and public posture wording."
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--changed-file-list",
        type=Path,
        help="Read newline-separated changed paths from this file.",
    )
    parser.add_argument(
        "--changed-from-stdin",
        action="store_true",
        help="Read newline-separated changed paths from stdin.",
    )
    return parser.parse_args()


def run_git(repo: Path, *args: str) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def changed_files(repo: Path, args: argparse.Namespace) -> list[str]:
    if args.changed_from_stdin:
        return sorted({line.strip() for line in sys.stdin if line.strip()})
    if args.changed_file_list:
        return sorted(
            {
                line.strip()
                for line in args.changed_file_list.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            }
        )

    names: set[str] = set()
    for git_args in (
        ("diff", "--name-only", "HEAD"),
        ("diff", "--name-only"),
        ("ls-files", "--others", "--exclude-standard"),
    ):
        names.update(run_git(repo, *git_args))
    return sorted(names)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def read_json(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def read_toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def normalize_license(value: object) -> str | None:
    if isinstance(value, dict):
        value = value.get("id") or value.get("text")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip().upper()


def normalize_person_name(value: str) -> str:
    name = value.strip().strip("\"'")
    if "," in name:
        family, given = [part.strip() for part in name.split(",", 1)]
        name = f"{given} {family}".strip()
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def pyproject_metadata(path: Path) -> tuple[str | None, set[str]]:
    project = read_toml(path).get("project", {})
    if not isinstance(project, dict):
        return None, set()

    license_value = normalize_license(project.get("license"))
    author_names: set[str] = set()
    authors = project.get("authors", [])
    if isinstance(authors, list):
        for author in authors:
            if isinstance(author, dict) and isinstance(author.get("name"), str):
                author_names.add(normalize_person_name(author["name"]))
    return license_value, author_names


def version_from_python(path: Path) -> str | None:
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", read_text(path), re.M)
    return match.group(1) if match else None


def pyproject_status(path: Path) -> tuple[str, str | None]:
    text = read_text(path)
    if re.search(
        r"version\s*=\s*\{[^}]*attr\s*=\s*['\"]gwexpy\._version\.__version__['\"]",
        text,
    ):
        return "dynamic-gwexpy-version", None
    match = re.search(r"^version\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
    if match:
        return "literal", match.group(1)
    return "missing", None


def cff_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key in ("version", "date-released", "license"):
        match = re.search(
            rf"^{re.escape(key)}:\s*['\"]?([^'\"\n]+)", read_text(path), re.M
        )
        if match:
            fields[key] = match.group(1).strip()
    return fields


def cff_author_names(path: Path) -> set[str]:
    text = read_text(path)
    names: set[str] = set()
    for match in re.finditer(
        r"family-names:\s*['\"]?([^'\"\n]+).*?given-names:\s*['\"]?([^'\"\n]+)",
        text,
        re.S,
    ):
        family = match.group(1).strip()
        given = match.group(2).strip()
        names.add(normalize_person_name(f"{given} {family}"))
    return names


def zenodo_author_names(data: dict[str, object]) -> set[str]:
    creators = data.get("creators", [])
    names: set[str] = set()
    if isinstance(creators, list):
        for creator in creators:
            if isinstance(creator, dict) and isinstance(creator.get("name"), str):
                names.add(normalize_person_name(creator["name"]))
    return names


def changelog_has_version(path: Path, version: str) -> bool:
    text = read_text(path)
    if not text:
        return False
    return (
        re.search(rf"(^|[^0-9A-Za-z])v?{re.escape(version)}([^0-9A-Za-z]|$)", text)
        is not None
    )


def metadata_issues(repo: Path, changed: list[str]) -> list[str]:
    if not METADATA_FILES.intersection(changed):
        return []

    issues: list[str] = []
    package_version = version_from_python(repo / "gwexpy/_version.py")
    pyproject_kind, pyproject_version = pyproject_status(repo / "pyproject.toml")
    pyproject_license, pyproject_authors = pyproject_metadata(repo / "pyproject.toml")
    citation = cff_fields(repo / "CITATION.cff")
    citation_authors = cff_author_names(repo / "CITATION.cff")
    zenodo = read_json(repo / ".zenodo.json")

    if package_version is None:
        issues.append("gwexpy/_version.py does not expose __version__.")

    if pyproject_kind == "literal" and pyproject_version != package_version:
        issues.append(
            f"pyproject.toml version {pyproject_version!r} differs from __version__ {package_version!r}."
        )
    elif pyproject_kind == "missing":
        issues.append(
            "pyproject.toml has neither a literal version nor the gwexpy dynamic version attr."
        )

    citation_version = citation.get("version")
    if citation_version and package_version and citation_version != package_version:
        issues.append(
            f"CITATION.cff version {citation_version!r} differs from __version__ {package_version!r}."
        )

    zenodo_version = zenodo.get("version")
    if zenodo_version and package_version and str(zenodo_version) != package_version:
        issues.append(
            f".zenodo.json version {zenodo_version!r} differs from __version__ {package_version!r}."
        )

    citation_date = citation.get("date-released")
    zenodo_date = zenodo.get("publication_date")
    if citation_date and zenodo_date and str(zenodo_date) != citation_date:
        issues.append(
            f".zenodo.json publication_date {zenodo_date!r} differs from CITATION.cff date-released {citation_date!r}."
        )

    if package_version and not changelog_has_version(
        repo / "CHANGELOG.md", package_version
    ):
        issues.append(
            f"CHANGELOG.md does not appear to mention version {package_version}."
        )

    citation_license = normalize_license(citation.get("license"))
    zenodo_license = normalize_license(zenodo.get("license"))
    known_licenses = {
        "pyproject.toml": pyproject_license,
        "CITATION.cff": citation_license,
        ".zenodo.json": zenodo_license,
    }
    licenses = {source: value for source, value in known_licenses.items() if value}
    if len(set(licenses.values())) > 1:
        rendered = ", ".join(f"{source}={value}" for source, value in licenses.items())
        issues.append(f"License metadata differs across release files: {rendered}.")

    zenodo_authors = zenodo_author_names(zenodo)
    known_authors = {
        "pyproject.toml": pyproject_authors,
        "CITATION.cff": citation_authors,
        ".zenodo.json": zenodo_authors,
    }
    author_sets = {source: value for source, value in known_authors.items() if value}
    if len({tuple(sorted(value)) for value in author_sets.values()}) > 1:
        rendered = ", ".join(
            f"{source}={'; '.join(sorted(value))}"
            for source, value in author_sets.items()
        )
        issues.append(f"Author metadata differs across release files: {rendered}.")

    return issues


def is_public_doc(path: str) -> bool:
    return path in PUBLIC_POSTURE_DOCS


def posture_hits(repo: Path, changed: list[str]) -> list[str]:
    hits: list[str] = []
    for rel_path in changed:
        if not is_public_doc(rel_path):
            continue
        path = repo / rel_path
        if path.suffix and path.suffix not in TEXT_SUFFIXES:
            continue
        text = read_text(path)
        if not text:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if POSTURE_RE.search(line):
                snippet = line.strip()
                if len(snippet) > 120:
                    snippet = snippet[:117] + "..."
                hits.append(f"{rel_path}:{lineno}: {snippet}")
                if len(hits) >= 8:
                    return hits
    return hits


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    changed = changed_files(repo, args)
    if not changed:
        return 0

    issues = metadata_issues(repo, changed)
    hits = posture_hits(repo, changed)
    if not issues and not hits:
        return 0

    print("\n[gwexpy/release-posture]", file=sys.stderr)
    if issues:
        print("  Release metadata drift detected:", file=sys.stderr)
        for issue in issues:
            print(f"    - {issue}", file=sys.stderr)
    if hits:
        print("  Public docs changed with release/posture wording:", file=sys.stderr)
        for hit in hits:
            print(f"    - {hit}", file=sys.stderr)
        print(
            "  Check that wording matches the current PyPI/conda-forge/Zenodo state.",
            file=sys.stderr,
        )
    print("  Run /metadata-checker before release or PR handoff.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
