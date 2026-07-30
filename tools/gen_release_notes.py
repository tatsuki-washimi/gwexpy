#!/usr/bin/env python3
"""Generate GitHub release notes from CHANGELOG.md.

The CHANGELOG is the single source of truth. This script extracts the stable
release sections and writes Markdown files that can be pasted into GitHub
Releases or consumed by tools/publish_releases.sh.

Usage
-----
Generate a *single* version (preferred for release workflows)::

    python tools/gen_release_notes.py --version 0.1.12

Generate *all* configured versions (legacy bulk mode)::

    python tools/gen_release_notes.py

Flags
-----
--version VERSION
    Only generate release_notes/vVERSION.md.  Must be a dated, stable version
    that appears exactly once in CHANGELOG.md.  "Unreleased" is explicitly
    rejected.  Exits non-zero if the section is absent or duplicated.
--root ROOT
    Project root directory (default: parent of this script).  Useful for
    testing with a temporary directory.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _build_root(root_arg: str | None) -> Path:
    if root_arg is not None:
        return Path(root_arg).resolve()
    return Path(__file__).resolve().parents[1]


SECTION_RE = re.compile(
    r"^## \[(?P<version>[^\]]+)\] - (?P<date>\d{4}-\d{2}-\d{2})\s*$",
    re.MULTILINE,
)

FOOTER_TEMPLATE = """\
---
**Install:** `pip install gwexpy=={version}`
**Full changelog:** https://github.com/tatsuki-washimi/gwexpy/blob/main/CHANGELOG.md
"""

# Legacy bulk-mode version list (kept for backward compatibility).
_LEGACY_VERSIONS = (
    "0.1.0",
    "0.1.1",
    "0.1.2",
    "0.1.3",
    "0.1.4",
    "0.1.5",
    "0.1.6",
    "0.1.7",
    "0.1.8",
    "0.1.9",
    "0.1.10",
    "0.1.11",
    "0.1.12",
)


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------


def _parse_all_matches(changelog_text: str) -> list[re.Match[str]]:
    """Return all dated section matches in document order."""
    return list(SECTION_RE.finditer(changelog_text))


def extract_single_version(changelog_text: str, version: str) -> str:
    """Extract the body of *exactly one* version section from CHANGELOG text.

    Parameters
    ----------
    changelog_text:
        Full text of CHANGELOG.md (LF or CRLF; normalised internally).
    version:
        Bare semantic version string, e.g. ``"0.1.12"``.
        ``"Unreleased"`` is explicitly rejected (non-zero exit).

    Returns
    -------
    str
        Section body, stripped of leading/trailing whitespace.

    Raises
    ------
    SystemExit
        On any contract violation: Unreleased requested, section absent,
        or section appearing more than once.

    """
    if version.lower() == "unreleased" or version == "Unreleased":
        raise SystemExit(
            "error: 'Unreleased' is not a valid release version. "
            "Specify a dated stable version such as '0.1.12'."
        )

    # Normalise CRLF → LF for consistent processing.
    text = changelog_text.replace("\r\n", "\n").replace("\r", "\n")

    matches = _parse_all_matches(text)
    hits = [m for m in matches if m.group("version") == version]

    if len(hits) == 0:
        raise SystemExit(
            f"error: version '{version}' not found in CHANGELOG.md. "
            f"Available dated versions: "
            f"{', '.join(m.group('version') for m in matches)}"
        )
    if len(hits) > 1:
        raise SystemExit(
            f"error: version '{version}' appears {len(hits)} times in "
            f"CHANGELOG.md. Each version must appear exactly once."
        )

    hit = hits[0]
    hit_idx = matches.index(hit)
    start = hit.end()
    end = matches[hit_idx + 1].start() if hit_idx + 1 < len(matches) else len(text)
    return text[start:end].strip()


def build_release_note(version: str, body: str) -> str:
    """Append the standard release footer and enforce LF + single trailing newline."""
    footer = FOOTER_TEMPLATE.format(version=version)
    content = f"{body.rstrip()}\n\n{footer}"
    # Normalise CRLF → LF (defensive; body should already be LF-only).
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    # Ensure exactly one trailing newline.
    content = content.rstrip("\n") + "\n"
    return content


# ---------------------------------------------------------------------------
# Single-version mode
# ---------------------------------------------------------------------------


def generate_one(version: str, root: Path) -> None:
    """Extract *version* from CHANGELOG and write release_notes/vVERSION.md.

    Exits non-zero on any contract violation.
    """
    changelog = root / "CHANGELOG.md"
    if not changelog.exists():
        raise SystemExit(f"error: CHANGELOG.md not found at {changelog}")

    changelog_text = changelog.read_text(encoding="utf-8")
    body = extract_single_version(changelog_text, version)

    output_dir = root / "release_notes"
    output_dir.mkdir(exist_ok=True)
    path = output_dir / f"v{version}.md"
    path.write_text(build_release_note(version, body), encoding="utf-8")
    print(path.relative_to(root))


# ---------------------------------------------------------------------------
# Legacy bulk mode (backward-compatible)
# ---------------------------------------------------------------------------


def _parse_sections_bulk(changelog_text: str) -> dict[str, str]:
    """Return release-note bodies keyed by stable semantic version (bulk mode)."""
    # Normalise CRLF → LF.
    text = changelog_text.replace("\r\n", "\n").replace("\r", "\n")
    matches = _parse_all_matches(text)
    sections: dict[str, str] = {}

    for index, match in enumerate(matches):
        version = match.group("version")
        if version not in _LEGACY_VERSIONS:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[version] = text[start:end].strip()

    missing = [v for v in _LEGACY_VERSIONS if v not in sections]
    if missing:
        raise SystemExit(f"Missing CHANGELOG sections: {', '.join(missing)}")

    return sections


def generate_all(root: Path) -> None:
    """Generate release notes for every version in _LEGACY_VERSIONS."""
    changelog = root / "CHANGELOG.md"
    if not changelog.exists():
        raise SystemExit(f"error: CHANGELOG.md not found at {changelog}")

    changelog_text = changelog.read_text(encoding="utf-8")
    sections = _parse_sections_bulk(changelog_text)

    output_dir = root / "release_notes"
    output_dir.mkdir(exist_ok=True)
    for version in _LEGACY_VERSIONS:
        path = output_dir / f"v{version}.md"
        path.write_text(build_release_note(version, sections[version]), encoding="utf-8")
        print(path.relative_to(root))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GitHub release notes from CHANGELOG.md.",
    )
    parser.add_argument(
        "--version",
        metavar="VERSION",
        help=(
            "Generate release notes for this single version only "
            "(e.g. '0.1.12').  Cannot be 'Unreleased'."
        ),
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        default=None,
        help="Project root directory (default: parent of this script).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parse arguments and dispatch to generate_one or generate_all."""
    args = _parse_args(argv)
    root = _build_root(args.root)

    if args.version is not None:
        generate_one(args.version, root)
    else:
        generate_all(root)


if __name__ == "__main__":
    main()
