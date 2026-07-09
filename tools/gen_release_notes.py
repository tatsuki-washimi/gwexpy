#!/usr/bin/env python3
"""Generate GitHub release notes from CHANGELOG.md.

The CHANGELOG is the single source of truth. This script extracts the stable
release sections and writes Markdown files that can be pasted into GitHub
Releases or consumed by tools/publish_releases.sh.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
OUTPUT_DIR = ROOT / "release_notes"

VERSIONS = (
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
)

SECTION_RE = re.compile(
    r"^## \[(?P<version>[^\]]+)\] - (?P<date>\d{4}-\d{2}-\d{2})\s*$",
    re.MULTILINE,
)

FOOTER_TEMPLATE = """---
**Install:** `pip install gwexpy=={version}`
**Full changelog:** https://github.com/tatsuki-washimi/gwexpy/blob/main/CHANGELOG.md
"""


def parse_sections(changelog_text: str) -> dict[str, str]:
    """Return release-note bodies keyed by stable semantic version."""
    matches = list(SECTION_RE.finditer(changelog_text))
    sections: dict[str, str] = {}

    for index, match in enumerate(matches):
        version = match.group("version")
        if version not in VERSIONS:
            continue

        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(changelog_text)
        sections[version] = changelog_text[start:end].strip()

    missing = [version for version in VERSIONS if version not in sections]
    if missing:
        raise SystemExit(f"Missing CHANGELOG sections: {', '.join(missing)}")

    return sections


def build_release_note(version: str, body: str) -> str:
    """Append the standard release footer to a CHANGELOG section body."""
    footer = FOOTER_TEMPLATE.format(version=version)
    return f"{body.rstrip()}\n\n{footer}"


def main() -> None:
    changelog_text = CHANGELOG.read_text(encoding="utf-8")
    sections = parse_sections(changelog_text)

    OUTPUT_DIR.mkdir(exist_ok=True)
    for version in VERSIONS:
        tag = f"v{version}"
        path = OUTPUT_DIR / f"{tag}.md"
        path.write_text(build_release_note(version, sections[version]), encoding="utf-8")
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
