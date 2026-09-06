"""Contracts for the public docs release-status substitutions."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE_STATUS = ROOT / "docs_redesign/release_status.json"
VERSION_SOURCE = ROOT / "gwexpy/_version.py"
DOCUMENTATION_VERSION = ROOT / "docs_redesign/about/documentation_version.md"
JAPANESE_CATALOGUE = (
    ROOT / "docs_redesign/locales/ja/LC_MESSAGES/about/documentation_version.po"
)


def _package_version() -> str:
    match = re.search(
        r"__version__\s*=\s*[\"\']([^\"\']+)[\"\']",
        VERSION_SOURCE.read_text(encoding="utf-8"),
    )
    assert match is not None
    return match.group(1)


def test_release_status_tracks_the_current_published_package() -> None:
    """Stable and introductory docs targets must follow the released package."""
    status = json.loads(RELEASE_STATUS.read_text(encoding="utf-8"))
    assert status == {
        "stable_release": "0.2.3",
        "intro_examples_release": "0.2.3",
    }
    assert status["stable_release"] == _package_version()


def test_release_status_is_consumed_by_both_language_documents() -> None:
    """EN and JA pages keep the version values substitution-driven."""
    source = DOCUMENTATION_VERSION.read_text(encoding="utf-8")
    assert "{{ stable_release }}" in source
    assert "{{ intro_examples_release }}" in source

    catalogue = JAPANESE_CATALOGUE.read_text(encoding="utf-8")
    assert "{{ stable_release }}" in catalogue
    assert "{{ intro_examples_release }}" in catalogue
