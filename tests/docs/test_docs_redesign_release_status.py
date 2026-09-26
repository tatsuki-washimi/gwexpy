"""Contracts for the public docs release-status substitutions."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RELEASE_STATUS = ROOT / "docs_redesign/release_status.json"
DOCUMENTATION_VERSION = ROOT / "docs_redesign/about/documentation_version.md"
KNOWN_LIMITATIONS = ROOT / "docs_redesign/about/known_limitations.md"
INSTALLATION = ROOT / "docs_redesign/tutorials/installation.md"
REDESIGN_CONF = ROOT / "docs_redesign/conf.py"
JAPANESE_CATALOGUES = (
    ROOT / "docs_redesign/locales/ja/LC_MESSAGES/about/documentation_version.po",
    ROOT / "docs_redesign/locales/ja/LC_MESSAGES/about/known_limitations.po",
    ROOT / "docs_redesign/locales/ja/LC_MESSAGES/tutorials/installation.po",
)

# Construct the legacy name so this test can forbid it in live sources without
# reintroducing the literal key into the source it audits.
_LEGACY_KEY = "stable_" + "release"


def test_release_status_records_latest_published_and_example_releases() -> None:
    """Published-release metadata is independent from the development version."""
    status = json.loads(RELEASE_STATUS.read_text(encoding="utf-8"))
    assert status == {
        "latest_release": "0.2.4",
        "intro_examples_release": "0.2.4",
    }
    assert _LEGACY_KEY not in status


def test_release_status_is_consumed_by_both_language_documents() -> None:
    """EN and JA pages keep both version values substitution-driven."""
    source = DOCUMENTATION_VERSION.read_text(encoding="utf-8")
    assert "Latest release: **{{ latest_release }}**." in source
    assert "{{ intro_examples_release }}" in source

    install_source = INSTALLATION.read_text(encoding="utf-8")
    assert "GWexpy {{ latest_release }} is available on [PyPI]" in install_source
    assert "The latest conda-forge package is v0.2.3." in install_source
    assert "https://zenodo.org/records/22978439" in install_source

    catalogue = (
        ROOT / "docs_redesign/locales/ja/LC_MESSAGES/about/documentation_version.po"
    ).read_text(encoding="utf-8")
    assert "Latest release: " in catalogue
    assert "{{ latest_release }}" in catalogue
    assert "{{ intro_examples_release }}" in catalogue
    assert "最新リリース: **{{ latest_release }}**" in catalogue

    install_catalogue = (
        ROOT / "docs_redesign/locales/ja/LC_MESSAGES/tutorials/installation.po"
    ).read_text(encoding="utf-8")
    assert "GWexpy {{ latest_release }}" in install_catalogue
    assert "conda-forge の最新パッケージは v0.2.3 です。" in install_catalogue
    assert "https://zenodo.org/records/22978439" in install_catalogue


def test_legacy_release_name_is_absent_from_live_docs_contract() -> None:
    """Historical records may retain old terminology; live docs may not."""
    live_sources = (
        RELEASE_STATUS,
        DOCUMENTATION_VERSION,
        KNOWN_LIMITATIONS,
        INSTALLATION,
        REDESIGN_CONF,
        *JAPANESE_CATALOGUES,
    )
    for path in live_sources:
        assert _LEGACY_KEY not in path.read_text(encoding="utf-8"), path
