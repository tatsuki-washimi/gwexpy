from __future__ import annotations

import json
import re
from pathlib import Path

from babel.messages import pofile

REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_TITLE = (
    "GWexpy: Extending GWpy with metadata-preserving multidimensional "
    "abstractions for detector commissioning"
)
RELEASE_PAGE = "https://github.com/tatsuki-washimi/gwexpy/releases"
PYPI_PAGE = "https://pypi.org/project/gwexpy/"
EXACT_TAG_CITATION = "blob/<exact tag>/CITATION.cff"

INSTALLATION_SOURCES = (
    "docs_redesign/tutorials/installation.md",
    "docs/web/en/user_guide/installation.md",
    "docs/web/ja/user_guide/installation.md",
)
CLI_SOURCES = (
    "docs_redesign/how-to/cli.md",
    "docs/web/en/user_guide/cli.md",
    "docs/web/ja/user_guide/cli.md",
)
CITATION_SOURCES = (
    "docs_redesign/about/citation.md",
    "docs/web/en/user_guide/citation.md",
    "docs/web/ja/user_guide/citation.md",
)
JAPANESE_CATALOGUES = (
    "docs_redesign/locales/ja/LC_MESSAGES/tutorials/installation.po",
    "docs_redesign/locales/ja/LC_MESSAGES/how-to/cli.po",
    "docs_redesign/locales/ja/LC_MESSAGES/about/citation.po",
)
PUBLIC_DOC_INVENTORY = (
    *INSTALLATION_SOURCES,
    *CLI_SOURCES,
    *CITATION_SOURCES,
    *JAPANESE_CATALOGUES,
)


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_public_release_docs_are_a_closed_version_neutral_inventory() -> None:
    """Reject stale release, CLI, and citation facts in every public variant."""
    assert len(PUBLIC_DOC_INVENTORY) == 12
    contents = {path: _read(path) for path in PUBLIC_DOC_INVENTORY}

    for path, source in contents.items():
        assert "current release:" not in source, path
        assert "v0.1.13" not in source, path
        assert (
            "GWexpy: Extended Analysis Utilities for Gravitational Wave Data"
            not in source
        ), path
        assert "blob/main/CITATION.cff" not in source, path
        assert not re.search(r"gwexpy 0\.1\.(?:0|10)\b", source), path

    for path in INSTALLATION_SOURCES:
        source = contents[path]
        assert PYPI_PAGE in source, path
        assert RELEASE_PAGE in source, path
        assert "Python 3.11 or later" in source or "Python 3.11 以上" in source, path

    for path in CLI_SOURCES:
        assert "gwexpy <installed version>" in contents[path], path

    for path in CITATION_SOURCES:
        source = contents[path]
        assert OFFICIAL_TITLE in source, path
        assert "<version used>" in source, path
        assert EXACT_TAG_CITATION in source, path


def test_japanese_catalogues_translate_the_version_neutral_source_messages() -> None:
    required_messages = {
        JAPANESE_CATALOGUES[0]: (
            "GWexpy {{ stable_release }} is available from both [PyPI](https://pypi.org/project/gwexpy/) "
            "and [conda-forge](https://anaconda.org/conda-forge/gwexpy). Check PyPI, "
            "conda-forge, or the [release page](https://github.com/tatsuki-washimi/gwexpy/releases) "
            "for available versions."
        ),
        JAPANESE_CATALOGUES[1]: "gwexpy <installed version>",
        JAPANESE_CATALOGUES[2]: (
            "For a reproducible citation, use the `CITATION.cff` file from the exact release "
            "tag you used (for example, `https://github.com/tatsuki-washimi/gwexpy/blob/<exact "
            "tag>/CITATION.cff`), rather than the mutable `main` branch."
        ),
    }

    for relative_path, message_id in required_messages.items():
        with (REPO_ROOT / relative_path).open(encoding="utf-8") as stream:
            catalogue = pofile.read_po(stream, locale="ja")
        message = catalogue.get(message_id)
        assert message is not None, relative_path
        assert message.string, relative_path
        assert "fuzzy" not in message.flags, relative_path


def test_citation_and_zenodo_metadata_agree_with_the_changelog() -> None:
    """Keep public citation facts aligned without editing release metadata here."""
    cff = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    zenodo = json.loads((REPO_ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    cff_title = re.search(r'^title: "(.+?)"$', cff, re.MULTILINE)
    cff_version = re.search(r"^version: (.+)$", cff, re.MULTILINE)
    cff_date = re.search(r"^date-released: (.+)$", cff, re.MULTILINE)
    assert cff_title and cff_title.group(1) == OFFICIAL_TITLE
    assert cff_version and cff_date
    assert "license: MIT" in cff
    assert 'family-names: "Washimi"' in cff
    assert 'given-names: "Tatsuki"' in cff
    assert 'url: "https://github.com/tatsuki-washimi/gwexpy"' in cff
    assert 'repository-code: "https://github.com/tatsuki-washimi/gwexpy"' in cff

    assert zenodo["title"] == OFFICIAL_TITLE
    assert zenodo["version"] == cff_version.group(1)
    assert zenodo["publication_date"] == cff_date.group(1)
    assert zenodo["license"]["id"] == "MIT"
    assert zenodo["creators"] == [
        {
            "name": "Washimi, Tatsuki",
            "affiliation": "National Astronomical Observatory of Japan (NAOJ)",
            "orcid": "0000-0001-5792-4907",
        }
    ]
    assert zenodo["related_identifiers"] == [
        {
            "identifier": "https://github.com/tatsuki-washimi/gwexpy",
            "relation": "isSupplementTo",
            "resource_type": "software",
            "scheme": "url",
        }
    ]
    assert f"## [{cff_version.group(1)}] - {cff_date.group(1)}" in changelog
