from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from babel.messages import pofile

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_VERSION = "0.2.2"
RELEASE_DATE = "2026-09-01"
RELEASE_HISTORY_ENTRY = f"[{RELEASE_VERSION}] - {RELEASE_DATE}"
ACTIVITY_RELEASE_VERSION = "0.2.0"
ACTIVITY_RELEASE_SHA = "5c91cf2d1087616c9815d0cbcc082c5f21bb36e9"
ACTIVITY_CSV_SHA256 = "8411c5acff7beea282d5f24a105ef0573012a923241d83dd4208a0af2fbf63b0"
RELEASE_SCOPE_HISTORY = """\
### Update history

```mermaid
flowchart LR
    baseline[\"v0.1.14 baseline\"] --> integration[\"v0.2 contract integration\"]
    integration --> median_mean[\"#686 median-mean spectral dispatch\"]
    median_mean --> source[\"v0.2.0 release-source metadata\"]
```
"""


def test_released_changelog_publishes_the_v020_activity_snapshot() -> None:
    """Keep the public changelog tied to the immutable v0.2.0 release."""
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    source = (REPO_ROOT / "docs_redesign/about/changelog.md").read_text(
        encoding="utf-8"
    )
    activity_svg = (
        REPO_ROOT / "docs_redesign/_static/images/development-activity-v0.2.0.svg"
    )
    activity_csv = (
        REPO_ROOT
        / "docs_redesign/_static/downloads/development-activity-v0.2.0-weekly.csv"
    )

    assert not re.search(r"^## \[Unreleased\]\s*$", changelog, re.MULTILINE)
    assert "[Unreleased]:" not in changelog
    assert activity_svg.is_file()
    assert activity_csv.is_file()
    assert hashlib.sha256(activity_csv.read_bytes()).hexdigest() == ACTIVITY_CSV_SHA256

    svg_text = activity_svg.read_text(encoding="utf-8")
    assert (
        f"Target ref: v{ACTIVITY_RELEASE_VERSION}; resolved SHA: {ACTIVITY_RELEASE_SHA}"
    ) in svg_text
    assert f"canonical CSV SHA-256: {ACTIVITY_CSV_SHA256}" in svg_text
    assert "/_static/images/development-activity-v0.2.0.svg" in source
    assert "/_static/downloads/development-activity-v0.2.0-weekly.csv" in source


def test_v020_activity_snapshot_has_japanese_public_copy() -> None:
    """Keep the changelog figure readable on the Japanese docs site."""
    catalogue_path = (
        REPO_ROOT / "docs_redesign/locales/ja/LC_MESSAGES/about/changelog.po"
    )
    with catalogue_path.open(encoding="utf-8") as stream:
        catalogue = pofile.read_po(stream, locale="ja")

    expected_translations = {
        "Weekly development activity": "週次開発活動",
        "[Download the weekly CSV data](/_static/downloads/development-activity-v0.2.0-weekly.csv)": "[週次 CSV データをダウンロード](/_static/downloads/development-activity-v0.2.0-weekly.csv)",
    }
    for source, translation in expected_translations.items():
        message = catalogue.get(source)
        assert message is not None
        assert message.string == translation


def test_current_release_facts_match_the_approved_values() -> None:
    """Pin the approved v0.2.2 facts in every canonical public source."""
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    zenodo = json.loads((REPO_ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    catalogue_path = (
        REPO_ROOT / "docs_redesign/locales/ja/LC_MESSAGES/about/changelog.po"
    )
    with catalogue_path.open(encoding="utf-8") as stream:
        catalogue = pofile.read_po(stream, locale="ja")

    assert re.search(
        rf"^## {re.escape(RELEASE_HISTORY_ENTRY)}$", changelog, re.MULTILINE
    )
    assert re.search(
        rf"^version: {re.escape(RELEASE_VERSION)}$", citation, re.MULTILINE
    )
    assert re.search(
        rf"^date-released: {re.escape(RELEASE_DATE)}$", citation, re.MULTILINE
    )
    assert zenodo["version"] == RELEASE_VERSION
    assert zenodo["publication_date"] == RELEASE_DATE

    release_message = catalogue.get(RELEASE_HISTORY_ENTRY)
    assert release_message is not None
    assert release_message.id == RELEASE_HISTORY_ENTRY
    assert release_message.string == RELEASE_HISTORY_ENTRY

    release_note = REPO_ROOT / "release_notes" / f"v{RELEASE_VERSION}.md"
    assert release_note.is_file()
    release_note_text = release_note.read_text(encoding="utf-8")
    assert f"pip install gwexpy=={RELEASE_VERSION}" in release_note_text
    assert "`TimeSeries.crop()` delegates GWpy-supported bounds" in release_note_text


def test_redesign_changelog_includes_the_canonical_release_history() -> None:
    source = (REPO_ROOT / "docs_redesign/about/changelog.md").read_text(
        encoding="utf-8"
    )
    canonical = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert ":::{include} ../../CHANGELOG.md" in source
    assert ':start-after: "# Changelog"' in source
    canonical_releases = re.findall(
        r"^## (\[[^\]]+\] - \d{4}-\d{2}-\d{2})$", canonical, re.MULTILINE
    )
    assert canonical_releases == [
        RELEASE_HISTORY_ENTRY,
        "[0.2.1] - 2026-08-31",
        "[0.2.0] - 2026-08-26",
        "[0.1.14] - 2026-08-15",
        "[0.1.13] - 2026-08-08",
        "[0.1.12] - 2026-07-31",
        "[0.1.11] - 2026-07-25",
        "[0.1.10] - 2026-07-18",
        "[0.1.9] - 2026-07-11",
        "[0.1.8] - 2026-07-04",
        "[0.1.7] - 2026-06-27",
        "[0.1.6] - 2026-06-11",
        "[0.1.5] - 2026-06-10",
        "[0.1.4] - 2026-05-20",
        "[0.1.3] - 2026-05-12",
        "[0.1.2] - 2026-05-08",
        "[0.1.1] - 2026-04-28",
        "[0.1.0] - 2026-03-15",
        "[0.1.0b2] - 2026-02-23",
        "[0.1.0b1] - 2026-02-01",
    ]
    assert not re.search(r"^## \[0\.1", source, re.MULTILINE)


def test_redesign_changelog_japanese_catalogue_translates_every_source_message() -> (
    None
):
    catalogue_path = (
        REPO_ROOT / "docs_redesign/locales/ja/LC_MESSAGES/about/changelog.po"
    )
    with catalogue_path.open(encoding="utf-8") as stream:
        catalogue = pofile.read_po(stream, locale="ja")

    messages = [message for message in catalogue if message.id]
    assert messages
    assert all(message.string for message in messages)
    assert all("fuzzy" not in message.flags for message in messages)
    changelog_message = catalogue.get("Changelog")
    assert changelog_message is not None
    assert changelog_message.string == "更新履歴"
    for release in re.findall(
        r"^## (\[[^\]]+\] - \d{4}-\d{2}-\d{2})$",
        (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
        re.MULTILINE,
    ):
        release_message = catalogue.get(release)
        assert release_message is not None
        assert release_message.string == release


def test_noise_tutorial_declares_the_packaged_gwinc_dependency() -> None:
    notebook = json.loads(
        (REPO_ROOT / "docs_redesign/tutorials/intro_noise.ipynb").read_text(
            encoding="utf-8"
        )
    )
    first_cell = "".join(notebook["cells"][0]["source"])

    assert "`gwinc` (optional, for detector models)" in first_cell
    assert "pygwinc" not in first_cell


def test_redesign_external_links_use_current_documentation_locations() -> None:
    """Keep links in the redesign source on their verified public locations."""
    source_paths = (
        REPO_ROOT / "docs_redesign/conf.py",
        REPO_ROOT / "docs_redesign/how-to/cli.md",
        REPO_ROOT / "docs_redesign/how-to/case-studies/case_dttxml_calibration.ipynb",
        REPO_ROOT / "gwexpy/interop/openems_.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)

    assert "https://gwpy.github.io/docs/stable/cli/" not in source
    assert "https://gwpy.github.io/docs/stable/" not in source
    assert "https://lscsoft.docs.ligo.org/lalsuite/lal/\n" not in source
    assert (
        "https://docs.ligo.org/lscsoft/lalsuite/lal/group___x_l_a_l_time__c.html"
        in source
    )
    assert "https://dtt.ligo.org/" not in source
    assert "https://openems.de/index.php/HDF5_Field_Dumps.html" not in source
