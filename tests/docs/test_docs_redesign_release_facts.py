from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import yaml
from babel.messages import pofile

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_VERSION = "0.2.2"
RELEASE_DATE = "2026-09-01"
RELEASE_HISTORY_ENTRY = f"[{RELEASE_VERSION}] - {RELEASE_DATE}"
RELEASE_DOI_URL = "https://doi.org/10.5281/zenodo.22228340"
ACTIVITY_RELEASE_VERSION = "0.2.2"
ACTIVITY_RELEASE_SHA = "2503743cf654606a5baa83c7b7e7c8b8e1e06596"
ACTIVITY_CSV_SHA256 = "cd72102029af78b05dcb002051365092f128df341125164e4ba7c8a96d4203e3"
RELEASE_CLOSURE_MANIFEST = (
    REPO_ROOT
    / "docs/developers/plans/manifests/audit-manifest-v0.2.2-release-closure.yaml"
)
RELEASE_SCOPE_HISTORY = """\
### Update history

```mermaid
flowchart LR
    baseline[\"v0.1.14 baseline\"] --> integration[\"v0.2 contract integration\"]
    integration --> median_mean[\"#686 median-mean spectral dispatch\"]
    median_mean --> source[\"v0.2.0 release-source metadata\"]
```
"""


def test_released_changelog_publishes_the_v022_activity_snapshot() -> None:
    """Keep the public changelog tied to the immutable v0.2.2 release."""
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    source = (REPO_ROOT / "docs_redesign/about/changelog.md").read_text(
        encoding="utf-8"
    )
    handover = (
        REPO_ROOT
        / "docs/developers/plans/notes/development-activity-visualization-handover.md"
    ).read_text(encoding="utf-8")
    activity_svg = (
        REPO_ROOT / "docs_redesign/_static/images/development-activity-v0.2.2.svg"
    )
    activity_csv = (
        REPO_ROOT
        / "docs_redesign/_static/downloads/development-activity-v0.2.2-weekly.csv"
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
    assert "/_static/images/development-activity-v0.2.2.svg" in source
    assert "/_static/downloads/development-activity-v0.2.2-weekly.csv" in source
    assert f"`v{ACTIVITY_RELEASE_VERSION}`" in handover
    assert f"(`{ACTIVITY_RELEASE_SHA}`)" in handover
    assert f"`{ACTIVITY_CSV_SHA256}`" in handover


def test_v022_activity_snapshot_has_japanese_public_copy() -> None:
    """Keep the changelog figure readable on the Japanese docs site."""
    source_text = (REPO_ROOT / "docs_redesign/about/changelog.md").read_text(
        encoding="utf-8"
    )
    catalogue_path = (
        REPO_ROOT / "docs_redesign/locales/ja/LC_MESSAGES/about/changelog.po"
    )
    with catalogue_path.open(encoding="utf-8") as stream:
        catalogue = pofile.read_po(stream, locale="ja")

    expected_translations = {
        "Weekly development activity": "週次開発活動",
        "[Download the weekly CSV data](/_static/downloads/development-activity-v0.2.2-weekly.csv)": "[週次 CSV データをダウンロード](/_static/downloads/development-activity-v0.2.2-weekly.csv)",
        "The v0.2.2 release is available from [GitHub Releases](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.2) and archived under [Zenodo DOI 10.5281/zenodo.22228340](https://doi.org/10.5281/zenodo.22228340).": "v0.2.2 リリースは [GitHub Releases](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.2) から取得でき、[Zenodo DOI 10.5281/zenodo.22228340](https://doi.org/10.5281/zenodo.22228340) でアーカイブされています。",
    }
    assert RELEASE_DOI_URL in source_text
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
    assert "Publication remains on HOLD" not in release_note_text
    assert "Published on PyPI" in release_note_text
    assert RELEASE_DOI_URL in release_note_text


def test_legacy_web_changelogs_publish_v022_in_each_language() -> None:
    english = (REPO_ROOT / "docs/web/en/user_guide/changelog.md").read_text(
        encoding="utf-8"
    )
    japanese = (REPO_ROOT / "docs/web/ja/user_guide/changelog.md").read_text(
        encoding="utf-8"
    )

    assert "## [0.2.2] - 2026-09-01" in english
    assert "restores GWpy-compatible default sample selection" in english
    assert RELEASE_DOI_URL in english
    assert "## [0.2.2] - 2026-09-01" in japanese
    assert "GWpy と互換な既定のサンプル選択を復元しました" in japanese
    assert RELEASE_DOI_URL in japanese


def test_v022_release_closure_manifest_binds_distribution_channels() -> None:
    """Keep the post-publication audit tied to the accepted release bytes."""
    manifest = yaml.safe_load(RELEASE_CLOSURE_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["schema"] == "gwexpy-v022-release-closure-v1"
    assert manifest["release"] == {
        "version": RELEASE_VERSION,
        "source_sha": ACTIVITY_RELEASE_SHA,
        "tag": "v0.2.2",
        "tag_object_sha": "83b1916537214a4446d76d3773c57c35bb3cd6a5",
        "tagger_utc": "2026-09-01T08:45:28Z",
    }
    assert manifest["publication"]["workflow_run_id"] == 33488802448
    assert manifest["publication"]["wheel_sha256"] == (
        "64e517b906366d30b96560e1149f39fa343b8e24be977bdb18dbc40868c38126"
    )
    assert manifest["publication"]["sdist_sha256"] == (
        "3448af15e417187f201f1d910e92fc11e04224607b9cb77849e6d9e172383636"
    )
    assert manifest["github_release"] == {
        "id": 380369377,
        "url": "https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.2",
        "published_at": "2026-09-01T09:29:42Z",
        "draft": False,
        "prerelease": False,
        "assets": 0,
        "notes_source": "release_notes/v0.2.2.md",
    }
    assert manifest["zenodo"]["record_id"] == 22228340
    assert manifest["zenodo"]["version_doi"] == "10.5281/zenodo.22228340"
    assert manifest["zenodo"]["concept_doi"] == "10.5281/zenodo.19059422"
    assert manifest["conda_forge"]["feedstock"] == "conda-forge/gwexpy-feedstock"
    assert manifest["documentation"]["activity_csv_sha256"] == ACTIVITY_CSV_SHA256


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
