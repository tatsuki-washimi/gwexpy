from __future__ import annotations

import csv
import gettext
import hashlib
import importlib.util
import io
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from babel.messages import mofile, pofile

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE_VERSION = "0.2.2"
RELEASE_DATE = "2026-09-01"
RELEASE_HISTORY_ENTRY = f"[{RELEASE_VERSION}] - {RELEASE_DATE}"
PUBLISHED_V023_VERSION = "0.2.3"
PUBLISHED_V023_DATE = "2026-09-05"
PUBLISHED_V023_HISTORY_ENTRY = f"[{PUBLISHED_V023_VERSION}] - {PUBLISHED_V023_DATE}"
PUBLISHED_V024_VERSION = "0.2.4"
PUBLISHED_V024_DATE = "2026-09-26"
PUBLISHED_V024_HISTORY_ENTRY = f"[{PUBLISHED_V024_VERSION}] - {PUBLISHED_V024_DATE}"
PUBLISHED_V025_VERSION = "0.2.5"
PUBLISHED_V025_DATE = "2026-10-04"
PUBLISHED_V025_HISTORY_ENTRY = f"[{PUBLISHED_V025_VERSION}] - {PUBLISHED_V025_DATE}"
PUBLISHED_V026_VERSION = "0.2.6"
PUBLISHED_V026_DATE = "2026-10-27"
PUBLISHED_V026_METADATA_DATE = "2026-10-24"
PUBLISHED_V026_HISTORY_ENTRY = (
    f"[{PUBLISHED_V026_VERSION}] - {PUBLISHED_V026_METADATA_DATE}"
)
CANDIDATE_V027_VERSION = "0.2.7"
CANDIDATE_V027_METADATA_DATE = "2026-11-03"
CANDIDATE_V027_HISTORY_ENTRY = (
    f"[{CANDIDATE_V027_VERSION}] - {CANDIDATE_V027_METADATA_DATE}"
)
PUBLISHED_V027_VERSION = "0.2.7"
PUBLISHED_V027_DATE = "2026-10-31"
RELEASE_DOI_URL = "https://doi.org/10.5281/zenodo.22228340"
ACTIVITY_RELEASE_VERSION = "0.2.2"
ACTIVITY_RELEASE_SHA = "2503743cf654606a5baa83c7b7e7c8b8e1e06596"
ACTIVITY_CSV_SHA256 = "cd72102029af78b05dcb002051365092f128df341125164e4ba7c8a96d4203e3"
CANDIDATE_ACTIVITY_SHA = "3ade51de26c9adf21c4a7ad076837967e61c8038"
CANDIDATE_ACTIVITY_CSV_SHA256 = (
    "0a0453910f6944e53312b9e36f811e40e608698f2b75eee8171d1a26b64abdd8"
)
PUBLISHED_V024_SHA = "522e52a082925da4dd37966d82a7616bdd2a5248"
PUBLISHED_V024_ACTIVITY_CSV_SHA256 = (
    "1a449a3cb1be659698dd6ab02850e3b483249540475c71afe42994438d2f34f9"
)
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


def load_v027_review_validator():
    path = REPO_ROOT / "scripts/ci/validate_release_review_evidence.py"
    spec = importlib.util.spec_from_file_location(
        "v027_review_evidence_for_docs_test", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_v027_release_contract():
    path = REPO_ROOT / "scripts/ci/release_contract.py"
    spec = importlib.util.spec_from_file_location("v027_contract_for_docs_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_v027_release_validator():
    path = REPO_ROOT / "scripts/validate_release.py"
    spec = importlib.util.spec_from_file_location("v027_release_for_docs_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.historical_audit
def test_changelog_preserves_the_historical_v022_activity_snapshot() -> None:
    """Keep the earlier published snapshot and its provenance available."""
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


def test_candidate_activity_uses_the_same_data_in_both_languages() -> None:
    static = REPO_ROOT / "docs_redesign/_static"
    data = static / "downloads/development-activity-v0.2.3-candidate-weekly.csv"
    assert (
        hashlib.sha256(data.read_bytes()).hexdigest() == CANDIDATE_ACTIVITY_CSV_SHA256
    )
    for suffix in ("", "-ja"):
        svg = (
            static / f"images/development-activity-v0.2.3-candidate{suffix}.svg"
        ).read_text(encoding="utf-8")
        assert f"resolved SHA: {CANDIDATE_ACTIVITY_SHA}" in svg
        assert f"canonical CSV SHA-256: {CANDIDATE_ACTIVITY_CSV_SHA256}" in svg
        if suffix:
            assert "週ごとのコミット数" in svg
            assert "月曜日始まりの週（UTC）" in svg
            assert "Commits per week" not in svg
    source = (REPO_ROOT / "docs_redesign/about/changelog.md").read_text(
        encoding="utf-8"
    )
    assert "The candidate has not been tagged or published." not in source
    assert CANDIDATE_ACTIVITY_SHA in source


@pytest.mark.historical_audit
def test_published_v023_activity_snapshot_and_hash_are_immutable() -> None:
    static = REPO_ROOT / "docs_redesign/_static"
    source = (REPO_ROOT / "docs_redesign/about/changelog.md").read_text(
        encoding="utf-8"
    )
    # Preserve the published asset as an immutable snapshot after the current
    # figure advances to v0.2.4.
    published_sha = "75d3d1a89ebc8942af1f3228152fea99d2d3420e"
    published_csv = static / "downloads/development-activity-v0.2.3-weekly.csv"
    digest = hashlib.sha256(published_csv.read_bytes()).hexdigest()
    assert digest == "612292f6e17e1aadf13642646873ea68f75b34f182966416c32d5fb8bc910885"
    for suffix in ("", "-ja"):
        svg = (static / f"images/development-activity-v0.2.3{suffix}.svg").read_text(
            encoding="utf-8"
        )
        assert f"Target ref: v0.2.3; resolved SHA: {published_sha}" in svg
        assert f"canonical CSV SHA-256: {digest}" in svg
        if suffix:
            assert "週ごとのコミット数" in svg
            assert "Commits per week" not in svg
    assert published_sha in source
    assert "https://zenodo.org/records/22978439" in source
    historical_changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "https://pypi.org/project/gwexpy/0.2.3/" in historical_changelog


@pytest.mark.historical_audit
def test_current_activity_matches_the_published_v024_tag() -> None:
    static = REPO_ROOT / "docs_redesign/_static"
    release_status = json.loads(
        (REPO_ROOT / "docs_redesign/release_status.json").read_text(encoding="utf-8")
    )
    version = PUBLISHED_V024_VERSION
    assert release_status["latest_release"] == PUBLISHED_V027_VERSION
    source = (REPO_ROOT / "docs_redesign/about/changelog.md").read_text(
        encoding="utf-8"
    )
    csv_path = static / f"downloads/development-activity-v{version}-weekly.csv"

    assert f"## v{version} weekly development activity" in source
    assert (
        f":::{{figure}} /_static/images/development-activity-v{version}.svg" in source
    )
    assert f"/_static/downloads/development-activity-v{version}-weekly.csv" in source
    assert PUBLISHED_V024_SHA in source
    assert hashlib.sha256(csv_path.read_bytes()).hexdigest() == (
        PUBLISHED_V024_ACTIVITY_CSV_SHA256
    )
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["target_ref"] for row in rows} == {f"v{version}"}
    assert {row["target_sha"] for row in rows} == {PUBLISHED_V024_SHA}
    assert sum(int(row["commit_count"]) for row in rows) == 1962
    for suffix in ("", "-ja"):
        svg = (
            static / f"images/development-activity-v{version}{suffix}.svg"
        ).read_text(encoding="utf-8")
        assert f"Target ref: v{version}; resolved SHA: {PUBLISHED_V024_SHA}" in svg
        assert f"canonical CSV SHA-256: {PUBLISHED_V024_ACTIVITY_CSV_SHA256}" in svg
        assert f"v{version}" in svg
        assert all(not line.endswith((" ", "\t")) for line in svg.splitlines())
        if suffix:
            assert "週ごとのコミット数" in svg
            assert "Commits per week" not in svg


@pytest.mark.historical_audit
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
        "v0.2.4 weekly development activity": "v0.2.4 の週次開発活動",
        "Weekly GWexpy development activity through v0.2.4, showing non-merge commits and edited source lines by category.": "v0.2.4 までの GWexpy の週次開発活動。カテゴリ別に、マージを除くコミット数と編集されたソース行数を示します。",
        "[Download the weekly CSV data](/_static/downloads/development-activity-v0.2.4-weekly.csv)": "[週次 CSV データをダウンロード](/_static/downloads/development-activity-v0.2.4-weekly.csv)",
        "Published on 2026-09-26 UTC: [PyPI 0.2.4](https://pypi.org/project/gwexpy/0.2.4/), [conda-forge](https://anaconda.org/conda-forge/gwexpy), [GitHub Release](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.4), and [Zenodo DOI 10.5281/zenodo.22978439](https://zenodo.org/records/22978439).": "2026-09-26（UTC）に公開しました：[PyPI 0.2.4](https://pypi.org/project/gwexpy/0.2.4/)、[conda-forge](https://anaconda.org/conda-forge/gwexpy)、[GitHub Release](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.4)、[Zenodo DOI 10.5281/zenodo.22978439](https://zenodo.org/records/22978439)。",
        "Published on 2026-09-05 UTC: [PyPI 0.2.3](https://pypi.org/project/gwexpy/0.2.3/), [conda-forge](https://anaconda.org/conda-forge/gwexpy), [GitHub Release](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.3), and [Zenodo DOI 10.5281/zenodo.22344992](https://doi.org/10.5281/zenodo.22344992).": "2026-09-05（UTC）に公開しました：[PyPI 0.2.3](https://pypi.org/project/gwexpy/0.2.3/)、[conda-forge](https://anaconda.org/conda-forge/gwexpy)、[GitHub Release](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.3)、[Zenodo DOI 10.5281/zenodo.22344992](https://doi.org/10.5281/zenodo.22344992)。",
        "[Download the weekly CSV data](/_static/downloads/development-activity-v0.2.2-weekly.csv)": "[週次 CSV データをダウンロード](/_static/downloads/development-activity-v0.2.2-weekly.csv)",
        "The v0.2.2 release is available from [GitHub Releases](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.2) and archived under [Zenodo DOI 10.5281/zenodo.22228340](https://doi.org/10.5281/zenodo.22228340).": "v0.2.2 リリースは [GitHub Releases](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.2) から取得でき、[Zenodo DOI 10.5281/zenodo.22228340](https://doi.org/10.5281/zenodo.22228340) でアーカイブされています。",
    }
    assert RELEASE_DOI_URL in source_text
    assert "https://zenodo.org/records/22978439" in source_text
    for source, translation in expected_translations.items():
        message = catalogue.get(source)
        assert message is not None
        assert message.string == translation


def test_published_history_and_v027_source_metadata_remain_distinct():
    """Keep actual publication facts distinct from frozen source metadata."""
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    citation = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    zenodo = json.loads((REPO_ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    release_status = json.loads(
        (REPO_ROOT / "docs_redesign/release_status.json").read_text(encoding="utf-8")
    )
    catalogue_path = (
        REPO_ROOT / "docs_redesign/locales/ja/LC_MESSAGES/about/changelog.po"
    )
    with catalogue_path.open(encoding="utf-8") as stream:
        catalogue = pofile.read_po(stream, locale="ja")

    for history_entry in (
        RELEASE_HISTORY_ENTRY,
        PUBLISHED_V023_HISTORY_ENTRY,
        PUBLISHED_V024_HISTORY_ENTRY,
        PUBLISHED_V025_HISTORY_ENTRY,
        PUBLISHED_V026_HISTORY_ENTRY,
    ):
        assert re.search(rf"^## {re.escape(history_entry)}$", changelog, re.MULTILINE)
    assert re.search(
        rf"^version: {re.escape(CANDIDATE_V027_VERSION)}$",
        citation,
        re.MULTILINE,
    )
    release_plan = (
        REPO_ROOT / "docs/developers/plans/20260927_v0.2.5_release_plan.md"
    ).read_text(encoding="utf-8")
    release_plan_text = " ".join(release_plan.split())
    assert f"Release date: **{PUBLISHED_V025_DATE} UTC**" in release_plan_text
    assert "Status: corrected-note source `S` frozen" in release_plan_text
    assert (
        "fresh same-`S` reviews, owner approval, and exact-`R` qualification pending"
        in release_plan_text
    )
    assert "Release decision: **HOLD**" in release_plan_text
    assert (
        "The original 09:00 JST pre-tag planning target has elapsed."
        in release_plan_text
    )
    assert (
        "not a release authorization or a reason to backdate a tag" in release_plan_text
    )
    assert release_plan_text.count("| SELECTED |") == 1
    assert release_plan_text.count("| NOT SELECTED |") == 3
    assert (
        "#584 nine-path candidate-A overlay has not been applied" in release_plan_text
    )
    assert (
        "serial #588 optimization is already present in the current base"
        in release_plan_text
    )
    assert "#588 patch is a separate selection and is deferred" in release_plan_text
    assert (
        "The valid historical primary performance result remains 31/32 HOLD"
        in release_plan_text
    )
    assert "Cold timing cannot replace valid live-process PSS" in release_plan_text
    checkboxes = re.findall(r"^- \[([ xX])\]", release_plan, re.MULTILINE)
    assert checkboxes

    validator_path = REPO_ROOT / "scripts/validate_release.py"
    spec = importlib.util.spec_from_file_location(
        "release_validator_for_docs", validator_path
    )
    assert spec is not None and spec.loader is not None
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)
    readiness = (
        REPO_ROOT
        / "docs/developers/plans/manifests/audit-manifest-v0.2.5-release-readiness.yaml"
    )
    readiness_bytes = readiness.read_bytes()
    if readiness_bytes == validator.V025_EMPTY_REVIEW_EVIDENCE_PLACEHOLDER:
        # Source S must remain an exact empty-evidence placeholder with no
        # completed release gates.
        assert set(checkboxes) == {" "}
    else:
        # An evidence-filled descendant R is accepted only through the official
        # validator, which binds its evidence and permits only S-to-R checkbox
        # transitions on this plan.
        source_sha = subprocess.check_output(
            ["git", "rev-parse", "refs/tags/v0.2.5^{commit}"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
        reviewed_commit = validator.validate_review_evidence(
            REPO_ROOT,
            readiness,
            source_sha=source_sha,
            expected_tag="v0.2.5",
        )
        source_plan = subprocess.check_output(
            [
                "git",
                "show",
                f"{reviewed_commit}:docs/developers/plans/20260927_v0.2.5_release_plan.md",
            ],
            cwd=REPO_ROOT,
            text=True,
        )
        source_checkboxes = re.findall(r"^- \[([ xX])\]", source_plan, re.MULTILINE)
        assert source_checkboxes and set(source_checkboxes) == {" "}

    assert "Three independent reviews—" in release_plan_text
    assert "fresh human approval bound to final `S`" in release_plan_text
    assert "the exact disposition-document SHA-256" in release_plan_text
    assert "both scoped exceptions" in release_plan_text
    assert "twelve historical disposition IDs" in release_plan_text
    assert (
        "Only after all reviews and human approval pass may a distinct `R` be created"
        in release_plan_text
    )
    assert (
        "The `S`-to-`R` diff is limited to filling that readiness manifest"
        in release_plan_text
    )
    assert "exact final-`R` installed artifacts" in release_plan_text

    roadmap = (REPO_ROOT / "ROADMAP.md").read_text(encoding="utf-8")
    roadmap_text = " ".join(roadmap.split())
    assert "planned release date is 2026-10-04 UTC" in roadmap_text
    assert "Task 6 selected case 1" in roadmap_text
    assert "both deferred independently" in roadmap_text
    assert "Retain the R8 serial #588 implementation" in roadmap_text
    assert "The PyPI environment has no manual approval requirement" in roadmap_text
    assert "Trusted Publisher tuple remains unconfirmed" in roadmap_text

    assert "September cycle records are historical only" in release_plan_text
    assert (
        "they do not identify or authorize the current `S` or `R`" in release_plan_text
    )
    assert re.search(
        rf"^date-released: {re.escape(CANDIDATE_V027_METADATA_DATE)}$",
        citation,
        re.MULTILINE,
    )
    assert zenodo["version"] == CANDIDATE_V027_VERSION
    assert zenodo["publication_date"] == CANDIDATE_V027_METADATA_DATE
    candidate_plan = (
        REPO_ROOT / "docs/developers/plans/20261029_v0.2.7_release_plan.md"
    ).read_text(encoding="utf-8")
    candidate_plan_text = " ".join(candidate_plan.split())
    assert (
        "Target release date: **2026-11-03 (conditional target only; not a completion date or publication authorization).**"
        in candidate_plan_text
    )
    assert "Status: **Preparation only.**" in candidate_plan_text
    assert "Tracking Issue [#844]" in candidate_plan_text
    candidate_checkboxes = re.findall(r"^- \[([ xX])\]", candidate_plan, re.MULTILINE)
    assert candidate_checkboxes
    candidate_readiness = yaml.safe_load(
        (
            REPO_ROOT
            / "docs/developers/plans/manifests/audit-manifest-v0.2.7-release-readiness.yaml"
        ).read_text(encoding="utf-8")
    )
    candidate_evidence = json.loads(candidate_readiness["review_evidence_json"])
    assert candidate_evidence["schema"] == "gwexpy-v027-review-evidence-v2"
    candidate_manifest_path = (
        REPO_ROOT
        / "docs/developers/plans/manifests/audit-manifest-v0.2.7-release-readiness.yaml"
    )
    contract = load_v027_release_contract().release_contract("v0.2.7")
    expected_placeholder = (
        load_v027_release_validator()._v027_empty_review_evidence_placeholder(
            contract["review_base_sha"]
        )
    )
    if candidate_evidence["source_sha"] == "":
        # Source S retains the empty evidence placeholder and no completed steps.
        assert candidate_manifest_path.read_bytes() == expected_placeholder
        assert set(candidate_checkboxes) == {" "}
    else:
        # Preserve the Candidate R readiness manifest as a pre-publication
        # phase snapshot; later qualification and publication evidence is tracked
        # separately and must not rewrite the tagged source.
        load_v027_review_validator().validate_review_evidence(
            candidate_manifest_path,
            None,
            None,
            REPO_ROOT,
            expected_tag="v0.2.7",
            phase="candidate",
        )
        assert candidate_checkboxes == ["x"] * 4 + [" "] * 4
    releasing_text = " ".join(
        (REPO_ROOT / "RELEASING.md").read_text(encoding="utf-8").split()
    )
    assert "A planned date does not expire a qualified candidate." in releasing_text
    assert (
        "Record GitHub Release and PyPI times from their service readbacks"
        in releasing_text
    )
    release_note = (REPO_ROOT / "release_notes/v0.2.5.md").read_text(encoding="utf-8")
    release_note_text = " ".join(release_note.split())
    changelog_text = " ".join(changelog.split())
    assert release_note_text.startswith(
        "This patch improves the reliability of scientific-data I/O"
    )
    assert "release qualification pending" not in release_note_text.lower()
    assert "three fresh reviews" not in changelog_text.lower()
    assert (
        "candidate records are historical evidence only" not in changelog_text.lower()
    )
    assert "candidate records are historical" not in release_note_text.lower()

    assert release_status["latest_release"] == PUBLISHED_V027_VERSION
    assert release_status["intro_examples_release"] == PUBLISHED_V024_VERSION
    for language in ("en", "ja"):
        public_changelog = (
            REPO_ROOT / f"docs/web/{language}/user_guide/changelog.md"
        ).read_text(encoding="utf-8")
        assert f"## {PUBLISHED_V024_HISTORY_ENTRY}" in public_changelog
        assert f"## {PUBLISHED_V023_HISTORY_ENTRY}" in public_changelog
        current_status = public_changelog.split("## [0.2.4]", 1)[0]
        assert "conda-forge" in current_status
        assert "0.2.3" not in current_status

    redesigned_changelog = (REPO_ROOT / "docs_redesign/about/changelog.md").read_text(
        encoding="utf-8"
    )
    assert f"Published on {PUBLISHED_V026_DATE} UTC" in redesigned_changelog
    assert "conda-forge feedstock [PR #16]" in redesigned_changelog
    assert "merged on 2026-10-28 UTC" in redesigned_changelog
    assert "0.2.6 is available on conda-forge" in redesigned_changelog
    assert "open for review" not in redesigned_changelog
    assert "development-activity-v0.2.7.svg" in redesigned_changelog
    assert "development-activity-v0.2.6.svg" in redesigned_changelog
    japanese_po = (
        REPO_ROOT / "docs_redesign/locales/ja/LC_MESSAGES/about/changelog.po"
    ).read_text(encoding="utf-8")
    assert (
        'msgid "[0.2.7] - 2026-11-03"\nmsgstr "[0.2.7] - 2026-11-03"'
    ) in japanese_po
    assert (
        'msgid "[0.2.6] - 2026-10-24"\nmsgstr "[0.2.6] - 2026-10-24"'
    ) in japanese_po
    assert "development-activity-v0.2.4.svg" in redesigned_changelog
    assert ':start-after: "## [0.2.7] - 2026-11-03"' in redesigned_changelog
    assert ':end-before: "## [0.2.6] - 2026-10-24"' in redesigned_changelog
    assert ':start-after: "## [0.2.6] - 2026-10-24"' in redesigned_changelog
    assert ':end-before: "## [0.2.5] - 2026-10-04"' in redesigned_changelog
    assert ':start-after: "[#778]' in redesigned_changelog

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

    # The redesign follows the actual publication readback while the frozen
    # source metadata retains its original planned date of 2026-11-03.
    assert release_status["latest_release"] == PUBLISHED_V027_VERSION
    assert release_status["intro_examples_release"] == PUBLISHED_V024_VERSION
    assert CANDIDATE_V027_HISTORY_ENTRY in changelog
    assert PUBLISHED_V026_HISTORY_ENTRY in changelog
    assert f"Published on {PUBLISHED_V027_DATE} UTC" in redesigned_changelog
    assert "planned date of 2026-11-03" in redesigned_changelog
    assert f"Published on {PUBLISHED_V026_DATE} UTC" in redesigned_changelog
    assert "10.5281/zenodo.23680359" in redesigned_changelog
    assert "conda-forge" in redesigned_changelog
    assert "0.2.6 is available on conda-forge" in redesigned_changelog
    activity_csv = REPO_ROOT / (
        "docs_redesign/_static/downloads/development-activity-v0.2.6-weekly.csv"
    )
    activity_sha = "a1e0f56593545c6aa0f8e5916b681edabc6f1b85cb640fbc08811be8932fbe26"
    assert hashlib.sha256(activity_csv.read_bytes()).hexdigest() == activity_sha
    for suffix in ("", "-ja"):
        svg = (
            REPO_ROOT
            / f"docs_redesign/_static/images/development-activity-v0.2.6{suffix}.svg"
        ).read_text(encoding="utf-8")
        assert (
            "Target ref: v0.2.6; resolved SHA: a1ff8e2f18ef4e599727d6df6fdff61fce7019bd"
            in svg
        )
        assert f"canonical CSV SHA-256: {activity_sha}" in svg
        if suffix:
            assert "週ごとのコミット数" in svg
            assert "Commits per week" not in svg
    activity_v027_csv = REPO_ROOT / (
        "docs_redesign/_static/downloads/development-activity-v0.2.7-weekly.csv"
    )
    activity_v027_sha = (
        "186fd03f1b46fe9a20193c731d0cfe929561487ad1c6d2db8bc53c35d300c5b6"
    )
    assert hashlib.sha256(activity_v027_csv.read_bytes()).hexdigest() == (
        activity_v027_sha
    )
    for suffix in ("", "-ja"):
        svg = (
            REPO_ROOT
            / f"docs_redesign/_static/images/development-activity-v0.2.7{suffix}.svg"
        ).read_text(encoding="utf-8")
        assert (
            "Target ref: v0.2.7; resolved SHA: "
            "bcf6e0c98b01249b742124676330d8f98a3adab2" in svg
        )
        assert f"canonical CSV SHA-256: {activity_v027_sha}" in svg


def test_v026_changelog_reconciliation_manifest_is_valid_yaml() -> None:
    manifest_path = (
        REPO_ROOT
        / "docs/developers/plans/manifests/audit-manifest-v0.2.6-changelog-reconcile.yaml"
    )

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema"] == "gwexpy-agent-audit-v1"
    assert manifest["live_readback"]["status"] == (
        "Docs correction review and Pages deployment remain pending."
    )


def test_japanese_inline_examples_preserve_matrix_and_pad_tokens() -> None:
    modal_path = (
        REPO_ROOT
        / "docs_redesign/locales/ja/LC_MESSAGES/how-to/control/advanced_modal_analysis.po"
    )
    io_path = (
        REPO_ROOT / "docs_redesign/locales/ja/LC_MESSAGES/reference/io_capabilities.po"
    )
    with modal_path.open(encoding="utf-8") as stream:
        modal_catalogue = pofile.read_po(stream, locale="ja")
    with io_path.open(encoding="utf-8") as stream:
        io_catalogue = pofile.read_po(stream, locale="ja")

    matrix_messages = [
        message
        for message in modal_catalogue
        if isinstance(message.id, str) and "H[i,j](f)" in message.id
    ]
    pad_messages = [
        message
        for message in io_catalogue
        if isinstance(message.id, str) and "pad=<value>" in message.id
    ]

    assert len(matrix_messages) == 1
    assert "H[i,j](f)" in matrix_messages[0].string
    assert len(pad_messages) == 1
    assert pad_messages[0].string.count("pad=<値>") == 2
    assert "pad= <値>" not in pad_messages[0].string


def test_v027_release_status_and_v026_conda_forge_history_are_current() -> None:
    english = (REPO_ROOT / "docs_redesign/about/changelog.md").read_text(
        encoding="utf-8"
    )
    catalogue_path = (
        REPO_ROOT / "docs_redesign/locales/ja/LC_MESSAGES/about/changelog.po"
    )
    with catalogue_path.open(encoding="utf-8") as stream:
        catalogue = pofile.read_po(stream, locale="ja")
    message = catalogue.get(
        "Published on 2026-10-27 UTC: [PyPI 0.2.6](https://pypi.org/project/gwexpy/0.2.6/), "
        "[GitHub Release](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.6), "
        "and [Zenodo DOI 10.5281/zenodo.23680359](https://zenodo.org/records/23680359). "
        "The conda-forge feedstock [PR #16](https://github.com/conda-forge/gwexpy-feedstock/pull/16) "
        "merged on 2026-10-28 UTC, and version 0.2.6 is available on conda-forge."
    )

    assert "merged on 2026-10-28 UTC" in english
    assert "version 0.2.6 is available on conda-forge" in english
    assert message is not None
    assert message.string == (
        "2026-10-27（UTC）に公開しました：[PyPI 0.2.6](https://pypi.org/project/gwexpy/0.2.6/)、"
        "[GitHub Release](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.6)、"
        "[Zenodo DOI 10.5281/zenodo.23680359](https://zenodo.org/records/23680359)。"
        "conda-forge feedstock の [PR #16](https://github.com/conda-forge/gwexpy-feedstock/pull/16) は"
        "2026-10-28（UTC）にマージされ、conda-forge では0.2.6のパッケージを利用できます。"
    )

    published_message = catalogue.get(
        "Published on 2026-10-31 UTC: [PyPI 0.2.7](https://pypi.org/project/gwexpy/0.2.7/) "
        "and [GitHub Release](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.7). "
        "The [Zenodo DOI 10.5281/zenodo.23828442](https://zenodo.org/records/23828442) record "
        "was created on 2026-10-31 UTC and lists metadata `publication_date` 2026-11-03; "
        "the Release Owner's date disposition remains pending. As of 2026-10-31 UTC, "
        "no v0.2.7 conda-forge feedstock PR or package was observed, and the latest "
        "conda-forge package is v0.2.6."
    )
    assert "Published on 2026-10-31 UTC" in english
    assert published_message is not None
    assert "2026-11-03" in published_message.string
    assert "この日付に関するRelease Ownerの判断は未完了" in published_message.string
    planned_date_message = catalogue.get(
        "The detailed v0.2.7 release-notes heading below retains the release source's planned date of "
        "2026-11-03; service readbacks show the actual publication date as "
        "2026-10-31 UTC."
    )
    assert planned_date_message is not None
    assert "実際の公開日は2026-10-31（UTC）です" in planned_date_message.string

    roadmap = (REPO_ROOT / "docs_redesign/explanation/roadmap.md").read_text(
        encoding="utf-8"
    )
    roadmap_catalogue_path = (
        REPO_ROOT / "docs_redesign/locales/ja/LC_MESSAGES/explanation/roadmap.po"
    )
    with roadmap_catalogue_path.open(encoding="utf-8") as stream:
        roadmap_catalogue = pofile.read_po(stream, locale="ja")
    roadmap_msgid = (
        "v0.2.7 was published on 2026-10-31 UTC to "
        "[PyPI](https://pypi.org/project/gwexpy/0.2.7/) and "
        "[GitHub Releases](https://github.com/tatsuki-washimi/gwexpy/releases/tag/v0.2.7). "
        "The [Zenodo record](https://zenodo.org/records/23828442) was created on "
        "2026-10-31 UTC and lists metadata `publication_date` 2026-11-03; the "
        "Release Owner's date disposition remains pending. As of 2026-10-31 UTC, "
        "no v0.2.7 feedstock PR or package was observed on conda-forge; version "
        "0.2.6 remains available. No next-minor release theme is committed; "
        "future work remains directional."
    )
    roadmap_message = roadmap_catalogue.get(roadmap_msgid)

    assert "2026-10-31 UTC" in roadmap
    assert "no v0.2.7 feedstock PR or package was observed on conda-forge" in roadmap
    assert "version 0.2.6 remains available" in roadmap
    assert "open for review" not in roadmap
    assert "tops at 0.2.4" not in roadmap
    assert roadmap_message is not None
    assert "2026-11-03" in roadmap_message.string
    assert "日付に関するリリース責任者の判断は未完了" in roadmap_message.string
    assert "v0.2.6です" in roadmap_message.string

    installation = (REPO_ROOT / "docs_redesign/tutorials/installation.md").read_text(
        encoding="utf-8"
    )
    installation_catalogue_path = (
        REPO_ROOT / "docs_redesign/locales/ja/LC_MESSAGES/tutorials/installation.po"
    )
    with installation_catalogue_path.open(encoding="utf-8") as stream:
        installation_catalogue = pofile.read_po(stream, locale="ja")
    installation_msgid = (
        "GWexpy {{ latest_release }} is available on "
        "[PyPI](https://pypi.org/project/gwexpy/) and the "
        "[GitHub releases page](https://github.com/tatsuki-washimi/gwexpy/releases). "
        "The latest package on [conda-forge](https://anaconda.org/conda-forge/gwexpy) "
        "is v0.2.6; no v0.2.7 package was observed "
        "as of 2026-10-31 UTC. The [v0.2.7 Zenodo record]"
        "(https://zenodo.org/records/23828442) was created on 2026-10-31 UTC and "
        "lists metadata `publication_date` 2026-11-03. The Release Owner's date "
        "disposition is pending. Check each install source for its available versions."
    )
    installation_message = installation_catalogue.get(installation_msgid)
    installation_text = " ".join(installation.split())

    assert "The latest package on [conda-forge]" in installation_text
    assert "is v0.2.6;" in installation_text
    assert "no v0.2.7 package was observed as of 2026-10-31 UTC" in installation_text
    assert "publication_date` 2026-11-03" in installation_text
    assert "date disposition is pending" in installation_text
    assert installation_message is not None
    assert (
        "[conda-forge](https://anaconda.org/conda-forge/gwexpy) "
        "で確認できる最新パッケージはv0.2.6です"
    ) in installation_message.string
    assert "2026-11-03" in installation_message.string
    assert "リリース責任者の判断は未完了" in installation_message.string


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


@pytest.mark.historical_audit
def test_v022_release_closure_manifest_binds_distribution_channels() -> None:
    """Keep the post-publication audit tied to the accepted release bytes."""
    manifest = yaml.safe_load(RELEASE_CLOSURE_MANIFEST.read_text(encoding="utf-8"))

    assert manifest["schema"] == "gwexpy-v022-release-closure-v1"
    assert manifest["status"] == "closure-complete"
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
    assert manifest["conda_forge"] == {
        "feedstock": "conda-forge/gwexpy-feedstock",
        "state": "complete",
        "pull_request": {
            "number": 12,
            "url": "https://github.com/conda-forge/gwexpy-feedstock/pull/12",
            "branch": "0.2.2_hff071a",
            "merged_at": "2026-09-01T12:31:04Z",
            "merge_commit": "5eea7e9a7d1b15f6a8e625df33595b957ac9e8db",
        },
        "recipe": {
            "version": RELEASE_VERSION,
            "source_url": "https://pypi.org/packages/source/g/gwexpy/gwexpy-0.2.2.tar.gz",
            "source_sha256": (
                "3448af15e417187f201f1d910e92fc11e04224607b9cb77849e6d9e172383636"
            ),
            "build_number": 0,
        },
        "ci": {
            "package_build": "pass",
            "linter": "pass",
        },
        "channel_package": {
            "basename": "noarch/gwexpy-0.2.2-pyhc364b38_0.conda",
            "version": RELEASE_VERSION,
            "build": "pyhc364b38_0",
            "build_number": 0,
            "uploaded_at": "2026-09-01T12:33:10.289000Z",
        },
        "fresh_install_smoke": {
            "result": "pass",
            "python": "3.12.14",
            "metadata_version": RELEASE_VERSION,
            "module_version": RELEASE_VERSION,
            "user_site": "disabled",
            "timeseries_shape": [3],
        },
    }
    assert manifest["documentation"]["deployment"] == {
        "pull_request": 697,
        "merge_commit": "c325f6fa0b4ee8de0ca5f6203880504a2af0f9d9",
        "pages_workflow_run_id": 33498679298,
        "gh_pages_commit": "202ec1c9cbf5d570b1022d8b593deddd19efece8",
        "live_readback": "pass",
    }
    assert manifest["documentation"]["activity_csv_sha256"] == ACTIVITY_CSV_SHA256
    assert "pending" not in manifest


def test_redesign_changelog_includes_the_published_release_history() -> None:
    source = (REPO_ROOT / "docs_redesign/about/changelog.md").read_text(
        encoding="utf-8"
    )
    canonical = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    published_heading = f"## {PUBLISHED_V023_HISTORY_ENTRY}"
    latest_history_heading = f"## {CANDIDATE_V027_HISTORY_ENTRY}"
    previous_heading = f"## {PUBLISHED_V026_HISTORY_ENTRY}"
    assert ":::{include} ../../CHANGELOG.md" in source
    assert latest_history_heading in source
    assert f':start-after: "{latest_history_heading}"' in source
    assert f':end-before: "{previous_heading}"' in source
    assert ':start-after: "[#778]' in source
    canonical_lines = canonical.splitlines()
    latest_line = canonical_lines.index(latest_history_heading)
    previous_line = canonical_lines.index(previous_heading)
    assert latest_line < previous_line
    assert "Missing local GWF files" in "\n".join(
        canonical_lines[latest_line:previous_line]
    )
    assert published_heading not in source
    current_section = canonical.split(latest_history_heading, 1)[1].split(
        published_heading, 1
    )[0]
    assert "Release status: candidate." not in current_section
    assert "Publication is on HOLD" not in current_section
    candidate_section = canonical.split(f"## {CANDIDATE_V027_HISTORY_ENTRY}", 1)[
        1
    ].split(f"## {PUBLISHED_V026_HISTORY_ENTRY}", 1)[0]
    assert "Release status: candidate." not in candidate_section
    assert "Publication is on HOLD" not in candidate_section

    rendered_history = canonical.split("# Changelog", 1)[1]
    rendered_headings = re.findall(
        r"^## (\[[^\]]+\] - (?:TBD|\d{4}-\d{2}-\d{2}))$",
        rendered_history,
        re.MULTILINE,
    )
    assert rendered_headings[:4] == [
        CANDIDATE_V027_HISTORY_ENTRY,
        PUBLISHED_V026_HISTORY_ENTRY,
        PUBLISHED_V025_HISTORY_ENTRY,
        PUBLISHED_V024_HISTORY_ENTRY,
    ]
    canonical_releases = re.findall(
        r"^## (\[[^\]]+\] - (?:TBD|\d{4}-\d{2}-\d{2}))$",
        canonical,
        re.MULTILINE,
    )
    assert canonical_releases[0] == CANDIDATE_V027_HISTORY_ENTRY
    assert canonical_releases[1:] == [
        PUBLISHED_V026_HISTORY_ENTRY,
        PUBLISHED_V025_HISTORY_ENTRY,
        PUBLISHED_V024_HISTORY_ENTRY,
        PUBLISHED_V023_HISTORY_ENTRY,
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
    release_entries = re.findall(
        r"^## (\[[^\]]+\] - (?:TBD|\d{4}-\d{2}-\d{2}))$",
        (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert release_entries[0] == CANDIDATE_V027_HISTORY_ENTRY
    for release in release_entries[1:]:
        release_message = catalogue.get(release)
        assert release_message is not None
        assert release_message.string == release

    # Inspect the current source, so a missing entry cannot pass merely because
    # all of the older entries already in the catalogue have translations.
    current_release = (
        (REPO_ROOT / "CHANGELOG.md")
        .read_text(encoding="utf-8")
        .split(f"## {CANDIDATE_V027_HISTORY_ENTRY}", 1)[1]
        .split(f"## {PUBLISHED_V026_HISTORY_ENTRY}", 1)[0]
    )
    # This release uses headings, paragraphs and list items. Split those
    # blocks without requiring a docs-only Markdown parser in the PR gate.
    # Unrecognized markup fails the catalogue lookup instead of being skipped.
    blocks = re.split(r"\n\s*\n|\n(?=\s*(?:#{1,6}|[-+*]|\d+\.)\s)", current_release)
    for block in blocks:
        content = re.sub(r"^\s*(?:#{1,6}|[-+*]|\d+\.)\s+", "", block)
        message_id = " ".join(content.split())
        if not message_id or message_id in {PUBLISHED_V024_DATE, PUBLISHED_V023_DATE}:
            continue
        message = catalogue.get(message_id)
        assert message is not None, message_id
        assert message.string and "fuzzy" not in message.flags, message_id
        assert re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", message.string), message_id


def test_v025_changelog_matches_release_note_and_gettext_compiles() -> None:
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release_note = (REPO_ROOT / "release_notes/v0.2.5.md").read_text(encoding="utf-8")
    section = changelog.split(f"## {PUBLISHED_V025_HISTORY_ENTRY}", 1)[1].split(
        f"## {PUBLISHED_V024_HISTORY_ENTRY}", 1
    )[0]
    blocks = re.split(r"\n\s*\n|\n(?=\s*(?:#{1,6}|[-+*]|\d+\.)\s)", section)
    ids = []
    for block in blocks:
        content = re.sub(r"^\s*(?:#{1,6}|[-+*]|\d+\.)\s+", "", block)
        message_id = " ".join(content.split())
        if message_id:
            ids.append(message_id)
    assert len(ids) > 8  # Current changelog summary, headings, and release claims.
    assert ids[0].startswith(
        "This patch improves the reliability of scientific-data I/O"
    )
    assert "It adds no public API, dependency, or storage schema." in ids[0]
    assert ids[1] == "Fixed"
    release_note_text = " ".join(release_note.split())
    assert all(message_id in release_note_text for message_id in ids)
    assert any("**Native DTTXML selection**" in item for item in ids)
    assert any("**SDB during concurrent writes**" in item for item in ids)
    assert "release qualification pending" not in release_note.lower()
    assert "preserve serialized values and metadata" not in release_note

    path = REPO_ROOT / "docs_redesign/locales/ja/LC_MESSAGES/about/changelog.po"
    with path.open(encoding="utf-8") as stream:
        catalogue = pofile.read_po(stream, locale="ja")
    output = io.BytesIO()
    mofile.write_mo(output, catalogue)
    output.seek(0)
    translations = gettext.GNUTranslations(output)
    # The Japanese redesign catalog retains the published v0.2.5 history.
    for message_id in (PUBLISHED_V025_HISTORY_ENTRY, "Fixed"):
        message = catalogue.get(message_id)
        assert message is not None, message_id
        assert message.string and "fuzzy" not in message.flags, message_id
        if message_id != PUBLISHED_V025_HISTORY_ENTRY:
            assert re.search(r"[\u3040-\u30ff\u4e00-\u9fff]", message.string), (
                message_id
            )
        assert translations.gettext(message_id) == message.string, message_id
        if message_id != PUBLISHED_V025_HISTORY_ENTRY:
            assert translations.gettext(message_id) != message_id, message_id


def test_japanese_changelog_catalog_has_no_placeholder_diagnostics() -> None:
    path = REPO_ROOT / "docs_redesign/locales/ja/LC_MESSAGES/about/changelog.po"
    with path.open(encoding="utf-8") as stream:
        catalogue = pofile.read_po(stream, locale="ja")

    diagnostics = [
        f"{message.id!r}: {', '.join(str(error) for error in errors)}"
        for message, errors in catalogue.check()
    ]
    assert not diagnostics, "\n".join(diagnostics)


def test_v026_known_limitations_are_linked_in_release_artifacts() -> None:
    issue_ids = (766, 767, 768, 769, 770, 771, 773, 774, 777, 778)
    release_note = (REPO_ROOT / "release_notes/v0.2.6.md").read_text(encoding="utf-8")
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for issue_id in issue_ids:
        issue_link = f"https://github.com/tatsuki-washimi/gwexpy/issues/{issue_id}"
        assert issue_link in release_note
        assert issue_link in changelog
    assert "remain unresolved in v0.2.6" in release_note
    assert "remain unresolved in v0.2.6" in changelog
    assert "No workaround has been verified" in " ".join(release_note.split())
    assert "No workaround has been verified" in " ".join(changelog.split())
    release_note_text = " ".join(release_note.split()).lower()
    changelog_text = " ".join(changelog.split()).lower()
    assert "no commitment that it will be included in v0.2.7" in release_note_text
    assert "not a commitment to include them in v0.2.7" in changelog_text


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
