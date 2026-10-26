"""Check that published interop docs stay aligned with the contract."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs/developers/contracts/public_interop_contract.json"
EN_GUIDE = ROOT / "docs_redesign/reference/interop_capabilities.md"
JA_GUIDE = (
    ROOT / "docs_redesign/locales/ja/LC_MESSAGES/reference/interop_capabilities.po"
)
EN_INSTALL = ROOT / "docs_redesign/tutorials/installation.md"
JA_INSTALL = ROOT / "docs_redesign/locales/ja/LC_MESSAGES/tutorials/installation.po"

STATUS_LABELS_EN = {
    "public": "Public",
    "implemented": "Implemented",
    "implemented_partial": "Implemented, some paths still in progress",
    "in_progress": "In progress",
    "planned": "Planned",
}

STATUS_LABELS_JA = {
    "public": "公開済み",
    "implemented": "実装済み",
    "implemented_partial": "実装済み（一部経路は対応中）",
    "in_progress": "対応中",
    "planned": "対応予定",
}


def _load_contract() -> list[dict]:
    data = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return data["targets"]


def _read(path: Path) -> str:
    if path.suffix == ".po":
        from scripts.po_utils import read_po

        return "\n".join(
            str(value)
            for entry in read_po(path)
            if "fuzzy" not in entry["flags"]
            for value in (entry["msgstr"], *entry["plural"])
        )
    return path.read_text(encoding="utf-8")


def _find_row(text: str, marker: str, required: tuple[str, ...] = ()) -> str:
    candidates = [
        line for line in text.splitlines() if line.startswith("| ") and marker in line
    ]
    for line in candidates:
        if all(part in line for part in required):
            return line
    raise AssertionError(f"Could not find guide row containing {marker!r}")


def _find_contract_row(text: str, entry: dict) -> str:
    status = STATUS_LABELS_EN[entry["status"]]
    candidates = [
        line
        for line in text.splitlines()
        if line.startswith("| ") and entry["row_match_en"] in line and status in line
    ]
    for line in candidates:
        if all(
            f"`{name}()`" in line or f"`{name}`" in line for name in entry["guide_api"]
        ):
            return line
    raise AssertionError(f"Could not find contract row for {entry['name']!r}")


def _row_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip("|").split("|")]


def _po_entries() -> list[dict]:
    from scripts.po_utils import read_po

    return [
        entry
        for entry in read_po(JA_GUIDE)
        if "fuzzy" not in entry["flags"] and entry["msgstr"]
    ]


def _collect_contract_rows(text: str, labels: set[str]) -> set[str]:
    rows: set[str] = set()
    for line in text.splitlines():
        if not line.startswith("| "):
            continue

        cols = [col.strip() for col in line.strip("|").split("|")]
        if len(cols) != 5:
            continue

        if any(cols[2].startswith(label) for label in labels):
            rows.add(line)
    return rows


def test_english_guide_rows_match_contract():
    guide = _read(EN_GUIDE)

    for entry in _load_contract():
        line = _find_contract_row(guide, entry)
        for func_name in entry["guide_api"]:
            assert f"`{func_name}()`" in line or f"`{func_name}`" in line, (
                f"Expected `{func_name}` (with or without ()) in guide row: {line!r}"
            )
        details_link = entry.get("details_link")
        if details_link:
            redesigned_link = details_link.removesuffix(".rst")
            assert f"[API]({redesigned_link})" in line


def test_japanese_guide_rows_match_contract():
    english = _read(EN_GUIDE)
    translations = _po_entries()

    for entry in _load_contract():
        cells = _row_cells(_find_contract_row(english, entry))
        assert len(cells) == 5
        for cell in cells:
            matches = [
                translation
                for translation in translations
                if translation["msgid"] == cell
            ]
            assert matches, f"No Japanese catalog entry for source cell {cell!r}"
        status = cells[2]
        assert any(
            STATUS_LABELS_JA[entry["status"]] in item["msgstr"]
            for item in translations
            if item["msgid"] == status
        )
        api_cell = cells[1]
        for func_name in entry["guide_api"]:
            assert func_name in api_cell
            assert any(
                func_name in item["msgstr"]
                for item in translations
                if item["msgid"] == api_cell
            )


def test_mth5_row_is_now_public_in_both_guides():
    contract = {entry["name"]: entry for entry in _load_contract()}

    assert contract["mth5"]["status"] == "public"

    mth5 = contract["mth5"]
    en_line = _find_contract_row(_read(EN_GUIDE), mth5)
    ja_entries = _po_entries()

    assert "Public" in en_line
    assert any(item["msgstr"] == "公開済み" for item in ja_entries)
    assert any("to_mth5()" in item["msgstr"] for item in ja_entries)
    assert any("from_mth5()" in item["msgstr"] for item in ja_entries)
    assert "[API](../reference/api/gwexpy.interop.mt_)" in en_line
    assert any("gwexpy.interop.mt_" in item["msgid"] for item in ja_entries)


def test_contract_covers_all_documented_interop_rows():
    contract = _load_contract()
    en_guide = _read(EN_GUIDE)
    ja_translations = _po_entries()

    contract_en_rows = {_find_contract_row(en_guide, entry) for entry in contract}

    expected_en_rows = _collect_contract_rows(en_guide, set(STATUS_LABELS_EN.values()))
    assert contract_en_rows == expected_en_rows
    for row in contract_en_rows:
        for cell in _row_cells(row):
            assert any(item["msgid"] == cell for item in ja_translations), cell


def test_optional_dependency_policy_matches_contract():
    contract = _load_contract()
    en_docs = _read(EN_GUIDE)
    ja_docs = _read(JA_GUIDE)
    en_install = _read(EN_INSTALL)
    ja_install = _read(JA_INSTALL)

    for entry in contract:
        for dependency in entry.get("source_dependencies", []):
            assert f"`{dependency}`" in en_docs
            assert f"`{dependency}`" in ja_docs
        for dependency in entry["optional_dependencies"]:
            assert f"`{dependency}`" in en_docs
            assert f"`{dependency}`" in ja_docs
        for extra in entry["extras"]:
            assert f"`{extra}`" in en_install
            assert f"`{extra}`" in ja_install

    mth5 = {entry["name"]: entry for entry in contract}["mth5"]
    assert mth5["optional_dependencies"] == ["mth5"]
    assert mth5["extras"] == ["seismic"]
    assert "Install the matching extra from PyPI" in en_docs
    assert 'pip install "gwexpy[control]"' in en_docs
    assert "対応する extra を PyPI からインストール" in ja_docs
    assert 'pip install "gwexpy[control]"' in ja_docs

    en_normalized = " ".join(en_docs.split())
    ja_normalized = " ".join(ja_docs.split())
    assert "`gwexpy[all]` installs the declared GWexpy extras" in en_normalized
    assert "does not install every public interop backend" in en_normalized
    assert "`gwexpy[all]` は GWexpy が宣言している extra" in ja_docs
    assert "すべての public interop backend" in ja_normalized
