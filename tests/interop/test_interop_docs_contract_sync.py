"""Check that published interop docs stay aligned with the contract."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs/developers/contracts/public_interop_contract.json"
EN_GUIDE = ROOT / "docs/web/en/user_guide/interop.md"
JA_GUIDE = ROOT / "docs/web/ja/user_guide/interop.md"
EN_INSTALL = ROOT / "docs/web/en/user_guide/installation.md"
JA_INSTALL = ROOT / "docs/web/ja/user_guide/installation.md"

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
    return path.read_text(encoding="utf-8")


def _find_row(text: str, marker: str) -> str:
    for line in text.splitlines():
        if marker in line:
            return line
    raise AssertionError(f"Could not find guide row containing {marker!r}")


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
        line = _find_row(guide, entry["row_match_en"])
        assert STATUS_LABELS_EN[entry["status"]] in line
        for func_name in entry["guide_api"]:
            assert (
                f"`{func_name}()`" in line or f"`{func_name}`" in line
            ), f"Expected `{func_name}` (with or without ()) in guide row: {line!r}"
        details_link = entry.get("details_link")
        if details_link:
            assert f"[API]({details_link})" in line


def test_japanese_guide_rows_match_contract():
    guide = _read(JA_GUIDE)

    for entry in _load_contract():
        line = _find_row(guide, entry["row_match_ja"])
        assert STATUS_LABELS_JA[entry["status"]] in line
        for func_name in entry["guide_api"]:
            assert (
                f"`{func_name}()`" in line or f"`{func_name}`" in line
            ), f"Expected `{func_name}` (with or without ()) in guide row: {line!r}"
        details_link = entry.get("details_link")
        if details_link:
            assert f"[API]({details_link})" in line


def test_mth5_row_is_now_public_in_both_guides():
    contract = {entry["name"]: entry for entry in _load_contract()}

    assert contract["mth5"]["status"] == "public"

    en_line = _find_row(_read(EN_GUIDE), contract["mth5"]["row_match_en"])
    ja_line = _find_row(_read(JA_GUIDE), contract["mth5"]["row_match_ja"])

    assert "Public" in en_line
    assert "公開済み" in ja_line
    assert "[API](../reference/api/gwexpy.interop.mt_.rst)" in en_line
    assert "[API](../reference/api/gwexpy.interop.mt_.rst)" in ja_line


def test_contract_covers_all_documented_interop_rows():
    contract = _load_contract()
    en_guide = _read(EN_GUIDE)
    ja_guide = _read(JA_GUIDE)

    contract_en_rows = {_find_row(en_guide, entry["row_match_en"]) for entry in contract}
    contract_ja_rows = {_find_row(ja_guide, entry["row_match_ja"]) for entry in contract}

    expected_en_rows = _collect_contract_rows(en_guide, set(STATUS_LABELS_EN.values()))
    expected_ja_rows = _collect_contract_rows(ja_guide, set(STATUS_LABELS_JA.values()))

    assert contract_en_rows == expected_en_rows
    assert contract_ja_rows == expected_ja_rows


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
    assert "pip install 'gwexpy[seismic]'" in en_docs
    assert "pip install 'gwexpy[seismic]'" in ja_docs

    en_normalized = " ".join(en_docs.split())
    ja_normalized = " ".join(ja_docs.split())
    assert "`gwexpy[all]` installs the declared GWexpy extras" in en_normalized
    assert "does not install every public interop backend" in en_normalized
    assert "`gwexpy[all]` は GWexpy が宣言している extra" in ja_docs
    assert "すべての public interop backend" in ja_normalized
