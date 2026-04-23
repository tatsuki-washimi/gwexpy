"""Check that published interop docs stay aligned with the contract."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs/developers/contracts/public_interop_contract.json"
EN_GUIDE = ROOT / "docs/web/en/user_guide/interop.md"
JA_GUIDE = ROOT / "docs/web/ja/user_guide/interop.md"

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


def test_english_guide_rows_match_contract():
    guide = _read(EN_GUIDE)

    for entry in _load_contract():
        line = _find_row(guide, entry["row_match_en"])
        assert STATUS_LABELS_EN[entry["status"]] in line
        for func_name in entry["guide_api"]:
            assert f"`{func_name}()`" in line
        details_link = entry.get("details_link")
        if details_link:
            assert f"[API]({details_link})" in line


def test_japanese_guide_rows_match_contract():
    guide = _read(JA_GUIDE)

    for entry in _load_contract():
        line = _find_row(guide, entry["row_match_ja"])
        assert STATUS_LABELS_JA[entry["status"]] in line
        for func_name in entry["guide_api"]:
            assert f"`{func_name}()`" in line
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
