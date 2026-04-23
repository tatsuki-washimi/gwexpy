"""Check that published I/O docs and the public contract stay aligned."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs/developers/contracts/public_io_contract.json"
EN_GUIDE = ROOT / "docs/web/en/user_guide/io_formats.md"
JA_GUIDE = ROOT / "docs/web/ja/user_guide/io_formats.md"


def _load_contract() -> dict[str, dict]:
    data = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return {entry["canonical"]: entry for entry in data["formats"]}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_docs_list_only_published_general_exchange_formats():
    contract = _load_contract()
    en = _read(EN_GUIDE)
    ja = _read(JA_GUIDE)

    for fmt in ("csv", "txt", "nc", "zarr", "root"):
        assert contract[fmt]["public_api"]["read"]

    assert "| **Pickle** (`.pkl`) |" not in en
    assert "| **Pickle** (`.pkl`) |" not in ja


def test_docs_loggers_table_matches_current_public_boundary():
    contract = _load_contract()
    en = _read(EN_GUIDE)
    ja = _read(JA_GUIDE)

    assert contract["gbd"]["required_args"]["read"] == ["timezone"]
    assert "timezone=..." in en
    assert "timezone=..." in ja

    assert contract["tdms"]["public_api"]["write"] == []
    assert "requires `nptdms`" in en
    assert "`nptdms` が必要" in ja

    assert contract["wav"]["public_api"]["write"] == ["TimeSeries"]
    assert "Public write is single-series only" in en
    assert "public write は単一路のみ" in ja

    for fmt in ("flac", "ogg", "mp3", "m4a"):
        assert contract[fmt]["public_api"]["write"] == ["TimeSeries", "TimeSeriesDict"]
    assert "TimeSeriesDict.read(..., format=...)" in en
    assert "TimeSeriesDict.read(..., format=...)" in ja


def test_docs_seismic_table_matches_current_public_boundary():
    contract = _load_contract()
    en = _read(EN_GUIDE)
    ja = _read(JA_GUIDE)

    assert contract["mseed"]["aliases"] == ["miniseed"]
    assert "legacy alias: `miniseed`" in en
    assert "旧 alias: `miniseed`" in ja

    for fmt in ("mseed", "sac", "gse2"):
        assert contract[fmt]["public_api"]["read"] == ["TimeSeriesDict"]
        assert contract[fmt]["public_api"]["write"] == ["TimeSeriesDict"]

    for fmt in ("knet", "win"):
        assert contract[fmt]["public_api"]["write"] == []

    assert contract["ats"]["public_api"]["read"] == ["TimeSeries", "TimeSeriesDict"]
    assert contract["ats.mth5"]["public_api"]["read"] == ["TimeSeries"]
    assert "The only direct path today is `ats.mth5`" in en
    assert "使える direct path は `ats.mth5` のみ" in ja
