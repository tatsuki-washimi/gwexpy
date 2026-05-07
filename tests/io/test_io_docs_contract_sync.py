"""Check that published I/O docs and the public contract stay aligned."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs/developers/contracts/public_io_contract.json"
EN_GUIDE = ROOT / "docs/web/en/user_guide/io_formats.md"
JA_GUIDE = ROOT / "docs/web/ja/user_guide/io_formats.md"
EN_INSTALL = ROOT / "docs/web/en/user_guide/installation.md"
JA_INSTALL = ROOT / "docs/web/ja/user_guide/installation.md"
LANDING_PAGES = (
    ROOT / "docs/index.rst",
    ROOT / "docs/web/en/index.rst",
    ROOT / "docs/web/ja/index.rst",
)
SCHUMANN_TUTORIAL = (
    ROOT / "docs/web/en/user_guide/tutorials/case_schumann_resonance.ipynb"
)
SEISMIC_TUTORIAL = ROOT / "docs/web/en/user_guide/tutorials/case_seismic_obspy.ipynb"


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


def test_docs_mark_frequency_dttxml_as_implementation_only():
    contract = _load_contract()
    en = _read(EN_GUIDE)
    ja = _read(JA_GUIDE)

    assert contract["xml.diaggui"]["public_api"]["read"] == ["TimeSeriesDict"]
    assert (
        "Frequency-domain DTTXML direct shims and registry adapters are implementation-only"
        in en
    )
    assert (
        "周波数領域の DTTXML direct shim と registry adapter は implementation-only"
        in ja
    )


def test_optional_dependency_matrix_matches_contract():
    contract = _load_contract()
    en = _read(EN_GUIDE)
    ja = _read(JA_GUIDE)
    en_install = _read(EN_INSTALL)
    ja_install = _read(JA_INSTALL)

    for entry in contract.values():
        for dependency in entry["optional_dependencies"]:
            assert f"`{dependency}`" in en
            assert f"`{dependency}`" in ja
        for extra in entry["extras"]:
            assert f"`{extra}`" in en_install
            assert f"`{extra}`" in ja_install

    win = contract["win"]
    assert win["unavailable_behavior"]["read"] == "conditional_registration"
    assert "conditional registration" in en
    assert "条件付き登録" in ja

    ats_mth5 = contract["ats.mth5"]
    assert ats_mth5["extras"] == ["seismic"]
    assert "required `seismic` extra" in en
    assert "必要な `seismic` extra" in ja

    wav = contract["wav"]
    assert wav["unavailable_behavior"]["read"] == "warns_and_skips_optional_metadata"
    assert "`.read(..., extract_metadata=True)` warns and skips metadata" in en
    assert (
        "`.read(..., extract_metadata=True)` は警告を出し、metadata を省略します" in ja
    )
    assert "`audio` or `all` source-extra syntax" in en
    assert "`audio` または `all` のソース導入形式" in ja


def test_landing_pages_import_frequency_matrix_from_frequencyseries():
    for page in LANDING_PAGES:
        text = _read(page)
        assert "from gwexpy.timeseries import FrequencySeriesMatrix" not in text
        assert "from gwexpy.frequencyseries import FrequencySeriesMatrix" in text


def test_tutorials_do_not_present_channel_names_as_read_paths():
    text = _read(SCHUMANN_TUTORIAL)

    assert "TimeSeries.read('K1:PEM-MAG_EXV_EAST_X_DQ', start, end)" not in text
    assert "NDS/GWOSC fetch workflow" in text


def test_seismic_tutorial_uses_public_dict_reader_boundary():
    text = _read(SEISMIC_TUTORIAL)

    stale_examples = (
        "TimeSeries.read(format='miniseed')",
        "TimeSeries.read(\"seismic.mseed\", format='miniseed')",
        "TimeSeries.read(path, format='miniseed')",
        "TimeSeries.read(path, format='sac')",
        "TimeSeries.read(\"path/to/seismic.sac\", format='sac')",
    )

    for example in stale_examples:
        assert example not in text

    assert "TimeSeriesDict.read(format='mseed')" in text
    assert "TimeSeriesDict.read(path, format='mseed')" in text
    assert "TimeSeriesDict.read(path, format='sac')" in text
