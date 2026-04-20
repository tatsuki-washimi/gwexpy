from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAYOUT_PATH = ROOT / "docs" / "_templates" / "layout.html"
ROOT_INDEX_SOURCE = ROOT / "docs" / "index.rst"


def test_root_docs_index_uses_the_shipped_branding_hook():
    layout = LAYOUT_PATH.read_text(encoding="utf-8")
    root_index = ROOT_INDEX_SOURCE.read_text(encoding="utf-8")

    assert "if pagename == 'index'" in layout
    assert '<div class="gw-home-brand">' in layout
    assert '<img src="_static/branding/logo.svg" alt="GWexpy" />' in layout
    assert root_index.startswith("GWexpy\n======")
    assert ".. toctree::" in root_index
    assert "web/en/index" in root_index
    assert "web/ja/index" in root_index
    assert ".wy-nav-side          { display: none !important; }" in root_index
    assert ".wy-nav-content-wrap  { margin-left: 0 !important; }" in root_index
