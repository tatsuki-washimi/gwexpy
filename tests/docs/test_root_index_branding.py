import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROOT_LAYOUT = ROOT / "docs" / "_templates" / "layout.html"
ROOT_INDEX_RST = ROOT / "docs" / "index.rst"


def test_root_docs_index_keeps_branding_source_contract():
    layout = ROOT_LAYOUT.read_text(encoding="utf-8")
    index_rst = ROOT_INDEX_RST.read_text(encoding="utf-8")

    match = re.search(
        r"\{% if pagename == 'index' %\}.*?<div class=\"gw-home-brand\">.*?<img src=\"_static/branding/logo\.svg\" alt=\"GWexpy\" />.*?</div>",
        layout,
        flags=re.DOTALL,
    )
    assert match is not None
    assert ".wy-nav-side          { display: none !important; }" in index_rst
    assert ".wy-nav-content-wrap  { margin-left: 0 !important; }" in index_rst
