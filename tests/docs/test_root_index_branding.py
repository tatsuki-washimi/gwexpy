import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROOT_INDEX_HTML = ROOT / "docs" / "_build" / "html" / "index.html"


def test_root_docs_index_renders_home_brand_anchor():
    html = ROOT_INDEX_HTML.read_text(encoding="utf-8")

    match = re.search(
        r'<div class="gw-home-brand">\s*<img src="_static/branding/logo.svg" alt="GWexpy" />\s*</div>',
        html,
    )
    assert match is not None
    assert 'href="web/en/index.html"' in html
