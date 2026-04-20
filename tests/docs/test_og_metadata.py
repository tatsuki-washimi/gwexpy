import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
CONF_PATH = ROOT / "docs" / "conf.py"


def _load_conf_module(name: str):
    spec = importlib.util.spec_from_file_location(name, CONF_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_app(conf):
    return SimpleNamespace(
        config=SimpleNamespace(
            html_baseurl=conf.html_baseurl,
            html_context=conf.html_context,
            html_title=conf.html_title,
        )
    )


def test_html_page_context_adds_page_level_og_fields(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("NBS_EXECUTE", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    conf = _load_conf_module("gwexpy_docs_conf_og_page")
    context = {
        "title": "Verification and Quality Signals",
        "metatags": (
            '<meta content="Understand what GWexpy validates publicly for '
            'notebooks, direct I/O formats, algorithm audit notes, and '
            'repository coverage signals, and where each evidence source '
            'lives." name="description" />'
        ),
    }

    conf.html_page_context(
        _make_app(conf),
        "web/en/user_guide/verification_and_quality",
        "page.html",
        context,
        doctree=None,
    )

    assert context["og_title"] == "Verification and Quality Signals — GWexpy Documentation"
    assert (
        context["og_description"]
        == "Understand what GWexpy validates publicly for notebooks, direct I/O "
        "formats, algorithm audit notes, and repository coverage signals, and "
        "where each evidence source lives."
    )
    assert (
        context["og_url"]
        == "https://tatsuki-washimi.github.io/gwexpy/docs/web/en/user_guide/"
        "verification_and_quality.html"
    )
    assert context["og_image"] == APPROVED_OG_IMAGE
    assert context["og_site_name"] == conf.html_title
    assert context["twitter_card"] == "summary_large_image"
    assert context["twitter_title"] == context["og_title"]
    assert context["twitter_description"] == context["og_description"]
    assert context["twitter_image"] == APPROVED_OG_IMAGE


def test_html_page_context_sets_ja_language_and_description_fallback(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("NBS_EXECUTE", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    conf = _load_conf_module("gwexpy_docs_conf_og_ja")
    context = {
        "title": "GWexpy Documentation",
        "metatags": "",
    }

    conf.html_page_context(
        _make_app(conf),
        "web/ja/index",
        "page.html",
        context,
        doctree=None,
    )

    assert context["language"] == "ja"
    assert context["og_title"] == "GWexpy Documentation"
    assert context["og_description"] == conf.html_context["og_description"]
    assert context["og_url"] == "https://tatsuki-washimi.github.io/gwexpy/docs/web/ja/index.html"
    assert context["og_image"] == APPROVED_OG_IMAGE


def test_docs_branding_configuration_points_to_brand_assets(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("NBS_EXECUTE", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)

    conf = _load_conf_module("gwexpy_docs_conf_branding")

    approved_og_image = (
        "https://tatsuki-washimi.github.io/gwexpy/docs/_static/branding/og-card.png"
    )

    assert conf.html_logo == conf.BRANDING_LOGO
    assert conf.html_favicon == conf.BRANDING_FAVICON
    assert conf.html_title == conf.BRANDING_SITE_TITLE
    assert conf.html_short_title == conf.BRANDING_SITE_SHORT_TITLE
    assert conf.social_og_image == approved_og_image
    assert conf.html_context["og_title"] == conf.BRANDING_OG_TITLE
    assert conf.html_context["og_description"] == conf.BRANDING_OG_DESCRIPTION
    assert conf.html_context["og_image"] == approved_og_image
