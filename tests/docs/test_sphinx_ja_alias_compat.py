import io
from types import SimpleNamespace

from docutils.core import publish_doctree
from sphinx import locale as sphinx_locale

from docs import conf as docs_conf

ALIAS_MESSAGE = "alias of %s"
BAD_ALIAS = "%sの別名です。"
FIXED_ALIAS = "%s の別名です。"
INLINE_ALIAS = ":emphasis:`gwexpy.TimeSeries`"


def _warning_text(text: str) -> str:
    warning_stream = io.StringIO()
    publish_doctree(text, settings_overrides={"warning_stream": warning_stream})
    return warning_stream.getvalue()


def test_ja_alias_catalog_compatibility_rewrites_only_exact_bad_entry(
    monkeypatch,
) -> None:
    patcher = getattr(docs_conf, "_patch_sphinx_ja_alias_catalog", None)
    assert callable(patcher)

    translator = SimpleNamespace(_catalog={ALIAS_MESSAGE: BAD_ALIAS})
    monkeypatch.setitem(sphinx_locale.translators, ("general", "sphinx"), translator)
    app = SimpleNamespace(
        config=SimpleNamespace(language="ja"),
    )

    patcher(app)

    assert translator._catalog == {ALIAS_MESSAGE: FIXED_ALIAS}

    untouched_value = SimpleNamespace(_catalog={ALIAS_MESSAGE: "別名: %s"})
    monkeypatch.setitem(
        sphinx_locale.translators, ("general", "sphinx"), untouched_value
    )
    patcher(app)
    assert untouched_value._catalog == {ALIAS_MESSAGE: "別名: %s"}

    app.config.language = "en"
    monkeypatch.setitem(
        sphinx_locale.translators,
        ("general", "sphinx"),
        SimpleNamespace(_catalog={ALIAS_MESSAGE: BAD_ALIAS}),
    )
    patcher(app)
    assert sphinx_locale.get_translator("sphinx", "general")._catalog == {
        ALIAS_MESSAGE: BAD_ALIAS
    }


def test_ja_alias_bad_form_warns_but_fixed_form_parses_cleanly(monkeypatch) -> None:
    patcher = getattr(docs_conf, "_patch_sphinx_ja_alias_catalog", None)
    assert callable(patcher)

    translator = SimpleNamespace(_catalog={ALIAS_MESSAGE: BAD_ALIAS})
    monkeypatch.setitem(sphinx_locale.translators, ("general", "sphinx"), translator)
    app = SimpleNamespace(
        config=SimpleNamespace(language="ja"),
    )
    bad_text = translator._catalog[ALIAS_MESSAGE] % INLINE_ALIAS

    assert "Inline interpreted text or phrase reference start-string" in _warning_text(
        bad_text
    )

    patcher(app)
    fixed_text = translator._catalog[ALIAS_MESSAGE] % INLINE_ALIAS

    assert _warning_text(fixed_text) == ""
