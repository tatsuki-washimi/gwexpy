# Configuration file for the Sphinx documentation builder.
# Redesign Variant B1: pydata-sphinx-theme + Diataxis (greenfield prototype).

# The real ``gwexpy`` package is installed in the build environment, so
# autodoc/autosummary document the actual API directly (no demo stub).

import os

from gwexpy._version import __version__

# Default matches the confirmed production URL (Step F-2: keep the existing
# /docs/ path rather than moving docs_redesign to the site root).
_DEFAULT_SITE_ROOT = "https://tatsuki-washimi.github.io/gwexpy/docs/"

# Shared site root the language switcher uses to compute the counterpart-
# language page URL. Identical for the EN and JA builds of the same deploy
# (e.g. both point at .../docs/, never at .../docs/ja/).
LANG_BASEURL = os.environ.get("GWEXPY_DOCS_LANG_BASEURL", _DEFAULT_SITE_ROOT)

# Canonical base for *this* build only. Differs per language (EN serves at
# the site root, JA under /ja/) -- keep this separate from LANG_BASEURL or
# the JA build's <link rel="canonical"> ends up pointing at the EN page.
SITE_BASEURL = os.environ.get("GWEXPY_DOCS_BASEURL", LANG_BASEURL)

# Set to "1" for preview/staging deploys so search engines don't index them
# ahead of the production cutover. Left unset (falsy) in production.
NOINDEX = os.environ.get("GWEXPY_DOCS_NOINDEX") == "1"

# -- Project information -----------------------------------------------------
project = "GWexpy"
copyright = "2026, GWexpy Developers"
author = "GWexpy Developers"
release = __version__

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
]

# Strip interactive prompts from copy-pasted code (affects the ">>> "/"... "
# examples that autodoc renders from gwexpy's own docstrings on API pages).
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# MyST extensions for richer Markdown authoring.
myst_enable_extensions = [
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "tasklist",
]
# Resolve in-page Markdown anchor links (e.g. [text](#section)).
myst_heading_anchors = 3

# Execute clean notebook sources through MyST-NB's build cache.  The cache is
# an untracked build artifact: rendered HTML therefore includes plots and
# other cell outputs, while the .ipynb files committed to Git remain clean.
# ``cache`` also lets the JA build reuse the executed EN notebooks instead of
# running the same code a second time.
nb_execution_mode = "cache"
nb_execution_cache_path = "_build/jupyter-cache"
# Fitting and file-I/O examples exceed MyST-NB's 30-second default on the
# GitHub Pages runner.
nb_execution_timeout = 180
nb_execution_allow_errors = False
nb_execution_raise_on_error = True

# -- Internationalization (gettext single-source) ----------------------------
# English is the single source; Japanese is delivered via gettext catalogs in
# ``locales/ja/LC_MESSAGES/*.po`` instead of a parallel source tree.
#   sphinx-build -b gettext .  _build/gettext     # extract .pot
#   sphinx-intl update -p _build/gettext -l ja    # create/update .po
#   sphinx-build -b html -D language=ja . _build/html/ja
language = "en"
locale_dirs = ["locales/"]
gettext_compact = False  # one .pot per source doc (enables selective seeding)
gettext_uuid = True

# -- autodoc / autosummary ---------------------------------------------------
# Document the real, installed ``gwexpy`` package.
autosummary_generate = True
autosummary_imported_members = False
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_admonition_for_notes = True

# Custom roles / substitutions used in gwexpy and upstream GWpy docstrings.
rst_prolog = r"""
.. role:: dcc(code)
.. role:: mpltype(code)
.. role:: doi(code)
.. |lal.LIGOTimeGPS| replace:: ``lal.LIGOTimeGPS``
.. _lal.ligotimegps: https://docs.ligo.org/lscsoft/lalsuite/lal/group___x_l_a_l_time__c.html
"""

# Optional third-party backends that need not be installed to build the docs.
autodoc_mock_imports = [
    "mictools", "dcor", "hurst", "specutils", "pyspeckit", "obspy",
    "mth5", "mtpy", "librosa", "pydub", "mne", "neo", "elephant",
    "cupy", "pygmt", "PyQt5", "qtpy", "pyqtgraph",
]

# Keep the build signal meaningful: silence external cross-reference noise
# (offline intersphinx, upstream GWpy docstrings) but keep broken internal
# references visible.
suppress_warnings = [
    "ref.ref", "ref.obj", "ref.meth", "ref.func", "ref.class",
    "ref.footnote", "intersphinx.broken_domain",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "gwpy": ("https://gwpy.readthedocs.io/en/stable/", None),
}

# The LALSuite publisher serves this valid public reference to browsers, but
# returns 404 to Sphinx linkcheck's HTTP client. Keep the authoritative target
# while excluding that false negative from the automated external-link gate.
linkcheck_ignore = [
    r"https://docs\.ligo\.org/lscsoft/lalsuite/lal/group___x_l_a_l_time__c\.html",
]

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/branding/logo.svg"
html_favicon = "_static/images/favicon.svg"
html_title = "GWexpy"
# Base URL lets the language switcher compute the counterpart-language page URL.
html_baseurl = SITE_BASEURL
# Expose raw .md/.ipynb sources (instead of a Sphinx-appended .txt) so the
# "Show Source" sidebar link doubles as a real download of the source file.
html_sourcelink_suffix = ""

html_theme_options = {
    # Top navbar layout (Diataxis sections live in the center). The logo alone
    # keeps the header compact; the version switcher/warning banner are omitted
    # while the project is pre-release to avoid an oversized top band.
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links", "language-switcher.html"],
    "navbar_persistent": ["search-button"],
    # Secondary (right) sidebar with in-page table of contents.
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "show_toc_level": 2,
    # Icon links.
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/tatsuki-washimi/gwexpy",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
    "logo": {
        "text": "GWexpy",
        "image_light": "_static/branding/logo.svg",
        "image_dark": "_static/branding/logo.svg",
    },
    "header_links_before_dropdown": 6,
}

html_context = {
    "github_user": "tatsuki-washimi",
    "github_repo": "gwexpy",
    "github_version": "main",
    "doc_path": "docs_redesign",
    "default_mode": "auto",
    # Languages offered by the navbar switcher. EN is served at the site root,
    # JA under the ``/ja/`` subdirectory (a single source, two builds).
    "languages": [
        ("en", "English", ""),
        ("ja", "日本語", "ja/"),
    ],
    # Base URL the switcher uses to build the counterpart-language page URL.
    "lang_base": LANG_BASEURL,
    # Also used by _templates/layout.html to compute hreflang alternates and
    # to gate the preview-only noindex meta tag.
    "noindex": NOINDEX,
}

# Sidebar: keep the left sidebar clean (navigation only).
html_sidebars = {
    "index": [],  # Homepage: no left sidebar for a clean hero.
}


# -- i18n fix for notebooks --------------------------------------------------
# Sphinx's i18n transform re-parses each translated ``msgstr`` with the source
# document's parser. For ``.ipynb`` docs that is myst-nb's notebook reader,
# which expects JSON and therefore crashes on a translated Markdown string
# (myst-nb keys the reader off ``env.doc2path(docname)``, ignoring source_path).
# A msgstr is always inline Markdown, never a notebook, so for notebook docs we
# re-parse it with the plain MyST Markdown parser instead.
def setup(app):
    from sphinx.transforms import i18n as _i18n

    _orig_publish_msgstr = _i18n.publish_msgstr

    def _publish_msgstr(app, source, source_path, source_line, config, settings):
        if source_path and str(source_path).endswith(".ipynb"):
            import contextlib

            from docutils.io import StringInput
            from myst_parser.parsers.sphinx_ import MystParser
            from sphinx.io import SphinxI18nReader

            rst_prolog = config.rst_prolog
            config.rst_prolog = None
            try:
                reader = SphinxI18nReader()
                reader.setup(app)
                parser = MystParser()
                parser.set_application(app)
                doc = reader.read(
                    source=StringInput(
                        source=source,
                        source_path=f"{source_path}:{source_line}:<translated>",
                    ),
                    parser=parser,
                    settings=settings,
                )
                with contextlib.suppress(IndexError):
                    return doc[0]
                return doc
            finally:
                config.rst_prolog = rst_prolog
        return _orig_publish_msgstr(
            app, source, source_path, source_line, config, settings
        )

    _i18n.publish_msgstr = _publish_msgstr

    # Localize theme_options string defaults that pydata-sphinx-theme does not
    # run through gettext (they're plain ``theme.conf`` option strings, e.g.
    # ``search_bar_text``/``icon_links_label``, not template ``_()`` calls).
    # Hooked on config-inited (not set unconditionally above) so it sees the
    # real target language after a ``-D language=ja`` build-time override.
    def _localize_theme_option_strings(app, config):
        if config.language == "ja":
            config.html_theme_options = {
                **config.html_theme_options,
                "search_bar_text": "ドキュメントを検索...",
                "icon_links_label": "アイコンリンク",
            }

    app.connect("config-inited", _localize_theme_option_strings)

    return {"parallel_read_safe": True, "parallel_write_safe": True}
