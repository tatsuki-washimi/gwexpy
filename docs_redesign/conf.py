# Configuration file for the Sphinx documentation builder.
# Redesign Variant B1: pydata-sphinx-theme + Diataxis (greenfield prototype).

# The real ``gwexpy`` package is installed in the build environment, so
# autodoc/autosummary document the actual API directly (no demo stub).

import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path

from gwexpy import __file__ as _gwexpy_module_path
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
_docs_root = Path(__file__).resolve().parent
_release_status = json.loads((_docs_root / "release_status.json").read_text())
_identity_path = _docs_root / "_build_identity.json"
if _identity_path.exists():
    _build_identity = json.loads(_identity_path.read_text())
else:
    _repo = _docs_root.parent
    _build_identity = {
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_repo, text=True
        ).strip(),
        "source_ref": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=_repo, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=_repo, text=True
            )
        ),
    }
# Used by prose; the build environment's package version is separate from a released version.
myst_substitutions = _release_status


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
    "substitution",
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
# Never reuse numerical outputs across a different interpreter, dependency set,
# or package source revision merely because notebook cell text is unchanged.
_runtime_source_hash = hashlib.sha256()
for _source_path in sorted(Path(_gwexpy_module_path).parent.rglob("*.py")):
    _runtime_source_hash.update(
        str(_source_path.relative_to(Path(_gwexpy_module_path).parent)).encode()
    )
    _runtime_source_hash.update(_source_path.read_bytes())
_execution_environment = {
    "runtime_source_sha256": _runtime_source_hash.hexdigest(),
    "python": sys.version,
    "source_revision": _build_identity["source_revision"],
    "packages": sorted(
        (d.metadata["Name"], d.version) for d in distributions() if d.metadata["Name"]
    ),
}
_execution_fingerprint = hashlib.sha256(
    json.dumps(_execution_environment, sort_keys=True).encode()
).hexdigest()[:16]
nb_execution_cache_path = str(
    _docs_root / "_build/jupyter-cache" / _execution_fingerprint
)
# Fitting and file-I/O examples exceed MyST-NB's 30-second default on the
# GitHub Pages runner. Seasonal auto-ARIMA search (m=50) and per-bin
# Student-t MLE spectrogram fits measured at ~170-225s locally, right at
# the previous 180s ceiling and prone to timing out under CI load
# variance; 600s gives real headroom without approaching the 60-minute
# job timeout.
nb_execution_timeout = 600
nb_execution_allow_errors = False
nb_execution_raise_on_error = True
nb_execution_show_tb = True

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

# Reader-facing labels in the v0.2.0 changelog's Mermaid code fence are
# literal source and therefore outside gettext. Keep the English release
# record immutable, while substituting just these rendered labels in the
# Japanese changelog page.
_CHANGELOG_MERMAID_JA_LABELS = {
    'baseline["v0.1.14 baseline"]': 'baseline["v0.1.14 ベースライン"]',
    'integration["v0.2 contract integration"]': 'integration["v0.2 契約統合"]',
    'median_mean["#686 median-mean spectral dispatch"]': 'median_mean["#686 median-mean スペクトル振り分け"]',
    'source["v0.2.0 release-source metadata"]': 'source["v0.2.0 リリースソースのメタデータ"]',
}

# -- autodoc / autosummary ---------------------------------------------------
# Document the real, installed ``gwexpy`` package.
autosummary_generate = True
autosummary_imported_members = False
autodoc_typehints = "signature"
autodoc_typehints_format = "short"
autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_admonition_for_notes = True
# Render the Attributes section as instance-variable fields; autodoc separately
# indexes the actual dataclass/property members on the same API page.
napoleon_use_ivar = True

# Custom roles / substitutions used in gwexpy and upstream GWpy docstrings.
rst_prolog = r"""
.. role:: dcc(code)
.. role:: mpltype(code)
.. role:: doi(code)
.. |lal.LIGOTimeGPS| replace:: ``lal.LIGOTimeGPS``
.. _lal.ligotimegps: https://docs.ligo.org/lscsoft/lalsuite/lal/group___x_l_a_l_time__c.html
.. |lalframe| replace:: LALFrame
.. _lalframe: https://docs.ligo.org/lscsoft/lalsuite/lalframe/
.. |nds2| replace:: NDS2
.. _nds2: https://nds.ligo.org/
"""

# Optional third-party backends that need not be installed to build the docs.
autodoc_mock_imports = [
    "mictools",
    "dcor",
    "hurst",
    "specutils",
    "pyspeckit",
    "obspy",
    "mth5",
    "mtpy",
    "librosa",
    "pydub",
    "mne",
    "neo",
    "elephant",
    "cupy",
    "pygmt",
    "PyQt5",
    "qtpy",
    "pyqtgraph",
]

# Keep the build signal meaningful: silence external cross-reference noise
# (offline intersphinx, upstream GWpy docstrings) but keep broken internal
# references visible.
suppress_warnings = [
    "ref.ref",
    "ref.obj",
    "ref.meth",
    "ref.func",
    "ref.class",
    "ref.footnote",
    "intersphinx.broken_domain",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "README.md"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

intersphinx_timeout = 15
linkcheck_timeout = 20

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
    # keeps the header compact; build identity is displayed above the content.
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
    "source_revision": _build_identity["source_revision"],
    "source_dirty": _build_identity["dirty"],
    **_release_status,
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
html_sidebars: dict[str, list[str]] = {
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
    """Register localization, API presentation, and build evidence hooks."""
    from pygments.lexers.special import TextLexer
    from sphinx.ext.autodoc import ClassDocumenter
    from sphinx.transforms import i18n as _i18n

    # Historical release notes preserve Mermaid source as a literal record.
    app.add_lexer("mermaid", TextLexer)

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

    class _AliasClassDocumenter(ClassDocumenter):
        def add_line(self, line, source, *lineno):
            """Separate the RST role from Sphinx's Japanese alias suffix."""
            if self.config.language == "ja" and line.endswith("`の別名です。"):
                line = line.removesuffix("の別名です。") + r"\ の別名です。"
            super().add_line(line, source, *lineno)

    app.add_autodocumenter(_AliasClassDocumenter, override=True)

    def _localize_changelog_mermaid(app, doctree, docname):
        if app.config.language != "ja" or docname != "about/changelog":
            return
        from docutils import nodes

        for block in doctree.findall(nodes.literal_block):
            if block.get("language") != "mermaid":
                continue
            text = block.astext()
            if 'baseline["v0.1.14 baseline"]' not in text:
                continue
            for english, japanese in _CHANGELOG_MERMAID_JA_LABELS.items():
                text = text.replace(english, japanese)
            block.rawsource = text
            block.children[:] = [nodes.Text(text)]

    app.connect("doctree-resolved", _localize_changelog_mermaid)

    def _write_build_identity(app, exception):
        if exception is None and app.builder.format == "html":
            info = {
                **_build_identity,
                **_release_status,
                "package_version": release,
                "language": app.config.language,
                "language_baseurl": LANG_BASEURL,
                "canonical_baseurl": SITE_BASEURL,
                "notebook_execution": {
                    name: {
                        key: details.get("exec_data", {}).get(key)
                        for key in ("succeeded", "runtime", "method")
                    }
                    for name, details in getattr(app.env, "nb_metadata", {}).items()
                },
            }
            (Path(app.outdir) / "build-info.json").write_text(
                json.dumps(info, indent=2) + "\n"
            )

    app.connect("build-finished", _write_build_identity)

    def _case_information(app, pagename, templatename, context, doctree):
        cases = json.loads((_docs_root / "case_studies.json").read_text())
        if pagename in cases:
            context["case_details"] = cases[pagename]
            context["case_execution"] = (
                getattr(app.env, "nb_metadata", {}).get(pagename, {}).get("exec_data")
            )

    app.connect("html-page-context", _case_information)

    def _localize_changelog_activity(app, docname, source):
        if app.config.language == "ja" and docname == "about/changelog":
            source[0] = source[0].replace(
                ":::{figure} /_static/images/development-activity-v0.2.3.svg",
                ":::{figure} /_static/images/development-activity-v0.2.3-ja.svg",
            ).replace(
                ":::{figure} /_static/images/development-activity-v0.2.3-candidate.svg",
                ":::{figure} /_static/images/development-activity-v0.2.3-candidate-ja.svg",
            )

    app.connect("source-read", _localize_changelog_activity)

    def _format_api_docstring(app, what, name, obj, options, lines):
        import re

        # Resolve legacy tutorial links in the current documentation layout.
        tutorial_paths = {
            "case_transfer_function": "/how-to/case-studies/case_transfer_function",
            "case_bootstrap_gls_fitting": "/how-to/case-studies/case_bootstrap_gls_fitting",
            "advanced_hht": "/how-to/spectral/advanced_hht",
        }
        for index, line in enumerate(lines):
            for tutorial, path in tutorial_paths.items():
                line = line.replace("../../../user_guide/tutorials/" + tutorial, path)
            if name == "gwexpy.analysis.response.ResponseFunctionResult":
                line = re.sub(
                    r"(?<![\w.])SegmentTable(?![\w.])",
                    "gwexpy.table.SegmentTable",
                    line,
                )
            lines[index] = line

        # Keep the approved runtime tree unchanged while making this existing
        # bullet list valid reStructuredText in the generated API reference.
        if (
            getattr(obj, "__module__", None) == "gwexpy.timeseries._signal"
            and getattr(obj, "__name__", None) == "lock_in"
        ):
            for index in range(len(lines) - 1, -1, -1):
                if lines[index].strip() == "**Edge Handling**":
                    if index + 1 < len(lines) and lines[index + 1].strip():
                        lines.insert(index + 1, "")

    app.connect("autodoc-process-docstring", _format_api_docstring)

    return {"parallel_read_safe": True, "parallel_write_safe": True}
