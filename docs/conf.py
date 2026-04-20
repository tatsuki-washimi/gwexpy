import os
import tempfile
from html.parser import HTMLParser
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

from pygments.lexers.python import PythonLexer

DOCS_DIR = Path(__file__).resolve().parent
if str(DOCS_DIR) not in sys.path:
    sys.path.insert(0, str(DOCS_DIR))

from intersphinx_registry import build_intersphinx_mapping

sys.path.insert(0, os.path.abspath(".."))

project = "GWexpy"
author = "GWexpy contributors"
copyright = f"{datetime.now():%Y}, GWexpy contributors"

BRANDING_STATIC_DIR = "_static/branding"
BRANDING_LOGO = f"{BRANDING_STATIC_DIR}/logo.svg"
BRANDING_FAVICON = f"{BRANDING_STATIC_DIR}/icon.svg"
BRANDING_SOCIAL_CARD = f"{BRANDING_STATIC_DIR}/og-card.png"
BRANDING_SITE_TITLE = "GWexpy Documentation"
BRANDING_SITE_SHORT_TITLE = "GWexpy"
BRANDING_OG_TITLE = "GWexpy Documentation"
BRANDING_OG_DESCRIPTION = (
    "GWexpy documentation hub for installation, quickstart, tutorials, case "
    "studies, and API reference entry points."
)

# Version variables — automatically resolved by Sphinx |release| and |version|
try:
    from importlib.metadata import version as _get_version
    release = _get_version("gwexpy")
except Exception:
    # Fallback for development environments
    release = "dev"
version = ".".join(release.split(".")[:2]) if release != "dev" else "dev"

# Sitemap URL
sitemap_url = "https://tatsuki-washimi.github.io/gwexpy/docs/"

# Date format for last updated
html_last_updated_fmt = "%Y-%m-%d %H:%M:%S"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_last_updated_by_git",
]


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _has_pandoc() -> bool:
    return shutil.which("pandoc") is not None


def _prefer_remote_intersphinx() -> bool:
    return _env_flag("INTERSPHINX_USE_REMOTE")


def _default_nbsphinx_execute() -> str:
    """Choose a safe default execution policy for the current environment."""
    if "NBS_EXECUTE" in os.environ:
        return os.environ["NBS_EXECUTE"]
    # Local/sandbox builds cannot reliably start Jupyter kernels.
    if _env_flag("GITHUB_ACTIONS"):
        return "always"
    return "never"


def _default_nbsphinx_allow_errors() -> bool:
    """Allow notebook errors locally, but fail them by default in GitHub Actions."""
    if "NBS_ALLOW_ERRORS" in os.environ:
        return _env_flag("NBS_ALLOW_ERRORS")
    return not _env_flag("GITHUB_ACTIONS")


DOCS_MPL_FONT_FAMILY = ["sans-serif"]
DOCS_MPL_FONT_SANS_SERIF = [
    "Noto Sans CJK JP",
    "IPAexGothic",
    "IPAGothic",
    "DejaVu Sans",
]


def _configure_mplconfig_dir() -> Path:
    """Prepare a shared Matplotlib config directory for docs and notebook kernels."""
    if "MPLCONFIGDIR" in os.environ:
        mplconfig_dir = Path(os.environ["MPLCONFIGDIR"])
    else:
        mplconfig_dir = Path(tempfile.gettempdir()) / "gwexpy-mplconfig"
        os.environ["MPLCONFIGDIR"] = str(mplconfig_dir)

    mplconfig_dir.mkdir(parents=True, exist_ok=True)
    (mplconfig_dir / "matplotlibrc").write_text(
        "\n".join(
            [
                "backend: Agg",
                f"font.family: {', '.join(DOCS_MPL_FONT_FAMILY)}",
                f"font.sans-serif: {', '.join(DOCS_MPL_FONT_SANS_SERIF)}",
                "axes.unicode_minus: False",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return mplconfig_dir


try:
    _configure_mplconfig_dir()
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib as mpl
except Exception:  # pragma: no cover - docs env should have matplotlib
    mpl = None
else:
    mpl.rcParams["font.family"] = DOCS_MPL_FONT_FAMILY
    mpl.rcParams["font.sans-serif"] = DOCS_MPL_FONT_SANS_SERIF
    mpl.rcParams["axes.unicode_minus"] = False

# Only include sphinx_sitemap for environments that actually need sitemap generation.
# The extension unconditionally creates a multiprocessing.Manager() queue during
# ``builder-inited``, which fails in restricted local sandboxes with
# ``PermissionError: [Errno 1] Operation not permitted``.
#
# GitHub Actions builds keep sitemap generation enabled by default. For other
# environments, opt in explicitly with ``ENABLE_SITEMAP=1``.
enable_sitemap = _env_flag("ENABLE_SITEMAP") or _env_flag("GITHUB_ACTIONS")

# Without this, `sphinx-build -b linkcheck` errors out with
# "No pages generated for sitemap.xml".
if "linkcheck" not in sys.argv and enable_sitemap:
    extensions.append("sphinx_sitemap")

# Notebook support is optional for local builds:
# - sandboxed environments cannot open sockets for kernel execution
# - nbconvert requires pandoc to convert notebook markdown cells
notebook_build_enabled = _has_pandoc()
if notebook_build_enabled:
    extensions.append("nbsphinx")

# nbsphinx configuration
# Local development defaults to "never" to avoid sandbox/kernel failures.
# CI/production can opt in explicitly with NBS_EXECUTE=always or rely on the
# GitHub Actions default below.
nbsphinx_execute = _default_nbsphinx_execute()

# Allow errors locally, but fail notebook errors by default in GitHub Actions.
nbsphinx_allow_errors = _default_nbsphinx_allow_errors()

autosummary_generate = True
autosummary_imported_members = False
autodoc_typehints = "signature"
autodoc_typehints_format = "short"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
    "no-index": True,
}

# Avoid pulling in docstrings from external base classes that contain Sphinx
# substitutions or formatting we don't control.
autodoc_inherit_docstrings = False

# Napoleon settings for NumPy/Google docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_admonition_for_notes = True

# Add a consistent download link on notebook pages - language-aware using Jinja2
nbsphinx_prolog = r"""
{% if '/ja/' in env.docname %}
.. note::
   このページは Jupyter Notebook から生成されました。
   `ノートブックをダウンロード (.ipynb) <https://github.com/tatsuki-washimi/gwexpy/raw/main/docs/{{ env.doc2path(env.docname, base=None) }}>`_
{% else %}
.. note::
   This page was generated from a Jupyter Notebook.
   `Download the notebook (.ipynb) <https://github.com/tatsuki-washimi/gwexpy/raw/main/docs/{{ env.doc2path(env.docname, base=None) }}>`_
{% endif %}
"""

rst_prolog = r"""
.. role:: dcc(code)
.. role:: mpltype(code)
.. role:: doi(code)
.. |lal.LIGOTimeGPS| replace:: ``lal.LIGOTimeGPS``
.. _lal.ligotimegps: https://lscsoft.docs.ligo.org/lalsuite/lal/
"""

# Whitelist internal or temporarily unreachable links
linkcheck_ignore = [
    # LIGO internal DTT documentation (requires internal network access)
    r"https://dtt\.ligo\.org/.*",
    # LIGO internal LALSuite class documentation (often redirects/requires VPN)
    r"https://lscsoft\.docs\.ligo\.org/lalsuite/lal/classlal_1_1_l_i_g_o_time_g_p_s\.html",
]

# Ignore nitpick errors for well-known external symbols to keep -n builds manageable.
nitpick_ignore = []
nitpick_ignore_regex = []

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "NOTEBOOK_POLICY.md",
    "developers/**",
    "repro/**",
    "superpowers/plans/**",
]
if not notebook_build_enabled:
    exclude_patterns.append("**/*.ipynb")

# Silence cross-reference warnings from external docstrings (e.g. GWpy upstream).
# Avoid suppressing broad categories like "autodoc" or "docutils" so that
# broken internal references remain visible during build.
suppress_warnings = [
    "ref.ref",
    "ref.obj",
    "ref.meth",
    "ref.func",
    "ref.class",
    "ref.doc",
    "ref.footnote",
    "toc.not_included",
    "toc.not_readable",
    "nbsphinx.localfile",
    "intersphinx.broken_domain",
]

language = "en"
locale_dirs = ["locales/"]
gettext_compact = False

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_js_files = ["copybutton.js", "external-links.js"]
html_logo = BRANDING_LOGO
html_favicon = BRANDING_FAVICON
html_baseurl = "https://tatsuki-washimi.github.io/gwexpy/docs/"
html_theme_options = {
    "logo_only": False,
    "prev_next_buttons_location": "bottom",
    "style_external_links": True,
    "vcs_pageview_mode": "blob",
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# User-facing site title/branding (package name remains `gwexpy`).
html_title = BRANDING_SITE_TITLE
html_short_title = BRANDING_SITE_SHORT_TITLE
# Keep social images absolute so OGP/Twitter metadata stays valid on GitHub Pages.
social_og_image = urljoin(html_baseurl, BRANDING_SOCIAL_CARD)
html_context = {
    "display_github": True,
    "github_user": "tatsuki-washimi",
    "github_repo": "gwexpy",
    "github_version": "main",
    "conf_py_path": "/docs/",
    "current_language": language,
    # GitHub Pages publishes docs under /gwexpy/; user-facing docs live in /docs/web/{en,ja}/.
    "languages": [("en", "/tatsuki-washimi/gwexpy/docs/web/en/"), ("ja", "/tatsuki-washimi/gwexpy/docs/web/ja/")],
    # OGP / Metadata
    "og_title": BRANDING_OG_TITLE,
    "og_description": BRANDING_OG_DESCRIPTION,
    "og_type": "website",
    "og_url": "https://tatsuki-washimi.github.io/gwexpy/",
    "og_image": social_og_image,
    "twitter_card": "summary_large_image",
}

# docs/conf.py (add near top or appropriate section)
autodoc_mock_imports = getattr(globals(), "autodoc_mock_imports", []) + [
    "mictools",
    "dcor",
    "hurst",
    "bottleneck",
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
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# MyST: enable heading anchors so internal markdown `#...` links work.
myst_heading_anchors = 3
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
]

# nbformat emits noisy warnings when notebooks omit cell IDs (valid for older nbformat).
try:  # pragma: no cover
    from nbformat.warnings import MissingIDFieldWarning

    warnings.filterwarnings("ignore", category=MissingIDFieldWarning)
except Exception:
    pass

intersphinx_mapping = build_intersphinx_mapping(
    prefer_remote=_prefer_remote_intersphinx()
)

nitpick_ignore = [
    ("py:class", "u.Quantity"),  # Sphinx sometimes misses the alias
    ("py:class", "UnitLike"),
    ("py:class", "IndexLike"),
    ("py:class", "MetaDataCollectionType"),
    ("py:class", "MetaDataLike"),
    ("py:class", "MetaDataMatrix"),
    ("py:class", "MetaDataDictLike"),
    ("py:class", "Quantity"),
    ("py:class", "subset"),
    ("py:class", "file-like"),
    ("py:class", "copy"),
    ("py:class", "self"),
    # Mixins are handled by nitpick_ignore_regex below
    # External not mapped (most handled by regex)
    # Docstring fragments that Sphinx misinterprets as cross-references
    ("py:class", "fmin"),
    ("py:class", "fmax"),
    ("py:class", "Series"),
    ("py:class", "Unit"),
    ("py:class", "u.Unit"),
    ("py:class", "Colormap"),
    ("py:class", "Plot"),
    ("py:class", "Array4D"),
    ("py:class", "AverageTFR"),
    # GWPy internal (some not in intersphinx)
    ("py:class", "gwpy.plot.Plot"),
    ("py:class", "gwpy.types.array2d.Array2D"),
    ("py:class", "gwpy.types.array.Array"),
    ("py:class", "a shallow copy of od"),
    # Fragments from dict.update etc.
    ("py:class", "None.  Update D from mapping/iterable E and F."),
    # Type aliases and internal classes
    ("py:class", "ArrayLike"),
    ("py:class", "GwpyTimeSeries"),
    ("py:class", "ArimaResult"),
    ("py:class", "BrucoResult"),
    ("py:class", "mne_object"),
    ("py:class", "data"),
    # gwexpy internal classes
    ("py:class", "gwexpy.timeseries._timeseries_legacy.TimeSeries"),
    ("py:class", "gwexpy.analysis.bruco.BrucoResult"),
    ("py:class", "gwexpy.signal.preprocessing.whitening.WhiteningModel"),
]

# Use regex patterns to suppress large categories of unresolved references
nitpick_ignore_regex = [
    # ----------------------------------------------------------------
    # Standard library and well-known external libraries
    # ----------------------------------------------------------------
    (r"py:class", r"collections(\.abc)?\..+"),
    (r"py:class", r"numpy(\..+)?"),
    (r"py:class", r"matplotlib(\..+)?"),
    (r"py:class", r"astropy\..+"),
    (r"py:class", r"gwpy\..+"),
    (r"py:class", r"pandas(\..+)?"),
    (r"py:class", r"scipy(\..+)?"),
    (r"py:meth", r"gwpy\..+"),
    (r"py:obj", r"gwpy\..+"),
    (r"py:class", r"enum\.Enum"),
    (r"py:class", r"abc\.ABC"),
    (r"py:class", r"astropy\.units\.core\.UnitBase"),
    (r"py:class", r"astropy\.units\.quantity\.Quantity"),
    # ----------------------------------------------------------------
    # gwexpy internal mixins, private classes, TypeVars, base classes
    # ----------------------------------------------------------------
    (r"py:class", r"gwexpy\..*Mixin$"),
    (r"py:class", r"gwexpy\..*\._[A-Za-z_]+"),  # private modules/classes e.g. _FS
    (r"py:class", r"gwexpy\.frequencyseries\.collections\._FS"),
    (r"py:obj", r"gwexpy\.frequencyseries\.collections\._FS"),
    (r"py:class", r"gwexpy\.types\.seriesmatrix_base\.SeriesMatrix"),
    (r"py:class", r"gwexpy\.fields\.base\.FieldBase"),
    # gwexpy internal functions referenced in docstrings
    (r"py:(func|obj)", r"gwexpy\.fields\.signal\..*"),
    # ----------------------------------------------------------------
    # numpy internal typing helpers (exposed by type hints)
    # ----------------------------------------------------------------
    (r"py:class", r"numpy\._typing\..*"),
    # ----------------------------------------------------------------
    # External libraries without intersphinx mapping
    # ----------------------------------------------------------------
    (r"py:class", r"control\..*"),
    (r"py:class", r"emcee\..*"),
    (r"py:class", r"mne\..*"),
    (r"py:class", r"obspy\..*"),
    (r"py:class", r"polars\..*"),
    (r"py:class", r"pyspeckit\..*"),
    (r"py:class", r"quantities\..*"),
    (r"py:class", r"simpeg\..*"),
    (r"py:class", r"specutils\..*"),
    (r"py:class", r"torch\..*"),
    # ----------------------------------------------------------------
    # Docstring fragments with curly braces or standard types
    # ----------------------------------------------------------------
    (r"py:class", r"\{.*"),  # starts with {
    (r"py:class", r".*\}$"),  # ends with }
    (r"py:class", r"'[a-zA-Z_/]+'\}?$"),  # e.g. 'Hz'}, 'rad/s'}
    (r"py:class", r"(array_like|ndarray|np\.ndarray|2D ndarray)"),
    (r"py:class", r"(callable|scalar|tuples?|dicts?|class|instance|cls|Object|module)"),
    # ----------------------------------------------------------------
    # default / description fragments
    # ----------------------------------------------------------------
    (r"py:class", r"default .*"),  # e.g. default "datetime"
    (r"py:class", r"default=.*"),
    (r"py:class", r"[0-9]+"),  # Any standalone number fragment
    # misc fragments
    (r"py:class", r"mne[ -]object"),
    (r"py:class", r"gwexpy[ -]object"),
    (r"py:class", r"type cls\."),
    # Double-quoted strings in docstrings (e.g. "gwpy", "velocity")
    (r"py:class", r'"[a-zA-Z0-9_/]+"'),
]


def html_page_context(app, pagename, templatename, context, doctree):
    """Set lang="ja" for Japanese pages so <html lang="ja"> is rendered correctly."""
    if "/ja/" in pagename or pagename.startswith("web/ja"):
        context["language"] = "ja"

    site_context = getattr(app.config, "html_context", {}) or {}
    site_title = getattr(app.config, "html_title", "")
    page_title = _build_social_title(context.get("title"), site_title)
    page_description = _extract_meta_content(
        context.get("metatags", ""),
        "description",
    ) or site_context.get("og_description", "")
    page_url = _build_page_url(getattr(app.config, "html_baseurl", ""), pagename)
    page_image = site_context.get("og_image", "")

    context.update(
        {
            "og_title": page_title,
            "og_description": page_description,
            "og_type": site_context.get("og_type", "website"),
            "og_url": page_url or site_context.get("og_url", ""),
            "og_image": page_image,
            "og_site_name": site_title,
            "twitter_card": site_context.get("twitter_card", "summary_large_image"),
            "twitter_title": page_title,
            "twitter_description": page_description,
            "twitter_image": page_image,
        }
    )


class _MetaTagParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.meta: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "meta":
            return

        attr_map = {name.lower(): value for name, value in attrs if value is not None}
        key = attr_map.get("name") or attr_map.get("property")
        content = attr_map.get("content")
        if key and content:
            self.meta[key] = content


def _extract_meta_content(metatags: str, key: str) -> str | None:
    parser = _MetaTagParser()
    parser.feed(metatags or "")
    return parser.meta.get(key)


def _build_social_title(page_title: str | None, site_title: str) -> str:
    normalized_page_title = (page_title or "").strip()
    normalized_site_title = site_title.strip()
    if not normalized_page_title:
        return normalized_site_title
    if not normalized_site_title or normalized_site_title in normalized_page_title:
        return normalized_page_title
    return f"{normalized_page_title} — {normalized_site_title}"


def _build_page_url(baseurl: str, pagename: str) -> str | None:
    normalized_baseurl = (baseurl or "").rstrip("/")
    if not normalized_baseurl:
        return None
    return urljoin(f"{normalized_baseurl}/", f"{pagename}.html")


def setup(app):
    app.connect("html-page-context", html_page_context)
    app.add_lexer("ipython3", PythonLexer)
