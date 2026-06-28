# Configuration file for the Sphinx documentation builder.
# Redesign Variant B1: pydata-sphinx-theme + Diataxis (greenfield prototype).

# The real ``gwexpy`` package is installed in the build environment, so
# autodoc/autosummary document the actual API directly (no demo stub).

# -- Project information -----------------------------------------------------
project = "GWexpy"
copyright = "2026, GWexpy Developers"
author = "GWexpy Developers"
release = "0.0.0"

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

# MyST extensions for richer Markdown authoring.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "tasklist",
]
# Resolve in-page Markdown anchor links (e.g. [text](#section)).
myst_heading_anchors = 3

# Do not execute notebooks in this prototype.
nb_execution_mode = "off"

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
.. _lal.ligotimegps: https://lscsoft.docs.ligo.org/lalsuite/lal/
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
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".rst": "restructuredtext",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "gwpy": ("https://gwpy.github.io/docs/stable/", None),
}

# -- HTML output -------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/branding/logo.svg"
html_favicon = "_static/images/favicon.svg"
html_title = "GWexpy"

html_theme_options = {
    # Top navbar layout (Diataxis sections live in the center).
    "navbar_start": ["navbar-logo", "version-switcher"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links", "language-switcher.html"],
    "navbar_persistent": ["search-button"],
    # Stub version switcher (proves the control renders).
    "switcher": {
        "json_url": "_static/switcher.json",
        "version_match": release,
    },
    "show_version_warning_banner": True,
    # Secondary (right) sidebar with in-page table of contents.
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "show_toc_level": 2,
    # Icon links.
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/gwexpy/gwexpy",
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
    "github_user": "gwexpy",
    "github_repo": "gwexpy",
    "github_version": "main",
    "doc_path": "docs_redesign",
    "default_mode": "auto",
}

# Sidebar: keep the left sidebar clean (navigation only).
html_sidebars = {
    "index": [],  # Homepage: no left sidebar for a clean hero.
}
