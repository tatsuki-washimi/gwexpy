# Configuration file for the Sphinx documentation builder.
#
# Redesign Variant B2: furo (minimal) — greenfield prototype.
# This is a standalone prototype; it does NOT import gwexpy. A tiny dummy
# module under _demo_api/ provides real autosummary/autodoc content.

import os
import sys
from datetime import datetime
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent

# Make the dummy demo API importable so autosummary/autodoc can document it.
sys.path.insert(0, str(DOCS_DIR / "_demo_api"))

# -- Project information -----------------------------------------------------

project = "GWexpy"
author = "GWexpy contributors"
copyright = f"{datetime.now():%Y}, GWexpy contributors"
release = "dev"
version = "dev"

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

# MyST configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "smartquotes",
]
myst_heading_anchors = 3

# myst-nb: do not execute notebooks during the build.
nb_execution_mode = "off"

# autosummary / autodoc
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

source_suffix = {
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".rst": "restructuredtext",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "_demo_api"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "GWexpy"
html_logo = "_static/branding/logo.svg"
html_favicon = "_static/branding/icon.svg"

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# Brand accent — a calm GW-violet/indigo, tuned for furo light + dark.
_BRAND = "#6d6acb"
_BRAND_DARK = "#9b97ff"

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_buttons": [],
    "light_css_variables": {
        "color-brand-primary": _BRAND,
        "color-brand-content": _BRAND,
        "font-stack": (
            "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, "
            "sans-serif, 'Noto Sans CJK JP', 'Hiragino Sans', sans-serif"
        ),
        "color-admonition-title-background--note": "rgba(109, 106, 203, 0.1)",
    },
    "dark_css_variables": {
        "color-brand-primary": _BRAND_DARK,
        "color-brand-content": _BRAND_DARK,
        "color-background-primary": "#16161d",
        "color-background-secondary": "#1c1c25",
    },
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/tatsuki-washimi/gwexpy",
            "html": (
                '<svg stroke="currentColor" fill="currentColor" '
                'stroke-width="0" viewBox="0 0 16 16">'
                '<path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 '
                "2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 "
                "0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-"
                ".94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82."
                "72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-."
                "89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 "
                "0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 "
                "1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 "
                "1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 "
                "0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 "
                '8c0-4.42-3.58-8-8-8z"></path></svg>'
            ),
            "class": "",
        },
    ],
}

# A minimal announcement banner (furo feature).
html_theme_options["announcement"] = (
    "Prototype redesign &mdash; Variant B2 (furo, minimal)."
)

html_show_sourcelink = False
