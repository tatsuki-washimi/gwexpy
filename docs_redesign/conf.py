# Configuration file for the Sphinx documentation builder.
# Redesign Variant B1: pydata-sphinx-theme + Diataxis (greenfield prototype).

import os
import sys

# Make the self-contained demo module importable so autodoc/autosummary can
# generate real API tables WITHOUT installing gwexpy.
sys.path.insert(0, os.path.abspath("_demo_api"))

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

# Do not execute notebooks in this prototype.
nb_execution_mode = "off"

# Autosummary generates stub pages automatically.
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False

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
