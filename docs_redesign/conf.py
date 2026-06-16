# Configuration file for the Sphinx documentation builder.
#
# Redesign Variant B3: sphinx-book-theme (Jupyter Book style)
# Greenfield prototype -- does NOT import gwexpy.

import os
import sys

# Make the tiny dummy demo API importable for autosummary/autodoc.
sys.path.insert(0, os.path.abspath("_demo_api"))

# -- Project information -----------------------------------------------------

project = "GWexpy"
copyright = "2026, GWexpy contributors"
author = "GWexpy contributors"

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

# myst-nb: do not execute notebooks during the build.
nb_execution_mode = "off"

# MyST extensions used in the prototype pages.
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "linkify",
    "substitution",
]

# Treat .md as MyST and .ipynb / MyST-notebooks via myst-nb.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "gwpy": ("https://gwpy.github.io/docs/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "GWexpy"
html_logo = "_static/logo.svg"
html_favicon = "_static/favicon.svg"

html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    # Repository / source buttons in the header.
    "repository_url": "https://github.com/tatsuki-washimi/gwexpy",
    "repository_branch": "main",
    "path_to_docs": "docs_redesign",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "use_edit_page_button": True,
    # Launch buttons (Colab / Binder-style) on notebook pages.
    "launch_buttons": {
        "notebook_interface": "jupyterlab",
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
        "thebe": False,
    },
    # Book-style left TOC heading + right in-page TOC depth.
    "toc_title": "On this book",
    "show_toc_level": 2,
    "home_page_in_toc": True,
    "show_navbar_depth": 1,
    "max_navbar_depth": 3,
    "announcement": (
        "This is a greenfield design prototype (Variant B3, sphinx-book-theme)."
    ),
}

html_baseurl = "https://example.org/gwexpy/"
