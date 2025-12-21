import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))

project = "gwexpy"
author = "gwexpy contributors"
copyright = f"{datetime.now():%Y}, gwexpy contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
