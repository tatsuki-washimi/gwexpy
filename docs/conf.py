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
    "myst_parser",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"
locale_dirs = ["locales/"]
gettext_compact = False

html_theme = "sphinx_rtd_theme"
html_context = {
    "display_github": True,
    "github_user": "tatsuki-washimi",
    "github_repo": "gwexpy",
    "github_version": "main",
    "conf_py_path": "/docs/",
    "current_language": language,
    "languages": [("en", "/gwexpy/en/"), ("ja", "/gwexpy/ja/")],
}

autodoc_mock_imports = [
    "pycbc",
    "gwinc",
    "polars",
    "sklearn",
    "statsmodels",
    "pmdarima",
    "mictools",
    "dcor",
    "hurst",
    "bottleneck",
    "iminuit",
    "emcee",
    "corner",
    "specutils",
    "pyspeckit",
    "obspy",
    "mth5",
    "mtpy",
    "librosa",
    "pydub",
    "torchaudio",
    "mne",
    "neo",
    "elephant",
    "torch",
    "tensorflow",
    "jax",
    "dask",
    "zarr",
    "cupy",
    "xarray",
    "control",
    "PyQt5",
    "pyqtgraph",
    "qtpy",
    "pygmt",
    "PyEMD",
    "pywt",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "gwpy": ("https://gwpy.github.io/docs/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
}

nitpick_ignore = [
    ("py:class", "numpy.dtype"),  # Often fails even with intersphinx
    ("py:class", "numpy.typing.ArrayLike"),
    ("py:class", "numpy.typing.DTypeLike"),
]
