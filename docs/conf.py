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

autodoc_mock_imports = [
    "pycbc",
    "nds2",
    "framel",
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
