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
    ("py:class", "numpy.dtype"),
    ("py:class", "numpy.typing.ArrayLike"),
    ("py:class", "numpy.typing.DTypeLike"),
    ("py:class", "optional"),
    ("py:class", "iterable"),
    ("py:class", "array-like"),
    ("py:class", "u.Quantity"),  # Sphinx sometimes misses the alias
    ("py:class", "UnitLike"),
    ("py:class", "IndexLike"),
    ("py:class", "MetaDataCollectionType"),
    ("py:class", "MetaDataLike"),
    ("py:class", "MetaDataMatrix"),
    ("py:class", "MetaDataDictLike"),
    ("py:class", "Quantity"),
    ("py:class", "subset"),
    ("py:class", "copy"),
    ("py:class", "self"),
    # Mixins (often not exported to top level documentation)
    ("py:class", "gwexpy.types.mixin.mixin_legacy.RegularityMixin"),
    ("py:class", "gwexpy.types.mixin.signal_interop.InteropMixin"),
    ("py:class", "gwexpy.types.series_matrix_core.SeriesMatrixCoreMixin"),
    ("py:class", "gwexpy.types.series_matrix_indexing.SeriesMatrixIndexingMixin"),
    ("py:class", "gwexpy.types.series_matrix_io.SeriesMatrixIOMixin"),
    ("py:class", "gwexpy.types.series_matrix_math.SeriesMatrixMathMixin"),
    ("py:class", "gwexpy.types.series_matrix_analysis.SeriesMatrixAnalysisMixin"),
    ("py:class", "gwexpy.types.series_matrix_structure.SeriesMatrixStructureMixin"),
    (
        "py:class",
        "gwexpy.types.series_matrix_visualization.SeriesMatrixVisualizationMixin",
    ),
    (
        "py:class",
        "gwexpy.types.series_matrix_validation_mixin.SeriesMatrixValidationMixin",
    ),
    ("py:class", "gwexpy.types._stats.StatisticalMethodsMixin"),
    # External not mapped
    ("py:class", "pandas.core.frame.DataFrame"),
    ("py:class", "torch.Tensor"),
    ("py:class", "torch.dtype"),
]
