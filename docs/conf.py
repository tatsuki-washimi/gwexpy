import os
import sys
import warnings
from datetime import datetime

from pygments.lexers.python import PythonLexer

sys.path.insert(0, os.path.abspath(".."))

project = "GWexpy"
author = "GWexpy contributors"
copyright = f"{datetime.now():%Y}, GWexpy contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "nbsphinx",
]

# nbsphinx configuration
# Default to skipping execution so docs builds don't require a registered Jupyter kernel.
nbsphinx_execute = os.environ.get("NBS_EXECUTE", "never")
nbsphinx_allow_errors = True  # Allow notebooks with execution errors to be included

autosummary_generate = True

# Add a consistent download link on notebook pages.
nbsphinx_prolog = r"""

.. note::
   This page was generated from a Jupyter Notebook.
   `Download the notebook (.ipynb) <https://github.com/tatsuki-washimi/gwexpy/raw/main/docs/{{ env.doc2path(env.docname, base=None) }}>`_.

"""

# Ignore nitpick errors for well-known external symbols to keep -n builds manageable.
nitpick_ignore = []
nitpick_ignore_regex = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "developers/**"]

language = "en"
locale_dirs = ["locales/"]
gettext_compact = False

html_theme = "sphinx_rtd_theme"

# User-facing site title/branding (package name remains `gwexpy`).
html_title = "GWexpy Documentation"
html_short_title = "GWexpy"
html_context = {
    "display_github": True,
    "github_user": "tatsuki-washimi",
    "github_repo": "gwexpy",
    "github_version": "main",
    "conf_py_path": "/docs/",
    "current_language": language,
    # GitHub Pages publishes docs under /gwexpy/; user-facing docs live in /docs/web/{en,ja}/.
    "languages": [("en", "/gwexpy/docs/web/en/"), ("ja", "/gwexpy/docs/web/ja/")],
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

# MyST: enable heading anchors so internal markdown `#...` links work.
myst_heading_anchors = 3

# nbformat emits noisy warnings when notebooks omit cell IDs (valid for older nbformat).
try:  # pragma: no cover
    from nbformat.warnings import MissingIDFieldWarning

    warnings.filterwarnings("ignore", category=MissingIDFieldWarning)
except Exception:
    pass

intersphinx_mapping = {}

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
    ("py:class", "file-like"),
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
    # Docstring fragments that Sphinx misinterprets as cross-references
    ("py:class", "array_like"),
    ("py:class", "array"),
    ("py:class", "ndarray"),
    ("py:class", "np.ndarray"),
    ("py:class", "2D ndarray"),
    ("py:class", "callable"),
    ("py:class", "scalar"),
    ("py:class", "tuples"),
    ("py:class", "dicts"),
    ("py:class", "class"),
    ("py:class", "instance"),
    ("py:class", "cls"),
    ("py:class", "Object"),
    ("py:class", "module"),
    ("py:class", "fmin"),
    ("py:class", "fmax"),
    ("py:class", "Series"),
    ("py:class", "Unit"),
    ("py:class", "u.Unit"),
    ("py:class", "Colormap"),
    ("py:class", "Plot"),
    ("py:class", "Array4D"),
    ("py:class", "AverageTFR"),
    # Default value fragments
    ("py:class", "default=True"),
    ("py:class", "default=95"),
    ("py:class", "default=2.0"),
    ("py:class", "0"),
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
    ("py:class", "1"),
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
    # Docstring fragments with curly braces (Napoleon parses as refs)
    # e.g. {'Hz', 'rad/s'} -> {'Hz' and 'rad/s'} become refs
    # ----------------------------------------------------------------
    (r"py:class", r"\{.*"),  # starts with {
    (r"py:class", r".*\}$"),  # ends with }
    (r"py:class", r"'[a-zA-Z_/]+'\}?$"),  # e.g. 'Hz'}, 'rad/s'}
    # ----------------------------------------------------------------
    # default / description fragments
    # ----------------------------------------------------------------
    (r"py:class", r"default .*"),  # e.g. default "datetime"
    (r"py:class", r"default=.*"),
    # misc fragments
    (r"py:class", r"mne object"),
    (r"py:class", r"gwexpy object"),
    (r"py:class", r"type cls\."),
    # Double-quoted strings in docstrings (e.g. "gwpy", "velocity")
    (r"py:class", r'"[a-zA-Z0-9_/]+"'),
]


def setup(app):
    app.add_lexer("ipython3", PythonLexer)
