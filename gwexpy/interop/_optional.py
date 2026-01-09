
import importlib
from typing import Any

_OPTIONAL_DEPENDENCIES = {
    "pandas": "pandas",
    "xarray": "xarray",
    "h5py": "h5py",
    "obspy": "obspy",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "tensorflow": "tensorflow",
    "dask": "dask",
    "dask.array": "dask.array",
    "zarr": "zarr",
    "netCDF4": "netCDF4",
    "control": "control",
    "jax": "jax",
    "cupy": "cupy",
    "librosa": "librosa",
    "pydub": "pydub",
    "astropy": "astropy",
    "mth5": "mth5",
    "mt_metadata": "mt_metadata",
    "mtpy": "mtpy",
    "mne": "mne",
    "neo": "neo",
    "dttxml": "dttxml",
    "gwinc": "gwinc",
    "joblib": "joblib",
}

def require_optional(name: str) -> Any:
    """
    Import an optional dependency or raise an informative ImportError.

    Parameters
    ----------
    name : str
        Key name of the dependency (e.g., 'pandas').

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    ImportError
        If the package is not installed.
    """
    if name not in _OPTIONAL_DEPENDENCIES:
        # Fallback: assume package name matches request if not in map
        pkg_name = name
    else:
        pkg_name = _OPTIONAL_DEPENDENCIES[name]

    # Map package name to help message for installation
    # Aligned with pyproject.toml [project.optional-dependencies]
    _EXTRA_MAP = {
        # interop: Deep Learning & Big Data frameworks
        "torch": "interop",
        "torchaudio": "audio",
        "tensorflow": "interop",
        "jax": "interop",
        "dask": "interop",
        "zarr": "interop",
        "cupy": "interop",
        "xarray": "interop",
        # stats: Statistical analysis
        "polars": "stats",
        "statsmodels": "stats",
        # fitting: Curve fitting and MCMC
        "iminuit": "fitting",
        # audio: Audio processing
        "librosa": "audio",
        "pydub": "audio",
        # geophysics: Earth science
        "obspy": "geophysics",
        "mth5": "geophysics",
        "netCDF4": "geophysics",
        # bio: Bioscience
        "mne": "bio",
        "neo": "bio",
        # control: Control systems
        "control": "control",
        # analysis: Signal analysis
        "PyEMD": "analysis",
        "pywt": "analysis",
        # gw: Gravitational wave specific
        "dttxml": "gw",
        "gwinc": "gw",
        "joblib": "stats",
    }

    try:
        return importlib.import_module(pkg_name)
    except ImportError as e:
        extra = _EXTRA_MAP.get(name) or _EXTRA_MAP.get(pkg_name)
        if extra:
            install_cmd = f"pip install 'gwexpy[{extra}]'"
        else:
            install_cmd = f"pip install {pkg_name}"

        raise ImportError(
            f"The '{name}' package is required for this feature but is not installed. "
            f"You can install it via '{install_cmd}' or 'pip install \"gwexpy[all]\"'."
        ) from e
