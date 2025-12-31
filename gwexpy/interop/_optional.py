
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
    _EXTRA_MAP = {
        "torch": "torch",
        "torchaudio": "audio",
        "tensorflow": "tensorflow",
        "jax": "jax",
        "dask": "dask",
        "zarr": "zarr",
        "polars": "polars",
        "xarray": "data",
        "h5py": "data",
        "netCDF4": "data",
        "librosa": "audio",
        "pydub": "audio",
        "obspy": "geophysics",
        "mth5": "geophysics",
        "mne": "bio",
        "neo": "bio",
        "control": "control",
        "iminuit": "stats",
        "statsmodels": "stats",
        "PyEMD": "analysis",
        "hurst": "analysis",
        "pywt": "analysis",
        "dttxml": "gw",
        "gwinc": "gw",
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
