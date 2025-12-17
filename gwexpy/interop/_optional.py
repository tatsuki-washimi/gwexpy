
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
    "zarr": "zarr",
    "netCDF4": "netCDF4",
    "control": "control",
    "jax": "jax",
    "cupy": "cupy",
    "librosa": "librosa",
    "pydub": "pydub",
    "wintools": "wintools",
    "win2ndarray": "win2ndarray",
    "astropy": "astropy",
    "mth5": "mth5",
    "mt_metadata": "mt_metadata",
    "mtpy": "mtpy",
    "mne": "mne",
    "neo": "neo",
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
        
    try:
        return importlib.import_module(pkg_name)
    except ImportError as e:
        raise ImportError(
            f"The '{name}' package is required for this feature but is not installed. "
            f"Please install it via 'pip install {pkg_name}' or 'conda install {pkg_name}'."
        ) from e
