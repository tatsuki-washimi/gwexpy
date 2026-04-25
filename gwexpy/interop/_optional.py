from __future__ import annotations

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
    "finesse": "finesse",
    "joblib": "joblib",
    "PySpice": "PySpice",
    "skrf": "skrf",
    "pyroomacoustics": "pyroomacoustics",
    "pycbc": "pycbc",
    "meep": "meep",
    "openems": "openems",
    "emg3d": "emg3d",
    "meshio": "meshio",
    "scipy": "scipy",
    "metpy": "metpy",
    "wrf": "wrf",
    "harmonica": "harmonica",
    "pint": "pint",
    "sdynpy": "sdynpy",
    "pyuff": "pyuff",
    "pyOMA": "pyOMA",
    "openseespy": "openseespy",
    "exudyn": "exudyn",
    "multitaper": "multitaper",
    "mtspec": "mtspec",
}


def require_optional(name: str) -> Any:
    """Import an optional dependency or raise an informative ImportError.

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
        # analysis: Signal analysis and statistics
        "scikit-learn": "analysis",
        "sklearn": "analysis",
        "statsmodels": "analysis",
        "pmdarima": "analysis",
        "dcor": "analysis",
        "hurst": "analysis",
        "hurst-exponent": "analysis",
        "exp-hurst": "analysis",
        "EMD-signal": "analysis",
        "PyEMD": "analysis",
        "PyWavelets": "analysis",
        "pywt": "analysis",
        # fitting: Curve fitting and MCMC
        "iminuit": "fitting",
        "emcee": "fitting",
        "corner": "fitting",
        # control: Control systems
        "control": "control",
        # seismic: Seismic and magnetotelluric data
        "obspy": "seismic",
        "mth5": "seismic",
        "mtpy": "seismic",
        "mt_metadata": "seismic",
        # gw: Gravitational wave data access and tools
        "lalsuite": "gw",
        "lal": "gw",
        "pycbc": "gw",
        "gwdatafind": "gw",
        "gwosc": "gw",
        "dqsegdb2": "gw",
        "dttxml": "gw",
        "gwinc": "gw",
        "finesse": "gw",
        "ligo.skymap": "gw",
        # io: Experimental data I/O
        "nptdms": "io",
        # netcdf4: NetCDF4 / xarray time-series I/O
        "netCDF4": "netcdf4",
        "xarray": "netcdf4",
        # zarr: Zarr store I/O
        "zarr": "zarr",
        # plotting: Advanced plotting
        "pygmt": "plotting",
        # audio: Audio processing
        "pydub": "audio",
        "tinytag": "audio",
        "librosa": "audio",
        "pyroomacoustics": "audio",
        # gui: GUI components
        "PyQt5": "gui",
        "pyqtgraph": "gui",
        "qtpy": "gui",
        "sounddevice": "gui",
        # No declared extra — install these packages directly
        "PySpice": None,
        "skrf": None,
        "torch": None,
        "torchaudio": None,
        "tensorflow": None,
        "jax": None,
        "dask": None,
        "cupy": None,
        "mne": None,
        "neo": None,
        "polars": None,
        "joblib": None,
    }

    try:
        return importlib.import_module(pkg_name)
    except ImportError as e:
        # Prefer name lookup; fall back to pkg_name only if name is not explicitly mapped.
        _SENTINEL = object()
        _raw = _EXTRA_MAP.get(name, _SENTINEL)
        if _raw is _SENTINEL:
            _raw = _EXTRA_MAP.get(pkg_name)
        extra = _raw  # may be a string or None

        if extra:
            install_cmd = f"pip install 'gwexpy[{extra}]'"
            also_all = " or 'pip install \"gwexpy[all]\"'"
        else:
            install_cmd = f"pip install {pkg_name}"
            also_all = ""

        raise ImportError(
            f"The '{name}' package is required for this feature but is not installed. "
            f"You can install it via '{install_cmd}'{also_all}."
        ) from e
