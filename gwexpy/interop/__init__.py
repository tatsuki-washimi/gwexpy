from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("LAL_DEBUG_LEVEL", "0")
# Force JAX to CPU if GPU is not fully setup to avoid warnings
if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"
from ._optional import require_optional
from .astropy_ import from_astropy_timeseries, to_astropy_timeseries
from .gwinc_ import from_gwinc_budget
from .meep_ import from_meep_hdf5
from .openems_ import from_openems_hdf5
from .emg3d_ import from_emg3d_field, from_emg3d_h5, to_emg3d_field
from .meshio_ import from_fenics_vtk, from_fenics_xdmf, from_meshio
from .metpy_ import from_metpy_dataarray
from .wrf_ import from_wrf_variable
from .harmonica_ import from_harmonica_grid
from .sdynpy_ import from_sdynpy_frf, from_sdynpy_shape, from_sdynpy_timehistory
from .sdypy_ import from_uff_dataset55, from_uff_dataset58
from .pyoma_ import from_pyoma_results
from .opensees_ import from_opensees_recorder
from .exudyn_ import from_exudyn_sensor
from .lal_ import (
    from_lal_frequencyseries,
    from_lal_timeseries,
    to_lal_frequencyseries,
    to_lal_timeseries,
)
from .pycbc_ import (
    from_pycbc_frequencyseries,
    from_pycbc_timeseries,
    to_pycbc_frequencyseries,
    to_pycbc_timeseries,
)
from .control_ import from_control_frd, from_control_response, to_control_frd
from .cupy_ import from_cupy, is_cupy_available, to_cupy
from .dask_ import from_dask, to_dask
from .finesse_ import from_finesse_frequency_response, from_finesse_noise
from .frequency import (
    from_hdf5_frequencyseries,
    from_pandas_frequencyseries,
    from_xarray_frequencyseries,
    to_hdf5_frequencyseries,
    to_pandas_frequencyseries,
    to_xarray_frequencyseries,
)
from .hdf5_ import from_hdf5, to_hdf5
from .jax_ import from_jax, to_jax
from .json_ import from_dict, from_json, to_dict, to_json
from .mne_ import from_mne, from_mne_raw, to_mne, to_mne_rawarray
from .neo_ import from_neo, to_neo
from .netcdf4_ import from_netcdf4, to_netcdf4
from .obspy_ import from_obspy, from_obspy_trace, to_obspy, to_obspy_trace
from .pandas_ import (
    from_pandas_dataframe,
    from_pandas_series,
    to_pandas_dataframe,
    to_pandas_series,
)
from .polars_ import (
    from_polars_dataframe,
    from_polars_dict,
    from_polars_series,
    to_polars_dataframe,
    to_polars_dict,
    to_polars_frequencyseries,
    to_polars_series,
)

# P2
from .pydub_ import from_pydub, to_librosa, to_pydub
from .pyroomacoustics_ import (
    from_pyroomacoustics_field,
    from_pyroomacoustics_mic_signals,
    from_pyroomacoustics_rir,
    from_pyroomacoustics_source,
    from_pyroomacoustics_stft,
    to_pyroomacoustics_source,
    to_pyroomacoustics_stft,
)
from .pyspeckit_ import from_pyspeckit, to_pyspeckit
from .pyspice_ import (
    from_pyspice_ac,
    from_pyspice_distortion,
    from_pyspice_noise,
    from_pyspice_transient,
)
from .quantities_ import from_quantity, to_quantity
from .root_ import (
    from_root,
    to_tgraph,
    to_th1d,
    to_th2d,
    to_tmultigraph,
    write_root_file,
)
from .simpeg_ import from_simpeg, to_simpeg
from .skrf_ import (
    from_skrf_impulse_response,
    from_skrf_network,
    from_skrf_step_response,
    to_skrf_network,
)
from .specutils_ import from_specutils, to_specutils
from .sqlite_ import from_sqlite, to_sqlite
from .tensorflow_ import from_tf, to_tf

# P1
from .torch_ import from_torch, to_torch
from .torch_dataset import (
    TimeSeriesWindowDataset,
    to_torch_dataloader,
    to_torch_dataset,
)
from .xarray_ import from_xarray, from_xarray_field, to_xarray, to_xarray_field
from .zarr_ import from_zarr, to_zarr

__all__ = [
    # pandas
    "to_pandas_series",
    "from_pandas_series",
    "to_pandas_dataframe",
    "from_pandas_dataframe",
    # xarray
    "to_xarray",
    "from_xarray",
    "to_xarray_field",
    "from_xarray_field",
    # hdf5
    "to_hdf5",
    "from_hdf5",
    # obspy
    "to_obspy_trace",
    "from_obspy_trace",
    "to_obspy",
    "from_obspy",
    # specutils
    "to_specutils",
    "from_specutils",
    # pyspeckit
    "to_pyspeckit",
    "from_pyspeckit",
    # polars
    "to_polars_series",
    "to_polars_dataframe",
    "from_polars_series",
    "from_polars_dataframe",
    "to_polars_frequencyseries",
    "to_polars_dict",
    "from_polars_dict",
    # root
    "to_tgraph",
    "to_th1d",
    "to_th2d",
    "from_root",
    "to_tmultigraph",
    "write_root_file",
    # sqlite
    "to_sqlite",
    "from_sqlite",
    # optional
    "require_optional",
    # frequency series
    "to_pandas_frequencyseries",
    "from_pandas_frequencyseries",
    "to_xarray_frequencyseries",
    "from_xarray_frequencyseries",
    "to_hdf5_frequencyseries",
    "from_hdf5_frequencyseries",
    # P1 - torch
    "to_torch",
    "from_torch",
    # P1 - tensorflow
    "to_tf",
    "from_tf",
    # P1 - dask
    "to_dask",
    "from_dask",
    # P1 - zarr
    "to_zarr",
    "from_zarr",
    # P1 - netcdf4
    "to_netcdf4",
    "from_netcdf4",
    # P1 - control
    "to_control_frd",
    "from_control_frd",
    "from_control_response",
    # P1 - finesse
    "from_finesse_frequency_response",
    "from_finesse_noise",
    # P1 - pyspice
    "from_pyspice_transient",
    "from_pyspice_ac",
    "from_pyspice_noise",
    "from_pyspice_distortion",
    # P1 - skrf
    "from_skrf_network",
    "to_skrf_network",
    "from_skrf_impulse_response",
    "from_skrf_step_response",
    # P1 - pyroomacoustics
    "from_pyroomacoustics_rir",
    "from_pyroomacoustics_mic_signals",
    "from_pyroomacoustics_source",
    "from_pyroomacoustics_stft",
    "from_pyroomacoustics_field",
    "to_pyroomacoustics_source",
    "to_pyroomacoustics_stft",
    # P1 - jax
    "to_jax",
    "from_jax",
    # P1 - cupy
    "to_cupy",
    "from_cupy",
    "is_cupy_available",
    # P2 - audio
    "to_librosa",
    "to_pydub",
    "from_pydub",
    # P2 - astropy
    "to_astropy_timeseries",
    "from_astropy_timeseries",
    # P2 - mne
    "to_mne_rawarray",
    "from_mne_raw",
    "to_mne",
    "from_mne",
    # P2 - simpeg
    "to_simpeg",
    "from_simpeg",
    # P2 - neo
    "to_neo",
    "from_neo",
    # P2 - quantities
    "to_quantity",
    "from_quantity",
    # P2 - json
    "to_json",
    "from_json",
    "to_dict",
    "from_dict",
    # gw - gwinc
    "from_gwinc_budget",
    # meep
    "from_meep_hdf5",
    # openems
    "from_openems_hdf5",
    # emg3d
    "from_emg3d_field",
    "from_emg3d_h5",
    "to_emg3d_field",
    # meshio
    "from_meshio",
    "from_fenics_xdmf",
    "from_fenics_vtk",
    # metpy / wrf / harmonica
    "from_metpy_dataarray",
    "from_wrf_variable",
    "from_harmonica_grid",
    # gw - lal
    "from_lal_timeseries",
    "to_lal_timeseries",
    "from_lal_frequencyseries",
    "to_lal_frequencyseries",
    # gw - pycbc
    "from_pycbc_timeseries",
    "to_pycbc_timeseries",
    "from_pycbc_frequencyseries",
    "to_pycbc_frequencyseries",
    # torch dataset
    "TimeSeriesWindowDataset",
    "to_torch_dataset",
    "to_torch_dataloader",
    # modal - sdynpy
    "from_sdynpy_shape",
    "from_sdynpy_frf",
    "from_sdynpy_timehistory",
    # modal - sdypy/pyuff
    "from_uff_dataset58",
    "from_uff_dataset55",
    # modal - pyoma
    "from_pyoma_results",
    # modal - opensees
    "from_opensees_recorder",
    # modal - exudyn
    "from_exudyn_sensor",
]
