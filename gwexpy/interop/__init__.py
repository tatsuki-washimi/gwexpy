from .pandas_ import to_pandas_series, from_pandas_series, to_pandas_dataframe, from_pandas_dataframe
from .xarray_ import to_xarray, from_xarray
from .hdf5_ import to_hdf5, from_hdf5
from .obspy_ import to_obspy_trace, from_obspy_trace, to_obspy, from_obspy
from .specutils_ import to_specutils, from_specutils
from .pyspeckit_ import to_pyspeckit, from_pyspeckit
from .polars_ import (
    to_polars_series,
    to_polars_dataframe,
    from_polars_series,
    from_polars_dataframe,
    to_polars_frequencyseries,
    to_polars_dict,
    from_polars_dict,
)
from .root_ import to_tgraph, to_th1d, to_th2d, from_root, to_tmultigraph, write_root_file
from .sqlite_ import to_sqlite, from_sqlite
from ._optional import require_optional
from .frequency import (
    to_pandas_frequencyseries,
    from_pandas_frequencyseries,
    to_xarray_frequencyseries,
    from_xarray_frequencyseries,
    to_hdf5_frequencyseries,
    from_hdf5_frequencyseries,
)

# P1
from .torch_ import to_torch, from_torch
from .tensorflow_ import to_tf, from_tf
from .dask_ import to_dask, from_dask
from .zarr_ import to_zarr, from_zarr
from .netcdf4_ import to_netcdf4, from_netcdf4
from .control_ import to_control_frd, from_control_frd, from_control_response
from .jax_ import to_jax, from_jax
from .cupy_ import to_cupy, from_cupy, is_cupy_available

# P2
from .pydub_ import to_librosa, to_pydub, from_pydub
from .astropy_ import to_astropy_timeseries, from_astropy_timeseries
from .mne_ import to_mne_rawarray, from_mne_raw, to_mne, from_mne
from .simpeg_ import to_simpeg, from_simpeg
from .neo_ import to_neo, from_neo
from .quantities_ import to_quantity, from_quantity
from .json_ import to_json, from_json, to_dict, from_dict
from .torch_dataset import TimeSeriesWindowDataset, to_torch_dataset, to_torch_dataloader

__all__ = [
    # pandas
    "to_pandas_series", "from_pandas_series", "to_pandas_dataframe", "from_pandas_dataframe",
    # xarray
    "to_xarray", "from_xarray",
    # hdf5
    "to_hdf5", "from_hdf5",
    # obspy
    "to_obspy_trace", "from_obspy_trace", "to_obspy", "from_obspy",
    # specutils
    "to_specutils", "from_specutils",
    # pyspeckit
    "to_pyspeckit", "from_pyspeckit",
    # polars
    "to_polars_series", "to_polars_dataframe", "from_polars_series", "from_polars_dataframe",
    "to_polars_frequencyseries", "to_polars_dict", "from_polars_dict",
    # root
    "to_tgraph", "to_th1d", "to_th2d", "from_root", "to_tmultigraph", "write_root_file",
    # sqlite
    "to_sqlite", "from_sqlite",
    # optional
    "require_optional",
    # frequency series
    "to_pandas_frequencyseries", "from_pandas_frequencyseries",
    "to_xarray_frequencyseries", "from_xarray_frequencyseries",
    "to_hdf5_frequencyseries", "from_hdf5_frequencyseries",
    # P1 - torch
    "to_torch", "from_torch",
    # P1 - tensorflow
    "to_tf", "from_tf",
    # P1 - dask
    "to_dask", "from_dask",
    # P1 - zarr
    "to_zarr", "from_zarr",
    # P1 - netcdf4
    "to_netcdf4", "from_netcdf4",
    # P1 - control
    "to_control_frd", "from_control_frd", "from_control_response",
    # P1 - jax
    "to_jax", "from_jax",
    # P1 - cupy
    "to_cupy", "from_cupy", "is_cupy_available",
    # P2 - audio
    "to_librosa", "to_pydub", "from_pydub",
    # P2 - astropy
    "to_astropy_timeseries", "from_astropy_timeseries",
    # P2 - mne
    "to_mne_rawarray", "from_mne_raw", "to_mne", "from_mne",
    # P2 - simpeg
    "to_simpeg", "from_simpeg",
    # P2 - neo
    "to_neo", "from_neo",
    # P2 - quantities
    "to_quantity", "from_quantity",
    # P2 - json
    "to_json", "from_json", "to_dict", "from_dict",
    # torch dataset
    "TimeSeriesWindowDataset", "to_torch_dataset", "to_torch_dataloader",
]
