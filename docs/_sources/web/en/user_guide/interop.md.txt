---
orphan: true
myst:
  html_meta:
    description: "Find the right GWexpy interop path for converting objects to and from storage containers, analysis libraries, ML backends, and domain-specific tools."
---

# Interop / Conversion Guide

> **Page Role:** Guide

This page is the dedicated **interop guide** for `gwexpy`.  
Here, interop means **conversion and bridging** centered on `to_*()` / `from_*()` APIs.

This page covers:

- object conversion via `to_*()` / `from_*()`
- bridges to external libraries and external data models
- object-level file-bridge helpers
- asymmetric conversion APIs
- conversions that apply only to some classes

This page does not cover:

- `Class.read(..., format=...)`
- `obj.write(..., format=...)`
- the choice of local file formats themselves

For local file formats and direct I/O, see the [File I/O Supported Formats Guide](io_formats).

## At a Glance

| Item | Details |
| --- | --- |
| **Audience** | Users and contributors who need to move `gwexpy` objects into other libraries, containers, or storage representations |
| **Prerequisites** | Basic familiarity with `gwexpy` object types, direct I/O versus interop, and the target library or container you want to use |
| **Use Cases** | Choose a `to_*()` / `from_*()` path, check whether a bridge is public yet, and distinguish storage conversion from object conversion |
| **Search Keywords** | interop, conversion, `to_*`, `from_*`, xarray, pandas, ROOT, Zarr, NetCDF4, PyTorch |

**Search hints:** interop, conversion, `to_*`, `from_*`, xarray, pandas, ROOT, Zarr, NetCDF4, PyTorch

## Jump Links

- [How to Read This Page](#how-to-read-this-page)
- [Status Labels](#status-labels)
- [A. Storage Formats and Container Conversion](#a-storage-formats-and-container-conversion)
- [B. Analysis Library and Object Conversion](#b-analysis-library-and-object-conversion)
- [C. Machine Learning, Acceleration, and Array Backends](#c-machine-learning-acceleration-and-array-backends)
- [D. Physics and Domain-Specific Libraries](#d-physics-and-domain-specific-libraries)
- [What to Prioritize First](#what-to-prioritize-first)

(interop-en-how-to-read)=
## How to Read This Page

- If you want to convert to a **storage format or container**, start with A.
- If you want to convert to **analysis objects** such as pandas, xarray, astropy, or dask, start with B.
- If you want to hand data to **PyTorch, TensorFlow, JAX, or CuPy**, start with C.
- If you want to connect to **ROOT, ObsPy, LAL, PyCBC**, or other domain-specific libraries, start with D.
- If you want to move **Field** objects into xarray, NetCDF4, or Zarr workflows, treat that as interop, not as direct I/O.

(interop-en-status-labels)=
## Status Labels

- `Public`: implemented and reachable from `reference/api/interop`
- `Implemented, public cleanup pending`: implemented, but the public documentation is not fully organized yet
- `Implemented, some paths still in progress`: the main route works, but some conversion paths are still incomplete
- `In progress`: the implementation or the public presentation is not finished yet
- `Planned`: explicitly in scope, but not implemented yet

(interop-en-storage-conversion)=
## A. Storage Formats and Container Conversion

This section is for conversions where the target is a **file format, container, or storage representation**.  
Use it when the question is “what storage representation do I bridge to?”

- Purpose: identify object-level bridges whose target is a storage representation
- Input: a `gwexpy` object plus a destination container or storage backend
- Output: a converted object, container, or storage-facing representation via `to_*()` / `from_*()`

| Target | API / Entry | Status | Notes | Details |
| --- | --- | --- | --- | --- |
| HDF5 | `to_hdf5()`, `from_hdf5()` | Public | object-level conversion | [API](../reference/api/gwexpy.interop.hdf5_.rst) |
| JSON | `to_json()`, `from_json()` | Public | JSON string conversion | [API](../reference/api/gwexpy.interop.json_.rst) |
| Python dict | `to_dict()`, `from_dict()` | Public | dict conversion | — |
| SQLite | `to_sqlite()`, `from_sqlite()` | Implemented, public cleanup pending | object-level bridge | — |
| Zarr | `to_zarr()`, `from_zarr()` | Public | array/store bridge | [API](../reference/api/gwexpy.interop.zarr_.rst) |
| NetCDF4 | `to_netcdf4()`, `from_netcdf4()` | Public | object-level bridge | [API](../reference/api/gwexpy.interop.netcdf4_.rst) |

(interop-en-analysis-conversion)=
## B. Analysis Library and Object Conversion

This section is for conversions where the target is a **Python object model** rather than a storage format.  
Use it when the question is “which analysis-library object do I bridge to?”

- Purpose: pick the correct bridge into analysis-oriented Python objects
- Input: a `gwexpy` object or an external analysis object such as pandas, xarray, or astropy
- Output: an analysis-library object or a reconstructed `gwexpy` object

| Target | API / Entry | Status | Notes | Details |
| --- | --- | --- | --- | --- |
| NumPy | no dedicated `to_*()` / `from_*()` API | Implemented as infrastructure | widely used as the internal array basis | — |
| pandas | `to_pandas_series()`, `from_pandas_series()`, `to_pandas_dataframe()`, `from_pandas_dataframe()` | Public | Series / DataFrame | [API](../reference/api/gwexpy.interop.pandas_.rst) |
| polars | `to_polars_series()`, `from_polars_series()`, `to_polars_dataframe()`, `from_polars_dataframe()`, `to_polars_dict()`, `from_polars_dict()` | Implemented, public cleanup pending | Series / DataFrame / dict | — |
| xarray | `to_xarray()`, `from_xarray()` | Public | DataArray / Dataset | [API](../reference/api/gwexpy.interop.xarray_.rst) |
| xarray Field | `to_xarray_field()`, `from_xarray_field()` | Public | ScalarField / VectorField | [API](../reference/api/gwexpy.interop.xarray_.rst) |
| astropy | `to_astropy_timeseries()`, `from_astropy_timeseries()` | Public | `astropy.timeseries.TimeSeries` | [API](../reference/api/gwexpy.interop.astropy_.rst) |
| dask | `to_dask()`, `from_dask()` | Public | dask array bridge | [API](../reference/api/gwexpy.interop.dask_.rst) |

(interop-en-ml-conversion)=
## C. Machine Learning, Acceleration, and Array Backends

This section is for accelerated computing and ML-oriented bridges.  
Check whether only the array payload moves, or whether metadata can also be reconstructed.

- Purpose: decide whether an ML or accelerated-array bridge matches your workflow
- Input: a `gwexpy` object and an ML / GPU / array backend target
- Output: tensors or accelerated arrays, and in some cases a route back into `gwexpy`

| Target | API / Entry | Status | Notes | Details |
| --- | --- | --- | --- | --- |
| PyTorch | `to_torch()`, `from_torch()` | Implemented, public cleanup pending | tensor conversion | — |
| TensorFlow | `to_tf()`, `from_tf()` | Implemented, public cleanup pending | tensor conversion | — |
| JAX | `to_jax()`, `from_jax()` | Implemented, public cleanup pending | JAX array conversion | — |
| CuPy | `to_cupy()`, `from_cupy()` | Implemented, public cleanup pending | GPU array conversion | — |

(interop-en-domain-conversion)=
## D. Physics and Domain-Specific Libraries

This section is for domain-specific libraries and specialized objects.  
Read the status carefully: some targets are full round-trips, some are mainly import paths, and some are still being organized publicly.

- Purpose: find bridges into domain-specific tools without confusing them with direct file I/O
- Input: a `gwexpy` object or a domain-library object such as ObsPy, ROOT, LAL, or PyCBC
- Output: a target-library object, imported data, or a partial round-trip depending on status

| Target | API / Entry | Status | Notes | Details |
| --- | --- | --- | --- | --- |
| ROOT | `to_tgraph()`, `to_th1d()`, `to_th2d()`, `to_tmultigraph()`, `from_root()`, `write_root_file()` | Implemented, some paths still in progress | `TH1 -> non-Histogram` is incomplete | [API](../reference/api/gwexpy.interop.root_.rst) |
| ObsPy | `to_obspy()`, `from_obspy()`, `to_obspy_trace()`, `from_obspy_trace()` | Public | seismic bridge | [API](../reference/api/gwexpy.interop.obspy_.rst) |
| LAL | `to_lal_timeseries()`, `from_lal_timeseries()`, `to_lal_frequencyseries()`, `from_lal_frequencyseries()` | Public | GW time / frequency series | [API](../reference/api/gwexpy.interop.lal_.rst) |
| PyCBC | `to_pycbc_timeseries()`, `from_pycbc_timeseries()`, `to_pycbc_frequencyseries()`, `from_pycbc_frequencyseries()` | Public | GW time / frequency series | [API](../reference/api/gwexpy.interop.pycbc_.rst) |
| GWINC | `from_gwinc_budget()` | Public | budget import | [API](../reference/api/gwexpy.interop.gwinc_.rst) |
| Finesse | `from_finesse_frequency_response()`, `from_finesse_noise()` | Public | optics / response | [API](../reference/api/gwexpy.interop.finesse_.rst) |
| python-control | `to_control_frd()`, `from_control_frd()`, `from_control_response()` | Public | FRD / response. Requires `pip install gwexpy[control]`. FRD conversion is available from `FrequencySeries` / `FrequencySeriesDict`; time-response import is available via `TimeSeries.from_control()` / `TimeSeriesDict.from_control()`. | [API](../reference/api/gwexpy.interop.control_.rst) |
| SimPEG | `to_simpeg()`, `from_simpeg()` | Implemented, public cleanup pending | geophysics | — |
| MTH5 | `to_mth5()`, `from_mth5()` | Implemented, public cleanup pending | magnetotellurics | — |
| MTpy | dedicated `to_*()` / `from_*()` API still in progress | In progress | MTH5-adjacent organization is incomplete | — |
| MNE-Python | `to_mne()`, `from_mne()`, `to_mne_rawarray()`, `from_mne_raw()` | Implemented, public cleanup pending | EEG / biosignal | — |
| Neo | `to_neo()`, `from_neo()` | Implemented, public cleanup pending | electrophysiology | — |
| Elephant | dedicated `to_*()` / `from_*()` API still in progress | In progress | organization with `Neo` and `quantities` is incomplete | — |
| quantities | `to_quantity()`, `from_quantity()` | Implemented, public cleanup pending | quantity bridge | — |
| pyroomacoustics | `to_pyroomacoustics_source()`, `to_pyroomacoustics_stft()`, `from_pyroomacoustics_rir()`, `from_pyroomacoustics_mic_signals()`, `from_pyroomacoustics_source()`, `from_pyroomacoustics_stft()`, `from_pyroomacoustics_field()` | Implemented, public cleanup pending | room acoustics | — |
| pydub | `to_pydub()`, `from_pydub()` | Implemented, public cleanup pending | audio object bridge | — |
| librosa | `to_librosa()` | Implemented, public cleanup pending | mainly export | — |
| Specutils | `to_specutils()`, `from_specutils()` | Implemented, public cleanup pending | astronomy spectra | — |
| pyspeckit | `to_pyspeckit()`, `from_pyspeckit()` | Implemented, public cleanup pending | spectral analysis | — |
| PySpice | `from_pyspice_transient()`, `from_pyspice_ac()`, `from_pyspice_noise()`, `from_pyspice_distortion()` | Implemented, public cleanup pending | mainly import | — |
| scikit-rf | `to_skrf_network()`, `from_skrf_network()`, `from_skrf_impulse_response()`, `from_skrf_step_response()` | Implemented, public cleanup pending | RF network analysis | — |
| pyOMA | `from_pyoma_results()` | Implemented, public cleanup pending | mainly import | — |
| multitaper | `from_mtspec()` | Implemented, public cleanup pending | mainly import | — |
| mtspec | `from_mtspec_array()` | Implemented, public cleanup pending | mainly import | — |
| pySDy | `from_uff_dataset55()`, `from_uff_dataset58()` | Implemented, public cleanup pending | mainly import | — |
| SDynPy | `from_sdynpy_frf()`, `from_sdynpy_shape()`, `from_sdynpy_timehistory()` | Implemented, public cleanup pending | mainly import | — |
| Meep | `from_meep_hdf5()` | Implemented, public cleanup pending | mainly import | — |
| openEMS | `from_openems_hdf5()` | Implemented, public cleanup pending | mainly import | — |
| emg3d | `to_emg3d_field()`, `from_emg3d_field()`, `from_emg3d_h5()` | Implemented, public cleanup pending | EM field import/export | — |
| meshio | `from_meshio()`, `from_fenics_xdmf()`, `from_fenics_vtk()` | Implemented, public cleanup pending | mainly import | — |
| MetPy | `from_metpy_dataarray()` | Implemented, public cleanup pending | mainly import | — |
| WRF | `from_wrf_variable()` | Implemented, public cleanup pending | mainly import | — |
| Harmonica | `from_harmonica_grid()` | Implemented, public cleanup pending | mainly import | — |
| Exudyn | `from_exudyn_sensor()` | Implemented, public cleanup pending | mainly import | — |
| OpenSees | `from_opensees_recorder()` | Implemented, public cleanup pending | mainly import | — |

(interop-en-priorities)=
## What to Prioritize First

The following targets are especially important because they sit close to the direct-I/O boundary or because they are high-value public entry points:

- **ROOT**: `io_formats` keeps only EventTable direct I/O; ROOT object conversion belongs here
- **xarray / Field**: the main route for ScalarField / VectorField bridges
- **Zarr**: easy to confuse with direct I/O, so the boundary matters
- **NetCDF4**: needs a clean line between direct I/O and xarray-backed workflows
- **ObsPy**: common and easy-to-understand round-trip examples
- **pandas / polars / astropy**: frequent analysis entry points

## Related Pages

- [Interop tutorial](tutorials/intro_interop)
- [Interop API reference](../reference/api/interop)
- [File I/O Supported Formats Guide](io_formats)

## Next to Read

- [File I/O Supported Formats Guide](io_formats) if your real question is about `Class.read(..., format=...)` or `obj.write(...)`
- [GPS Time Utility Functions](time_utilities) if conversion workflows depend on GPS or timezone handling
- [Interop tutorial](tutorials/intro_interop) for worked examples before dropping into the API reference
