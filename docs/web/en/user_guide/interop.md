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

For local file formats and direct `.read()` / `.write()` / `fetch()` paths, see the [File I/O Supported Formats Guide](io_formats).

## Direct I/O Names

These are the canonical direct-I/O format names used by `gwexpy`.
Legacy aliases remain supported during the transition, but new examples should prefer the canonical spellings below.

| Canonical name | Legacy aliases | Typical direct-I/O entry points | External package / schema |
| --- | --- | --- | --- |
| `mseed` | `miniseed` | `TimeSeriesDict.read(..., format="mseed")`, `.write(..., format="mseed")` | ObsPy |
| `nc` | `netcdf4` | `TimeSeries.read(..., format="nc")`, `TimeSeriesDict.read(..., format="nc")`, `TimeSeriesMatrix.read(..., format="nc")` | [netCDF4](https://unidata.github.io/netcdf4-python/), [xarray](https://docs.xarray.dev/) |
| `hdf.ndscope` | `ndscope-hdf5`, `ndscope_hdf5`, `ndscopehdf5` | `TimeSeriesDict.read(..., format="hdf.ndscope")`, `.write(..., format="hdf.ndscope")` | ndscope HDF5 schema |
| `xml.diaggui` | `dttxml` | `TimeSeriesDict.read(..., format="xml.diaggui", products="...")` | DiagGUI / DTT XML |

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

- If you want to convert a live object to a **storage format or container**, start with A.
- If you want to convert to **analysis objects** such as pandas, xarray, astropy, or dask, start with B.
- If you want to hand data to **PyTorch, TensorFlow, JAX, or CuPy**, start with C.
- If you want to connect to **ROOT, ObsPy, LAL, PyCBC**, or other domain-specific libraries, start with D.
- If you want to move **Field** objects into xarray, NetCDF4, or Zarr workflows via `to_*()` / `from_*()`, treat that as interop, not as direct I/O.

(interop-en-status-labels)=
## Status Labels

- `Public`: implemented and reachable from `reference/api/interop`
- `Implemented`: implemented, but the page still needs concrete navigation or reference polish
- `Implemented, some paths still in progress`: the main route works, but some conversion paths are still incomplete
- `In progress`: the implementation or the public presentation is not finished yet
- `Planned`: explicitly in scope, but not implemented yet

## Optional Dependency Policy

Interop bridges import optional runtime backends lazily. If a runtime backend is
missing, the bridge raises `ImportError` with installation guidance instead of
importing the backend at `import gwexpy` time.

Some import-only adapters consume objects or dictionaries that were produced by
external packages, but do not import those producer packages themselves. For
example, `from_pyoma_results()`, `from_mtspec()`, `from_mtspec_array()`,
`from_uff_dataset55()`, `from_uff_dataset58()`, the `from_sdynpy_*()`,
`from_metpy_dataarray()`, `from_wrf_variable()`, and `from_harmonica_grid()`
helpers accept caller-supplied pyOMA, multitaper, mtspec, pyuff, SDynPy-style,
MetPy-enhanced, wrf-python, or Harmonica objects. Install those packages when
you need to create the source objects, not because the adapter imports them
directly.

| Policy | Dependencies | Install guidance |
| --- | --- | --- |
| Declared GWexpy extras | `zarr`, `netCDF4`, `xarray`, `obspy`, `mth5`, `lalsuite`, `gwinc`, `control`, `pydub` | Use `pip install 'gwexpy[zarr]'`, `pip install 'gwexpy[netcdf4]'`, `pip install 'gwexpy[seismic]'`, `pip install 'gwexpy[gw]'`, `pip install 'gwexpy[control]'`, or `pip install 'gwexpy[audio]'` as appropriate. |
| Bare package installs | `ROOT`, `polars`, `dask`, `torch`, `tensorflow`, `jax`, `cupy`, `pycbc`, `finesse`, `simpeg`, `mne`, `neo`, `quantities`, `pyroomacoustics`, `specutils`, `pyspeckit`, `PySpice`, `skrf`, `pyOMA`, `multitaper`, `mtspec`, `pyuff`, `sdynpy`, `metpy`, `wrf-python`, `harmonica`, `emg3d`, `meshio` | Install the named backend directly when the bridge imports it or when you need it to create accepted source objects. |

The xarray-backed interop bridges list the `netcdf4` extra because GWexpy does
not currently publish a standalone `xarray` extra. If you only need in-memory
xarray conversion and want to avoid installing `netCDF4`, install `xarray`
directly.

(interop-en-storage-conversion)=
## A. Storage Formats and Container Conversion

This section is for conversions where the target is a **file format, container, or storage representation**.  
Use it when the question is “what storage representation do I bridge to?”

- Purpose: identify object-level bridges whose target is a storage representation
- Input: a `gwexpy` object plus a destination container or storage backend
- Output: a converted object, container, or storage-facing representation via `to_*()` / `from_*()`

| Target | API / Entry | Status | Notes | Details |
| --- | --- | --- | --- | --- |
| [HDF5](https://www.hdfgroup.org/solutions/hdf5/) | `to_hdf5()`, `from_hdf5()` | Public | object-level conversion | [API](../reference/api/gwexpy.interop.hdf5_.rst) |
| HDF5 FrequencySeries | `to_hdf5_frequencyseries()`, `from_hdf5_frequencyseries()` | Public | FrequencySeries HDF5 helpers | [API](../reference/api/gwexpy.interop.frequency.rst) |
| JSON | `to_json()`, `from_json()` | Public | JSON string conversion | [API](../reference/api/gwexpy.interop.json_.rst) |
| Python dict | `to_dict()`, `from_dict()` | Public | dict conversion | — |
| [SQLite](https://www.sqlite.org/index.html) | `to_sqlite()`, `from_sqlite()` | Public | object-level bridge | [API](../reference/api/gwexpy.interop.sqlite_.rst) |
| [Zarr](https://zarr.readthedocs.io/en/stable/) | `to_zarr()`, `from_zarr()` | Public | array/store bridge | [API](../reference/api/gwexpy.interop.zarr_.rst) |
| [NetCDF4](https://unidata.github.io/netcdf4-python/) | `to_netcdf4()`, `from_netcdf4()` | Public | object-level bridge | [API](../reference/api/gwexpy.interop.netcdf4_.rst) |

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
| [pandas](https://pandas.pydata.org/) | `to_pandas_series()`, `from_pandas_series()`, `to_pandas_dataframe()`, `from_pandas_dataframe()` | Public | Series / DataFrame | [API](../reference/api/gwexpy.interop.pandas_.rst) |
| pandas FrequencySeries | `to_pandas_frequencyseries()`, `from_pandas_frequencyseries()` | Public | FrequencySeries ⇔ pandas.Series | [API](../reference/api/gwexpy.interop.frequency.rst) |
| [polars](https://pola.rs/) | `to_polars_series()`, `from_polars_series()`, `to_polars_dataframe()`, `from_polars_dataframe()`, `to_polars_frequencyseries()`, `to_polars_dict()`, `from_polars_dict()` | Public | Series / DataFrame / dict / FrequencySeries | [API](../reference/api/gwexpy.interop.polars_.rst) |
| [xarray](https://docs.xarray.dev/) | `to_xarray()`, `from_xarray()` | Public | DataArray / Dataset | [API](../reference/api/gwexpy.interop.xarray_.rst) |
| [xarray](https://docs.xarray.dev/) Field | `to_xarray_field()`, `from_xarray_field()` | Public | ScalarField / VectorField | [API](../reference/api/gwexpy.interop.xarray_.rst) |
| xarray FrequencySeries | `to_xarray_frequencyseries()`, `from_xarray_frequencyseries()` | Public | FrequencySeries ⇔ xarray.DataArray | [API](../reference/api/gwexpy.interop.frequency.rst) |
| [astropy](https://www.astropy.org/) | `to_astropy_timeseries()`, `from_astropy_timeseries()` | Public | `astropy.timeseries.TimeSeries` | [API](../reference/api/gwexpy.interop.astropy_.rst) |
| [dask](https://www.dask.org/) | `to_dask()`, `from_dask()` | Public | dask array bridge | [API](../reference/api/gwexpy.interop.dask_.rst) |

(interop-en-ml-conversion)=
## C. Machine Learning, Acceleration, and Array Backends

This section is for accelerated computing and ML-oriented bridges.  
Check whether only the array payload moves, or whether metadata can also be reconstructed.

- Purpose: decide whether an ML or accelerated-array bridge matches your workflow
- Input: a `gwexpy` object and an ML / GPU / array backend target
- Output: tensors or accelerated arrays, and in some cases a route back into `gwexpy`

| Target | API / Entry | Status | Notes | Details |
| --- | --- | --- | --- | --- |
| [PyTorch](https://pytorch.org/) | `to_torch()`, `from_torch()` | Public | tensor conversion | [API](../reference/api/gwexpy.interop.torch_.rst) |
| [PyTorch](https://pytorch.org/) Dataset | `TimeSeriesWindowDataset`, `to_torch_dataset()`, `to_torch_dataloader()` | Public | windowed dataset for training | [API](../reference/api/gwexpy.interop.torch_dataset.rst) |
| [TensorFlow](https://www.tensorflow.org/) | `to_tf()`, `from_tf()` | Public | tensor conversion | [API](../reference/api/gwexpy.interop.tensorflow_.rst) |
| [JAX](https://jax.readthedocs.io/en/latest/) | `to_jax()`, `from_jax()` | Public | JAX array conversion | [API](../reference/api/gwexpy.interop.jax_.rst) |
| [CuPy](https://cupy.dev/) | `to_cupy()`, `from_cupy()`, `is_cupy_available()` | Public | GPU array conversion | [API](../reference/api/gwexpy.interop.cupy_.rst) |

(interop-en-domain-conversion)=
## D. Physics and Domain-Specific Libraries

This section is for domain-specific libraries and specialized objects.  
Read the status carefully: some targets are full round-trips, some are mainly import paths, and some are still being organized publicly.

- Purpose: find bridges into domain-specific tools without confusing them with direct file I/O
- Input: a `gwexpy` object or a domain-library object such as ObsPy, ROOT, LAL, or PyCBC
- Output: a target-library object, imported data, or a partial round-trip depending on status

| Target | API / Entry | Status | Notes | Details |
| --- | --- | --- | --- | --- |
| [ROOT](https://root.cern/) | `to_tgraph()`, `to_th1d()`, `to_th2d()`, `to_tmultigraph()`, `from_root()`, `write_root_file()` | Implemented, some paths still in progress | `TH1 -> non-Histogram` is incomplete | [API](../reference/api/gwexpy.interop.root_.rst) |
| [ObsPy](https://docs.obspy.org/) | `to_obspy()`, `from_obspy()`, `to_obspy_trace()`, `from_obspy_trace()` | Public | seismic bridge | [API](../reference/api/gwexpy.interop.obspy_.rst) |
| [LALSuite](https://lscsoft.docs.ligo.org/lalsuite/) | `to_lal_timeseries()`, `from_lal_timeseries()`, `to_lal_frequencyseries()`, `from_lal_frequencyseries()` | Public | GW time / frequency series | [API](../reference/api/gwexpy.interop.lal_.rst) |
| [PyCBC](https://pycbc.org/) | `to_pycbc_timeseries()`, `from_pycbc_timeseries()`, `to_pycbc_frequencyseries()`, `from_pycbc_frequencyseries()` | Public | GW time / frequency series | [API](../reference/api/gwexpy.interop.pycbc_.rst) |
| [GWINC](https://git.ligo.org/gwinc/pygwinc) | `from_gwinc_budget()` | Public | budget import | [API](../reference/api/gwexpy.interop.gwinc_.rst) |
| [Finesse](https://finesse.ifosim.org/) | `from_finesse_frequency_response()`, `from_finesse_noise()` | Public | optics / response | [API](../reference/api/gwexpy.interop.finesse_.rst) |
| [python-control](https://python-control.readthedocs.io/en/latest/) | `to_control_frd()`, `from_control_frd()`, `from_control_response()` | Public | FRD / response. Requires `pip install gwexpy[control]`. FRD conversion is available from `FrequencySeries` / `FrequencySeriesDict`; time-response import is available via `TimeSeries.from_control()` / `TimeSeriesDict.from_control()`. | [API](../reference/api/gwexpy.interop.control_.rst) |
| [SimPEG](https://simpeg.xyz/) | `to_simpeg()`, `from_simpeg()` | Public | geophysics | [API](../reference/api/gwexpy.interop.simpeg_.rst) |
| [MTH5](https://mth5.readthedocs.io/en/latest/) | `to_mth5()`, `from_mth5()` | Public | magnetotellurics | [API](../reference/api/gwexpy.interop.mt_.rst) |
| MTpy | dedicated `to_*()` / `from_*()` API still in progress | In progress | MTH5-adjacent organization is incomplete | — |
| [MNE-Python](https://mne.tools/stable/index.html) | `to_mne()`, `from_mne()`, `to_mne_rawarray()`, `from_mne_raw()` | Public | EEG / biosignal | [API](../reference/api/gwexpy.interop.mne_.rst) |
| [Neo](https://neo.readthedocs.io/en/latest/) | `to_neo()`, `from_neo()` | Public | electrophysiology | [API](../reference/api/gwexpy.interop.neo_.rst) |
| Elephant | dedicated `to_*()` / `from_*()` API still in progress | In progress | organization with `Neo` and `quantities` is incomplete | — |
| [quantities](https://python-quantities.readthedocs.io/en/latest/) | `to_quantity()`, `from_quantity()` | Public | quantity bridge | [API](../reference/api/gwexpy.interop.quantities_.rst) |
| [pyroomacoustics](https://pyroomacoustics.readthedocs.io/en/stable/) | `to_pyroomacoustics_source()`, `to_pyroomacoustics_stft()`, `from_pyroomacoustics_rir()`, `from_pyroomacoustics_mic_signals()`, `from_pyroomacoustics_source()`, `from_pyroomacoustics_stft()`, `from_pyroomacoustics_field()` | Public | room acoustics | [API](../reference/api/gwexpy.interop.pyroomacoustics_.rst) |
| [pydub](https://www.pydub.com/) | `to_pydub()`, `from_pydub()` | Public | audio object bridge | [API](../reference/api/gwexpy.interop.pydub_.rst) |
| [librosa](https://librosa.org/doc/latest/index.html) | `to_librosa()` | Public | mainly export | [API](../reference/api/gwexpy.interop.pydub_.rst) |
| [specutils](https://specutils.readthedocs.io/en/stable/) | `to_specutils()`, `from_specutils()` | Public | astronomy spectra | [API](../reference/api/gwexpy.interop.specutils_.rst) |
| [pyspeckit](https://pyspeckit.readthedocs.io/en/latest/) | `to_pyspeckit()`, `from_pyspeckit()` | Public | spectral analysis | [API](../reference/api/gwexpy.interop.pyspeckit_.rst) |
| PySpice | `from_pyspice_transient()`, `from_pyspice_ac()`, `from_pyspice_noise()`, `from_pyspice_distortion()` | Public | mainly import | [API](../reference/api/gwexpy.interop.pyspice_.rst) |
| [scikit-rf](https://scikit-rf.readthedocs.io/en/latest/) | `to_skrf_network()`, `from_skrf_network()`, `from_skrf_impulse_response()`, `from_skrf_step_response()` | Public | RF network analysis | [API](../reference/api/gwexpy.interop.skrf_.rst) |
| [pyOMA](https://py-oma.readthedocs.io/en/latest/) | `from_pyoma_results()` | Public | mainly import | [API](../reference/api/gwexpy.interop.pyoma_.rst) |
| multitaper | `from_mtspec()` | Public | mainly import | [API](../reference/api/gwexpy.interop.multitaper_.rst) |
| mtspec | `from_mtspec_array()` | Public | mainly import | [API](../reference/api/gwexpy.interop.multitaper_.rst) |
| pySDy | `from_uff_dataset55()`, `from_uff_dataset58()` | Public | mainly import | [API](../reference/api/gwexpy.interop.sdypy_.rst) |
| SDynPy | `from_sdynpy_frf()`, `from_sdynpy_shape()`, `from_sdynpy_timehistory()` | Public | mainly import | [API](../reference/api/gwexpy.interop.sdynpy_.rst) |
| Meep | `from_meep_hdf5()` | Public | mainly import | [API](../reference/api/gwexpy.interop.meep_.rst) |
| openEMS | `from_openems_hdf5()`, `DUMP_TYPE_MAP` | Public | mainly import | [API](../reference/api/gwexpy.interop.openems_.rst) |
| emg3d | `to_emg3d_field()`, `from_emg3d_field()`, `from_emg3d_h5()` | Public | EM field import/export | [API](../reference/api/gwexpy.interop.emg3d_.rst) |
| meshio | `from_meshio()`, `from_fenics_xdmf()`, `from_fenics_vtk()` | Public | mainly import | [API](../reference/api/gwexpy.interop.meshio_.rst) |
| MetPy | `from_metpy_dataarray()` | Public | mainly import | [API](../reference/api/gwexpy.interop.metpy_.rst) |
| WRF | `from_wrf_variable()` | Public | mainly import | [API](../reference/api/gwexpy.interop.wrf_.rst) |
| Harmonica | `from_harmonica_grid()` | Public | mainly import | [API](../reference/api/gwexpy.interop.harmonica_.rst) |
| Exudyn | `from_exudyn_sensor()` | Public | mainly import | [API](../reference/api/gwexpy.interop.exudyn_.rst) |
| OpenSees | `from_opensees_recorder()` | Public | mainly import | [API](../reference/api/gwexpy.interop.opensees_.rst) |

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

- [Interop tutorial](tutorials/intro_interop.ipynb)
- [Interop API reference](../reference/api/interop)
- [File I/O Supported Formats Guide](io_formats)

## Next to Read

- [File I/O Supported Formats Guide](io_formats) if your real question is about `Class.read(..., format=...)` or `obj.write(...)`
- [GPS Time Utility Functions](time_utilities) if conversion workflows depend on GPS or timezone handling
- [Interop tutorial](tutorials/intro_interop.ipynb) for worked examples before dropping into the API reference
