# File I/O Supported Formats Guide

This guide provides a comprehensive list of all file formats supported by `gwexpy`, along with methods for reading and writing data.
This page focuses on the end-user API (`.read()` / `.write()` class methods) and omits internal implementation details.

:::{warning}
**Security Warning: Handling Pickle Files**

The **Pickle** (:term:`Pickle`) format is highly convenient but reading a Pickle file from an untrusted source is extremely dangerous, as malicious files can execute arbitrary code on your system.
We strongly recommend using structured and secure formats like **HDF5**, **GWF**, or **Zarr** for data sharing and long-term storage.

:::
## format Comparison Table

Choose the best format for your data and needs. Refer to each section for detailed usage.

| Format | R / W | Supported Class | Auto-detect | Required Extras | Typical Use | Warnings |
| :--- | :---: | :--- | :---: | :--- | :--- | :--- |
| **GWF** (.gwf) | ○ / ○ | `TimeSeries(Dict)` | ○ | — | Standard LIGO/KAGRA analysis | Reliable and standard |
| **HDF5** (.h5, .hdf5) | ○ / ○ | `TimeSeries(Dict)`, `Field` | ○ | — | Long-term metadata storage | Secure. Recommended |
| **Pickle** (.pkl) | ○ / ○ | All Classes | ○ | — | Python object serialization | **Security warning applies** |
| **LIGO_LW** (.xml) | ○ / × | `TimeSeries(Dict)` | ○ | — | DTT diagnostic products | `products` argument required |
| **CSV / TXT** (.csv, .txt) | ○ / ○ | `TimeSeries(Dict)` | △ | — | Plain text, general purpose | Supports directory bulk loading |
| **WAV** (.wav) | ○ / ○ | `TimeSeries(Dict)` | ○ | `scipy` | Audio analysis | Does not store absolute time |
| **MiniSEED** (.mseed) | ○ / ○ | `TimeSeries(Dict)` | ○ | `obspy` | Standard for seismic waveforms | — |
| **SAC** (.sac) | ○ / ○ | `TimeSeries(Dict)` | ○ | `obspy` | Seismic analysis | Rich header definitions |
| **GSE2** (.gse2) | ○ / ○ | `TimeSeries(Dict)` | ○ | `obspy` | Seismic waveform | — |
| **K-NET** (.knet) | ○ / × | `TimeSeries(Dict)` | ○ | `obspy` | NIED strong motion records | Read-only |
| **GBD** (.gbd) | ○ / × | `TimeSeriesDict` | ○ | — | GRAPHTEC Data Logger | `timezone` is required |
| **WIN / 32** (.win) | ○ / × | `TimeSeriesDict` | ○ | `obspy` | Standard for Japanese seismic nets | Uses improved custom parser |
| **MTH5** (.h5) | ○ / ○ | `TimeSeries` | × | `mth5` | Magnetic/MT observations | Under development/Design support |
| **ATS** (.ats) | ○ / × | `TimeSeriesDict` | ○ | — | Metronix MT observations | Binary parser |
| **ROOT** (.root) | ○ / ○ | `EventTable` | ○ | — | CERN ROOT (Event analysis) | Via gwpy |
| **SQLite** (.sdb) | ○ / × | `TimeSeriesDict` | ○ | — | WeeWX weather data | — |
| **NetCDF4** (.nc) | ○ / ○ | `TimeSeries(Dict)` | ○ | `xarray` | Cloud-ready, multi-dimensional | — |
| **Zarr** (.zarr) | ○ / ○ | All Classes | ○ | `zarr` | Cloud-optimized, parallel processing | Recommended modern format |
| **Audio** (.mp3 etc) | ○ / ○ | `TimeSeriesDict` | ○ | `pydub` | Compressed audio files | Requires ffmpeg, no time info |
| **NDS2** | ○ / × | `TimeSeries` | — | — | Network Data Server | Remote data access |
| **TDMS** (.tdms) | ○ / × | `TimeSeriesDict` | ○ | `npTDMS` | National Instruments format | — |

> **Note**: Formats marked as "gwpy standard" (GWF, HDF5, CSV/TXT, Pickle) are processed through gwpy's built-in IO pathways. Since gwexpy extends gwpy, these are available natively.

## Format Specific Details

### GBD — GRAPHTEC Data Logger `.gbd`

**Extension**: `.gbd`
**Read/Write**: Read ○ / Write ×
**Recommended API**:

```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.gbd", timezone="Asia/Tokyo")
```

## Developer Guide

The following information is intended for developers or advanced users interested in the internal implementation.

### Design-level Support (No Dedicated Implementation)

The following formats are specified in the design but currently rely on external library pathways or stubs.

| Format | Extension | Notes |
| :--- | :--- | :--- |
| **MTH5** | `.h5` | Via `mth5` library. Partially supported through `ats.mth5` |

### Unimplemented Formats (Stubs)

These formats are currently registered in the IO registry as placeholders. Calling `.read()` will raise an `UnimplementedIOError`.

#### TimeSeries Stubs

| Format | Registered Class | Notes |
| :--- | :--- | :--- |
| `orf`, `mem`, `wvf`, `wdf`, `taffmat`, `lsf`, `li` | TimeSeries(Dict/Matrix) | Reserved for future use |

## Basic Usage of `.read()` / `.write()`

All `gwexpy` data classes use the `gwpy` IO registry pattern.

```python
from gwexpy.timeseries.collections import TimeSeriesDict

# Auto-detect format from extension
tsd = TimeSeriesDict.read("path/to/file.gbd", timezone="Asia/Tokyo")

# Explicitly specify format
tsd = TimeSeriesDict.read("path/to/file.dat", format="miniseed")
```

### Implementation Reference

| Module Path | Summary |
|---|---|
| `gwexpy/timeseries/io/gbd.py` | GBD reader. Requires `timezone` |
| `gwexpy/timeseries/io/ats.py` | ATS reader (binary parser) |
| `gwexpy/timeseries/io/seismic.py` | MiniSEED / SAC / GSE2 / KNET IO. Depends on ObsPy |

## Reference: Source Files

For a full list of IO implementation files, refer to the Japanese version of this guide or the `gwexpy/timeseries/io/` directory.
