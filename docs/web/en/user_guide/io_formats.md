# File I/O Supported Formats Guide

This guide provides a comprehensive list of all file formats supported by `gwexpy`, along with methods for reading and writing data.
This page focuses on the end-user API (`.read()` / `.write()` class methods) and omits internal implementation details.

> [!CAUTION]
> **Security Warning: Handling Pickle Files**
>
> The **Pickle** (:term:`Pickle`) format is highly convenient for saving Python objects as-is. However, reading a Pickle file from an untrusted source is extremely dangerous, as malicious files can execute arbitrary code on your system.
>
> We strongly recommend using structured and secure formats like **HDF5**, **GWF**, or **Zarr** for data sharing and long-term storage. For more details, see :term:`Pickle` in the glossary.

## Supported Formats Overview

| Format | Extension | R / W | Supported Class | Auto-detect | Required Extras | Notes |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| **GWF** | `.gwf` | ○ / ○ | `TimeSeries(Dict)` | ○ | — | Standard. Base for LIGO/KAGRA analysis |
| **HDF5** | `.h5`, `.hdf5` | ○ / ○ | `TimeSeries(Dict)`, `Field` | ○ | — | Secure. Best for maintaining metadata |
| **Pickle** | `.pkl` | ○ / ○ | All Classes | ○ | — | **Warning applies**. Python-only |
| **LIGO_LW** | `.xml`, `.xml.gz` | ○ / × | `TimeSeries(Dict)` | ○ | — | DTT format. `products` argument required |
| **CSV / TXT** | `.csv`, `.txt` | ○ / ○ | `TimeSeries(Dict)` | △ | — | ASCII. Supports directory bulk loading |
| **WAV** | `.wav` | ○ / ○ | `TimeSeries(Dict)` | ○ | `scipy` | Audio format. No absolute time info |
| **MiniSEED** | `.mseed` | ○ / ○ | `TimeSeries(Dict)` | ○ | `obspy` | Standard for seismic waveforms |
| **SAC** | `.sac` | ○ / ○ | `TimeSeries(Dict)` | ○ | `obspy` | Seismic analysis. Rich header definitions |
| **GSE2** | `.gse2` | ○ / ○ | `TimeSeries(Dict)` | ○ | `obspy` | Seismic waveform |
| **K-NET** | `.knet` | ○ / × | `TimeSeries(Dict)` | ○ | `obspy` | NIED strong motion records |
| **GBD** | `.gbd` | ○ / × | `TimeSeriesDict` | ○ | — | GRAPHTEC. `timezone` required |
| **WIN / 32** | `.win`, `.cnt` | ○ / × | `TimeSeriesDict` | ○ | `obspy` | NIED improved parser |
| **MTH5** | `.h5` | ○ / ○ | `TimeSeries` | × | `mth5` | Magnetometer data (design support) |
| **ATS** | `.ats` | ○ / × | `TimeSeriesDict` | ○ | — | Metronix binary |
| **ROOT** | `.root` | ○ / ○ | `EventTable` | ○ | — | CERN ROOT tables (via gwpy) |
| **SQLite** | `.sdb`, `.sqlite` | ○ / × | `TimeSeriesDict` | ○ | — | WeeWX weather data |
| **NetCDF4** | `.nc` | ○ / ○ | `TimeSeries(Dict)` | ○ | `xarray` | Multi-dimensional. Cloud-ready |
| **Zarr** | `.zarr` | ○ / ○ | All Classes | ○ | `zarr` | Parallel processing. Cloud-optimized |
| **Audio** | `.mp3`, `.flac` | ○ / ○ | `TimeSeriesDict` | ○ | `pydub` | Compressed audio (requires ffmpeg) |
| **NDS2** | (Network) | ○ / × | `TimeSeries` | — | — | Remote data access |
| **TDMS** | `.tdms` | ○ / × | `TimeSeriesDict` | ○ | `npTDMS` | National Instruments |

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

## Design-level Support (No Dedicated Implementation)

The following formats are specified in the design but currently rely on external library pathways or stubs.

| Format | Extension | Notes |
| :--- | :--- | :--- |
| **MTH5** | `.h5` | Via `mth5` library. Partially supported through `ats.mth5` |

## Unimplemented Formats (Stubs)

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

## Developer Guide

The following information is intended for developers or advanced users interested in the internal implementation.

### Implementation Reference
| Module Path | Summary |
|---|---|
| `gwexpy/timeseries/io/gbd.py` | GBD reader. Requires `timezone` |
| `gwexpy/timeseries/io/ats.py` | ATS reader (binary parser) |
| `gwexpy/timeseries/io/seismic.py` | MiniSEED / SAC / GSE2 / KNET IO. Depends on ObsPy |

## Reference: Source Files

For a full list of IO implementation files, refer to the Japanese version of this guide or the `gwexpy/timeseries/io/` directory.
