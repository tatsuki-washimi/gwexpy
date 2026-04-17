<a id="io-formats-en-top"></a>

# File I/O Supported Formats Guide

This is the end-user I/O guide for `gwexpy`.
This page only covers the public `.read()` / `.write()` / `fetch()` style APIs that users call directly.

It does **not** cover `to_*()` / `from_*()` conversions or object bridges to xarray, ROOT objects, or Zarr arrays. For those topics, see the [interop tutorial](tutorials/intro_interop) and the [interop API reference](../reference/api/interop).

:::{warning}
**Security Warning: Pickle Files**

**Pickle** (:term:`Pickle`) is convenient, but reading Pickle files from untrusted sources is dangerous. A malicious Pickle file can execute arbitrary code on your system.

For data sharing and long-term storage, prefer structured formats such as **HDF5**, **GWF**, or **Zarr**.
:::

## First: Decision Rules

- If you need a **default GW storage format**, start with **HDF5**. For existing seismic or geophysical assets, start with **MiniSEED / SAC / WIN / ATS**. For general interchange, start with **CSV / NetCDF4 / Zarr**. For logger- or device-specific data, start with **GBD / TDMS / SDB / WAV / Audio**. For **MTH5**, the current public direct-I/O story is only the single **`ats.mth5`** path. A generic standalone **`format="mth5"`** route is not published yet.
- **Auto-detect is fine** when the extension uniquely selects one reader.
- **Set `format=` explicitly** for ambiguous extensions such as `.xml`, for custom lab extensions, or whenever auto-detection is unclear.
- **Pass `timezone` explicitly** when the file stores local wall-clock time without embedded UTC/GPS. In the current user-facing guide, **GBD** is the main required case.
- **Read-only / write-only matters**: `○ / ×` means a format can be read but not written.
- If you need to preserve **Field** objects safely, this page treats **HDF5** and **Pickle** as the direct-I/O baseline. NetCDF4 / Zarr / ROOT object bridges belong to interop, not to this page.

## Jump Links

- <a href="#io-formats-en-quick">Quick Selection Table</a>
- <a href="#io-formats-en-basic">Basic `.read()` / `.write()` / `fetch()` Usage</a>
- <a href="#io-formats-en-a">A. GW Standards</a>
- <a href="#io-formats-en-b">B. Seismic and Geophysical Observation</a>
- <a href="#io-formats-en-c">C. General Analysis and Exchange</a>
- <a href="#io-formats-en-d">D. Loggers and Instrument Formats</a>
- <a href="#io-formats-en-dev">Developer Notes</a>

<a id="io-formats-en-quick"></a>

## Quick Selection Table

| Group | Choose this when... | First format to check | Formats covered here |
|---|---|---|---|
| **A. GW Standards** | You want standard GW storage, exchange, or acquisition paths | **HDF5** | GWF, HDF5, ndscope-hdf5, DTTXML, NDS2, GWOSC |
| **B. Seismic and Geophysical Observation** | You need to read existing seismic or EM observation data | **MiniSEED** | MiniSEED, SAC, GSE2, K-NET, WIN / WIN32, ATS, ATS.MTH5 (MTH5 standalone is status-only here) |
| **C. General Analysis and Exchange** | You need general-purpose storage or external analysis exchange | **CSV / TXT** or **Zarr** | CSV / TXT, NetCDF4, Zarr, Pickle, ROOT |
| **D. Loggers and Instrument Formats** | You are working with device- or logger-specific time series | **GBD** or **TDMS** | GBD, TDMS, SDB / SQLite / SQLite3, WAV, MP3, FLAC, OGG, M4A |

> **Note**: `NDS2` and `GWOSC` are not file formats. They are included in **A. GW Standards** because they are common GW data entry points. In the tables below, they are labeled as `network path`.

<a id="io-formats-en-basic"></a>

## Basic `.read()` / `.write()` / `fetch()` Usage

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.timeseries import TimeSeries

# Auto-detect from extension
tsd = TimeSeriesDict.read("path/to/data.mseed")

# Explicit format
tsd = TimeSeriesDict.read("path/to/data.dat", format="miniseed")

# Write out
tsd.write("output.h5", format="hdf5")

# Network path
ts = TimeSeries.fetch_open_data("H1", 1126259446, 1126259478)
```

- `.read()` / `.write()` uses the gwpy-style I/O registry.
- `.xml` is ambiguous, so **use `format="dttxml"` explicitly** for DTTXML data.
- `NDS2` and `GWOSC` are not file readers, so they use `fetch()` / `fetch_open_data()` instead of `.read()`.

## Supported Classes at a Glance

If the main question is whether a format is for a single channel or multiple channels, use this table first.

| Format / Family | Single-channel | Multi-channel | Other supported classes |
|---|---|---|---|
| **GWF / MiniSEED / SAC / GSE2 / K-NET / WIN / WIN32 / ATS / CSV / TXT / SDB / SQLite / SQLite3 / WAV / Audio** | `TimeSeries` | `TimeSeriesDict` | Baseline end-user direct I/O pattern |
| **NetCDF4 / Zarr / GBD / TDMS** | `TimeSeries` | `TimeSeriesDict`, `TimeSeriesMatrix` | Includes matrix-style direct I/O |
| **HDF5** | `TimeSeries`, `FrequencySeries`, and related classes | `TimeSeriesDict` and related collections | Also covers `Spectrogram`, `Histogram`, `EventTable`, and `Field` |
| **ndscope-hdf5** | - | `TimeSeriesDict` | ndscope-compatible schema |
| **DTTXML** | - | `TimeSeriesDict` | Requires `products` |
| **NDS2 / GWOSC** | `TimeSeries` | - | Use `fetch()` / `fetch_open_data()` |
| **ATS.MTH5** | `TimeSeries` | - | Partial single-path support |
| **Pickle** | Major classes broadly | Major classes broadly | Use only for trusted data |
| **ROOT** | `EventTable` | - | Direct I/O is limited to EventTable |

- If you are unsure, start by thinking in terms of `TimeSeries` and `TimeSeriesDict`.
- `TimeSeriesMatrix` mainly matters for `NetCDF4`, `Zarr`, `GBD`, and `TDMS`.
- If you need to preserve richer objects beyond Series classes, start with **HDF5** or **Pickle**.

<a id="io-formats-en-a"></a>

## A. GW Standards

These are the standard GW storage, exchange, and acquisition paths.
If you are unsure, start with **HDF5**. Use **GWF** when you need external standard compatibility, and **DTTXML** for diagnostic tool output.

| Format / Path | R / W | Main entry point | Best for | Notes |
|---|:---:|---|---|---|
| **GWF** (`.gwf`) | ○ / ○ | `TimeSeries.read()`, `TimeSeriesDict.read()`, `.write()` | Standard LIGO/KAGRA frame exchange | Standard format, via gwpy |
| **HDF5** (`.h5`, `.hdf5`) | ○ / ○ | `.read(..., format="hdf5")`, `.write(..., format="hdf5")` on major classes | Long-term storage with metadata | Main direct-I/O option for Field objects |
| **ndscope-hdf5** (`.h5`, `.hdf5`) | ○ / ○ | `TimeSeriesDict.read(..., format="ndscope-hdf5")`, `.write(..., format="ndscope-hdf5")` | ndscope-compatible HDF5 | `TimeSeriesDict` only |
| **DTTXML** (`.xml`, `.xml.gz`) | ○ / × | `TimeSeriesDict.read(..., format="dttxml", products="...")` | DTT outputs and diagnostics | `products` is required |
| **NDS2** | ○ / × | `TimeSeries.fetch()` | Detector data server access | Network path |
| **GWOSC** | ○ / × | `TimeSeries.fetch_open_data()` | Open data access | Network path |

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.timeseries import TimeSeries

tsd = TimeSeriesDict.read("data.h5", format="hdf5")
frame = TimeSeriesDict.read("data.gwf", format="gwf")
dtt = TimeSeriesDict.read("diag.xml", format="dttxml", products="TS")
open_data = TimeSeries.fetch_open_data("H1", 1126259446, 1126259478)
```

- **HDF5** is the safest general recommendation for structured GW data.
- **DTTXML** changes behavior depending on `products`. For complex transfer functions, prefer `native=True` on the frequency-domain side.
- **NDS2 / GWOSC** are shown inside group A, but explicitly marked as `network path` rather than file formats.

<a id="io-formats-en-b"></a>

## B. Seismic and Geophysical Observation

This group is for existing seismic and electromagnetic observation formats.
In practice, **MiniSEED** is the easiest starting point when you need to place a format in context.

| Format | R / W | Main entry point | Best for | Notes |
|---|:---:|---|---|---|
| **MiniSEED** (`.mseed`) | ○ / ○ | `TimeSeriesDict.read(..., format="miniseed")`, `.write(..., format="miniseed")` | Standard seismic waveform exchange | `gap` controls gap handling |
| **SAC** (`.sac`) | ○ / ○ | `TimeSeriesDict.read(..., format="sac")`, `.write(..., format="sac")` | Seismic waveform analysis | Via ObsPy |
| **GSE2** (`.gse2`) | ○ / ○ | `TimeSeriesDict.read(..., format="gse2")`, `.write(..., format="gse2")` | Seismic waveform exchange | Via ObsPy |
| **K-NET** (`.knet`) | ○ / × | `TimeSeriesDict.read(..., format="knet")` | Strong-motion records | Read-only |
| **WIN / WIN32** (`.win`, `.cnt`) | ○ / × | `TimeSeriesDict.read(..., format="win")`, `TimeSeriesDict.read(..., format="win32")` | Japanese WIN datasets | Improved parser, read-only |
| **ATS** (`.ats`) | ○ / × | `TimeSeries.read(..., format="ats")`, `TimeSeriesDict.read(..., format="ats")` | Metronix observation data | Native binary reader |
| **ATS.MTH5** (`format="ats.mth5"`) | ○ / × | `TimeSeries.read(..., format="ats.mth5")` | Single MTH5-backed path | Partial support |
| **MTH5 standalone** (`.h5`) | In progress | Dedicated `format="mth5"` not yet exposed | Future general MTH5 direct I/O | **Not currently a public direct-I/O format**. The only direct path today is `ats.mth5` |

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.timeseries import TimeSeries

tsd = TimeSeriesDict.read("data.mseed", format="miniseed", gap="pad")
win = TimeSeriesDict.read("data.cnt", format="win32")
ats = TimeSeries.read("data.atss", format="ats.mth5")
```

- **MiniSEED** pads gaps with `NaN` by default. Use `gap="raise"` if you want failures instead.
- **K-NET** and **WIN / WIN32** are intentionally read-only.
- **ATS.MTH5** is the limited current direct path.
- **MTH5 standalone** is still in design/publication cleanup. Read this as **"`ats.mth5` has partial support"**, not as **"MTH5 direct I/O is generally complete."**

<a id="io-formats-en-c"></a>

## C. General Analysis and Exchange

These formats are useful for analysis notebooks, interchange, and general storage.
The key rule here is not to mix up “format choice” with “library conversion.”

| Format | R / W | Main entry point | Best for | Notes |
|---|:---:|---|---|---|
| **CSV / TXT** (`.csv`, `.txt`) | ○ / ○ | `TimeSeries.read()`, `TimeSeriesDict.read()`, `.write()` | Lightweight exchange and inspection | Also supports directory bulk loading |
| **NetCDF4** (`.nc`) | ○ / ○ | `TimeSeries.read(..., format="netcdf4")`, `TimeSeriesDict.read(..., format="netcdf4")`, `TimeSeriesMatrix.read(..., format="netcdf4")`, `.write(..., format="netcdf4")` | Scientific storage for time-series-oriented data | Direct I/O here is centered on TimeSeries classes |
| **Zarr** (`.zarr`) | ○ / ○ | `TimeSeries.read(..., format="zarr")`, `TimeSeriesDict.read(..., format="zarr")`, `TimeSeriesMatrix.read(..., format="zarr")`, `.write(..., format="zarr")` | Chunked storage and parallel workflows | Direct I/O here is centered on TimeSeries classes |
| **Pickle** (`.pkl`) | ○ / ○ | `.read()` / `.write()` on major classes | Python object snapshots | Only for trusted data |
| **ROOT** (`.root`) | ○ / ○ | `EventTable.read(..., format="root")`, `EventTable.write(..., format="root")` | EventTable I/O | Direct I/O here is EventTable only |

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.table import EventTable

ascii_data = TimeSeriesDict.read("data.csv")
chunked = TimeSeriesDict.read("data.zarr", format="zarr")
events = EventTable.read("events.root", format="root")
```

- **CSV / TXT** remains useful for inspection and simple interchange.
- **NetCDF4 / Zarr** are treated here only as **direct TimeSeries-style I/O**. Field/xarray bridges belong to interop.
- **Zarr** still has an open API-behavior issue around missing sampling-rate metadata. **This page does not treat that issue as fixed**; it is tracked separately from the docs cleanup.
- **ROOT** object-level export/import belongs to interop. This page only covers EventTable direct I/O.

<a id="io-formats-en-d"></a>

## D. Loggers and Instrument Formats

This group is for logger and instrument-specific time-series formats.
Time handling, units, and audio `t0` semantics are the main points to watch.

| Format | R / W | Main entry point | Best for | Notes |
|---|:---:|---|---|---|
| **GBD** (`.gbd`) | ○ / × | `TimeSeries.read(..., format="gbd")`, `TimeSeriesDict.read(..., format="gbd")`, `TimeSeriesMatrix.read(..., format="gbd")` | GRAPHTEC loggers | `timezone` is required |
| **TDMS** (`.tdms`) | ○ / × | `TimeSeries.read(..., format="tdms")`, `TimeSeriesDict.read(..., format="tdms")`, `TimeSeriesMatrix.read(..., format="tdms")` | National Instruments data | Read-only |
| **SDB / SQLite / SQLite3** (`.sdb`, `.sqlite`, `.sqlite3`) | ○ / × | `TimeSeries.read(..., format="sdb" / "sqlite" / "sqlite3")`, `TimeSeriesDict.read(...)` | WeeWX and similar archives | Same reader family |
| **WAV** (`.wav`) | ○ / ○ | `TimeSeries.read(..., format="wav")`, `TimeSeriesDict.read(..., format="wav")`, `.write(..., format="wav")` | Uncompressed audio | Does not preserve absolute time |
| **MP3 / FLAC / OGG / M4A** | ○ / ○ | `TimeSeries.read(..., format="mp3" / "flac" / "ogg" / "m4a")`, `.write(...)` | Compressed audio | Uses `pydub`; some formats also need `ffmpeg` |

```python
from gwexpy.timeseries.collections import TimeSeriesDict

logger = TimeSeriesDict.read("data.gbd", timezone="Asia/Tokyo")
weather = TimeSeriesDict.read("archive.sqlite3", format="sqlite3")
audio = TimeSeriesDict.read("sound.flac", format="flac")
```

- **GBD** requires `timezone`.
- **SDB / SQLite / SQLite3** should all be named explicitly in the public page so users do not need to infer aliases.
- **WAV / Audio** do not preserve absolute timestamps. Reading with `t0=0.0` is a convenience convention, not a claim that the source had an absolute epoch.

<a id="io-formats-en-dev"></a>

## Developer Notes

Most users can skip this section.
It exists mainly to collect not-yet-prominent implementations and placeholders in one place.

### Managed in design, but not prominent in the public page

| Format | Current state | Notes |
|---|---|---|
| `ndscope-hdf5` | Implemented, not yet prominent | `TimeSeriesDict`-only HDF5 schema |
| `SQLite`, `SQLite3` | Implemented, not yet prominent | Aliases in the same family as `SDB` |
| `ATS.MTH5` | Implemented with partial scope | Current public direct path backed by MTH5 |
| `MTH5 standalone` | In progress | Dedicated `format="mth5"` is not exposed yet; not published as public direct I/O |

### Unimplemented Formats (Stubs)

These entries exist as placeholders only. Calling `.read()` on them is expected to fail because they are not ready for end users yet.

#### TimeSeries stubs

| Format | Status |
|---|---|
| `orf` | Planned |
| `mem` | Planned |
| `wvf` | Planned |
| `wdf` | Planned |
| `taffmat` | Planned |
| `lsf` | Planned |
| `li` | Planned |

#### FrequencySeries stubs

| Format | Status |
|---|---|
| `win` | Planned |
| `win32` | Planned |
| `sdb` | Planned |
| `orf` | Planned |
| `mem` | Planned |
| `wvf` | Planned |
| `wdf` | Planned |
| `taffmat` | Planned |
| `lsf` | Planned |
| `li` | Planned |

## Related Pages

- [Interop tutorial](tutorials/intro_interop)
- [Interop API reference](../reference/api/interop)
- [Installation guide](installation)

## Page-End Navigation

- <a href="#io-formats-en-quick">Back to Quick Selection Table</a>
- <a href="#io-formats-en-basic">Back to Basic Usage</a>
- <a href="#io-formats-en-top">Back to Top</a>
