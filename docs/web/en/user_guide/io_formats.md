---
myst:
  html_meta:
    description: "Choose the right direct file or network I/O path in GWexpy, including supported formats, read/write limits, fetch routes, and when to set format= explicitly."
---

<a id="io-formats-en-top"></a>

# File I/O Supported Formats Guide

> **Page Role:** Guide

This is the end-user I/O guide for `gwexpy`.
This page only covers the public `.read()` / `.write()` / `fetch()` style APIs that users call directly to read, write, or fetch data.

It does **not** cover `to_*()` / `from_*()` conversions or object bridges to xarray, ROOT objects, or Zarr arrays. If the question is "how do I convert this object into another library or container?", that belongs to interop instead. For those topics, see the [interop tutorial](tutorials/intro_interop.ipynb) and the [interop API reference](../reference/api/interop).

## At a Glance

| Item | Details |
| --- | --- |
| **Audience** | Users choosing a direct file or network I/O path for `gwexpy` objects |
| **Prerequisites** | Basic familiarity with `TimeSeries` / `TimeSeriesDict`, file extensions, and the difference between direct I/O and interop conversion |
| **Use Cases** | Pick a supported format, decide when to set `format=`, identify read/write limits, and avoid confusing direct I/O with object conversion |
| **Search Hints** | file I/O, direct I/O, `read`, `write`, `fetch`, HDF5, GWF, MiniSEED, Zarr, NDS2, GWOSC |

**Search hints:** file I/O, direct I/O, `read`, `write`, `fetch`, HDF5, GWF, MiniSEED, Zarr, NDS2, GWOSC

:::{warning}
**Security Warning: Pickle Files**

**Pickle** (:term:`Pickle`) is convenient, but reading Pickle files from untrusted sources is dangerous. A malicious Pickle file can execute arbitrary code on your system.

For data sharing and long-term storage, prefer structured formats such as **HDF5**, **GWF**, or **Zarr**.
:::

## First: Decision Rules

- If you need a **default GW storage format**, start with **HDF5**. For existing seismic or geophysical assets, start with **MiniSEED / SAC / WIN / ATS**. For general interchange, start with **CSV / NetCDF4 / Zarr**. For logger- or device-specific data, start with **GBD / TDMS / SDB / WAV / Audio**. For **MTH5**, the current public direct-I/O story is only the single **`ats.mth5`** path. A generic standalone **`format="mth5"`** route is not published yet.
- **Auto-detect is fine** when the extension uniquely selects one reader.
- For generic **HDF5**, set **`format="hdf5"` explicitly**. The `.h5` / `.hdf5` extensions overlap multiple HDF5-backed families, and auto-identification is not uniform across classes.
- **Set `format=` explicitly** for ambiguous extensions such as `.xml`, for custom lab extensions, or whenever auto-detection is unclear.
- **Pass `timezone` explicitly** when the file stores local wall-clock time without embedded UTC/GPS. In the current user-facing guide, **GBD** is the main required case.
- **Read-only / write-only matters**: `○ / ×` means a format can be read but not written.
- For richer direct-I/O objects beyond plain Series, start with **HDF5** for `Spectrogram`, `Histogram`, and `EventTable`. Field-class direct `.read()` / `.write()` is still under audit and is not published as a stable contract on this page.

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

| Group | Start here when... | First format | Formats covered here |
|---|---|---|---|
| **A. GW Standards** | You want standard GW storage, exchange, or acquisition paths | **HDF5** | GWF, HDF5, hdf.ndscope, xml.diaggui, NDS2, GWOSC |
| **B. Seismic and Geophysical Observation** | You need to read existing seismic or EM observation data | **mseed** | mseed, SAC, GSE2, K-NET, WIN / WIN32, ATS, ATS.MTH5 (MTH5 standalone is status-only here) |
| **C. General Analysis and Exchange** | You need general-purpose storage or external analysis exchange | **CSV / TXT** or **Zarr** | CSV / TXT, NetCDF4, Zarr, ROOT |
| **D. Loggers and Instrument Formats** | You are working with device- or logger-specific time series | **GBD** or **TDMS** | GBD, TDMS, SDB / SQLite / SQLite3, WAV, MP3, FLAC, OGG, M4A |

> **Note**: `NDS2` and `GWOSC` are not file formats. They are included in **A. GW Standards** because they are common GW data entry points. In the tables below, they are labeled as `network path`.

<a id="io-formats-en-basic"></a>

## Basic `.read()` / `.write()` / `fetch()` Usage

- Purpose: show the baseline direct-I/O entry points before format-specific details
- Input: file paths, an explicit `format=` when needed, or a detector/network query
- Output: `TimeSeries`, `TimeSeriesDict`, or other direct-I/O return objects

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.timeseries import TimeSeries

# Auto-detect from extension
tsd = TimeSeriesDict.read("path/to/data.mseed")

# Explicit format
tsd = TimeSeriesDict.read("path/to/data.dat", format="mseed")

# Write out
tsd.write("output.h5", format="hdf5")

# Network path
ts = TimeSeries.fetch_open_data("H1", 1126259446, 1126259478)
```

- `.read()` / `.write()` uses the gwpy-style I/O registry.
- `.xml` is ambiguous, so **use `format="xml.diaggui"` explicitly** for DiagGUI XML data.
- `NDS2` and `GWOSC` are not file readers, so they use `fetch()` / `fetch_open_data()` instead of `.read()`.

(io-formats-en-supported-classes)=
## Supported Classes at a Glance

If the main question is whether a format is for a single channel or multiple channels, use this table first.

| Format / Family | Single | Multi | Other classes |
|---|---|---|---|
| **GWF / mseed / SAC / GSE2 / K-NET / WIN / WIN32 / ATS / SDB / SQLite / SQLite3 / WAV / Audio** | `TimeSeries` | `TimeSeriesDict` | Baseline end-user direct I/O pattern |
| **CSV** | `TimeSeries` | `TimeSeriesDict` | `TimeSeriesDict` also supports manifest-backed collection directories |
| **TXT** | `TimeSeries` | `TimeSeriesDict` | Multi-channel direct I/O uses collection directories |
| **nc / Zarr / GBD / TDMS** | `TimeSeries` | `TimeSeriesDict`, `TimeSeriesMatrix` | Includes matrix-style direct I/O |
| **HDF5** | `TimeSeries`, `FrequencySeries`, and related classes | `TimeSeriesDict` and related collections | Also covers `Spectrogram`, `Histogram`, and `EventTable` |
| **hdf.ndscope** | - | `TimeSeriesDict` | ndscope-compatible schema; aliases: `ndscope-hdf5`, `ndscope_hdf5`, `ndscopehdf5` |
| **xml.diaggui** | - | `TimeSeriesDict` | Requires `products`; legacy alias: `dttxml` |
| **NDS2 / GWOSC** | `TimeSeries` | - | Use `fetch()` / `fetch_open_data()` |
| **ATS.MTH5** | `TimeSeries` | - | Partial single-path support |
| **ROOT** | `EventTable` | - | Direct I/O is limited to EventTable |

- If you are unsure, start by thinking in terms of `TimeSeries` and `TimeSeriesDict`.
- `TimeSeriesMatrix` mainly matters for `NetCDF4`, `Zarr`, `GBD`, and `TDMS`.
- If you need to preserve richer objects beyond Series classes, start with **HDF5**.

<a id="io-formats-en-a"></a>

## A. GW Standards

These are the standard GW storage, exchange, and acquisition paths.
If you are unsure, start with **HDF5**. Use **GWF** when you need external standard compatibility, and **DTTXML** for diagnostic tool output.

| Format / Path | R / W | Main entry | Best for | Notes |
|---|:---:|---|---|---|
| **GWF** (`.gwf`) | ○ / ○ | `TimeSeries.read()`, `TimeSeriesDict.read()`, `.write()` | Standard LIGO/KAGRA frame exchange | Standard format, via gwpy |
| **HDF5** (`.h5`, `.hdf5`) | ○ / ○ | `.read(..., format="hdf5")`, `.write(..., format="hdf5")` on major classes | Long-term storage with metadata | Prefer explicit `format="hdf5"` |
| **hdf.ndscope** (`.h5`, `.hdf5`) | ○ / ○ | `TimeSeriesDict.read(..., format="hdf.ndscope")`, `.write(..., format="hdf.ndscope")` | ndscope-compatible HDF5 | `TimeSeriesDict` only. Legacy aliases: `ndscope-hdf5`, `ndscope_hdf5`, `ndscopehdf5` |
| **xml.diaggui** (`.xml`, `.xml.gz`) | ○ / × | `TimeSeriesDict.read(..., format="xml.diaggui", products="...")` | DiagGUI / DTT outputs | `products` is required; legacy alias: `dttxml` |
| **NDS2** | ○ / × | `TimeSeries.fetch()` | Detector data server access | Network path |
| **GWOSC** | ○ / × | `TimeSeries.fetch_open_data()` | Open data access | Network path |

- Purpose: compare the main GW-oriented direct-I/O and network entry points
- Input: HDF5, GWF, DTTXML, or detector/open-data access parameters
- Output: `TimeSeries`, `TimeSeriesDict`, or fetched open data

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.timeseries import TimeSeries

tsd = TimeSeriesDict.read("data.h5", format="hdf5")
frame = TimeSeriesDict.read("data.gwf", format="gwf")
dtt = TimeSeriesDict.read("diag.xml", format="xml.diaggui", products="TS")
open_data = TimeSeries.fetch_open_data("H1", 1126259446, 1126259478)
```

- **HDF5** is the safest general recommendation for structured GW data.
- **DTTXML** changes behavior depending on `products`. For complex transfer functions, prefer `native=True` on the frequency-domain side.
- **NDS2 / GWOSC** are shown inside group A, but explicitly marked as `network path` rather than file formats.

<a id="io-formats-en-b"></a>

## B. Seismic and Geophysical Observation

This group is for existing seismic and electromagnetic observation formats.
In practice, **MiniSEED** is the easiest starting point when you need to place a format in context.

| Format | R / W | Main entry | Best for | Notes |
|---|:---:|---|---|---|
| **mseed** (`.mseed`) | ○ / ○ | `TimeSeriesDict.read(..., format="mseed")`, `.write(..., format="mseed")` | Standard seismic waveform exchange | `gap` controls gap handling; legacy alias: `miniseed` |
| **SAC** (`.sac`) | ○ / ○ | `TimeSeriesDict.read(..., format="sac")`, `.write(..., format="sac")` | Seismic waveform analysis | Via ObsPy |
| **GSE2** (`.gse2`) | ○ / ○ | `TimeSeriesDict.read(..., format="gse2")`, `.write(..., format="gse2")` | Seismic waveform exchange | Via ObsPy |
| **K-NET** (`.knet`) | ○ / × | `TimeSeriesDict.read(..., format="knet")` | Strong-motion records | Read-only |
| **WIN / WIN32** (`.win`, `.cnt`) | ○ / × | `TimeSeriesDict.read(..., format="win")`, `TimeSeriesDict.read(..., format="win32")` | Japanese WIN datasets | Improved parser, read-only |
| **ATS** (`.ats`) | ○ / × | `TimeSeries.read(..., format="ats")`, `TimeSeriesDict.read(..., format="ats")` | Metronix observation data | Native binary reader |
| **ATS.MTH5** (`format="ats.mth5"`) | ○ / × | `TimeSeries.read(..., format="ats.mth5")` | Single MTH5-backed path | Partial support |
| **MTH5 standalone** (`.h5`) | In progress | Dedicated `format="mth5"` not yet exposed | Future general MTH5 direct I/O | **Not currently a public direct-I/O format**. The only direct path today is `ats.mth5` |

- Purpose: compare common seismic and geophysical readers without overstating MTH5 support
- Input: existing waveform files such as MiniSEED, WIN/WIN32, or the limited `ats.mth5` path
- Output: `TimeSeries` or `TimeSeriesDict` objects depending on the reader

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.timeseries import TimeSeries

tsd = TimeSeriesDict.read("data.mseed", format="mseed", gap="pad")
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

| Format | R / W | Main entry | Best for | Notes |
|---|:---:|---|---|---|
| **CSV** (`.csv`) | ○ / ○ | `TimeSeries.read(..., format="csv")`, `TimeSeriesDict.read(..., format="csv")`, `TimeSeriesDict.write(..., format="csv")` | Lightweight exchange and inspection | `TimeSeriesDict` supports single-file reads and collection-directory layouts |
| **TXT** (`.txt`) | ○ / ○ | `TimeSeries.read(..., format="txt")`, `TimeSeriesDict.read(dir, format="txt")`, `TimeSeriesDict.write(dir, format="txt")` | Plain-text exchange | Multi-channel direct I/O uses collection directories |
| **nc** (`.nc`) | ○ / ○ | `TimeSeries.read(..., format="nc")`, `TimeSeriesDict.read(..., format="nc")`, `TimeSeriesMatrix.read(..., format="nc")`, `.write(..., format="nc")` | Scientific storage for time-series-oriented data | Direct I/O here is centered on TimeSeries classes; legacy alias: `netcdf4` |
| **Zarr** (`.zarr`) | ○ / ○ | `TimeSeries.read(..., format="zarr")`, `TimeSeriesDict.read(..., format="zarr")`, `TimeSeriesMatrix.read(..., format="zarr")`, `.write(..., format="zarr")` | Chunked storage and parallel workflows | Direct I/O here is centered on TimeSeries classes |
| **ROOT** (`.root`) | ○ / ○ | `EventTable.read(..., format="root")`, `EventTable.write(..., format="root")` | EventTable I/O | Direct I/O here is EventTable only |

- Purpose: show the general-purpose direct-I/O routes without mixing them with interop-only bridges
- Input: CSV, Zarr, ROOT, or other general exchange formats
- Output: `TimeSeriesDict`, `TimeSeriesMatrix`, or `EventTable`

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.table import EventTable

ascii_data = TimeSeriesDict.read("data.csv")
chunked = TimeSeriesDict.read("data.zarr", format="zarr")
events = EventTable.read("events.root", format="root")
```

- **CSV** remains useful for lightweight exchange and inspection.
- **TXT** direct I/O is more limited: single-series paths are explicit `format="txt"`, and multi-channel paths use collection directories.
- **Pickle** portability notes still exist in class references, but Pickle is not a published direct `.read()` / `.write()` format on this page.
- **NetCDF4 / Zarr** are treated here only as **direct TimeSeries-style I/O**. Field/xarray bridges belong to interop.
- **Zarr** direct I/O now expects per-array timing metadata explicitly. `sample_rate` is the primary key, `dt` is accepted as a fallback, and reads raise `ValueError` if neither is present unless you intentionally recover a legacy store with `sample_rate_override=...` or `dt_override=...`.
- **ROOT** object-level export/import belongs to interop. This page only covers EventTable direct I/O.

<a id="io-formats-en-d"></a>

## D. Loggers and Instrument Formats

This group is for logger and instrument-specific time-series formats.
Time handling, units, and audio `t0` semantics are the main points to watch.

| Format | R / W | Main entry | Best for | Notes |
|---|:---:|---|---|---|
| **GBD** (`.gbd`) | ○ / × | `TimeSeries.read(..., format="gbd", timezone=...)`, `TimeSeriesDict.read(..., format="gbd", timezone=...)`, `TimeSeriesMatrix.read(..., format="gbd", timezone=...)` | GRAPHTEC loggers | `timezone` is required for published reads |
| **TDMS** (`.tdms`) | ○ / × | `TimeSeries.read(..., format="tdms")`, `TimeSeriesDict.read(..., format="tdms")`, `TimeSeriesMatrix.read(..., format="tdms")` | National Instruments data | Read-only; requires `nptdms` |
| **SDB / SQLite / SQLite3** (`.sdb`, `.sqlite`, `.sqlite3`) | ○ / × | `TimeSeries.read(..., format="sdb" / "sqlite" / "sqlite3")`, `TimeSeriesDict.read(..., format="sdb" / "sqlite" / "sqlite3")` | WeeWX and similar archives | Same reader family; public direct I/O is read-only |
| **WAV** (`.wav`) | ○ / ○ | `TimeSeries.read(..., format="wav")`, `TimeSeriesDict.read(..., format="wav")`, `TimeSeries.write(..., format="wav")` | Uncompressed audio | Public write is single-series only; does not preserve absolute time |
| **MP3 / FLAC / OGG / M4A** | ○ / ○ | `TimeSeries.read(..., format="mp3" / "flac" / "ogg" / "m4a")`, `TimeSeriesDict.read(..., format=...)`, `.write(...)` | Compressed audio | Uses `pydub`; some formats also need `ffmpeg` |

- Purpose: highlight logger-specific and audio-specific direct-I/O requirements
- Input: logger data, SQLite-family archives, or audio files
- Output: `TimeSeries`, `TimeSeriesDict`, or `TimeSeriesMatrix`

```python
from gwexpy.timeseries.collections import TimeSeriesDict

logger = TimeSeriesDict.read("data.gbd", timezone="Asia/Tokyo")
weather = TimeSeriesDict.read("archive.sqlite3", format="sqlite3")
audio = TimeSeriesDict.read("sound.flac", format="flac")
```

- **GBD** requires `timezone`.
- **TDMS** requires the optional `nptdms` dependency.
- **MP3 / FLAC / OGG / M4A** require the optional `pydub` dependency. MP3/M4A commonly also need `ffmpeg`.
- **SDB / SQLite / SQLite3** should all be named explicitly in the public page so users do not need to infer aliases.
- **WAV / compressed-audio formats** do not preserve absolute timestamps. Reading with `t0=0.0` is a convenience convention, not a claim that the source had an absolute epoch.

<a id="io-formats-en-dev"></a>

## Developer Notes

Most users can skip this section.
It exists mainly to collect not-yet-prominent implementations and placeholders in one place.

### Managed in design, but not prominent in the public page

| Format | Status | Notes |
|---|---|---|
| `hdf.ndscope` | Implemented, not yet prominent | `TimeSeriesDict`-only HDF5 schema. Legacy aliases: `ndscope-hdf5`, `ndscope_hdf5`, `ndscopehdf5` |
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

- [Interop tutorial](tutorials/intro_interop.ipynb)
- [Interop API reference](../reference/api/interop)
- [Verification and Quality Signals](verification_and_quality.md)
- [Installation guide](installation)

## Next to Read

- [Interop / Conversion Guide](interop) for `to_*()` / `from_*()` bridges and object-level conversion
- [GPS Time Utility Functions](time_utilities) if your I/O workflow needs timezone or GPS-time handling
- [Installation guide](installation) if you need optional dependencies before using a format backend

## Page-End Navigation

- <a href="#io-formats-en-quick">Back to Quick Selection Table</a>
- <a href="#io-formats-en-basic">Back to Basic Usage</a>
- <a href="#io-formats-en-top">Back to Top</a>
