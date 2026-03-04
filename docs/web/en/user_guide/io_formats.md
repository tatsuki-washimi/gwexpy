# File I/O Supported Formats Guide

A comprehensive guide to all file formats supported by gwexpy, including how to read and write each format.
This page covers only the end-user API (`.read()` / `.write()` class methods) and does not expose internal implementation details.

---

## Supported Formats Overview

| Format | Extension | Read | Write | Recommended Class / Method | Dependencies | Notes |
|---|---|:---:|:---:|---|---|---|
| GWF | `.gwf` | Y | Y | `TimeSeriesDict.read()` / `.write()` | — (gwpy built-in) | gwpy standard format |
| HDF5 | `.h5`, `.hdf5` | Y | Y | `TimeSeriesDict.read(format="hdf5")` | — (gwpy built-in) | gwpy standard |
| LIGO_LW XML (DTTXML) | `.xml`, `.xml.gz` | Y | N | `TimeSeriesDict.read(format="dttxml")` / `FrequencySeriesDict.read(format="dttxml")` | — | `products` argument required |
| CSV / TXT | `.csv`, `.txt` | Y | Y | `TimeSeriesDict.read(format="csv")` | — (gwpy built-in) | ASCII format. Supports directory loading |
| Pickle | `.pkl` | Y | Y | `TimeSeries.read(format="pickle")` | — | Python serialization |
| WAV | `.wav` | Y | Y | `TimeSeriesDict.read("file.wav")` | scipy | `t0` always 0 (no absolute time) |
| MiniSEED | `.mseed` | Y | Y | `TimeSeriesDict.read(format="miniseed")` | ObsPy | Seismic waveform format |
| SAC | `.sac` | Y | Y | `TimeSeriesDict.read(format="sac")` | ObsPy | Seismic waveform format |
| GSE2 | `.gse2` | Y | Y | `TimeSeriesDict.read(format="gse2")` | ObsPy | Seismic waveform format |
| KNET | `.knet` | Y | N | `TimeSeriesDict.read(format="knet")` | ObsPy | K-NET strong motion records |
| GBD | `.gbd` | Y | N | `TimeSeriesDict.read("file.gbd", timezone=...)` | — | `timezone` required |
| WIN / WIN32 | `.win`, `.cnt` | Y | N | `TimeSeriesDict.read(format="win")` | ObsPy | NIED WIN format (improved parser) |
| MTH5 | `.h5` | Y | Y | `TimeSeries.read(format="ats.mth5")` | mth5 | Magnetometer data (design-level support) |
| ATS | `.ats` | Y | N | `TimeSeriesDict.read("file.ats")` | — | Metronix binary parser |
| ROOT | `.root` | Y | Y | `EventTable.read(format="root")` | — (via gwpy) | CERN ROOT tables |
| SQLite / SDB | `.sdb`, `.sqlite`, `.db` | Y | N | `TimeSeriesDict.read("file.sdb")` | — | WeeWX / Davis weather data |
| NetCDF4 | `.nc` | Y | Y | `TimeSeriesDict.read(format="netcdf4")` | xarray, netcdf4 | Auto-detects time dimension |
| Zarr | `.zarr` | Y | Y | `TimeSeriesDict.read(format="zarr")` | zarr | Cloud-optimized chunked arrays |
| Audio (MP3, FLAC, etc.) | `.mp3`, `.flac`, `.ogg`, `.m4a` | Y | Y | `TimeSeriesDict.read("file.mp3")` | pydub (+ffmpeg) | `t0` always 0 (same as WAV) |
| NDS2 | (network) | Y | N | `TimeSeries.fetch(...)` | nds2-client | Network data server |
| TDMS | `.tdms` | Y | N | `TimeSeriesDict.read("file.tdms")` | npTDMS | National Instruments |
| ORF | `.orf` | N | N | — | — | Not implemented (stub) |

> **Note**: Formats marked "gwpy built-in" (GWF, HDF5, CSV/TXT, Pickle) are handled through gwpy's built-in IO pathways. Since gwexpy extends gwpy, these are available out of the box.

---

## Format Details

---

### GBD — GRAPHTEC Data Logger `.gbd`

**Extension**: `.gbd`
**Read/Write**: Read Y / Write N
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/logger.gbd", timezone="Asia/Tokyo")
```

**Required arguments**:
- `timezone` (str or tzinfo) — Logger's local timezone (IANA name, e.g. `"Asia/Tokyo"` or UTC offset). **Required**. Omitting raises `ValueError`.

**Key optional arguments**:
- `channels` (iterable[str], optional) — List of channel names to load. Defaults to all channels.
- `digital_channels` (iterable[str], optional) — Channel names to treat as digital. Defaults to auto-detecting `ALARM`, `ALARMOUT`, `PULSE*`, `LOGIC*`.
- `unit` (str or Unit, optional) — Physical unit override. Default `'V'`.
- `epoch` (float or datetime, optional) — Epoch (GPS seconds) override. Datetimes are converted to GPS.
- `pad` (float, optional) — Padding value. Default `NaN`.

**Dependencies**: None (native implementation)

**Notes**:
- Digital channels (`ALARM`, `PULSE*`, etc.) are binarized to 0/1.
- If the `HeaderSiz` field is missing from the header, a `ValueError` is raised.
- Scale factors are automatically extracted from the AMP section.

**Implementation**: `gwexpy/timeseries/io/gbd.py`

---

### ATS — Metronix `.ats`

**Extension**: `.ats`
**Read/Write**: Read Y / Write N
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.ats")
```

Reading via the mth5 library (for `.atss` files):
```python
from gwexpy.timeseries import TimeSeries

ts = TimeSeries.read("path/to/data.atss", format="ats.mth5")
```

**Required arguments**: None

**Key optional arguments**: None (metadata is automatically extracted from the binary header)

**Dependencies**:
- Standard reader: None (native binary parser)
- `ats.mth5` format: Requires the `mth5` library. Raises `ImportError` if not installed.

**Notes**:
- Supports ATS header versions 80/81. CEA/sliced headers (version 1080) are not supported (`NotImplementedError`).
- LSB values (mV/count) are automatically converted to Volts (V).
- Channel names are auto-generated from header info in the format `Metronix_{system}_{serial}_{type}_{sensor}_{serial}`.
- The `ats.mth5` format requires mth5's filename conventions. Use the default binary parser if file names do not conform.

**Implementation**: `gwexpy/timeseries/io/ats.py`

---

### SDB — WeeWX / Davis Weather Station `.sdb`

**Extension**: `.sdb`, `.sqlite`, `.sqlite3`
**Read/Write**: Read Y / Write N
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/weewx.sdb")
```

**Required arguments**: None

**Key optional arguments**:
- `table` (str, optional) — Table name to read. Default `'archive'`.
- `columns` (list[str], optional) — Column names to read. Defaults to known weather columns (`barometer`, `outTemp`, `windSpeed`, etc.).

**Dependencies**: None (uses stdlib `sqlite3` + `pandas`)

**Notes**:
- Automatic conversion from Imperial to SI units (e.g. degF to degC, inHg to hPa, mph to m/s, inch to mm).
- The table must have a `dateTime` column (UNIX timestamp).
- Sample rate is auto-estimated from the median `dateTime` interval.

**Implementation**: `gwexpy/timeseries/io/sdb.py`

---

### TDMS — National Instruments `.tdms`

**Extension**: `.tdms`
**Read/Write**: Read Y / Write N
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.tdms")
```

**Required arguments**: None

**Key optional arguments**:
- `channels` (list[str], optional) — Channels to read. Channel names use the format `"GroupName/ChannelName"`. Defaults to all channels.
- `unit` (str, optional) — Physical unit override.

**Dependencies**: `npTDMS` — Raises `ImportError` if not installed, with a message suggesting `pip install nptdms`.

**Notes**:
- Channel names are stored in `"GroupName/ChannelName"` format.
- `wf_increment` (sample interval) and `wf_start_time` (start time) are read from TDMS properties.
- Start times as `numpy.datetime64` or `datetime` are automatically converted to GPS time.

**Implementation**: `gwexpy/timeseries/io/tdms.py`

---

### WAV — Audio File `.wav`

**Extension**: `.wav`
**Read/Write**: Read Y / Write Y
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/audio.wav")
```

**Required arguments**: None

**Key optional arguments**: None

**Dependencies**: `scipy` (`scipy.io.wavfile`) — included in gwexpy's recommended dependencies.

**Notes**:
- `t0` is always set to `0.0` (GPS seconds). WAV files do not carry absolute timestamps.
- Multi-channel files have channel names `channel_0`, `channel_1`, etc.
- Mono files are automatically loaded as a single channel.
- Writing uses the gwpy standard WAV writer pathway.

**Implementation**: `gwexpy/timeseries/io/wav.py`

---

### MiniSEED — Seismic Waveform `.mseed`

**Extension**: `.mseed`
**Read/Write**: Read Y / Write Y
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

# Read
tsd = TimeSeriesDict.read("path/to/data.mseed", format="miniseed")

# Write
tsd.write("output.mseed", format="miniseed")
```

**Required arguments**: None

**Key optional arguments**:
- `channels` (list[str], optional) — Channels to read. Specify by trace ID (`NET.STA.LOC.CHA`) or channel code.
- `unit` (str, optional) — Physical unit override.
- `epoch` (float or datetime, optional) — Epoch override.
- `timezone` (str, optional) — Timezone specification.
- `pad` (float, optional) — Gap fill value. Default `NaN`.
- `gap` (str, optional) — Gap handling method. `"pad"` (default) or `"raise"`.

**Dependencies**: `ObsPy` — Raises `ImportError` if not installed, with a message suggesting `pip install obspy`.

**Notes**:
- Data is read via ObsPy's `read()` function.
- Gaps are padded with `NaN` by default. Use `gap="raise"` to raise an error instead.
- Automatic trace merging (`merge(method=1)`) is applied.

**Implementation**: `gwexpy/timeseries/io/seismic.py`

---

### SAC — Seismic Waveform `.sac`

**Extension**: `.sac`
**Read/Write**: Read Y / Write Y
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.sac", format="sac")
tsd.write("output.sac", format="sac")
```

**Required arguments**: None

**Key optional arguments**: Same as MiniSEED (`channels`, `unit`, `epoch`, `timezone`, `pad`, `gap`).

**Dependencies**: `ObsPy`

**Notes**:
- SAC is typically one trace per file. Multi-trace writing behavior depends on ObsPy.

**Implementation**: `gwexpy/timeseries/io/seismic.py`

---

### GSE2 — Seismic Waveform `.gse2`

**Extension**: `.gse2`
**Read/Write**: Read Y / Write Y
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.gse2", format="gse2")
tsd.write("output.gse2", format="gse2")
```

**Required arguments**: None

**Key optional arguments**: Same as MiniSEED (`channels`, `unit`, `epoch`, `timezone`, `pad`, `gap`).

**Dependencies**: `ObsPy`

**Implementation**: `gwexpy/timeseries/io/seismic.py`

---

### KNET — K-NET Strong Motion Records `.knet`

**Extension**: `.knet`
**Read/Write**: Read Y / Write N
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.knet", format="knet")
```

**Required arguments**: None

**Key optional arguments**: Same as MiniSEED (`channels`, `unit`, `epoch`, `timezone`, `pad`, `gap`).

**Dependencies**: `ObsPy`

**Notes**:
- Read-only. No writer is registered.

**Implementation**: `gwexpy/timeseries/io/seismic.py`

---

### WIN / WIN32 — NIED WIN Format `.win` / `.cnt`

**Extension**: `.win`, `.cnt`
**Read/Write**: Read Y / Write N
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.win", format="win")
# or
tsd = TimeSeriesDict.read("path/to/data.cnt", format="win32")
```

**Required arguments**: None

**Key optional arguments**:
- `century` (str, optional) — Century part of the year. Default `"20"`.

**Dependencies**: `ObsPy` — If ObsPy is not installed, the reader is not registered and an `ImportError` is raised.

**Notes**:
- Uses gwexpy's improved parser over ObsPy's standard WIN reader, with the following fixes:
  - 0.5-byte (4-bit) delta decode: fixed lower nibble sign handling; skip unused nibble on odd delta counts.
  - 3-byte (24-bit) delta decode: fixed operator precedence and sign-preserving unpack/shift.
- Gaps are merged with `NaN`.

**Implementation**: `gwexpy/timeseries/io/win.py`

---

### DTTXML — Diag DTT XML (Time Series & Frequency Series)

**Extension**: `.xml`, `.xml.gz`
**Read/Write**: Read Y / Write N
**Recommended API (time series)**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/dtt_output.xml", format="dttxml", products="TS")
```

**Recommended API (frequency series)**:
```python
from gwexpy.frequencyseries.collections import FrequencySeriesDict

fsd = FrequencySeriesDict.read("path/to/dtt_output.xml", format="dttxml", products="PSD")
```

```python
from gwexpy.frequencyseries.matrix import FrequencySeriesMatrix

fsm = FrequencySeriesMatrix.read("path/to/dtt_output.xml", format="dttxml", products="TF")
```

**Required arguments**:
- `products` (str) — Product type to extract. **Required**. Raises `ValueError` if omitted.
  - Time series: `"TS"`
  - Frequency series: `"PSD"`, `"ASD"`, `"FFT"`
  - Matrix: `"TF"`, `"STF"`, `"CSD"`, `"COH"`

**Key optional arguments**:
- `channels` (iterable[str], optional) — List of channels to read.
- `unit` (str, optional) — Physical unit override.
- `epoch` (float or datetime, optional) — Epoch override.
- `timezone` (str, optional) — Timezone specification.
- `native` (bool, optional) — If `True`, uses gwexpy's native XML parser. Recommended for complex TF data (subtype 6 phase loss fix). Default `False`. (FrequencySeriesDict / FrequencySeriesMatrix only)
- `rows`, `cols`, `pairs` — Matrix filtering (FrequencySeriesMatrix only).

**Dependencies**: None (native implementation)

**Notes**:
- Supports both time-series and frequency-domain data. The output type is determined by the `products` value.
- The `.xml` extension is auto-identified (`format="dttxml"` can be omitted).

**Implementation**: `gwexpy/timeseries/io/dttxml.py`, `gwexpy/frequencyseries/io/dttxml.py`

---

### GWF — Gravitational Wave Frame `.gwf`

**Extension**: `.gwf`
**Read/Write**: Read Y / Write Y
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.gwf", format="gwf")
tsd.write("output.gwf", format="gwf")
```

**Required arguments**: None (follows gwpy standard arguments)

**Dependencies**: — (gwpy standard. Internally uses `python-ldas-tools-framecpp`, etc.)

**Notes**:
- Handled through gwpy's standard IO pathway. No custom reader/writer in gwexpy.

**Implementation**: gwpy standard (`gwpy/timeseries/io/gwf.py`)

---

### HDF5 — General Scientific Data `.h5` / `.hdf5`

**Extension**: `.h5`, `.hdf5`
**Read/Write**: Read Y / Write Y
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.h5", format="hdf5")
tsd.write("output.h5", format="hdf5")
```

**Required arguments**: None

**Dependencies**: `h5py` (included in gwexpy's required dependencies)

**Notes**:
- gwexpy's `TimeSeriesDict.read()` has extended HDF5 loading logic.
- Supports automatic layout detection (`LAYOUT_DATASET` / `LAYOUT_GROUP`).
- Supports keymap and ordering restoration.

**Implementation**: gwpy standard (`gwpy/timeseries/io/hdf5.py`) + gwexpy extension (`gwexpy/timeseries/collections.py`)

---

### CSV / TXT — ASCII Text `.csv` / `.txt`

**Extension**: `.csv`, `.txt`
**Read/Write**: Read Y / Write Y
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

# Single file
tsd = TimeSeriesDict.read("path/to/data.csv", format="csv")

# Load all files from a directory
tsd = TimeSeriesDict.read("path/to/data_dir/")
```

**Required arguments**: None

**Dependencies**: None (gwpy standard)

**Notes**:
- gwexpy supports loading all CSV/TXT files from a directory as a single `TimeSeriesDict`.

**Implementation**: gwpy standard (`gwpy/timeseries/io/ascii.py`) + gwexpy extension (`gwexpy/timeseries/collections.py`)

---

### Pickle — Python Serialization `.pkl`

**Extension**: `.pkl`
**Read/Write**: Read Y / Write Y
**Recommended API**:
```python
from gwexpy.timeseries import TimeSeries

ts = TimeSeries.read("path/to/data.pkl", format="pickle")
ts.write("output.pkl", format="pickle")
```

**Required arguments**: None

**Dependencies**: None

**Notes**:
- Uses gwpy's standard serialization pathway. Only use with files from trusted sources (standard pickle security considerations apply).

**Implementation**: gwpy standard

---

### ROOT — CERN ROOT `.root`

**Extension**: `.root`
**Read/Write**: Read Y / Write Y
**Recommended API**:
```python
from gwexpy.table import EventTable

table = EventTable.read("path/to/data.root", format="root")
```

**Dependencies**: — (via gwpy's table/root pathway)

**Notes**:
- gwexpy's `table/io/root.py` is a re-export of gwpy's module of the same name.
- Primarily used for event table data.

**Implementation**: gwpy standard (`gwpy/table/io/root.py`) — via gwexpy: `gwexpy/table/io/root.py`

---

### NDS2 — Network Data Server

**Extension**: None (network protocol)
**Read/Write**: Read Y / Write N
**Recommended API**:
```python
from gwexpy.timeseries import TimeSeries

ts = TimeSeries.fetch("channel_name", start, end)
```

**Dependencies**: `nds2-client` (Python bindings)

**Notes**:
- Network-based data retrieval, not file I/O.
- Uses gwpy's standard `fetch()` method.

**Implementation**: gwpy standard (`gwpy/timeseries/io/nds2.py`)

---

### Audio — MP3 / FLAC / OGG / M4A (via pydub)

**Extension**: `.mp3`, `.flac`, `.ogg`, `.m4a`
**Read/Write**: Read Y / Write Y
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

# Read (auto-detected by extension)
tsd = TimeSeriesDict.read("path/to/audio.mp3")

# Explicit format
tsd = TimeSeriesDict.read("path/to/audio.dat", format="flac")

# Write
tsd.write("output.flac", format="flac")
```

**Required arguments**: None

**Key optional arguments**:
- `channels` (iterable[str], optional) — Channel names to read (e.g. `"channel_0"`).
- `unit` (str, optional) — Physical unit override.

**Dependencies**: `pydub` — Raises `ImportError` if not installed, with a message suggesting `pip install pydub`. MP3/M4A encoding also requires `ffmpeg` (`apt install ffmpeg` or equivalent). FLAC may work without ffmpeg.

**Notes**:
- `t0` is always set to `0.0` (GPS seconds). Audio files carry no absolute timestamps (same behavior as WAV).
- On read, sample values are normalized to the `[-1.0, 1.0]` range.
- Multi-channel files use channel names `channel_0`, `channel_1`, etc.
- On write, data is rescaled to peak-normalized 16-bit PCM.

**Implementation**: `gwexpy/timeseries/io/audio.py`

---

### NetCDF4 — Scientific Data `.nc` (via xarray)

**Extension**: `.nc`
**Read/Write**: Read Y / Write Y
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.nc", format="netcdf4")
tsd.write("output.nc", format="netcdf4")
```

**Required arguments**: None

**Key optional arguments**:
- `channels` (list[str], optional) — Variable names to read. Defaults to all variables with a time dimension.
- `unit` (str, optional) — Physical unit override. Defaults to the file's `units` attribute.
- `time_coord` (str, optional) — Name of the time coordinate. Auto-detected if omitted (looks for `"time"`).

**Dependencies**: `xarray` + `netcdf4` — Raises `ImportError` if not installed, with a message suggesting `pip install xarray netcdf4`.

**Notes**:
- Only variables with a time dimension (`time`) are converted to `TimeSeries`. Variables without a time dimension are skipped.
- Time coordinates are automatically converted from `datetime64` to GPS seconds.
- Multi-dimensional variables (time + spatial, etc.) are flattened along non-time dimensions, resulting in `varname_0`, `varname_1`, etc.
- On write, the `datetime64` time coordinate is reconstructed from `t0` and `dt`.

**Implementation**: `gwexpy/timeseries/io/netcdf4_.py`

---

### Zarr — Cloud-Optimized Chunked Arrays `.zarr`

**Extension**: `.zarr` (directory store)
**Read/Write**: Read Y / Write Y
**Recommended API**:
```python
from gwexpy.timeseries.collections import TimeSeriesDict

tsd = TimeSeriesDict.read("path/to/data.zarr", format="zarr")
tsd.write("output.zarr", format="zarr")
```

**Required arguments**: None

**Key optional arguments**:
- `channels` (list[str], optional) — Array names to read. Defaults to all arrays in the root group.
- `unit` (str, optional) — Physical unit override.

**Dependencies**: `zarr` — Raises `ImportError` if not installed, with a message suggesting `pip install zarr`.

**Notes**:
- gwexpy Zarr convention: each array in the root group corresponds to one channel. Array `attrs` store `sample_rate` (Hz) and `t0` (GPS seconds).
- If `sample_rate` is not set, the inverse of `dt` is used; if neither is present, defaults to 1 Hz. `t0` defaults to `0.0`.
- Supports all store types supported by the zarr library (directory stores, zip stores, etc.).
- On write, `sample_rate`, `t0`, `dt`, and `unit` are saved as array attributes.

**Implementation**: `gwexpy/timeseries/io/zarr_.py`

---

## Design-Level Support (No Dedicated Implementation in gwexpy)

The following formats are listed in the design specification (`io_support.csv`) but do not have dedicated reader/writer implementations in the gwexpy repository at this time.

| Format | Extension | Designed Read/Write | Notes |
|---|---|:---:|---|
| MTH5 | `.h5` | Read Y / Write Y | Via `mth5` library. Partially supported through `ats.mth5` |

---

## Unimplemented (Stub) Formats

The following formats are registered as placeholders (stubs) in the IO registry. Calling `.read()` raises an `UnimplementedIOError` or `NotImplementedError`. These are reserved for future implementation pending specification or sample files.

### Time Series Stubs (`gwexpy/timeseries/io/stubs.py`)

| Format Name | Registered Classes | Notes |
|---|---|---|
| `orf` | TimeSeries, TimeSeriesDict, TimeSeriesMatrix | ORF format |
| `mem` | TimeSeries, TimeSeriesDict, TimeSeriesMatrix | MEM format |
| `wvf` | TimeSeries, TimeSeriesDict, TimeSeriesMatrix | WVF format |
| `wdf` | TimeSeries, TimeSeriesDict, TimeSeriesMatrix | WDF format |
| `taffmat` | TimeSeries, TimeSeriesDict, TimeSeriesMatrix | TAFFMAT format |
| `lsf` | TimeSeries, TimeSeriesDict, TimeSeriesMatrix | LSF format |
| `li` | TimeSeries, TimeSeriesDict, TimeSeriesMatrix | LI format |

### Frequency Series Stubs (`gwexpy/frequencyseries/io/stubs.py`)

| Format Name | Registered Classes | Notes |
|---|---|---|
| `win` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | Implemented for time series but not frequency domain |
| `win32` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | Same as above |
| `sdb` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | Implemented for time series but not frequency domain |
| `orf` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | Not implemented |
| `mem` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | Not implemented |
| `wvf` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | Not implemented |
| `wdf` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | Not implemented |
| `taffmat` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | Not implemented |
| `lsf` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | Not implemented |
| `li` | FrequencySeries, FrequencySeriesDict, FrequencySeriesMatrix | Not implemented |

---

## Basic Usage of `.read()` / `.write()`

All gwexpy data classes use gwpy's IO registry. Here is the basic usage pattern.

### Reading

```python
from gwexpy.timeseries.collections import TimeSeriesDict
from gwexpy.timeseries import TimeSeries

# Auto-detect format from extension
tsd = TimeSeriesDict.read("path/to/file.gbd", timezone="Asia/Tokyo")

# Explicitly specify format
tsd = TimeSeriesDict.read("path/to/file.dat", format="miniseed")

# Read a single channel
ts = TimeSeries.read("path/to/file.gbd", timezone="Asia/Tokyo")
```

### Writing

```python
# Write to a supported format
tsd.write("output.mseed", format="miniseed")
tsd.write("output.h5", format="hdf5")
```

### Frequency Series

```python
from gwexpy.frequencyseries.collections import FrequencySeriesDict
from gwexpy.frequencyseries.frequencyseries import FrequencySeries

fsd = FrequencySeriesDict.read("path/to/dtt.xml", format="dttxml", products="PSD")
```

### How the `format` Argument Works

- **Omitted**: The format is auto-detected from the file extension using identifier functions registered via `io_registry.register_identifier(...)`.
- **Specified**: The reader/writer registered for that format name is called directly.
- **When auto-detection fails** (e.g. `.xml` is used by formats other than DTTXML): specify `format` explicitly.

---

## Reference: Source Implementation Files

Implementation files referenced in the creation of this document.

| Module Path | Summary |
|---|---|
| `gwexpy/timeseries/io/__init__.py` | Time series IO module registration entry point |
| `gwexpy/timeseries/io/gbd.py` | GBD reader. `timezone` is required |
| `gwexpy/timeseries/io/ats.py` | ATS reader (binary parser). `ats.mth5` variant available |
| `gwexpy/timeseries/io/sdb.py` | SDB reader (WeeWX SQLite). Registered under `sdb`, `sqlite`, `sqlite3` |
| `gwexpy/timeseries/io/tdms.py` | TDMS reader. Depends on `npTDMS` |
| `gwexpy/timeseries/io/wav.py` | WAV reader. `scipy.io.wavfile` wrapper. `t0=0` fixed |
| `gwexpy/timeseries/io/seismic.py` | MiniSEED / SAC / GSE2 / KNET reader/writer. Depends on ObsPy |
| `gwexpy/timeseries/io/win.py` | WIN/WIN32 reader. Requires ObsPy. Improved 4bit/24bit delta decode |
| `gwexpy/timeseries/io/dttxml.py` | DTTXML time series reader. `products` is required |
| `gwexpy/timeseries/io/audio.py` | Audio reader/writer (MP3/FLAC/OGG/M4A). Depends on pydub. `t0=0` fixed |
| `gwexpy/timeseries/io/netcdf4_.py` | NetCDF4 reader/writer. Depends on xarray. Auto-detects time dimension |
| `gwexpy/timeseries/io/zarr_.py` | Zarr reader/writer. Depends on zarr. gwexpy store convention |
| `gwexpy/timeseries/io/stubs.py` | Time series stubs (`orf`, `mem`, `wvf`, `wdf`, `taffmat`, `lsf`, `li`) |
| `gwexpy/timeseries/io/hdf5.py` | HDF5 IO (gwpy re-export) |
| `gwexpy/timeseries/io/ascii.py` | ASCII IO (gwpy re-export) |
| `gwexpy/timeseries/io/nds2.py` | NDS2 IO (gwpy re-export) |
| `gwexpy/timeseries/io/cache.py` | Cache IO (gwpy re-export) |
| `gwexpy/timeseries/io/losc.py` | GWOSC IO (gwpy re-export) |
| `gwexpy/timeseries/io/core.py` | Core IO (gwpy re-export) |
| `gwexpy/frequencyseries/io/__init__.py` | Frequency series IO module registration entry point |
| `gwexpy/frequencyseries/io/dttxml.py` | DTTXML frequency series reader. `products` required. `native` option available |
| `gwexpy/frequencyseries/io/stubs.py` | Frequency series stubs (`win`, `win32`, `sdb`, `orf`, `mem`, `wvf`, `wdf`, `taffmat`, `lsf`, `li`) |
| `gwexpy/frequencyseries/io/hdf5.py` | HDF5 IO (gwpy re-export) |
| `gwexpy/frequencyseries/io/ascii.py` | ASCII IO (gwpy re-export) |
| `gwexpy/frequencyseries/io/ligolw.py` | LIGO_LW IO (gwpy re-export) |
| `gwexpy/table/io/root.py` | ROOT IO (gwpy re-export) |
| `gwexpy/timeseries/collections.py` | `TimeSeriesDict.read()` extensions (HDF5 layout detection, directory loading) |
| `docs/developers/design/design_data/io_support.csv` | Design data: supported format list |
