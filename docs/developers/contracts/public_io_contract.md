# Public File I/O Contract

This document defines how `docs/developers/contracts/public_io_contract.json`
should be interpreted.

## Purpose

The public File I/O contract is stricter than "whatever the registry currently
accepts." It exists to keep three things aligned:

- public documentation
- supported direct-I/O entry points
- CI gates that must fail when a published entry point regresses

The contract must therefore record both the user-facing surface and the broader
implementation surface.

## Schema

Each format entry contains these fields:

- `name`: human-facing label used in docs and reviews
- `canonical`: canonical format token
- `aliases`: accepted alias tokens
- `public_api`: classes that are part of the published direct-I/O contract
- `direct_api`: classes that implement direct `.read()` / `.write()` paths
  outside the astropy/gwpy registry
- `registry_api`: classes that are currently registered in the astropy/gwpy I/O registry
- `public_auto_identify`: whether published usage may rely on `format=None`
- `registry_auto_identify`: whether the current implementation can identify the format automatically
- `required_args`: operation-specific required keyword arguments
- `trusted_only`: whether the format must be restricted to trusted data
- `optional_dependencies`: optional import/package names used by the published
  route, or an empty list when the base install is sufficient
- `extras`: declared GWexpy extras that install those optional dependencies, or
  an empty list for base-install or bare-package policy
- `unavailable_behavior`: operation-specific behavior when the optional backend
  is unavailable
- `metadata_requirements`: extra metadata rules that are part of the public contract
- `notes`: short rationale and boundary notes

Rules:

- `public_api` must be a subset of `registry_api ∪ direct_api`.
- `public_auto_identify` may be stricter than `registry_auto_identify`.
- `required_args` and `metadata_requirements` are contract items, not optional commentary.
- `unavailable_behavior` uses a small vocabulary:
  `available_in_base_install`, `raises_import_error`,
  `raises_import_error_for_optional_metadata`,
  `warns_and_skips_optional_metadata`, `conditional_registration`, or `not_public`.
- Public registry entries may be absent only when
  `unavailable_behavior.<operation> = conditional_registration`.
- A registry adapter alone is not enough to publish a new direct-I/O surface.
- A class-level direct implementation alone is not enough to publish a new direct-I/O surface.

## Boundary Decisions

These decisions are fixed before expanding P1/P2/P3 coverage:

### `hdf.ndscope`

- Public contract: `TimeSeriesDict` read/write only.
- Registry surface: adapters for `TimeSeries` and `TimeSeriesMatrix` may exist.
- Reason: the ndscope schema is collection-oriented, and public docs already
  present it as a `TimeSeriesDict`-first HDF5 family.

### `xml.diaggui`

- Public contract: `TimeSeriesDict.read(..., format="xml.diaggui", products=...)`.
- Registry surface: `TimeSeries`, `TimeSeriesMatrix`, and frequency-domain
  `FrequencySeriesDict` / `FrequencySeriesMatrix` adapters may exist.
- Direct implementation surface: frequency-domain `FrequencySeries` /
  `FrequencySeriesDict` shims may exist for compatibility.
- Reason: the file is product-driven and `.xml` is ambiguous, so public guidance
  must stay explicit even if the registry can infer the identifier.
- Frequency-domain DTTXML readers are implementation-only and not part of the
  public direct-I/O contract.

### `root`

- `root` belongs to the `EventTable` contract, not to a generic timeseries-format
  expansion pass.
- Reason: ROOT direct I/O is intentionally limited to `EventTable`.

### `csv` / `txt`

- Public contract: `TimeSeries` file I/O plus `TimeSeriesDict` direct I/O.
- Reason: the user guide publishes these formats as baseline time-series
  exchange, but multi-channel helpers use collection-directory semantics rather
  than a uniform single-file contract.
- `txt` is stricter than `csv`: single-series `.txt` reads and writes require
  explicit `format="txt"`, and multi-channel `txt` uses manifest-backed
  collection directories.

### `sdb` / `sqlite` / `sqlite3`

- Public contract: `TimeSeries` and `TimeSeriesDict` read only.
- Registry surface: `TimeSeriesMatrix` read adapters may exist.
- Reason: the user guide publishes the SQLite-family weather/log archives as
  read-only direct I/O, and the alias family must behave consistently through
  the same public entry points.

### `wav`

- Public contract: `TimeSeries` read/write plus `TimeSeriesDict` read.
- Registry surface: `TimeSeriesMatrix` read adapters may exist, and dict-level
  write is intentionally not published.
- Reason: published WAV direct I/O is a simple audio exchange route, but the
  stable write surface is currently single-series only and does not preserve
  absolute timestamps.

### `mp3` / `flac` / `ogg` / `m4a`

- Public contract: `TimeSeries` and `TimeSeriesDict` read/write.
- Registry surface: `TimeSeriesMatrix` read/write adapters may exist as
  convenience paths.
- Optional dependency: published reads and writes require `pydub`; some codecs
  also require an external `ffmpeg`/`libav` backend.
- Reason: the user guide publishes compressed audio as direct exchange formats,
  but matrix adaptation is not part of the documented entry surface.

### `gbd`

- Public contract: `TimeSeries`, `TimeSeriesDict`, and `TimeSeriesMatrix` read
  only.
- Required args: `timezone` is mandatory for published reads.
- Reason: GBD stores local wall-clock timestamps, so published direct I/O must
  force explicit timezone resolution instead of silently assuming UTC.

### `tdms`

- Public contract: `TimeSeries`, `TimeSeriesDict`, and `TimeSeriesMatrix` read
  only.
- Optional dependency: published reads require `nptdms`; missing dependency
  should raise a format-specific `ImportError`.
- Reason: TDMS is a read-only instrument format in the current user guide, and
  its direct-I/O surface is entirely registry-backed.

### `mseed` / `sac` / `gse2`

- Public contract: `TimeSeriesDict` read/write only.
- Alias rule: `miniseed` must remain equivalent to canonical `mseed`.
- Registry surface: `TimeSeries` read/write and `TimeSeriesMatrix` read may
  exist as convenience adapters.
- Reason: the user guide publishes these ObsPy-backed seismic formats as
  collection-first direct I/O, even though registry adapters expose narrower
  single-series entry points.

### `knet` / `win` / `win32`

- Public contract: `TimeSeriesDict` read only.
- Registry surface: `TimeSeries` and `TimeSeriesMatrix` read adapters may
  exist.
- Optional dependency: `obspy` from the `seismic` extra.
- `win` / `win32` use conditional registration: when ObsPy is unavailable, the
  registry entry may be absent instead of a registered reader raising
  `ImportError`.
- Reason: these formats are intentionally documented as collection-first and
  read-only.

### `ats`

- Public contract: `TimeSeries` and `TimeSeriesDict` read only.
- Registry surface: `TimeSeriesMatrix` read adapters may exist.
- Reason: the binary ATS reader is exposed publicly for single-channel and
  dict-shaped reads, but matrix adaptation remains an implementation
  convenience.

### `ats.mth5`

- Public contract: `TimeSeries` read only.
- Optional dependency: published reads require `mth5`, available through the
  `seismic` extra; missing dependency should raise a format-specific
  `ImportError`.
- Reason: this is the only currently published MTH5-backed direct path and it
  remains intentionally narrow.

### `pickle`

- `pickle` is not a published direct `.read()` / `.write()` format.
- Reason: the current implementation provides `pickle.dumps()` /
  `pickle.loads()` / `shelve` portability helpers, but no public class-level
  direct-I/O entry points.
- User-facing security guidance for Pickle remains important, but it belongs to
  serialization compatibility notes rather than the direct-I/O contract.

### HDF5 Family

- Generic `hdf5` must be handled as an early class-family contract, not as a
  late catch-all after format-only phases.
- Reason: public docs already present HDF5 as the main storage path for
  `FrequencySeries`, `Spectrogram`, `Histogram`, `EventTable`, and related
  collection classes.
- Generic `hdf5` currently requires explicit `format="hdf5"` in the published
  contract.
- Reason: `.h5` / `.hdf5` overlaps multiple HDF5-backed families, and
  auto-identification is not uniform across the current class surface.
- `FrequencySeriesMatrix` and `SpectrogramMatrix` stay outside the published
  HDF5 contract until matrix-axis serialization is hardened.
- Field classes stay outside this schema slice until their direct-I/O story is
  audited separately.

### Built-in Direct I/O

- Some published direct-I/O paths are implemented by class methods or mixins
  instead of registry entries.
- Example classes include `SeriesMatrix`-based HDF5 readers/writers and custom
  HDF5 readers for histogram families.
- These paths must be recorded in `direct_api`, not squeezed into
  `registry_api`.

## Execution Rule

When a new phase extends the contract, the change must land as one logical unit:

1. update contract entries
2. add contract tests
3. put those tests behind a PR gate

Docs may follow in the same PR or the immediately following PR, but the gate
must not lag behind the new published contract.
