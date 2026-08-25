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

The raw JSON contract includes a top-level `io_conformance_v3` policy block.
Each on-disk format entry contains these fields:

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
- `time_semantics`: boundary classification for time-bearing formats;
  `absolute`, `fixed_zone`, `naive_civil`, or `relative`
- `epoch_arg`: explicit epoch surface; `none`, `override`, or the Zarr-specific
  `t0_override`
- `timezone_arg`: timezone policy; `rejected`, `required`,
  `epoch_localize_only`, `component_localize`, or `not_accepted`
- `time_routes`: optional route-level timing policy. CSV uses this because its
  component, numeric, and generated-index inputs have different semantics.

### Normalized v3 Fields

The conformance loader derives additional v3 policy fields in memory from each
on-disk format entry plus the top-level `io_conformance_v3` policy block. These
values are exposed under each normalized entry's `v3` surface; they are not
required as top-level fields in each JSON format entry.

- `fixture_generator`: normalized fixture-generation policy for the published route
- `coverage_status`: normalized coverage status for the published route
- `ci_jobs`: normalized CI gate mapping used to enforce the published route
- `missing_dependency_policy`: normalized per-operation policy for unavailable optional dependencies

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

### GWF and the optional FrameL backend

The canonical `gwf` format remains available in the base installation and has
no optional dependency declaration.  The explicit FrameL aliases `framel` and
`gwf.framel` require the optional `python-framel` backend.  Default tests skip
explicit FrameL rows when that backend is unavailable;
`GWEXPY_REQUIRE_GWF_FRAMEL=1` converts its absence into a required-gate
failure.  This does not make canonical `gwf` optional and does not prevent
importing the lazy FrameL compatibility proxy.

### Time semantics and timezone routing

`timezone` never reinterprets an absolute or fixed-zone timestamp. For an
`epoch_localize_only` format it may localize only a naive explicit `epoch`.
When the explicit epoch is already numeric or timezone-aware, the epoch is
preserved and a `UserWarning` reports that `timezone` was ignored. Invalid
timezone values are rejected before any epoch-type branch.

| Format | Time semantics | Epoch argument | Timezone argument |
|---|---|---|---|
| `gwf` | absolute | none | not accepted |
| `hdf.ndscope` | absolute | none | not accepted |
| `hdf5` | absolute | none | not accepted |
| `xml.diaggui` | absolute | override | epoch localize only |
| `csv` | absolute default; see routes below | none | component localize |
| `txt` | absolute | none | not accepted |
| `sdb` | absolute | none | rejected |
| `wav` | relative | override | not accepted |
| `flac` | relative | override | not accepted |
| `ogg` | relative | override | not accepted |
| `mp3` | relative | override | not accepted |
| `m4a` | relative | override | not accepted |
| `gbd` | naive civil | override | required |
| `tdms` | absolute | override | epoch localize only |
| `mseed` | absolute | override | epoch localize only |
| `sac` | absolute | override | epoch localize only |
| `gse2` | absolute | override | epoch localize only |
| `knet` | absolute | override | epoch localize only |
| `win` | fixed zone | none | rejected |
| `ats` | absolute | override | epoch localize only |
| `ats.mth5` | absolute | none | rejected |
| `nc` | absolute | none | not accepted |
| `zarr` | absolute | t0 override | not accepted |

CSV route details are machine-readable under `time_routes`:

| Route | Time semantics | Timezone behavior |
|---|---|---|
| component columns | naive civil | localize |
| numeric time column | absolute | ignore with one warning per top-level read |
| generated sample index | relative | ignore with one warning per top-level read |

Configured component columns fail closed when a naive civil timestamp is an
ambiguous daylight-saving fold or a nonexistent gap; the resulting
`ValueError` identifies the physical CSV line.
Their canonical instants use continuous GPS seconds, so a UTC component stream
that crosses a leap second without the unrepresentable `second=60` record is
rejected as a cadence gap rather than compressed onto a POSIX timeline.

Numeric and configured component-column routes validate regular source cadence
before any requested resampling. Numeric validation uses the source decimal
tokens, while component validation compares reconstructed continuous GPS
instants so a UTC leap-second gap cannot look like a regular civil-time grid.
Malformed rows raise `ValueError` with their physical CSV line number.
A finite, positive `sample_rate` declares source cadence and is honoured for a
single row; without it, the legacy one-second fallback remains.
`resample=` must be finite and positive and controls only the target cadence.
Output values for requested channels are evaluated on that exact target-rate
grid; existing source samples are not merely relabelled with a different `dt`.
An absolute float64 time axis is accepted only when both its rounding error and
local spacing are strictly below half the cadence. A top-level single- or
multi-file read is limited to 10,000,000 requested-channel resampled output
values in total before allocation.
Numeric timestamps retain the legacy GPS-second interpretation; v0.1.14 does
not add `time_scale=` or `time_unit=`.

### `hdf.ndscope`

- Public contract: `TimeSeriesDict` read/write only.
- Registry surface: adapters for `TimeSeries` and `TimeSeriesMatrix` may exist.
- Timing metadata: writers emit canonical `rate_hz` and `gps_start` group
  attributes. Readers also accept external files that use `sample_rate` in
  place of `rate_hz`; when both are present, their values must agree. A
  data-bearing group (one carrying `gps_start` and at least one of the
  `raw`/`mean`/`min`/`max` datasets) that has neither attribute raises
  `ValueError` naming the group, rather than being skipped: silently
  dropping it would return a `TimeSeriesDict` missing a channel with no
  indication anything went wrong. Groups without `gps_start`, and groups
  carrying no NDScope dataset, are not data-bearing and remain skipped.
- Format detection: auto-identification keys on `gps_start` plus at least one
  `raw`/`mean`/`min`/`max` dataset, and deliberately does not require the
  sampling-rate attribute. A file whose groups all lack it is a malformed
  NDScope file, not a different format, so the `ValueError` above is reachable
  through auto-detected `TimeSeriesDict.read()` / `TimeSeriesMatrix.read()`
  rather than being masked by a fall-through to another reader.
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
- Frequency-domain DTTXML direct shims (`FrequencySeries` /
  `FrequencySeriesDict`) and registry adapters (`FrequencySeriesDict` /
  `FrequencySeriesMatrix`) are implementation-only and not part of the public
  direct-I/O contract.

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

### `sdb`

- Public contract: `TimeSeries` and `TimeSeriesDict` read only.
- Registry surface: `TimeSeriesMatrix` read adapters may exist.
- Reason: the user guide publishes `.sdb` weather/log archives as read-only
  direct I/O.  `sdb` is the only public token and `.sdb` is the only
  auto-identified extension.
- Units: if an archive has a `usUnits` column, every row must have integer
  value `1`; otherwise reading fails with `ValueError`.  Archives without the
  column retain the legacy US customary unit assumption.
- Timing: `dateTime` values must be integer Unix seconds with regular cadence
  in database storage order. Duplicate, backward, missing, or overlarge gaps
  raise `ValueError` before a time axis is constructed. Rowid tables use SQLite
  rowid/B-tree order, which is not insertion chronology; `WITHOUT ROWID` tables
  use the declared primary-key order, including direction and collation.

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
- WIN packets must form one globally consecutive one-second sequence;
  duplicate timestamps, backward timestamps, and global gaps raise
  `ValueError`.
- A WIN channel may start after the first packet or end before the last packet.
  Its first appearance defines its channel-local `t0`. After a channel starts,
  reappearance following an internal packet gap, a duplicate block in one
  packet, or a sample-rate change raises `ValueError`.
- WIN packet and channel payload lengths and decoded sample counts are bounded
  and validated. Malformed, truncated, or overlong payloads fail before an
  oversized read, and cumulative delta decoding does not wrap at int32 bounds.
- Only a BCD year transition from `99` to `00` advances the supplied century.
  WIN header time is interpreted as UTC and emits one warning per top-level
  read, including a multi-file read.
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
- Optional dependency: published reads require `mth5>=0.6.8`, available through
  the `seismic` extra; missing or incompatible dependencies raise a
  format-specific `ImportError`.
- Backend route: `mth5.read_file(..., file_type="metronix")` returns one
  channel. Data must be one-dimensional and non-empty; start time and finite
  positive sample rate are required. Ex/Ey map to mV/km and Hx/Hy/Hz to nT
  without rescaling raw values.
- Source timing and units are authoritative: `epoch`, `timezone`, unit, and
  other reader overrides are rejected instead of being ignored.
- `TimeSeriesDict.read(..., format="ats.mth5")` is rejected before optional
  dependency lookup.
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
