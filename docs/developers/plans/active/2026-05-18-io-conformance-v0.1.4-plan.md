# File I/O Conformance Plan for v0.1.4+

Date: 2026-05-18
Target release: v0.1.4 for the first conformance harness baseline
Current released/runtime version in repository metadata: v0.1.3

## Authoritative Source Policy

When this plan and `docs/developers/contracts/public_io_contract.json` diverge:
- **JSON contract** is authoritative for machine-readable fields (aliases, class lists,
  tolerances, scenario status values).
- **This plan** is authoritative for prose policy, CI gate definitions, migration
  timelines, and acceptance criteria.
- Discrepancies must be resolved before v0.1.4 ships by updating the JSON contract to
  match the agreed policy.

## Purpose

GWexpy v0.1.1-v0.1.3 exposed repeated regressions in file I/O behavior. The
current test suite has many useful format-specific tests, but it does not yet
provide one systematic, contract-driven path that validates every public file
format through generation, read, write, and external readback.

This plan turns `docs/developers/contracts/public_io_contract.json` into an
executable conformance specification.

The required flow for every covered scenario is:

1. Generate mock data without importing or using GWexpy.
2. Read the mock file with GWexpy through all public read classes and required
   registry/direct convenience classes.
3. Cover the declared channel/file layout patterns for that format.
4. If public write exists, write to a local temporary file with GWexpy.
5. Read the written file back with GWexpy and, where applicable, with GWpy or
   the native third-party library.
6. Compare values and metadata under explicit per-format tolerances.

## Schema Migration (v2 → v3)

The public I/O contract is currently at schema-v2. Schema v3 introduces new fields
required by this conformance plan. The migration policy is:

- v2 is canonical until the v3 JSON schema validator lands in the repository.
- v3 is **additive-only** for v0.1.4: no existing v2 field is renamed or removed.
- The v3 validator must accept both v2 and v3 documents during the v0.1.4 cycle.
- Field removal and breaking changes to v2 are deferred to v0.1.6.
- Every new v3 field listed in "Contract Schema v3 Additions" must have a default
  value that does not break the v2 validator, or a separate v3-only schema file must
  be introduced alongside the v2 file.

## Non-Goals for v0.1.4

- v0.1.4 will not make every public format a blocking full-matrix gate.
- v0.1.4 will not promise stricter metadata preservation than the existing
  public contract.
- v0.1.4 will not remove existing targeted tests.
- v0.1.4 will not remove `gwexpy.register_all()`.
- v0.1.4 will not require all optional backends in the base CI job.

## Target Classes

The conformance harness must know every class referenced by the public I/O
contract. The classes are grouped by family.

### Time-Series Family

- `TimeSeries`
- `TimeSeriesDict`
- `TimeSeriesList`
- `TimeSeriesMatrix`

### Frequency-Series Family

- `FrequencySeries`
- `FrequencySeriesDict`
- `FrequencySeriesList`
- `FrequencySeriesMatrix`

### Spectrogram Family

- `Spectrogram`
- `SpectrogramDict`
- `SpectrogramList`

### Tracked but Out-of-Scope for v0.1.4

The following classes are known to GWexpy but are not referenced by the schema-v2
public I/O contract. They must not appear in blocking acceptance criteria until the
JSON contract explicitly includes them.

- `SpectrogramMatrix`: exclude from blocking scenarios; add to the conformance matrix
  only after the JSON contract references it.
- `FrequencySeriesMatrix`: tracked, not in any blocking HDF5 scenario for v0.1.4.

### Histogram Family

- `Histogram`
- `HistogramDict`
- `HistogramList`

### Segment Family

- `SegmentList`
- `DataQualityFlag`
- `DataQualityDict`

### Table Family

- `EventTable`

## Scenario Vocabulary

Every format entry in contract schema v3 must explicitly list supported and
unsupported scenarios. A missing scenario policy is a schema failure.

### Generation Scenarios

- `generate_single_channel_single_file`: create one channel in one file.
- `generate_multi_channel_single_file`: create multiple channels in one file.
- `generate_single_channel_multi_file`: create one logical channel split across
  multiple files.
- `generate_multi_channel_multi_file`: create multiple logical channels split
  across multiple files.
- `generate_collection_directory`: create a manifest-backed collection
  directory.
- `generate_native_external`: create with the native library for the format.
- `generate_manual_spec`: create with a documented binary/text schema and
  standard library or NumPy only.
- `generate_gwpy_external`: create with GWpy, allowed only where a practical
  independent writer is not available, such as GWF.

### Read Scenarios

- `read_explicit_format`: call `.read(..., format=canonical)`.
- `read_auto_identify_public`: call `.read(...)` with `format=None` when the format's
  public contract declares `public_auto_identify: true`. This is the user-visible
  documented behavior.
- `read_auto_identify_registry`: call `.read(...)` with `format=None` via the internal
  registry identifier when `public_auto_identify` is false but the registry can still
  identify the format. This is a nightly-only scenario and is not part of the public
  contract guarantee.
- `read_alias_format`: call `.read(..., format=alias)` for every alias.
- `read_required_args`: call `.read()` with required args such as `products` or
  `timezone`.
- `read_missing_required_args`: assert the documented error when required args
  are missing.
- `read_single_channel_single_file`: public single-channel read.
- `read_multi_channel_single_file`: public multi-channel read.
- `read_single_channel_multi_file`: public one-channel multi-file read.
- `read_multi_channel_multi_file`: public multi-channel multi-file read.
- `read_collection_directory`: public collection-directory read.
- `read_optional_dependency_missing`: assert documented missing-backend behavior.
- `read_trusted_only`: assert security policy behavior for formats that require
  explicit user opt-in (see `security_policy` field in schema v3).

### Write Scenarios

- `write_single_object`: write one public object.
- `write_collection`: write dict/list public objects.
- `write_matrix`: write matrix public objects.
- `write_alias_format`: write using every public write alias.
- `write_unsupported`: assert no public writer or documented unsupported error.
- `write_optional_dependency_missing`: assert documented missing-backend behavior.

### Readback and Interop Scenarios

- `readback_gwexpy`: read GWexpy-written file with GWexpy.
- `readback_gwpy`: read GWexpy-written file with GWpy where GWpy supports it.
- `readback_native`: read GWexpy-written file with the native library.
- `readback_registry_import_path`: repeat read in a fresh subprocess using the
  user-visible direct import path, without calling `register_all()`.
- `readback_top_level_import_path`: repeat read in a fresh subprocess after
  `import gwexpy`.

### Multi-File Policy Values

- `unsupported`: multi-file input is not part of the public contract.
- `single_file_only`: the format is inherently or currently single-file.
- `list_of_files`: `.read([file1, file2, ...])` is public behavior.
- `collection_directory`: multi-object I/O is represented by a directory and
  manifest, not by a list of independent format files.
- `backend_dependent`: scenario runs only when the backend supports it.

### Coverage Status Values

- `blocking`: must pass in the v0.1.4 primary CI gate.
- `optional_blocking`: must pass in the optional-backend CI job when required
  dependencies are installed.
- `nightly`: runs in nightly/extended CI and is not a PR blocker for v0.1.4.
- `planned`: explicitly out of scope for v0.1.4, but required before the
  conformance matrix is considered complete.
- `not_public`: not a public direct I/O scenario.

## Contract Schema v3 Additions

Every format entry must add these fields:

- `fixture_generator`: named generator under `tests/io_conformance/generators`.
- `scenario_matrix`: structured scenario list.
- `external_writer`: native or GWpy writer used to generate fixtures.
- `external_reader`: native or GWpy reader used for external readback.
- `roundtrip_policy`: exact, lossy, metadata-light, read-only, or unsupported.
- `comparison_tolerances`: value/time/frequency tolerance policy.
- `metadata_must_match`: metadata keys that must survive read/write/readback.
- `multi_file_policy`: one of the values listed above.
- `coverage_status`: default status for this format in v0.1.4.
- `ci_jobs`: one or more logical gate identifiers: `base`, `io-audio`, `io-seismic`,
  `io-netcdf4`, `io-zarr`, `io-tdms`, `io-root`, `nightly`. These identifiers are
  stable contract values, not GitHub Actions display names. The CI implementation
  must provide an explicit mapping from each logical gate identifier to a workflow
  job, workflow matrix entry, or `scripts/ci/run_gate.py` target.
- `missing_dependency_policy`: skip, must_raise_import_error,
  conditional_registration, or warns_and_skips_optional_metadata. Known bugs must
  be represented with `coverage_status=planned` or `coverage_status=nightly` plus
  an issue reference, not by weakening or skipping a blocking conformance row.
- `security_policy`: object with `trusted_only: bool` and `requires_warning: bool`.
  `trusted_only: true` means the format may only be read when the caller explicitly
  opts in. `requires_warning: true` means GWexpy must emit a visible warning on read.
  Default for all formats: `{trusted_only: false, requires_warning: false}`.

## Generator Determinism Contract

Every generator under `tests/io_conformance/generators/` must satisfy all of the
following rules.

**No GWexpy import.** Generators must not import `gwexpy` or any submodule of it.
Enforcement: `tests/io_conformance/conftest.py` runs an import-guard check at
session start that parses every generator module with `ast` and rejects:

- `ast.Import` / `ast.ImportFrom` nodes targeting `gwexpy` or `gwexpy.*`;
- literal dynamic imports such as `__import__("gwexpy")` and
  `importlib.import_module("gwexpy")`;
- any generator entry point that imports `gwexpy` while executed in a subprocess
  with an import hook that raises on `gwexpy` and `gwexpy.*`.

A text grep is insufficient because it misses `from gwexpy import ...`, misses lazy
dynamic imports, and can false-positive on comments.

**Seeded RNG.** All numeric payload generation must use `numpy.random.default_rng(SEED)`
where `SEED` is a per-format integer constant defined in `generators/__init__.py`.
No call to `random.random()`, `np.random.seed()`, or any global RNG state is allowed.

**Fixed metadata.** `t0`, `gps_start`, `f0`, and `sample_rate` must be hard-coded
per generator function, not derived from `time.time()` or any system clock.

**Pure function signature.** Every public generator function must accept only
`(tmp_path, seed=None, *, n_samples, n_channels, dtype)` or a strict subset of
those arguments. No side effects, no filesystem reads outside `tmp_path`.

**pytest-xdist safety.** All generators write exclusively under the `tmp_path`
fixture provided by pytest. No shared global state. Conformance tests are xdist-safe
by design.

Every scenario row must be executable as a unique tuple:

```text
(format, operation, scenario, class, layout, format_token, dependency_mode, status)
```

For example, `read_multi_channel_single_file` for `TimeSeriesDict` and
`read_multi_channel_single_file` for `TimeSeriesMatrix` are two separate rows,
not duplicate prose bullets. `format_token` is `canonical`, a concrete alias, or
`auto`; `dependency_mode` is `base`, `optional-present`, or `optional-missing`.

## Test Tree

Add a new conformance harness without replacing existing targeted tests.

```text
tests/io_conformance/
  __init__.py
  conftest.py
  contract.py
  scenarios.py
  generators/
    __init__.py
    audio.py
    ats.py
    csv_txt.py
    dttxml.py
    gbd.py
    gwf.py
    hdf5.py
    ndscope.py
    netcdf.py
    root.py
    sdb.py
    seismic.py
    tdms.py
    zarr.py
  validators/
    __init__.py
    common.py
    timeseries.py
    frequencyseries.py
    spectrogram.py
    histogram.py
    segments.py
    table.py
  test_contract_schema_v3.py
  test_read_conformance.py
  test_write_roundtrip.py
  test_external_interop.py
  test_registry_import_paths.py
  test_optional_dependency_behavior.py
```

## Test Parametrization

**Scenario expansion.** `tests/io_conformance/scenarios.py` reads the JSON contract
and expands every `(format, operation, scenario, class, layout, format_token,
dependency_mode, status)` tuple into a pytest parametrize ID. The harness must skip
rows where `status` is `planned` or `not_public`. Rows where `dependency_mode` is
`optional-missing` are ordinary executable tests; they must assert the documented
missing-backend behavior, such as a clear `ImportError`, conditional registration
absence, or a warning that optional metadata was skipped. `missing_dependency_policy=skip`
is valid only for rows that do not represent a public optional dependency contract.
The schema validator must reject `skip` for any format that has public read/write
coverage and non-empty `optional_dependencies`.

**Alias sampling policy.** Running the full scenario matrix for every alias would
multiply test count by the number of aliases. Instead:
- One complete scenario sweep runs under the canonical format name.
- Alias coverage is limited to `read_alias_format` (one read roundtrip per alias)
  and `write_alias_format` (one write roundtrip per writable alias).
- No alias receives the full matrix in blocking CI; full alias matrix is nightly only.

**Subprocess import-path sampling.** `readback_registry_import_path` and
`readback_top_level_import_path` start fresh Python processes, which are expensive.
Sampling rule for blocking CI:
- Run one subprocess per `(format, import_path_type)` pair using the simplest
  available class for that format (typically `TimeSeries` or `TimeSeriesDict`).
- Do not multiply subprocesses across all class/layout/alias combinations in blocking
  CI.
- Full per-scenario subprocess expansion is nightly only.

**pytest-xdist.** The full conformance suite must pass under `pytest -n auto`. Each
test function receives an isolated `tmp_path` and must not share file handles or
global state with other tests.

Generators under `tests/io_conformance/generators` must not import `gwexpy`.
Conformance tests should fail if a generator imports `gwexpy` accidentally.
Legacy fixtures under `tests/fixtures` remain available for targeted tests only.
Conformance tests must generate every file they consume in `tmp_path`; they must not
read from `tests/fixtures` or any pre-committed binary fixture. No silent stub
fallback is permitted: if a generator fails, the test must fail immediately with a
clear error, not skip silently.

## Complete Format Matrix

The matrix below lists all 25 current contract formats. No contract format is
omitted.

### 1. `gwf`

- Canonical: `gwf`
- Aliases: `frame`, `framecpp`, `framel`, `lalframe`, `gwf.framecpp`,
  `gwf.framel`, `gwf.lalframe`
- Public read classes: `TimeSeries`, `TimeSeriesDict`
- Public write classes: `TimeSeries`, `TimeSeriesDict`
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: `TimeSeries`, `TimeSeriesDict`
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: none in GWexpy contract, but runtime GWF backend
  availability must be detected
- Generator: `generators.gwf`
- External writer: GWpy `TimeSeriesDict.write(format="gwf")`
- External reader: GWpy GWF reader
- GWpy version policy: `gwpy` must be pinned in conformance test extras as
  `gwpy>=X.Y,<X.Z`. Version bumps require re-running the GWF scenario matrix and
  updating tolerances if outputs change.
- Multi-file policy: `list_of_files`
- Scenarios:
  - `generate_single_channel_single_file`: blocking
  - `generate_multi_channel_single_file`: blocking
  - `generate_single_channel_multi_file`: blocking
  - `generate_multi_channel_multi_file`: blocking
  - `read_explicit_format`: blocking
  - `read_auto_identify_public`: blocking
  - `read_alias_format`: blocking for every alias
  - `read_single_channel_single_file`: blocking for `TimeSeries`
  - `read_multi_channel_single_file`: blocking for `TimeSeriesDict`
  - `read_single_channel_multi_file`: blocking for `TimeSeries`
  - `read_multi_channel_multi_file`: blocking for `TimeSeriesDict`
  - `write_single_object`: blocking for `TimeSeries`
  - `write_collection`: blocking for `TimeSeriesDict`
  - `write_alias_format`: blocking for every writable alias
  - `readback_gwexpy`: blocking
  - `readback_gwpy`: blocking
  - `readback_registry_import_path`: blocking
  - `readback_top_level_import_path`: blocking
- Metadata must match: `sample_rate`, `dt`, `t0`, `duration`, `channel`,
  `name`, `unit`, channel keys
- Tolerances: exact sample count, exact channel keys, numeric `rtol=1e-12`,
  `atol=1e-12` for float64 fixtures
- v0.1.4 status: blocking

### 2. `hdf.ndscope`

- Canonical: `hdf.ndscope`
- Aliases: `ndscope-hdf5`, `ndscope_hdf5`, `ndscopehdf5`
- Public read classes: `TimeSeriesDict`
- Public write classes: `TimeSeriesDict`
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: none beyond base HDF5 stack used by project
- Generator: `generators.ndscope`
- External writer: `h5py`
- External reader: `h5py`
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: blocking
  - `generate_multi_channel_single_file`: blocking
  - `read_explicit_format`: blocking
  - `read_auto_identify_public`: blocking
  - `read_alias_format`: blocking for every alias
  - `read_single_channel_single_file`: blocking for `TimeSeriesDict`
  - `read_multi_channel_single_file`: blocking for `TimeSeriesDict`
  - `write_collection`: blocking for `TimeSeriesDict`
  - `write_alias_format`: blocking for every writable alias
  - `readback_gwexpy`: blocking
  - `readback_native`: blocking with `h5py` schema inspection
  - `readback_registry_import_path`: blocking
  - `readback_top_level_import_path`: blocking
- Metadata must match: `rate_hz`, `gps_start`, `unit`, channel group names,
  sample count
- Tolerances: numeric `rtol=1e-7`, `atol=1e-7` for float32 fixtures
- v0.1.4 status: blocking

### 3. `hdf5`

- Canonical: `hdf5`
- Aliases: none
- Public read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesList`,
  `TimeSeriesMatrix`, `FrequencySeries`, `FrequencySeriesDict`,
  `FrequencySeriesList`, `Spectrogram`, `SpectrogramDict`,
  `SpectrogramList`, `Histogram`, `HistogramDict`, `HistogramList`,
  `EventTable`
- Public write classes: same as public read classes
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `FrequencySeries`,
  `Spectrogram`, `SegmentList`, `DataQualityFlag`, `DataQualityDict`,
  `EventTable`
- Registry write classes: `TimeSeries`, `TimeSeriesDict`, `FrequencySeries`,
  `Spectrogram`, `SegmentList`, `DataQualityFlag`, `DataQualityDict`,
  `EventTable`
- Direct read classes: `TimeSeriesDict`, `TimeSeriesList`,
  `TimeSeriesMatrix`, `FrequencySeriesDict`, `FrequencySeriesList`,
  `SpectrogramDict`, `SpectrogramList`, `Histogram`, `HistogramDict`,
  `HistogramList`
- Direct write classes: same as direct read classes
- Required read args: none
- Required write args: none
- Optional dependencies: none beyond base HDF5 stack used by project
- Generator: `generators.hdf5`
- External writer: `h5py` for raw layout fixtures, GWpy for GWpy-native HDF5
  fixtures where applicable
- External reader: `h5py`, GWpy where applicable
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: blocking for time/frequency series
  - `generate_multi_channel_single_file`: blocking for dict/list/matrix families
  - `read_explicit_format`: blocking
  - `read_auto_identify_public`: not_public — the public contract requires explicit
    `format="hdf5"`; auto-identification is not a documented user-facing guarantee
  - `read_auto_identify_registry`: not_public — schema-v2 declares
    `registry_auto_identify=false`; make this nightly only if the JSON contract
    changes
  - `read_single_channel_single_file`: blocking for `TimeSeries`,
    `FrequencySeries`, `Spectrogram`, `Histogram`, `EventTable`
  - `read_multi_channel_single_file`: blocking for `TimeSeriesDict`,
    `TimeSeriesList`, `TimeSeriesMatrix`, `FrequencySeriesDict`,
    `FrequencySeriesList`, `SpectrogramDict`, `SpectrogramList`,
    `HistogramDict`, `HistogramList`
  - `write_single_object`: blocking for `TimeSeries`, `FrequencySeries`,
    `Spectrogram`, `Histogram`, `EventTable`
  - `write_collection`: blocking for dict/list classes
  - `write_matrix`: blocking for `TimeSeriesMatrix`; planned for
    `FrequencySeriesMatrix` and `SpectrogramMatrix` unless they become public
  - `readback_gwexpy`: blocking
  - `readback_gwpy`: blocking where GWpy supports the class
  - `readback_native`: blocking with `h5py` structure inspection
  - `readback_registry_import_path`: blocking for registry-backed classes
  - `readback_top_level_import_path`: blocking
- Metadata must match: family-specific axes, `sample_rate`/`dt`, `df`, `f0`,
  `frequencies`, `times`, `unit`, `name`, `channel`, collection keys,
  histogram edges, table columns
- Tolerances: exact shapes and keys; numeric `rtol=1e-12`, `atol=1e-12` for
  float64; family-specific relaxed tolerance only where existing writer stores
  float32
- v0.1.4 status: blocking for public classes listed above, except matrix classes
  not in public HDF5 contract remain not_public/planned

### 4. `xml.diaggui`

- Canonical: `xml.diaggui`
- Aliases: `dttxml`
- Public read classes: `TimeSeriesDict`
- Public write classes: none
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`,
  `FrequencySeriesDict`, `FrequencySeriesMatrix`
- Registry write classes: none
- Direct read classes: `FrequencySeries`, `FrequencySeriesDict`
- Direct write classes: none
- Required read args: `products`
- Required write args: none
- Optional dependencies: none; native parser fallback behavior must be checked
- Generator: `generators.dttxml`
- External writer: manual XML/LIGO_LW generation with standard library and NumPy
- External reader: optional `dttxml` package where installed, otherwise native
  parser expectations
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: optional_blocking
  - `generate_multi_channel_single_file`: optional_blocking
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: not_public — not a documented user-facing guarantee
    despite the registry identifier
  - `read_auto_identify_registry`: nightly — schema-v2 declares
    `registry_auto_identify=true`
  - `read_alias_format`: optional_blocking for `dttxml`
  - `read_required_args`: optional_blocking with `products`
  - `read_missing_required_args`: optional_blocking
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesDict`
  - `write_unsupported`: blocking
  - `readback_registry_import_path`: optional_blocking
  - `readback_top_level_import_path`: optional_blocking
- Metadata must match: product name, channel names, `dt`/`df`, `t0`/epoch,
  unit, sample/bin count
- Tolerances: XML float32 payload `rtol=1e-6`, `atol=1e-8`
- v0.1.4 status: planned or optional_blocking; not a core blocking gate

### 5. `csv`

- Canonical: `csv`
- Aliases: none
- Public read classes: `TimeSeries`, `TimeSeriesDict`
- Public write classes: `TimeSeries`, `TimeSeriesDict`
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`,
  `FrequencySeries`, `Spectrogram`, `EventTable`
- Registry write classes: `TimeSeries`, `FrequencySeries`, `Spectrogram`,
  `EventTable`
- Direct read classes: `TimeSeriesDict`, `TimeSeriesList`,
  `FrequencySeriesDict`, `FrequencySeriesList`
- Direct write classes: same as direct read classes
- Required read args: none
- Required write args: none
- Optional dependencies: none
- Generator: `generators.csv_txt`
- External writer: Python `csv` module / pandas-free text generation
- External reader: Python `csv` module / NumPy text loading for structure
- Multi-file policy: `collection_directory`
- Scenarios:
  - `generate_single_channel_single_file`: blocking
  - `generate_multi_channel_single_file`: blocking for single CSV where supported
  - `generate_collection_directory`: blocking for `TimeSeriesDict`
  - `read_explicit_format`: blocking
  - `read_auto_identify_public`: blocking
  - `read_single_channel_single_file`: blocking for `TimeSeries`
  - `read_multi_channel_single_file`: blocking for `TimeSeriesDict`
  - `read_collection_directory`: blocking for `TimeSeriesDict`
  - `write_single_object`: blocking for `TimeSeries`
  - `write_collection`: blocking for `TimeSeriesDict`
  - `readback_gwexpy`: blocking
  - `readback_native`: blocking with CSV structural checks
  - `readback_registry_import_path`: blocking
  - `readback_top_level_import_path`: blocking
- Metadata must match: sample count, values, time column where present,
  collection keys for manifest layout
- Metadata not guaranteed: complete unit/name/channel preservation across every
  GWpy-compatible CSV route
- Tolerances: numeric `rtol=1e-12`, `atol=1e-12`
- v0.1.4 status: blocking

### 6. `txt`

- Canonical: `txt`
- Aliases: none
- Public read classes: `TimeSeries`, `TimeSeriesDict`
- Public write classes: `TimeSeries`, `TimeSeriesDict`
- Registry read classes: `TimeSeries`, `FrequencySeries`
- Registry write classes: `TimeSeries`, `FrequencySeries`
- Direct read classes: `TimeSeriesDict`, `TimeSeriesList`,
  `FrequencySeriesDict`, `FrequencySeriesList`
- Direct write classes: same as direct read classes
- Required read args: none
- Required write args: none
- Optional dependencies: none
- Generator: `generators.csv_txt`
- External writer: plain text generation with NumPy arrays
- External reader: NumPy text loading for structure
- Multi-file policy: `collection_directory`
- Scenarios:
  - `generate_single_channel_single_file`: blocking
  - `generate_collection_directory`: blocking for `TimeSeriesDict`
  - `read_explicit_format`: blocking
  - `read_auto_identify_public`: not_public
  - `read_single_channel_single_file`: blocking for `TimeSeries`
  - `read_collection_directory`: blocking for `TimeSeriesDict`
  - `write_single_object`: blocking for `TimeSeries`
  - `write_collection`: blocking for `TimeSeriesDict`
  - `readback_gwexpy`: blocking
  - `readback_native`: blocking with text structural checks
  - `readback_registry_import_path`: blocking
  - `readback_top_level_import_path`: blocking
- Metadata must match: sample count, values, time column where present,
  collection manifest keys
- Metadata not guaranteed: full name/channel/unit preservation for basic text
- Tolerances: numeric `rtol=1e-12`, `atol=1e-12`
- v0.1.4 status: blocking

### 7. `pickle`

- Canonical: `pickle`
- Aliases: none
- Public read classes: none
- Public write classes: none
- Registry read classes: none
- Registry write classes: none
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: none
- Security policy: `{trusted_only: true, requires_warning: true}`
- Generator: none for public direct I/O
- External writer: Python `pickle` only for serialization compatibility tests
- External reader: Python `pickle`
- Multi-file policy: `unsupported`
- Scenarios:
  - `read_trusted_only`: blocking for security policy documentation tests
  - `write_unsupported`: blocking for direct I/O contract boundary
  - all public direct read/write scenarios: not_public
- Metadata must match: not applicable
- Tolerances: not applicable
- v0.1.4 status: not_public for direct I/O, blocking for boundary/security
  assertions

### 8. `root`

- Canonical: `root`
- Aliases: none
- Public read classes: `EventTable`
- Public write classes: `EventTable`
- Registry read classes: `EventTable`
- Registry write classes: `EventTable`
- Direct read classes: none
- Direct write classes: `TimeSeriesDict`, `TimeSeriesList`,
  `FrequencySeriesDict`, `FrequencySeriesList`, `SpectrogramDict`,
  `SpectrogramList`, `HistogramDict`, `HistogramList`
- Required read args: none
- Required write args: none
- Optional dependencies: `uproot`
- Generator: `generators.root`
- External writer: `uproot`
- External reader: `uproot`
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: optional_blocking for EventTable-like
    tree/table fixture
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking if public auto-identification
    is registered
  - `write_single_object`: optional_blocking for `EventTable`
  - `readback_gwexpy`: optional_blocking
  - `readback_native`: optional_blocking with `uproot`
  - `read_optional_dependency_missing`: blocking in base CI
  - `write_optional_dependency_missing`: blocking in base CI
- Metadata must match: table column names, row count, numeric column values
- Tolerances: numeric `rtol=1e-12`, `atol=1e-12`
- v0.1.4 status: planned/optional_blocking, not base blocking

### 9. `sdb`

- Canonical: `sdb`
- Aliases: `sqlite`, `sqlite3`
- Public read classes: `TimeSeries`, `TimeSeriesDict`
- Public write classes: none
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: none
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: none
- Generator: `generators.sdb`
- External writer: Python `sqlite3`
- External reader: Python `sqlite3`
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: optional_blocking
  - `generate_multi_channel_single_file`: optional_blocking
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking
  - `read_alias_format`: optional_blocking for `sqlite`, `sqlite3`
  - `read_single_channel_single_file`: optional_blocking for `TimeSeries`
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesDict`
  - `write_unsupported`: blocking
  - `readback_registry_import_path`: optional_blocking
  - `readback_top_level_import_path`: optional_blocking
- Metadata must match: selected channel/column names, sample count, time spacing
- Tolerances: numeric `rtol=1e-12`, `atol=1e-12`
- v0.1.4 status: planned/optional_blocking

### 10. `wav`

- Canonical: `wav`
- Aliases: none
- Public read classes: `TimeSeries`, `TimeSeriesDict`
- Public write classes: `TimeSeries`
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: `TimeSeries`
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: `tinytag`
- Generator: `generators.audio`
- External writer: `scipy.io.wavfile`
- External reader: `scipy.io.wavfile`, `tinytag` for metadata when installed
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: blocking
  - `generate_multi_channel_single_file`: blocking if stereo/multi-channel WAV is
    public for `TimeSeriesDict`
  - `read_explicit_format`: blocking
  - `read_auto_identify_public`: blocking
  - `read_single_channel_single_file`: blocking for `TimeSeries`
  - `read_multi_channel_single_file`: blocking for `TimeSeriesDict`
  - `write_single_object`: blocking for `TimeSeries`
  - `write_collection`: not_public
  - `readback_gwexpy`: blocking
  - `readback_native`: blocking with `scipy.io.wavfile`
  - `readback_registry_import_path`: blocking
  - `readback_top_level_import_path`: blocking
- Metadata must match: sample rate, sample count, channel count
- Metadata not guaranteed: absolute `t0` — WAV does not store GPS epoch. Roundtrip
  comparisons must exclude `t0`. Post-roundtrip tests assert `t0 == 0.0` (the
  writer default) rather than comparing against the original.
- Tolerances: PCM quantization tolerance based on sample width; `atol=1/32767` for
  int16 fixtures (full signed range is ±32767, not ±32768).
- v0.1.4 status: blocking

### 11. `flac`

- Canonical: `flac`
- Aliases: none
- Public read classes: `TimeSeries`, `TimeSeriesDict`
- Public write classes: `TimeSeries`, `TimeSeriesDict`
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: `pydub`, `tinytag`, external `ffmpeg`/`libav`
- Generator: `generators.audio`
- External writer: `pydub`
- External reader: `pydub`, `tinytag`
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: optional_blocking
  - `generate_multi_channel_single_file`: optional_blocking
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking
  - `write_single_object`: optional_blocking
  - `write_collection`: optional_blocking
  - `readback_gwexpy`: optional_blocking
  - `readback_native`: optional_blocking
  - `read_optional_dependency_missing`: blocking in base CI
  - `write_optional_dependency_missing`: blocking in base CI
- Metadata must match: sample rate, sample count within codec tolerance, channel
  count
- Metadata not guaranteed: absolute `t0` — excluded from roundtrip comparison;
  post-roundtrip tests assert `t0 == 0.0`.
- Tolerances: lossless codec should match PCM within quantization tolerance;
  fallback tolerance `atol=1/32767` for int16 fixtures.
- v0.1.4 status: planned/optional_blocking

### 12. `ogg`

- Canonical: `ogg`
- Aliases: none
- Public read classes: `TimeSeries`, `TimeSeriesDict`
- Public write classes: `TimeSeries`, `TimeSeriesDict`
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: `pydub`, `tinytag`, external `ffmpeg`/`libav`
- Generator: `generators.audio`
- External writer: `pydub`
- External reader: `pydub`, `tinytag`
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: optional_blocking
  - `generate_multi_channel_single_file`: optional_blocking
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking
  - `write_single_object`: optional_blocking
  - `write_collection`: optional_blocking
  - `readback_gwexpy`: optional_blocking
  - `readback_native`: optional_blocking
  - `read_optional_dependency_missing`: blocking in base CI
  - `write_optional_dependency_missing`: blocking in base CI
- Metadata must match: sample rate, approximate duration, channel count
- Metadata not guaranteed: absolute `t0` — excluded from roundtrip comparison;
  post-roundtrip tests assert `t0 == 0.0`.
- Tolerances: lossy codec, use duration/sample-rate checks and relaxed waveform
  comparison only if stable in CI
- v0.1.4 status: planned/optional_blocking

### 13. `mp3`

- Canonical: `mp3`
- Aliases: none
- Public read classes: `TimeSeries`, `TimeSeriesDict`
- Public write classes: `TimeSeries`, `TimeSeriesDict`
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: `pydub`, `tinytag`, external `ffmpeg`/`libav`
- Generator: `generators.audio`
- External writer: `pydub`
- External reader: `pydub`, `tinytag`
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: optional_blocking
  - `generate_multi_channel_single_file`: optional_blocking
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking
  - `write_single_object`: optional_blocking
  - `write_collection`: optional_blocking
  - `readback_gwexpy`: optional_blocking
  - `readback_native`: optional_blocking
  - `read_optional_dependency_missing`: blocking in base CI
  - `write_optional_dependency_missing`: blocking in base CI
- Metadata must match: sample rate, approximate duration, channel count
- Metadata not guaranteed: absolute `t0` — excluded from roundtrip comparison;
  post-roundtrip tests assert `t0 == 0.0`.
- Tolerances: lossy codec, avoid strict sample-by-sample equality
- v0.1.4 status: planned/optional_blocking

### 14. `m4a`

- Canonical: `m4a`
- Aliases: none
- Public read classes: `TimeSeries`, `TimeSeriesDict`
- Public write classes: `TimeSeries`, `TimeSeriesDict`
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: `pydub`, `tinytag`, external `ffmpeg`/`libav`
- Generator: `generators.audio`
- External writer: `pydub`
- External reader: `pydub`, `tinytag`
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: optional_blocking
  - `generate_multi_channel_single_file`: optional_blocking
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking
  - `write_single_object`: optional_blocking
  - `write_collection`: optional_blocking
  - `readback_gwexpy`: optional_blocking
  - `readback_native`: optional_blocking
  - `read_optional_dependency_missing`: blocking in base CI
  - `write_optional_dependency_missing`: blocking in base CI
- Metadata must match: sample rate, approximate duration, channel count
- Metadata not guaranteed: absolute `t0` — excluded from roundtrip comparison;
  post-roundtrip tests assert `t0 == 0.0`.
- Tolerances: lossy codec, avoid strict sample-by-sample equality
- v0.1.4 status: planned/optional_blocking

### 15. `gbd`

- Canonical: `gbd`
- Aliases: none
- Public read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Public write classes: none
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: none
- Direct read classes: none
- Direct write classes: none
- Required read args: `timezone`
- Required write args: none
- Optional dependencies: none
- Generator: `generators.gbd`
- External writer: manual Graphtec-style header/body writer
- External reader: structural manual parser for generated fixture
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: optional_blocking
  - `generate_multi_channel_single_file`: optional_blocking
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking
  - `read_required_args`: optional_blocking with `timezone`
  - `read_missing_required_args`: blocking
  - `read_single_channel_single_file`: optional_blocking for `TimeSeries`
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesDict`
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesMatrix`
  - `write_unsupported`: blocking
  - `readback_registry_import_path`: optional_blocking
  - `readback_top_level_import_path`: optional_blocking
- Metadata must match: channel names, digital/analog channel classification,
  sample count, `dt`, timezone-resolved start time
- Tolerances: integer raw counts exact; scaled analog values `rtol=1e-12`
- v0.1.4 status: planned/optional_blocking

### 16. `tdms`

- Canonical: `tdms`
- Aliases: none
- Public read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Public write classes: none
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: none
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: `nptdms`
- Generator: `generators.tdms`
- External writer: `nptdms`
- External reader: `nptdms`
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: optional_blocking
  - `generate_multi_channel_single_file`: optional_blocking
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking
  - `read_single_channel_single_file`: optional_blocking for `TimeSeries`
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesDict`
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesMatrix`
  - `write_unsupported`: blocking
  - `read_optional_dependency_missing`: blocking in base CI
  - `readback_native`: optional_blocking
- Metadata must match: group/channel names, `wf_increment`, `wf_start_time`,
  sample count
- Tolerances: numeric `rtol=1e-12`, `atol=1e-12`
- v0.1.4 status: planned/optional_blocking

### 17. `mseed`

- Canonical: `mseed`
- Aliases: `miniseed`
- Public read classes: `TimeSeriesDict`
- Public write classes: `TimeSeriesDict`
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: `TimeSeries`, `TimeSeriesDict`
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: `obspy`
- Generator: `generators.seismic`
- External writer: ObsPy
- External reader: ObsPy
- Multi-file policy: `backend_dependent`
- Scenarios:
  - `generate_single_channel_single_file`: optional_blocking
  - `generate_multi_channel_single_file`: optional_blocking
  - `generate_multi_channel_multi_file`: nightly
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking
  - `read_alias_format`: optional_blocking for `miniseed`
  - `read_single_channel_single_file`: optional_blocking for `TimeSeriesDict`
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesDict`
  - `write_collection`: optional_blocking for `TimeSeriesDict`
  - `write_alias_format`: optional_blocking for `miniseed`
  - `readback_gwexpy`: optional_blocking
  - `readback_native`: optional_blocking with ObsPy
  - `read_optional_dependency_missing`: blocking in base CI
- Metadata must match: network/station/channel mapping, start time, sample rate,
  sample count
- Tolerances: float32 seismic payload `rtol=1e-6`, `atol=1e-8`
- v0.1.4 status: planned/optional_blocking

### 18. `sac`

- Canonical: `sac`
- Aliases: none
- Public read classes: `TimeSeriesDict`
- Public write classes: `TimeSeriesDict`
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: `TimeSeries`, `TimeSeriesDict`
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: `obspy`
- Generator: `generators.seismic`
- External writer: ObsPy
- External reader: ObsPy
- Multi-file policy: `backend_dependent`
- Scenarios:
  - `generate_single_channel_single_file`: optional_blocking
  - `generate_multi_channel_single_file`: nightly
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking
  - `read_single_channel_single_file`: optional_blocking for `TimeSeriesDict`
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesDict`
  - `write_collection`: optional_blocking for `TimeSeriesDict`
  - `readback_gwexpy`: optional_blocking
  - `readback_native`: optional_blocking with ObsPy
  - `read_optional_dependency_missing`: blocking in base CI
- Metadata must match: station/channel, start time, sample rate, sample count
- Tolerances: float32 payload `rtol=1e-6`, `atol=1e-8`
- v0.1.4 status: planned/optional_blocking

### 19. `gse2`

- Canonical: `gse2`
- Aliases: none
- Public read classes: `TimeSeriesDict`
- Public write classes: `TimeSeriesDict`
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: `TimeSeries`, `TimeSeriesDict`
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: `obspy`
- Generator: `generators.seismic`
- External writer: ObsPy
- External reader: ObsPy
- Multi-file policy: `backend_dependent`
- Scenarios:
  - `generate_single_channel_single_file`: optional_blocking
  - `generate_multi_channel_single_file`: nightly
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking
  - `read_single_channel_single_file`: optional_blocking for `TimeSeriesDict`
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesDict`
  - `write_collection`: optional_blocking for `TimeSeriesDict`
  - `readback_gwexpy`: optional_blocking
  - `readback_native`: optional_blocking with ObsPy
  - `read_optional_dependency_missing`: blocking in base CI
- Metadata must match: station/channel, start time, sample rate, sample count
- Tolerances: format-dependent; prefer exact integer payload or relaxed float
  comparison `rtol=1e-6`
- v0.1.4 status: planned/optional_blocking

### 20. `knet`

- Canonical: `knet`
- Aliases: none
- Public read classes: `TimeSeriesDict`
- Public write classes: none
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: none
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: `obspy`
- Generator: `generators.seismic`
- External writer: ObsPy or manual K-NET text fixture if ObsPy writer support is
  insufficient
- External reader: ObsPy
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: planned
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesDict`
  - `write_unsupported`: blocking
  - `read_optional_dependency_missing`: blocking in base CI
- Metadata must match: channel/station identifier, sample rate, sample count
- Tolerances: generated numeric payload `rtol=1e-6`
- v0.1.4 status: planned

### 21. `win`

- Canonical: `win`
- Aliases: `win32`
- Public read classes: `TimeSeriesDict`
- Public write classes: none
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: none
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: `obspy`
- Generator: `generators.seismic`
- External writer: manual WIN fixture or ObsPy where available
- External reader: ObsPy where available
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: planned
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking
  - `read_alias_format`: optional_blocking for `win32`
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesDict`
  - `write_unsupported`: blocking
  - `read_optional_dependency_missing`: blocking or conditional registration
    according to contract
- Metadata must match: channel identifier, sample rate if present, sample count
- Tolerances: generated integer payload exact where possible
- v0.1.4 status: planned

### 22. `ats`

- Canonical: `ats`
- Aliases: none
- Public read classes: `TimeSeries`, `TimeSeriesDict`
- Public write classes: none
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: none
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: none
- Generator: `generators.ats`
- External writer: manual ATS binary writer using `struct`
- External reader: manual header/body parser for generated fixture
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: optional_blocking
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking
  - `read_single_channel_single_file`: optional_blocking for `TimeSeries`
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesDict`
    only if generated multi-channel variant exists
  - `write_unsupported`: blocking
  - `readback_registry_import_path`: optional_blocking
  - `readback_top_level_import_path`: optional_blocking
- Metadata must match: header version, sample frequency, start time, sample
  count, integer payload scaling
- Tolerances: integer payload exact before scaling; scaled values `rtol=1e-12`
- v0.1.4 status: planned/optional_blocking

### 23. `ats.mth5`

- Canonical: `ats.mth5`
- Aliases: none
- Public read classes: `TimeSeries`
- Public write classes: none
- Registry read classes: `TimeSeries`
- Registry write classes: none
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: `mth5`
- Generator: `generators.ats`
- External writer: `mth5` or minimal MTH5 fixture helper
- External reader: `mth5`
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: planned
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: not_public
  - `read_single_channel_single_file`: optional_blocking for `TimeSeries`
  - `write_unsupported`: blocking
  - `read_optional_dependency_missing`: blocking in base CI
  - `readback_native`: optional_blocking with `mth5`
- Metadata must match: station/run/channel mapping, sample rate, start time,
  unit where available
- Tolerances: generated numeric payload `rtol=1e-12`
- v0.1.4 status: planned

### 24. `nc`

- Canonical: `nc`
- Aliases: `netcdf4`
- Public read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Public write classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: `xarray`, `netCDF4`
- Generator: `generators.netcdf`
- External writer: `netCDF4` and/or `xarray`
- External reader: `netCDF4` and/or `xarray`
- Multi-file policy: `single_file_only`
- Scenarios:
  - `generate_single_channel_single_file`: optional_blocking
  - `generate_multi_channel_single_file`: optional_blocking
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking
  - `read_alias_format`: optional_blocking for `netcdf4`
  - `read_single_channel_single_file`: optional_blocking for `TimeSeries`
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesDict`
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesMatrix`
  - `write_single_object`: optional_blocking for `TimeSeries`
  - `write_collection`: optional_blocking for `TimeSeriesDict`
  - `write_matrix`: optional_blocking for `TimeSeriesMatrix`
  - `write_alias_format`: optional_blocking for `netcdf4`
  - `readback_gwexpy`: optional_blocking
  - `readback_native`: optional_blocking
  - `read_optional_dependency_missing`: blocking in base CI
  - `write_optional_dependency_missing`: blocking in base CI
- Metadata must match: explicit time coordinate, sample rate or `dt`, channel
  variable names, units, sample count
- Tolerances: numeric `rtol=1e-12`, `atol=1e-12`
- v0.1.4 status: optional_blocking in netcdf4 CI, planned in base CI

### 25. `zarr`

- Canonical: `zarr`
- Aliases: none
- Public read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Public write classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry read classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Registry write classes: `TimeSeries`, `TimeSeriesDict`, `TimeSeriesMatrix`
- Direct read classes: none
- Direct write classes: none
- Required read args: none
- Required write args: none
- Optional dependencies: `zarr`
- Generator: `generators.zarr`
- External writer: `zarr`
- External reader: `zarr`
- Multi-file policy: `single_file_only` for logical Zarr stores; directory store
  layout is the format container, not a multi-file input list
- Scenarios:
  - `generate_single_channel_single_file`: optional_blocking
  - `generate_multi_channel_single_file`: optional_blocking
  - `read_explicit_format`: optional_blocking
  - `read_auto_identify_public`: optional_blocking
  - `read_single_channel_single_file`: optional_blocking for `TimeSeries`
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesDict`
  - `read_multi_channel_single_file`: optional_blocking for `TimeSeriesMatrix`
  - `write_single_object`: optional_blocking for `TimeSeries`
  - `write_collection`: optional_blocking for `TimeSeriesDict`
  - `write_matrix`: optional_blocking for `TimeSeriesMatrix`
  - `readback_gwexpy`: optional_blocking
  - `readback_native`: optional_blocking
  - `read_optional_dependency_missing`: blocking in base CI
  - `write_optional_dependency_missing`: blocking in base CI
- Metadata must match: group/array keys, `t0`, `dt` or sample rate, unit,
  sample count, matrix channel ordering
- Tolerances: numeric `rtol=1e-12`, `atol=1e-12`
- v0.1.4 status: optional_blocking in zarr CI, planned in base CI

## Cross-Format CI Gates

### Base PR Gate for v0.1.4

The base gate should run without optional heavy backends beyond what is already
available in the default development environment.

Blocking formats:

- `gwf`
- `hdf.ndscope`
- `hdf5`
- `csv`
- `txt`
- `wav`

Blocking policy checks for all 25 formats:

- contract schema v3 is valid
- every format has `scenario_matrix`
- every format has `coverage_status`
- every public format has a generator policy, even if `coverage_status=planned`
- unsupported writes are asserted for read-only public formats
- missing optional dependencies follow documented behavior
- direct import subprocess checks run for core blocking formats

### Optional Backend CI Gates

Separate CI jobs or workflow matrix entries should run these groups. The logical
gate identifiers below are canonical and must match the `ci_jobs` field values in
the JSON contract. The GitHub Actions workflow may expose different display names,
but it must document the mapping from each logical identifier to the concrete job
or `scripts/ci/run_gate.py` target.

- `io-audio`: `flac`, `ogg`, `mp3`, `m4a`, extended `wav`
- `io-seismic`: `mseed`, `sac`, `gse2`, `knet`, `win`, `ats.mth5`
- `io-netcdf4`: `nc`, alias `netcdf4`
- `io-zarr`: `zarr`
- `io-tdms`: `tdms`
- `io-root`: `root`

### Nightly Gate

Nightly should run all scenarios marked `nightly`, including:

- lossy codec waveform comparisons
- seismic multi-file scenarios
- optional native readback for every optional backend
- import-path subprocess checks for every public class/format pair

## Release Milestones

### v0.1.4

Scope: conformance infrastructure plus core blocking baseline.

Required deliverables:

- contract schema v3
- `tests/io_conformance` harness skeleton
- scenario policies for all 25 formats
- blocking tests for `gwf`, `hdf.ndscope`, `hdf5`, `csv`, `txt`, `wav`
- optional dependency behavior checks for all optional formats
- no silent stub fallback in conformance generators
- release notes saying this is the first conformance baseline, not full
  exhaustive enforcement

### v0.1.5

Scope: optional backend expansion.

Required deliverables:

- `tdms`
- `mseed`
- `sac`
- `gse2`
- `ats`
- `ats.mth5`
- `nc`
- `zarr`
- compressed audio formats
- `root`

### v0.1.6 or Next Minor Release

Scope: full public I/O contract enforcement.

Required deliverables:

- eliminate or justify every `coverage_status=planned`
- make every public format at least optional-blocking in an appropriate CI job
- publish generated I/O conformance table in docs
- require scenario policy updates for every new public format PR

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| GWpy GWF write backend not available in base CI | Medium | High — GWF generator falls back to `generate_gwpy_external`; if GWpy is absent, all GWF blocking generation scenarios fail | Pin GWpy in conformance extras; add a CI step that asserts the GWF backend is importable before tests run |
| ffmpeg/libav absent in `io-audio` CI runner | Medium | Medium — FLAC/OGG/MP3/M4A generators fail silently | Require ffmpeg in `io-audio` job setup; assert `shutil.which("ffmpeg")` in audio conftest |
| #417 (FrequencySeries registry) slips past v0.1.4 | High | Low — only affects FrequencySeries auto-identify scenarios, which are `planned`; no blocking gate is broken | Keep FrequencySeries auto-identify scenarios `planned` until #417 merges |
| Subprocess import-path checks inflate CI time by > 5 min | Medium | Medium — may cause PR CI to time out | Apply alias + subprocess sampling policy defined in "Test Parametrization" |
| Schema v3 validator is not ready at v0.1.4 branch cut | Medium | Medium — schema policy checks can still run against v2 contract | Gate v3 field checks behind a version field in the JSON; fail the schema test only when `schema_version == 3` |
| GWpy silent upgrade shifts GWF tolerance behavior | Low | High — floating-point comparison at `rtol=1e-12` could flip | Pin GWpy `>=X.Y,<X.Z` in conformance extras; document the pinned version in `pyproject.toml` |

## Issue Tracking

Create or update tracking issues:

- I/O conformance schema v3 and harness
- core v0.1.4 blocking format coverage
- optional backend coverage expansion
- FrequencySeries registry auto-registration and registry normalization (#417) —
  **prerequisite** for `read_auto_identify_public` and `read_auto_identify_registry`
  scenarios on any FrequencySeries class; those scenarios must remain `planned` until
  #417 lands, even if other FrequencySeries I/O scenarios are `blocking`.
- docs generation from I/O contract (rendered as a Sphinx auto-generated table at
  `docs/io/conformance_matrix.rst`; the source of truth is the JSON contract)

## Acceptance Criteria for v0.1.4

v0.1.4 can ship when all of the following are true.

**Schema and policy:**
- The schema v3 contract validates against the new v3 JSON schema.
- Every format has explicit `scenario_matrix`, `coverage_status`, `ci_jobs`, and
  `missing_dependency_policy` entries.
- No format with public read or write classes lacks a `fixture_generator` reference.

**Quantitative blocking test targets** (CI must report `blocking_pass / blocking_total`
≥ 100% for the base gate):

| Format | Min blocking rows |
|---|---|
| `gwf` | ≥ 18 |
| `hdf.ndscope` | ≥ 13 |
| `hdf5` | ≥ 30 |
| `csv` | ≥ 14 |
| `txt` | ≥ 12 |
| `wav` | ≥ 11 |
| Policy checks (all 25 formats) | ≥ 25 |

Row counts are per-scenario-tuple as defined by the (format, operation, scenario,
class, layout, format_token, dependency_mode, status) matrix. Aliases are counted
only for `read_alias_format` / `write_alias_format` rows, not the full matrix.

**Functional gates:**
- GWF multi-file external-fixture reads pass for `TimeSeries` and `TimeSeriesDict`.
- GWF GWexpy single-file write/readback passes for `TimeSeries` and `TimeSeriesDict`.
- HDF5 public family read/write/readback passes for all v0.1.4 blocking public
  classes.
- Missing optional dependency behavior is asserted in base CI for every format with
  `optional_blocking` scenarios.

**Regression and documentation:**
- All existing targeted I/O tests in `tests/` (outside `io_conformance/`) remain
  green.
- Docs and changelog describe the conformance baseline accurately as the first
  systematic baseline, not full exhaustive enforcement.
- CI outputs a `blocking_pass / blocking_total` summary line per job.
