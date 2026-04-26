# Core Data Model Round-Trip Metadata Audit

Date: 2026-04-26
Issue: #247, "Audit Core Data Model round-trip metadata preservation"
Mode: audit-first only; no runtime behavior changes in this pass.

## Constraints

- `.agent/AGENTS.md` was reviewed. The relevant requirements are metadata
  preservation, explicit unit and axis handling, and conservative treatment of
  data-model behavior changes.
- This document records current copy/view/pickle/DataFrame/CSV/HDF5-facing
  contracts and intentional or suspected loss points.
- Runtime behavior, tests, and public docs are unchanged in this pass.

## Inputs Reviewed

- Issue #247 body.
- The latest audit comments for #244 and #246.
- The merged #243 / #258 MetaData and MetaDataMatrix audit plan.
- `gwexpy/types/metadata.py`
- `gwexpy/types/series_matrix_io.py`
- `gwexpy/types/series_matrix_structure.py`
- `gwexpy/types/seriesmatrix_base.py`
- Related SeriesMatrix indexing and math paths where they affect views or
  conversions.
- `gwexpy/types/io/`
- `gwexpy/io/pickle_compat.py`
- `gwexpy/io/hdf5_collection.py`
- `gwexpy/io/collection_dir.py`
- `gwexpy/fields/collections.py`
- `gwexpy/fields/base.py`
- `gwexpy/types/array4d.py`
- Spectrogram, SpectrogramMatrix, and collection round-trip implementations.
- Relevant tests under `tests/types/`, `tests/fields/`, and
  `tests/spectrogram/`.

## Current Contract Summary

### Copy And View

- `SeriesMatrix.copy()` is the strongest current in-memory preservation path. It
  copies numeric values, xindex, attrs, per-element `MetaDataMatrix`, and
  row/column `MetaDataDict` entries.
- Existing tests cover value independence and at least one metadata independence
  case, but they do not exhaustively cover every structural operation.
- Raw ndarray views and `SeriesMatrix.__array_finalize__` are weaker. Metadata,
  rows, cols, and attrs are assigned by reference, and xindex length checking is
  suppressed during finalize.
- Shape-changing views can therefore carry stale xindex or metadata until a
  constructor or setter validates them.
- `astype`, `real`, `imag`, `conj`, default transpose, and reshape should not be
  treated as equivalent to `copy()`. Several paths pass existing metadata
  objects through constructors or directly, so aliasing is possible.
- `reshape()` reshapes per-element metadata but regenerates row/column metadata.
- Custom-axis `transpose()` intentionally returns a plain ndarray and drops
  SeriesMatrix metadata.
- `MetaDataDict(MetaDataDict)` and `MetaDataMatrix` construction from existing
  `MetaData` objects can preserve object references instead of deep-copying each
  entry. This is useful for lightweight wrapping but is not an isolation
  guarantee.
- Field containers are mostly component-driven. `FieldDict.copy()` calls
  `v.copy()` for each field, while list/dict batch transforms preserve metadata
  only insofar as each component field method does.
- `Array4D` and `FieldBase.__array_finalize__` propagate axis names, axis arrays,
  domain metadata, and `_gwex_` attrs for compatible views. Some
  lower-dimensional indexing paths intentionally return non-field objects, while
  `ScalarField.__getitem__` keeps integer indexing 4D-preserving.

### Pickle

- `TimeSeries`, `FrequencySeries`, and `Spectrogram` define `__reduce_ex__`
  through `gwexpy.io.pickle_compat`. The goal is GWpy-compatible pickle output
  preserving values plus primary axes, unit, name, channel, and epoch.
- That portability goal may intentionally lose gwexpy-only subclass behavior.
- `SpectrogramMatrix` has custom `__reduce__` / `__setstate__` behavior that
  appends `__dict__`.
- Existing SpectrogramMatrix tests verify values, times, frequencies, row/column
  keys, and custom `_gwex_` attrs for a 4D matrix.
- Another SpectrogramMatrix test path still documents a limitation where pickle
  preserves shape/values but not xindex/frequencies. This looks like partial
  coverage of different construction paths or stale/contradictory expectations
  that should be reconciled.
- ScalarField pickle tests verify values, unit, axis domains, space domains, and
  custom `_gwex_` attrs for real-space and k/frequency-space fields.
- Equivalent explicit pickle regression tests were not found for
  SeriesMatrix/TimeSeriesMatrix/FrequencySeriesMatrix preserving xindex,
  per-element metadata, row/column metadata, attrs, names, and channels.

### DataFrame And CSV

- `MetaDataMatrix.to_dataframe()` / CSV uses long-form row/col records and
  stringifies units, so dense name/channel/unit values are intended to
  round-trip.
- Sparse DataFrames can leave uninitialized cells. Row and column key attrs are
  not part of the DataFrame contract.
- `MetaDataDict.to_dataframe()` on the audited baseline did not explicitly
  stringify units. CSV works through pandas/object string conversion plus
  `MetaData` normalization on read. This is one of the #258 plan areas.
- `SeriesMatrix.to_pandas()` and CSV/parquet export are value-oriented. Wide
  format preserves xindex values as the DataFrame index and flattens row/column
  labels into column names. Long format preserves row, col, index, and value
  columns.
- These formats do not preserve per-element names, units, channels, row/column
  metadata, attrs, epoch, or xindex unit in a machine-readable way. Treat them
  as lossy value exports unless a later PR changes the contract.
- Collection directory CSV/TXT helpers preserve a manifest with order,
  filenames, string keys, and a small metadata payload such as unit for
  TimeSeries/FrequencySeries collections. Name/channel/axis metadata depends on
  each per-entry writer and is not generally preserved by the collection
  manifest. Non-string keys are stringified.

### HDF5-Facing Helpers

- `SeriesMatrix.to_hdf5()` / `read()` is the most complete non-copy round-trip.
  It stores values, xindex values/unit, matrix name, epoch, JSON-serializable
  attrs, per-element units/names/channels, and row/column units/names/channels.
- Existing tests cover the basic path and one metadata-rich round-trip.
- Known losses and risks:
  - row/column key types are stringified,
  - non-JSON attrs are silently skipped,
  - epoch is coerced to float where possible,
  - blank strings normalize to empty/default metadata,
  - only Quantity xindex gets an explicit unit attr.
- `gwexpy/types/io/hdf5.py` and `ascii.py` are thin GWpy re-export wrappers and
  do not add Core Data Model preservation beyond type-specific code.
- `gwexpy/io/hdf5_collection.py` preserves collection layout/kind, order, and
  keymap, but it is a collection manifest only. Per-entry values, axes, unit,
  name, and channel preservation depends on each stored entry's own HDF5 writer.
- Spectrogram collection HDF5 tests cover keys/order plus values, times,
  frequencies, unit, and GWpy interoperability. They do not yet cover
  names/channels/custom attrs for entries or collection-level metadata beyond
  the manifest.

## Test Gaps

- SeriesMatrix, TimeSeriesMatrix, and FrequencySeriesMatrix pickle round-trip
  tests for values, xindex values/unit, per-element metadata units/names/channels,
  row/column metadata, attrs, name, and epoch.
- View and operation aliasing tests for raw `view`, slicing, `astype`,
  `real`/`imag`, `conj`, default transpose, and reshape.
- DataFrame/CSV contract tests distinguishing lossy value exports from
  metadata-preserving metadata exports.
- HDF5 edge-case tests for non-string row/column keys, non-JSON attrs,
  name/channel preservation, xindex unit/no-unit behavior, and epoch type
  expectations.
- Reconciliation of the two SpectrogramMatrix pickle expectations: one test path
  says axes may be lost, another verifies axes are preserved for 4D matrices.
- Collection round-trip tests for entry name/channel/custom attrs and explicit
  key-type stringification behavior.

## Recommended Next PRs

1. SeriesMatrix round-trip contract tests: cover copy, view, pickle, HDF5,
   DataFrame, and CSV with explicit assertions for values, axes, units, names,
   channels, attrs, row/column metadata, and known intentional loss.
2. Metadata CSV and shape follow-up from #243 / #258: strict
   `MetaDataMatrix` setter shape validation and explicit `MetaDataDict` unit
   serialization should stay coordinated with broader CSV claims.
3. Documentation update for lossy exports: state that
   `SeriesMatrix.to_pandas()` / CSV / parquet are value exports, while HDF5 is
   the stronger metadata-preserving path.
4. SpectrogramMatrix pickle cleanup: make the intended guarantee
   construction-independent, or update stale tests and comments if partial
   preservation is intentional.
5. Collection manifest follow-up: decide whether collection directory/HDF5
   manifests preserve only string keys/order/layout or also entry-level
   name/channel/custom metadata.

## Dependencies

- #243 / #258 defines MetaData and MetaDataMatrix setter shape semantics and CSV
  unit serialization.
- #244 defines SeriesMatrix row/column metadata, xindex, shape, slicing, reshape,
  transpose, and math invariants.
- #246 defines the `to_matrix()` axis/unit/per-element metadata contract before
  round-trip paths can promise preservation of those converted objects.

## Audit Notes

- This file records source/test inspection and current contract risks only.
- No runtime code, tests, public docs, commits, or behavior changed as part of
  the audit content.
- Follow-up changes should explicitly mark intentional metadata loss versus
  regression candidates.
