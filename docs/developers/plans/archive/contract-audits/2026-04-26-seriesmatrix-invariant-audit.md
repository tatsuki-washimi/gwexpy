# SeriesMatrix Metadata, Axis, And Shape Invariant Audit

Date: 2026-04-26
Issue: #244, "Audit SeriesMatrix shared metadata, axis, and shape invariants"
Mode: audit-first only; no runtime behavior changes in this pass.

## Constraints

- `.agent/AGENTS.md` was reviewed. The relevant requirements are metadata
  preservation, strict shape/axis consistency, explicit unit handling, and human
  review for data-model behavior changes.
- This document records observed contracts and follow-up candidates. It does not
  change `SeriesMatrix`, `TimeSeriesMatrix`, `FrequencySeriesMatrix`, or
  `SpectrogramMatrix` behavior.
- The #243 / #258 MetaData and MetaDataMatrix plan is now merged. Runtime
  changes in this area should still coordinate with that plan because
  SeriesMatrix delegates per-element metadata storage to MetaDataMatrix.

## Inputs Reviewed

- Issue #244 body and the merged #243 / #258 metadata audit context.
- `gwexpy/types/seriesmatrix_base.py`
- `gwexpy/types/series_matrix_core.py`
- `gwexpy/types/series_matrix_indexing.py`
- `gwexpy/types/series_matrix_math.py`
- `gwexpy/types/series_matrix_structure.py`
- Related mixins and helpers:
  - `gwexpy/types/series_matrix_analysis.py`
  - `gwexpy/types/series_matrix_io.py`
  - `gwexpy/types/seriesmatrix_validation.py`
- Matrix subclasses and related implementation:
  - `gwexpy/timeseries/matrix*.py`
  - `gwexpy/frequencyseries/matrix*.py`
  - `gwexpy/spectrogram/matrix*.py`
- Relevant tests under `tests/types/`, especially indexing, structure, math,
  I/O, validation, normalization, and known-fix coverage.

## Current Contract Summary

### Construction

- The shared SeriesMatrix contract is a numeric value shape of `(N, M, K)`, a
  `MetaDataMatrix` shape of `(N, M)`, `len(rows) == N`, `len(cols) == M`, and
  `len(xindex) == K`.
- `TimeSeriesMatrix` and `FrequencySeriesMatrix` mostly map time or frequency
  axes onto the shared `xindex` contract.
- `SpectrogramMatrix` is separate and custom. It accepts batch/time/frequency or
  row/column/time/frequency style data, with `times` matching `shape[-2]` and
  `frequencies` matching `shape[-1]`.
- Risk: constructing `SeriesMatrix(existing_series_matrix)` appears to rebuild a
  default xindex because `_handle_seriesmatrix()` reports `detected_xindex=None`
  unless the caller passes an explicit xindex.
- Risk: direct 2D ndarray normalization can reshape values as `(N, 1, M)` while
  attribute arrays are built for `(N, M)`. Full construction should fail later,
  but coverage mostly exercises the lower-level normalization helper.

### Slicing And Assignment

- Regular SeriesMatrix slicing is intended to preserve a 3D matrix result,
  slice `xindex` on the sample axis, and slice row, column, and per-element
  metadata on row/column axes.
- Scalar row plus scalar column returns the configured Series type rather than a
  one-cell SeriesMatrix.
- Risk: `submatrix()` resolves labels to integer lists and then calls
  `self[ri, ci, :]`. Current list handling treats lists as label lists, so the
  integer lists can be routed through `row_index(0)` / `col_index(0)`. Existing
  tests document this as a known issue instead of asserting success.
- Risk: `__setitem__` writes numeric values only. Assigning a Series or Quantity
  with different unit, name, or channel does not update element metadata, so
  stored values and metadata can diverge.

### Math And Reductions

- NumPy ufunc paths generally check shape and xindex compatibility and rely on
  `MetaDataMatrix.__array_ufunc__` for unit propagation.
- `__matmul__`, `trace`, `diagonal`, `det`, `inv`, `schur`, and `angle` have
  explicit handling.
- Risk: `__matmul__` checks only sample length, not xindex equality or xindex
  values. Same-length matrices on different sample axes can be multiplied
  silently while the result inherits `self.xindex`.
- Risk: matrix multiplication result metadata currently keeps unit information
  per result cell but loses result name/channel semantics. This may be
  intentional, but it should be made explicit and tested.
- Risk: `diagonal(output="matrix")` zeroes off-diagonal numeric values while
  reusing the original metadata matrix, leaving off-diagonal zero cells with the
  original units, names, and channels.

### Copy, View, And Structure Operations

- `copy()` is the strongest current preservation path. It deep-copies values,
  per-element metadata, row/column metadata, xindex, attrs, and matrix-level
  properties.
- `astype`, `real`, `imag`, `conj`, transpose, and reshape paths are less
  clearly isolated. Some paths pass existing metadata objects through
  constructors or directly into new matrices, creating possible aliasing unless
  construction deep-copies each entry.
- Risk: `reshape()` reconstructs value and metadata shapes but does not pass
  existing row/column labels or metadata, so row/column metadata is regenerated
  or lost.

### Analysis And Conversion

- `crop`, `append`, `diff`, `pad`, and `interpolate` use
  `_get_meta_for_constructor()`, so they generally preserve row, column, and
  per-element metadata while changing xindex.
- Risk: append paths verify shape and contiguity, but row/column key and
  metadata equality are not visibly enforced before concatenating.
- Risk: `TimeSeriesMatrix._apply_timeseries_method()` builds a metadata matrix
  from element-wise method results but assigns it to `new_mat._meta`. The shared
  SeriesMatrix surface uses `meta`, so this may fail to install the intended
  per-element metadata.
- `to_pandas()` and CSV flatten values and row/column keys but do not preserve
  full element, row, or column metadata. Treat these as value exports unless a
  later PR changes the contract.
- HDF5 read/write is the stronger metadata-preserving path and already has
  partial coverage.

### SpectrogramMatrix Specific Risks

- Invalid ndim construction does not consistently fail loudly.
- Binary ufuncs do not check equality of both `times` and `frequencies` between
  operands.
- Slicing with time/frequency keys appears able to reuse full
  `times`/`frequencies` arrays, and spatial metadata can become stale when a key
  includes more than row/column axes.
- This is the largest axis-invariant risk because SpectrogramMatrix does not use
  the shared 3D SeriesMatrix indexing path.

## Test Gaps

- Construction tests for `SeriesMatrix(existing_series_matrix)` preserving
  xindex and metadata.
- Full-constructor tests for 2D ndarray normalization plus metadata shape
  consistency.
- Positive regression coverage for `submatrix()` after label-to-integer
  resolution.
- Assignment tests for Series and Quantity values with unit/name/channel
  metadata.
- Metadata aliasing tests for `astype`, `real`, `imag`, `conj`, transpose, and
  reshape.
- Row/column preservation tests for reshape, or explicit tests documenting
  intentional row/column regeneration.
- `__matmul__` tests for same-length but different xindex values or units.
- `diagonal(output="matrix")` tests for off-diagonal metadata semantics.
- Append and append-exact tests for row/column key and metadata compatibility.
- TimeSeriesMatrix method-wrapper tests asserting that element metadata from
  returned Series objects survives.
- SpectrogramMatrix tests for invalid ndim, time/frequency slicing, binary ufunc
  axis equality, and row/column metadata slicing when time/frequency axes are
  involved.
- Conversion tests that distinguish value-only exports from metadata-preserving
  persistence formats.

## Recommended Next PRs

1. SeriesMatrix construction/slicing PR: preserve xindex when constructing from
   an existing SeriesMatrix, fix `submatrix()` integer-list handling, and add
   strict shape/xindex/meta invariant tests.
2. Structure/copy PR: decide and test whether structural operations deep-copy
   metadata or intentionally share it; fix or document reshape row/column
   metadata behavior.
3. Math/reduction PR: enforce xindex equality in `__matmul__`, define result
   name/channel semantics, and clarify diagonal-matrix off-diagonal metadata.
4. TimeSeriesMatrix wrapper PR: fix or verify `_apply_timeseries_method()`
   metadata assignment and add regression tests.
5. SpectrogramMatrix axis PR: fail loudly for invalid ndim, slice
   `times`/`frequencies` consistently, and check both time and frequency axes in
   binary operations.
6. Conversion PR: define metadata round-trip expectations for pandas, CSV, and
   HDF5 in coordination with the metadata serialization plan.

## Dependencies

- #243 / #258 defines the MetaData and MetaDataMatrix storage plan that
  SeriesMatrix uses for per-element metadata.
- #246 depends on these invariants because `to_matrix()` outputs become
  SeriesMatrix or SpectrogramMatrix objects.
- #247 depends on these invariants because round-trip preservation is only
  meaningful when slicing, copying, reshaping, and math keep axes and metadata
  coherent.

## Audit Notes

- This file records source/test inspection and current contract risks only.
- No runtime code, tests, or public docs behavior changed in this pass.
- Follow-up behavior changes touching metadata or physics-facing axes should be
  reviewed as data-model changes before merge.
