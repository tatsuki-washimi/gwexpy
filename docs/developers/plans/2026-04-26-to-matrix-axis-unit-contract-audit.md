# to_matrix Axis And Per-Element Unit Contract Audit

Date: 2026-04-26
Issue: #270, "Define and test family-specific to_matrix contracts"
Mode: audit-first only; no runtime behavior changes in this pass.

## Constraints

- `.agent/AGENTS.md` was reviewed. The relevant requirements are explicit unit
  conversion, axis consistency, metadata preservation, and human review for
  data-model behavior changes.
- This document records current `to_matrix()` and related collection conversion
  behavior. It does not change conversion semantics.
- The #244 SeriesMatrix audit and #243 / #258 MetaDataMatrix plan are treated as
  dependencies for later runtime changes.

## Inputs Reviewed

- Issue #270 scope and reviewer blockers for family-specific contract docs/tests.
- The latest #244 SeriesMatrix audit findings.
- The merged #243 / #258 MetaData and MetaDataMatrix audit plan.
- `gwexpy/timeseries/collections.py`
- `gwexpy/timeseries/preprocess.py`
- `gwexpy/timeseries/matrix.py`
- `gwexpy/frequencyseries/collections.py`
- `gwexpy/frequencyseries/matrix.py`
- `gwexpy/spectrogram/collections.py`
- `gwexpy/spectrogram/matrix.py`
- `gwexpy/types/seriesmatrix_base.py`
- `gwexpy/types/series_matrix_core.py`
- `gwexpy/types/seriesmatrix_validation.py`
- `gwexpy/types/metadata.py`
- Field container implementation in `gwexpy/fields/`.
- Relevant tests under `tests/timeseries/`, `tests/frequencyseries/`,
  `tests/spectrogram/`, `tests/types/`, and `tests/fields/`.

## Overall Finding

`to_matrix()` does not currently have one shared contract across container
families:

- TimeSeries collections use alignment and optional resampling.
- Spectrogram collections use strict axis equality after unit conversion.
- FrequencySeries collections currently check only sample length.
- Field containers do not expose a SeriesMatrix-style `to_matrix()` API.

This split may be intentional, but it should be documented as a family-specific
contract and tested accordingly.

## Developer Contract Table

This table records the current family-specific contract that downstream code
can rely on until a later, physics-reviewed runtime change explicitly narrows
or broadens it.

| Family | Supported collection entry point | Axis comparison and alignment | Resampling and tolerance | Per-element value-unit policy | Global matrix-unit policy | Row/column metadata and round trip |
| --- | --- | --- | --- | --- | --- | --- |
| `TimeSeries` | `TimeSeriesDict.to_matrix()` and `TimeSeriesList.to_matrix()` | Aligns every series onto a common time grid through `align_timeseries_collection()`; axes do not need exact equality when the requested overlap/alignment succeeds. | `align="intersection"` is the default. Alignment kwargs, including resampling tolerance where supported by the helper, are delegated to `align_timeseries_collection()`. There is no SeriesMatrix-style exact time-axis comparison at collection conversion time. | Source `TimeSeries.unit` values are not written into `MetaDataMatrix`; reconstructed elements from `to_dict()`/`to_list()` are dimensionless under the current path. Source channels are likewise not preserved in per-cell metadata. | No explicit global matrix unit is set by collection conversion. | Dict keys are written to per-cell names through `channel_names`, but row keys remain generated (`row0`, `row1`, ...), so `to_matrix().to_dict()` does not preserve original dict keys as dictionary keys. List element names become per-cell names when available; list rows are generated. |
| `FrequencySeries` | `FrequencySeriesDict.to_matrix()` only | Requires equal sample length only. Frequency coordinate values and frequency-axis units are not compared; the output frequency axis is taken from the first element. | No resampling and no tolerance parameter. Same-length, different frequency coordinates are accepted under the current contract. | Each element's `unit`, `name`, and `channel` are stored in `MetaDataMatrix` and restored by `to_dict()`/`to_list()`. | No explicit global matrix unit is set by collection conversion. | Dict keys become row keys and survive `to_matrix().to_dict()`. The single column is named `value`; `to_list()` flattens rows in insertion order. `FrequencySeriesList` currently has no `to_matrix()` method. |
| `Spectrogram` | `SpectrogramDict.to_matrix()` and `SpectrogramList.to_matrix()` | Requires identical shape, identical times after conversion to the first time-axis unit, and identical frequencies after conversion to the first frequency-axis unit. | No resampling and no tolerance parameter; axis values are compared with exact equality after unit conversion. | Each element's `unit`, `name`, and `channel` are stored in `MetaDataMatrix` and restored by `to_dict()`/`to_list()`. | The matrix `unit` is the first element unit only when all element units are equal; mixed units set the global matrix unit to `None`. | Dict keys become row keys and survive `to_matrix().to_dict()`. List rows are generated (`batch0`, `batch1`, ... for 3D matrices), while `to_list()` preserves element metadata in flattened order. |
| Field containers | No SeriesMatrix-style collection `to_matrix()` API | `FieldList`/`FieldDict` validation uses field-specific axis compatibility rather than SeriesMatrix conversion. | Field validation uses field tolerances for coordinate values; this is separate from collection `to_matrix()` semantics. | Component units remain on fields; `VectorField.to_array()`/`TensorField.to_array()` return raw arrays without a `MetaDataMatrix`. | Not applicable. | Not applicable to SeriesMatrix round trips. |

Current contract decisions for this audit pass:

- `TimeSeries` source value units and channels are documented as not preserved
  in `MetaDataMatrix` by collection `to_matrix()` today. Preserving them would
  be a runtime/data-model change and should be handled in a separate
  physics-reviewed PR.
- `FrequencySeriesDict.to_matrix()` is documented and tested as length-only
  today. Enforcing frequency-axis equality after unit conversion would be a
  runtime behavior change and should be split from this documentation/test
  contract PR.

## Current Contract Summary

### TimeSeriesDict And TimeSeriesList

- Axis behavior: `to_matrix(align="intersection", **kwargs)` delegates to
  `align_timeseries_collection()`. That helper computes a common time grid,
  standardizes time-like axes to seconds/common units, and can resample through
  `TimeSeries.asfreq(..., tolerance=tolerance)`.
- For mixed sampling steps, `align_timeseries_collection()` selects the
  coarsest available grid (`max(dt)`) as the target time grid before
  resampling.
- This is an alignment contract, not strict equality. Different but overlapping
  axes may succeed after alignment.
- Mismatch errors exist for empty input, no overlap, incompatible or non-time-like
  dt in time-mode, and unsupported alignment modes.
- Source `TimeSeries.unit` is not carried into `TimeSeriesMatrix` per-element
  metadata by the current conversion path. The alignment helper returns numeric
  values, then `TimeSeriesMatrix(data, times=times)` is called without unit,
  units, or meta.
- No explicit global matrix unit is set. The practical result is dimensionless
  per-cell metadata unless metadata is supplied separately.
- Dict/list identity behavior is uneven. Dict/list names are written through
  `matrix.channel_names`, which updates per-element metadata names. Row keys
  remain generated defaults, so `to_matrix().to_dict()` does not preserve the
  original dict keys as row metadata.

### FrequencySeriesDict And FrequencySeriesList

- Axis behavior: `FrequencySeriesDict.to_matrix()` checks only
  `len(fs) == n_samp`. It does not compare `fs.frequencies` values or frequency
  units between elements.
- HIGH RISK: same-length but different frequency coordinates are silently
  accepted today; this should be prioritized in the next contract PR.
- The output uses the first series' frequency axis unconditionally.
- Length mismatch raises a clear `ValueError`; same-length but different
  frequency coordinates or convertible units are silently accepted.
- Per-element value units are stronger than TimeSeries: each
  `FrequencySeries.unit` is stored in `MetaData(unit=fs.unit, name=fs.name,
  channel=fs.channel)`.
- `rows=keys`, `cols=["value"]`, and per-element name/channel are preserved.
- No explicit global matrix unit is set.
- No `FrequencySeriesList.to_matrix()` implementation was found. If the API is
  expected, this is an API or documentation gap.

### SpectrogramList And SpectrogramDict

- Axis behavior: both `times` and `frequencies` are checked through
  `check_shape_xindex_compatibility()`.
- Candidate axes are converted to the first axis unit using `.to_value()` and
  then compared with exact `np.array_equal`. There is no tolerance parameter.
- Shape, time, and frequency mismatches raise `ValueError`, with index/key
  context added by collection conversion methods.
- Differing spectrogram value units are allowed and stored per element in
  `MetaDataMatrix` with name/channel.
- Global matrix unit is set to the first unit only when all spectrogram units are
  equal. Mixed units produce `global_unit = None`.
- `SpectrogramDict.to_matrix()` passes `rows=keys`; `SpectrogramList.to_matrix()`
  uses default row labels.
- This is the best-covered family, but the exact no-tolerance behavior and
  frequency-axis unit conversion deserve explicit tests.

### Field Containers

- `FieldList` and `FieldDict` do not expose a SeriesMatrix-style `to_matrix()`
  conversion API.
- `FieldList(..., validate=True)` and `FieldDict(..., validate=True)` enforce a
  different compatibility contract:
  - value units must be equivalent,
  - axis names and domains must match,
  - axis units must match exactly,
  - axis coordinate values use `np.allclose` with field-level tolerances.
- This is stricter on axis units than Spectrogram/SeriesMatrix because
  convertible-but-not-identical axis units are rejected instead of converted.
- `VectorField.to_array()` and `TensorField.to_array()` stack raw values and do
  not return a metadata-bearing matrix object. Axes and units remain on the
  component fields, not in the returned ndarray.

### Shared SeriesMatrix Observations

- SeriesMatrix construction expects value shape `(N, M, K)`,
  `MetaDataMatrix.shape == (N, M)`, and xindex length `K`.
- `check_shape_xindex_compatibility()` performs exact equality after unit
  conversion and does not accept tolerance.
- #244 notes that downstream operations are not uniformly strict. For example,
  ufunc paths check xindex compatibility, while `__matmul__` checks only sample
  length.

## Test Gaps

### TimeSeries

- Tests deciding whether source `TimeSeries.unit` should be preserved in
  `TimeSeriesMatrix.meta.units` or intentionally dropped.
- Mixed value-unit tests through `to_matrix()`.
- Dict key preservation tests for row keys and `to_matrix().to_dict()`
  round-trip labels.
- Tests distinguishing alignment/resampling behavior from strict axis equality,
  including `tolerance` effects.
- Name/channel preservation tests.

### FrequencySeries

- Tests for same-length but different frequency values raising instead of using
  the first axis silently.
- Tests for convertible frequency units succeeding after conversion if that is
  the intended contract.
- Tests for non-convertible frequency-axis units raising clear errors.
- Tests for mixed per-element value units and global/no-global unit behavior.
- Decision and tests for whether `FrequencySeriesList.to_matrix()` should exist.

### Spectrogram

- Frequency-axis convertible-unit tests, not only time-axis conversion.
- Tests documenting no tolerance for tiny floating differences, or tests for a
  future tolerance if public behavior changes.
- Uniform-unit global unit tests versus mixed-unit `None` tests.
- Row label behavior tests for List versus Dict.

### Field Containers

- If no `to_matrix()` API is intended, docs/tests should state that fields use
  validation and `to_array()` rather than SeriesMatrix conversion.
- If a future field matrix conversion is intended, tests need an axis
  conversion/tolerance contract, per-component unit policy, and metadata
  preservation expectations.

### Docs

- Public docs mention `to_matrix()` as a conversion entry point but do not
  consistently explain the different family contracts: TimeSeries alignment,
  Spectrogram strict axes with per-element units, FrequencySeries length-only
  behavior, and field non-participation.

## Recommended Next PRs

1. Contract documentation PR: add a public/developer table for `to_matrix()`
   semantics by family, including axis comparison, alignment, tolerance,
   value-unit policy, global-unit policy, and row/column metadata policy.
2. TimeSeries conversion PR: decide whether units/channels should be preserved
   in `MetaDataMatrix`; if yes, pass metadata/units/rows during construction
   instead of setting only `channel_names`.
3. FrequencySeries conversion PR: enforce frequency-axis equality after unit
   conversion, matching the Spectrogram/SeriesMatrix rule, or document that
   length-only conversion is intentional.
4. Spectrogram polish PR: add explicit frequency-unit conversion tests,
   exact-comparison tests, and uniform/mixed global-unit tests.
5. Field container PR/docs: explicitly document that fields do not convert to
   SeriesMatrix, or design a separate field-to-array/matrix contract.
6. Round-trip PR: add `collection.to_matrix().to_dict()/to_list()` tests for
   labels, units, and channels across TimeSeries, FrequencySeries, and
   Spectrogram.

## Dependencies

- #244 defines the SeriesMatrix axis, row/column metadata, and operation
  invariant contracts that every `to_matrix()` result relies on.
- #243 / #258 defines MetaDataMatrix setter shape behavior and unit
  serialization. Per-element value units live in that layer.
- #247 depends on this issue because `to_matrix()` is a major conversion path
  into objects whose metadata should later survive copy, pickle, DataFrame, CSV,
  and HDF5 paths.

## Audit Notes

- This file records source/test inspection and current contract risks only.
- No runtime code behavior changed in this pass.
- This pass updates public/docs contract coverage and adds focused contract tests
  in `tests/types/test_to_matrix_family_contracts.py`.
- Later behavior changes should be split by container family so tests can encode
  the intended contract before broadening guarantees.
