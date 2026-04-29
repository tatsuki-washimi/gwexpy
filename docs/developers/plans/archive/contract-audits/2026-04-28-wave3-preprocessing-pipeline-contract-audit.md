# Wave 3 Preprocessing Pipeline Contract Audit

Date: 2026-04-28
Issue: #288, "Audit preprocessing pipeline decomposition and forecasting contracts"
Mode: audit-first; docs and regression tests only.
First-release follow-up owner: #346 (preprocessing runtime/statistical policy decisions).

## Scope For This Slice

This first #288 slice records deterministic preprocessing and decomposition
contracts that can be baselined without optional heavy dependencies or runtime
behavior changes:

- `standardize_timeseries()` TimeSeries metadata preservation and inverse-model
  return type;
- `impute_timeseries()` NaN versus Inf policy and Quantity `max_gap` conversion;
- `Pipeline` plus `StandardizeTransform` value and time-metadata round trips;
- `TimeSeriesMatrix.to_list()` flattening for a known multi-row, multi-column
  matrix shape;
- passive `PCAResult` metadata, alias, and summary contracts.

This slice does not change preprocessing algorithms, statistical defaults,
metadata restoration, optional dependency behavior, ARIMA/Hurst forecasting, or
PCA/ICA runtime paths.

## Contracts Recorded

### Standardization

- `standardize_timeseries()` returns a dimensionless `TimeSeries` while
  preserving `t0`, `dt`, `name`, and `channel`.
- The returned `StandardizationModel.inverse_transform()` accepts a
  `TimeSeries`-like object but returns a plain ndarray of restored values, not a
  metadata-bearing `TimeSeries`.
- `Pipeline([("standardize", StandardizeTransform())])` round-trips values and
  preserves `t0`, `dt`, `unit`, and `name` for the covered regular
  `TimeSeries` case.
- `Pipeline.inverse_transform()` defaults to strict mode. If a step such as
  `ImputeTransform` does not advertise inverse support, strict inverse raises
  `ValueError` with a "does not support inverse_transform" message.
- `StandardizeTransform.inverse_transform()` raises `TypeError` for an
  incompatible plain ndarray after fitting on a `TimeSeries`; this slice records
  the current unsupported-input error surface without changing accepted inputs.

### Imputation

- `impute_timeseries()` treats `NaN` as missing and does not treat `Inf` as
  missing. In the covered forward-fill case, a `NaN` is filled from the previous
  finite sample while an existing `Inf` remains unchanged.
- TimeSeries metadata (`t0`, `dt`, `unit`, `name`) is preserved in the covered
  imputation path.
- Quantity `max_gap` values are converted to the TimeSeries time-axis unit. In
  the covered millisecond-axis case, `500 ms` permits interpolation across a
  `400 ms` valid-sample gap while `0.2 s` blocks that same gap.

### Matrix And Pipeline Collection Handling

- `TimeSeriesMatrix.to_list()` currently flattens a `(rows, cols, time)` matrix
  in row-major order into a `TimeSeriesList`.
- Flattened elements preserve per-element value arrays, names, `t0`, `dt`, and
  units for the covered `(2, 2, 4)` matrix.
- In the covered multivariate `StandardizeTransform` path, fitting and
  transforming a `TimeSeriesList` currently returns a `TimeSeriesMatrix`, while
  inverse-transforming that matrix restores a flat `TimeSeriesList` through
  `to_list()` semantics. Values, names, `t0`, and `dt` round-trip in this path,
  but restored list elements are currently dimensionless rather than preserving
  the original physical unit.
- This test is intentionally separate from collection-restoration runtime
  changes. Follow-up work should decide whether row/column structure should be
  preserved or whether flat list semantics are the intended public contract.

### Decomposition Result Objects

- `PCAResult.components` and `PCAResult.components_` are aliases to the wrapped
  sklearn model's `components_`.
- `PCAResult.explained_variance_ratio` and
  `PCAResult.explained_variance_ratio_` are aliases to the wrapped model's
  `explained_variance_ratio_`.
- `PCAResult.summary_dict()` reports a list-valued
  `explained_variance_ratio` and `n_components`.
- `channel_labels`, `preprocessing`, and `input_meta` are currently passive
  stored metadata; this slice does not assert mutation, validation, or metadata
  propagation through transform/inverse-transform paths.
- `pca_inverse_transform()` and `ica_inverse_transform()` use
  `input_meta["original_shape"]` to restore the reconstructed matrix shape when
  the reconstructed feature count matches that shape. The returned matrix takes
  `t0` and `dt` from the scores/sources input object, not from `input_meta`.
- PCA/ICA inverse reconstruction applies stored `channel_labels` to the returned
  matrix when labels are present. The covered ICA inverse case records passive
  result metadata and the non-prewhitened reconstruction path only.

## Stable Versus Experimental Surfaces

For this audit, the stable preprocessing surface is the array and TimeSeries
contract around imputation, standardization, whitening entry points, and
pipeline composition. These paths should preserve physical metadata unless the
operation intentionally returns dimensionless data.

The ML and time-series analysis helpers are more experimental:

- PCA/ICA result objects expose useful metadata, but transform and inverse shape
  restoration still have edge-case fallbacks that should be documented before
  behavior changes.
- ARIMA/SARIMAX forecasting returns TimeSeries objects and intervals, but GPS
  timestamp assumptions, interval shape, optional dependency policy, and
  differencing semantics need a dedicated follow-up slice.
- Hurst and local-Hurst helpers depend on optional backends and convert window
  sizes to samples; backend errors, local-output metadata, and insufficient-data
  policy need a dedicated follow-up slice.

## Deferred Behavior Changes

These are intentionally not included in this docs/test-only slice:

- Change `standardize_timeseries()` inverse handling to return a TimeSeries or
  restore units.
- Change `StandardizeTransform` unit semantics, including whether transformed
  values should be dimensionless.
- Treat `Inf` as missing data, reject non-finite values, or add non-finite input
  validation to imputation, standardization, whitening, PCA, ICA, ARIMA, or
  Hurst helpers.
- Change `TimeSeriesMatrix.to_list()` flattening, collection restoration, or
  row/column metadata preservation.
- Change multivariate pipeline collection restoration to preserve original list
  units on inverse output.
- Change whitening epsilon validation or numerical regularization thresholds.
- Change PCA/ICA inverse shape fallback behavior, optional dependency errors,
  reproducibility knobs, or metadata propagation.
- Change ARIMA/SARIMAX forecast timestamps, interval labels/shapes, leap-second
  assumptions, optional dependency errors, or NaN/imputation behavior.
- Change Hurst/local-Hurst backend selection, window conversion, output
  metadata, or edge-case handling.

## Follow-Up Slices For #288

1. Pipeline semantics: unsupported input errors, fit versus transform behavior,
   strict/non-strict inverse ordering, collection restoration, and multivariate
   `TimeSeriesDict`/`TimeSeriesList` contracts.
2. Whitening and matrix preprocessing: epsilon validation, dimensionless unit
   policy, PCA versus ZCA shape contracts, covariance finite-ness, and metadata
   propagation.
3. PCA/ICA runtime contracts: optional dependency behavior, random-state and
   tolerance knobs, NaN/impute policy, component labels, inverse fallback shape,
   and channel metadata.
4. ARIMA/SARIMAX forecasting: statsmodels and pmdarima optional dependency
   behavior, prediction and forecast metadata, confidence interval names/shapes,
   differencing assumptions, and GPS/TAI timestamp notes.
5. Hurst/local-Hurst: optional backend fallback order, local window/step
   quantity conversion, output `TimeSeries` metadata, insufficient data, and
   NaN/imputation policy.

## Verification

Focused contract check:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q tests/timeseries/test_preprocessing_pipeline_contracts.py
```

Related preprocessing and decomposition regression checks:

```bash
rtk env XDG_CACHE_HOME=/tmp MPLCONFIGDIR=/tmp/matplotlib \
  pytest -q \
  tests/timeseries/test_preprocessing_pipeline_contracts.py \
  tests/timeseries/test_preprocess_impute.py \
  tests/timeseries/test_preprocess_standardize_whiten.py \
  tests/timeseries/test_pipeline.py \
  tests/timeseries/test_decomposition.py \
  tests/signal/test_imputation.py \
  tests/signal/test_preprocessing.py
```

Changed-file hygiene:

```bash
rtk ruff check tests/timeseries/test_preprocessing_pipeline_contracts.py
rtk ruff format --check tests/timeseries/test_preprocessing_pipeline_contracts.py
rtk python - <<'PY'
import yaml
from pathlib import Path
yaml.safe_load(Path("docs/developers/plans/manifests/audit-manifest-288-preprocessing-contracts.yaml").read_text())
PY
rtk git diff --check
```
