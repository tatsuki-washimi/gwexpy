# Wave 3 Q-Transform Contract Audit

Date: 2026-04-27
Issue: #285, "Audit time-frequency and space-transform contracts"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This slice records the current Q-transform boundary after the first #285
transform-axis slice and the special-transform follow-up:

- direct `TimeSeries.q_transform()` GWpy passthrough behavior;
- interpolated Q-transform time/frequency axes for a deterministic small signal;
- metadata policy for direct Q-transform outputs;
- `TimeSeriesList` and `TimeSeriesDict` Q-transform container return types;
- `TimeSeriesMatrix.q_transform()` aggregation into `SpectrogramMatrix`;
- known irregular-sampling error-policy gap.

This slice does not change Q-transform algorithms, normalization, whitening,
interpolation, metadata propagation, or return classes.

## Contracts Recorded

### Direct TimeSeries Q-Transform

- `TimeSeries.q_transform()` is currently inherited from GWpy, not wrapped by a
  gwexpy-owned override.
- The direct return object is a `gwpy.spectrogram.Spectrogram`, not a
  `gwexpy.spectrogram.Spectrogram`.
- With explicit `outseg`, `tres`, `fres`, `qrange`, and `frange`, the returned
  axes are deterministic:
  - time axis follows the requested output segment and resolution;
  - frequency axis follows the requested frequency range and resolution;
  - output unit is dimensionless normalized energy.
- Direct Q-transform output does not currently preserve the input `name` or
  `channel`; this metadata loss is recorded as current behavior, not changed in
  this slice.

### Collection Containers

- `TimeSeriesList.q_transform()` returns a `SpectrogramList`.
- `TimeSeriesDict.q_transform()` returns a `SpectrogramDict` and preserves key
  order.
- Collection values are coerced into `gwexpy.spectrogram.Spectrogram` objects by
  the collection containers, but they retain the direct Q-transform metadata
  policy: unit is dimensionless, while `name` and `channel` are absent.

### Matrix Container

- `TimeSeriesMatrix.q_transform()` returns a `SpectrogramMatrix`.
- Matrix shape is `(rows, cols, times, frequencies)`.
- Rows and columns are preserved.
- Common time and frequency axes are validated across elements.
- The output unit and per-element metadata units are dimensionless.
- Per-element `name` and `channel` metadata are absent because the direct GWpy
  Q-transform result does not provide them. In `MetaDataMatrix`, the absent
  channel is normalized to the existing `Channel("None")` sentinel rather than
  the original input channel.

### Known Gap

- Direct `TimeSeries.q_transform()` does not currently use gwexpy's regularity
  guardrail. An irregularly sampled series raises a passthrough GWpy-style
  `AttributeError` instead of gwexpy's usual `ValueError` contract. This is
  represented as a strict xfail and should be handled in a behavior-changing
  follow-up if the project decides to make Q-transform gwexpy-owned.

## Deferred Behavior Changes

These are intentionally not included in this docs/test-only slice:

- Add a gwexpy `TimeSeries.q_transform()` override.
- Preserve input `name`, `channel`, or source unit metadata through direct
  Q-transform outputs.
- Change Q-transform whitening, interpolation, normalization, or axis selection.
- Convert direct `TimeSeries.q_transform()` to return a gwexpy Spectrogram.
- Change irregular-sampling error type for Q-transform.

## Remaining Follow-Up Slices For #285

1. Spectrogram cleaning: shape, mask, units, and metadata preservation.
2. Field/space transforms: identify first-class contracts versus tutorial-only
   workflows.
3. HHT numerical behavior: only after optional dependency availability and
   physics-review expectations are explicit.

## Verification

Suggested focused command:

```bash
rtk proxy pytest tests/timeseries/test_q_transform_contracts.py -q
```

Suggested compatibility command:

```bash
rtk proxy pytest \
  tests/timeseries/test_q_transform_contracts.py \
  tests/timeseries/test_transform_axis_contracts.py \
  tests/timeseries/test_timeseries_matrix_slice.py -q
```

Suggested hygiene checks:

```bash
rtk ruff check tests/timeseries/test_q_transform_contracts.py
rtk ruff format --check tests/timeseries/test_q_transform_contracts.py
rtk git diff --check
```
