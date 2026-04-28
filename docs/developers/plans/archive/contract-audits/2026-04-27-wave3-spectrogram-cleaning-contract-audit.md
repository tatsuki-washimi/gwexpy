# Wave 3 Spectrogram Cleaning Contract Audit

Date: 2026-04-27
Issue: #285, "Audit time-frequency and space-transform contracts"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This slice records the current `Spectrogram.clean()` contract after the #285
transform-axis, special-transform, and Q-transform slices:

- threshold cleaning mask shape and replacement policy;
- threshold fill modes for `nan`, `zero`, and `interpolate`;
- line-removal mask semantics for persistent frequency bins;
- rolling-median mask and unit behavior;
- combined cleaning mask shape and metadata behavior;
- preservation of time/frequency axes, value unit, name, channel, epoch, `dt`,
  and `df`;
- immutability of the source spectrogram data.

This slice does not change cleaning algorithms, thresholds, replacement values,
unit semantics, or return classes.

## Contracts Recorded

### Shared Output Contract

- `Spectrogram.clean()` returns a `gwexpy.spectrogram.Spectrogram`.
- Returned data have the same shape as the source spectrogram.
- Time and frequency axes are preserved exactly.
- `dt`, `df`, value unit, `name`, `channel`, and epoch are preserved.
- Returned data do not share memory with the source data.
- When `return_mask=True`, the mask is a boolean NumPy array with the same shape
  as the source spectrogram.

### Threshold Cleaning

- `method="threshold"` uses a per-frequency-bin MAD threshold.
- The mask is `True` only at flagged outlier pixels.
- With `fill="median"`, flagged pixels are replaced by the per-frequency median.
- `fill="nan"`, `fill="zero"`, and `fill="interpolate"` preserve the same mask,
  output shape, axes, and metadata while using the selected replacement policy.
- Source data are not mutated.

### Line Removal

- `method="line_removal"` marks entire frequency columns when persistence and
  amplitude thresholds classify the column as a persistent line.
- The returned mask is `True` for every time sample in the detected frequency
  column and `False` for untouched columns.
- Detected columns are replaced with the time median of that frequency column.

### Rolling Median

- `method="rolling_median"` returns an all-false mask.
- It preserves the source unit even though the algorithm divides by a rolling
  median trend. This is current behavior and is not changed in this slice.

### Combined Cleaning

- `method="combined"` preserves the common output contract and returns a boolean
  mask shaped like the source spectrogram.
- The threshold stage contributes flagged pixels to the returned mask.
- With line-removal active, a detected frequency bin extends that same mask by the
  entire frequency column (all time bins), including any pre-existing threshold
  flags.

## Deferred Behavior Changes

These are intentionally not included in this docs/test-only slice:

- Change rolling-median cleaned units to dimensionless.
- Change threshold statistics or MAD scaling.
- Change line-removal replacement from column median to a background estimate.
- Add mask objects with axis metadata instead of plain boolean arrays.
- Add a `SpectrogramMatrix.clean()` aggregation contract.
- Change non-finite input handling in cleaning methods.

## Remaining Follow-Up Slices For #285

1. Field/space transforms: identify first-class contracts versus tutorial-only
   workflows.
2. HHT numerical behavior: only after optional dependency availability and
   physics-review expectations are explicit.

## Verification

Executed focused command:

```bash
rtk proxy pytest tests/spectrogram/test_cleaning_contracts.py -q
```

Result: `7 passed in 1.71s`

Executed compatibility command:

```bash
rtk proxy pytest \
  tests/spectrogram/test_cleaning_contracts.py \
  tests/spectrogram/test_cleaning.py \
  tests/spectrogram/test_spectrogram_metadata_propagation.py \
  tests/spectrogram/test_standalone_contracts.py -q
```

Result: `38 passed, 1 warning in 1.74s` (known warning from GWpy TimeSeries
 constructor: `xindex was given to TimeSeries(), x0 will be ignored`)

Executed hygiene checks:

```bash
rtk ruff check tests/spectrogram/test_cleaning_contracts.py
rtk ruff format --check tests/spectrogram/test_cleaning_contracts.py
rtk git diff --check
```

Results:

- `ruff check`: `Ruff: No issues found`
- `ruff format --check`: `1 file already formatted`
- `git diff --check`: pass (no output)
