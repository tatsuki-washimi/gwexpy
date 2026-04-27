# FrequencySeries And Spectrogram Standalone Contract Audit

Date: 2026-04-27
Issue: #292, "Audit FrequencySeries and Spectrogram standalone class contracts"
Mode: audit-first; docs and regression tests only.

## Scope And Non-Goals

This audit records current standalone public-class behavior for:

- `gwexpy/frequencyseries/frequencyseries.py`
- `gwexpy/frequencyseries/bifrequencymap.py`
- `gwexpy/spectrogram/spectrogram.py`

The scope intentionally excludes collection, matrix, I/O, and plot-specific
contract changes already covered by adjacent issues. This pass does not change
runtime behavior. Return-class changes, axis-unit conversion changes, metadata
policy changes, or physics-facing numerical interpretation changes should be
split into follow-up PRs with human review where appropriate.

## Inputs Reviewed

- `FrequencySeries` construction, indexing, view/copy behavior, `_gwex_*`
  attribute propagation, phase helpers, dB conversion, rebinning, and
  quadrature combination.
- `BifrequencyMap` axis construction, indexing, slicing, crop, propagation, and
  nearest-bin helper behavior.
- `Spectrogram` indexing, view/copy behavior, phase helpers, rebinning,
  normalization, cleaning, and conversion to one-dimensional list containers.
- Existing tests under `tests/frequencyseries/` and `tests/spectrogram/`.

## FrequencySeries Contracts

- `FrequencySeries.__new__()` drops transient `fmin` and `fmax` keyword
  arguments before delegating to the GWpy constructor.
- Scalar indexing returns an `astropy.units.Quantity`; slice and typed-view paths
  return gwexpy `FrequencySeries` objects.
- `view(np.ndarray)` returns a bare ndarray; `view(type(fs))` shares the same
  data buffer and preserves GWpy metadata such as name, channel, and epoch.
- Custom `_gwex_*` attributes propagate through slicing through
  `__array_finalize__`.
- `phase()`, `angle()`, `degree()`, and `to_db()` return gwexpy
  `FrequencySeries` objects and preserve frequency axis, channel, and epoch
  while updating unit/name according to the helper.
- `rebin(width)` averages only complete bins, truncates trailing incomplete
  bins, and preserves unit, name, channel, and epoch.
- `quadrature_sum()` returns a gwexpy `FrequencySeries` on the left-hand
  frequency axis and epoch, but currently drops channel metadata.

## BifrequencyMap Contracts

- `from_points(data, f2, f1)` maps rows to `frequency2`/`xindex` and columns to
  `frequency1`/`yindex`. Bare numeric axes default to Hz.
- Scalar indexing returns `Quantity`; two-dimensional slices return
  `BifrequencyMap`; one-dimensional row slices currently return GWpy `Series`.
- `propagate()` accepts GWpy-compatible `FrequencySeries` input and currently
  returns GWpy `FrequencySeries`, not gwexpy `FrequencySeries`.
- `propagate(interpolate=False)` checks only input length, multiplies by matrix
  rows, outputs on `frequency2`, sets unit to `map.unit * input.unit`, and does
  not preserve input channel or epoch.
- `get_slice(at, axis=...)` chooses the nearest bin using raw `.value`
  comparison. Quantity units are not converted before nearest-bin selection.
- `crop()` uses inclusive lower and upper bounds and preserves unit/name.
- `inverse()` swaps the two frequency axes and inverts non-dimensionless units.

## Spectrogram Contracts

- Indexing follows inherited GWpy return classes for one-dimensional slices:
  `sg[i]` returns GWpy `FrequencySeries`, `sg[:, j]` returns GWpy `TimeSeries`,
  `sg[i, j]` returns `Quantity`, and two-dimensional slices return gwexpy
  `Spectrogram`.
- `copy()` produces independent data; typed views share data; ndarray views are
  bare ndarrays.
- `radian()` and `degree()` return real-valued gwexpy `Spectrogram` objects,
  preserve axes/channel, set units to rad/deg, and apply phase suffixes to
  names.
- `rebin(dt, df)` averages only complete time/frequency bins and truncates
  trailing incomplete bins. It preserves unit/name/channel and constructs
  explicit rebinned axes from bin centers.
- `normalize()` returns a dimensionless gwexpy `Spectrogram` and preserves
  name/channel.
- `clean(return_mask=True)` returns `(Spectrogram, bool mask)` and preserves the
  original unit/name/channel on the cleaned spectrogram.
- `to_timeseries_list()` and `to_frequencyseries_list()` return gwexpy list
  containers with gwexpy element classes, copied element data, inherited
  unit/channel, epochs equivalent to the source GPS time, and generated names
  based on the fixed frequency or time bin.

## Current Behavior Baselines

This pass adds regression coverage in:

- `tests/frequencyseries/test_standalone_contracts.py`
- `tests/spectrogram/test_standalone_contracts.py`

The tests intentionally baseline current return-class boundaries where gwexpy
objects delegate to inherited GWpy behavior. These are not necessarily ideal
public API choices, but changing them would be user-visible and should not be
mixed into this audit-first PR.

## Known Risky Behavior Changes

Do not include these in the first #292 audit PR:

- Switch inherited one-dimensional `Spectrogram` slices from GWpy classes to
  gwexpy classes.
- Change `BifrequencyMap.propagate()`, `get_slice()`, `diagonal()`, or
  `convolute()` return classes from GWpy `FrequencySeries` to gwexpy
  `FrequencySeries`.
- Add unit conversion to `BifrequencyMap.get_slice()` nearest-bin selection.
- Change `BifrequencyMap` interpolation tolerance, inclusive crop bounds, or
  channel/epoch loss.
- Change `Spectrogram` row-slice epoch behavior inherited from GWpy.
- Change `FrequencySeries.quadrature_sum()` channel propagation.

## Recommended Follow-Ups

1. Decide whether standalone helpers that currently return GWpy series should
   instead return gwexpy classes.
2. Decide whether `BifrequencyMap.get_slice()` should convert quantity units
   before nearest-bin lookup.
3. Decide whether `FrequencySeries.quadrature_sum()` and `BifrequencyMap`
   projection helpers should preserve channel/epoch metadata.
4. Add release notes if any follow-up changes public return classes or metadata
   propagation.

## Verification

Suggested focused command:

```bash
rtk pytest -q \
  tests/frequencyseries/test_standalone_contracts.py \
  tests/spectrogram/test_standalone_contracts.py \
  tests/frequencyseries/test_fs_methods.py \
  tests/frequencyseries/test_bifrequencymap_logic.py \
  tests/spectrogram/test_getitem.py \
  tests/spectrogram/test_spectrogram_core.py \
  tests/spectrogram/test_to_series_list.py
```
