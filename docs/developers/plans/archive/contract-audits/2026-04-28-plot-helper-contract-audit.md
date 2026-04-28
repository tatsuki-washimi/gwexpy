# Plot Helper Contract Audit

Date: 2026-04-28
Issue: #283
Branch: `codex/wave3-283-plot-helper-contracts`
Mode: docs and tests only. No runtime modules under `gwexpy/` were changed.

## Scope

This pass adds a minimal headless contract baseline for extended plot helpers
that were identified as under-tested in the GUI and plotting metadata/unit
audit. The baseline is intentionally structural: it checks return object types,
axis titles, axis labels, panel titles, and optional backend error text without
asserting pixel output, screenshot layout, or broad GUI behavior.

Covered helpers:

- `FieldPlot.add_scalar`
- `plot_gauch_dashboard`
- `PairPlot`
- `GeoMap`

Out of scope:

- Runtime behavior changes in `gwexpy/`
- Colorbar axis-count assumptions for GWpy `Plot`
- Screenshot or image-diff visual regression
- GUI/core metadata work tracked separately by #274

## Current Contracts Covered

### FieldPlot.add_scalar

The current `FieldPlot.add_scalar` contract returns a Matplotlib
`QuadMesh`-like object for a 2D slice. With a deterministic 4D
`ScalarField` whose spatial axes are `x` in meters and `y` in centimeters,
the displayed axis labels are:

- x-axis: `x [m]`
- y-axis: `y [cm]`

The helper also attempts to create a colorbar label from the field name and
unit. This test baseline does not assert colorbar axes because this local
environment exposes GWpy `Plot` colorbars inconsistently.

### plot_gauch_dashboard

The current GauCh dashboard builds three structural panels:

- `GauCh p-value Map`
- `GauCh KS Statistic Map`
- `Time Series`

The frequency panels currently use the hard-coded y-label
`Frequency [Hz]`. The time-series panel currently uses `Time [s]` and
`Amplitude [m]` for a meter-valued input series. Colorbar labels are not
asserted in this slice because the current exposure path depends on
backend-specific Matplotlib axes.

### PairPlot

The current `PairPlot` corner mode for two dictionary-keyed equal-length
series returns a `(2, 2)` axes grid. The upper-right axes is invisible, the
bottom-left cell labels the x/y axes as `H1` and `L1`, and the bottom-right
diagonal cell labels the x-axis as `L1`.

### GeoMap

When the optional PyGMT backend is unavailable, `GeoMap()` currently raises
`ImportError` with both a feature-specific message and an install hint:

- `pygmt is required for GeoMap`
- `pip install pygmt`

## Deferred Follow-ups

- FieldPlot colorbar exposure and labels: decide whether colorbar labels are
  public contract for `FieldPlot.add_scalar`, then add assertions that do not
  depend on backend-specific axes exposure.
- GauCh unitless label risk and colorbar exposure: `plot_gauch_dashboard`
  formats the time-series y-label from `ts.unit`, and its colorbar labels are
  currently exposed through backend-specific axes.
- GauCh Rayleigh branch: add a separate structural test for `rayleigh_spec`
  once the intended Rayleigh title and colorbar contract is accepted.
- PairPlot unequal and aligned lengths: define whether the public policy is
  truncate, crop, resample, reject, or preserve inputs before adding behavior
  tests for mismatched spans and sample rates.
- GeoMap, SkyMap, and PairPlot public stability: decide which ancillary plot
  helpers are stable public API and which are best-effort convenience helpers.
- Optional backend install hints: standardize backend-unavailable errors across
  plotting helpers and package extras.
- Visual regression policy: keep this baseline headless and structural; add
  screenshot or image-diff checks only with an explicit visual testing policy.
- #274 GUI separation: avoid mixing GUI metadata/unit behavior and core plot
  helper contracts in this issue.

## Verification Plan

Focused checks for this PR:

```bash
rtk env MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp pytest -q tests/plot/test_plot_helper_contracts.py -p no:cacheprovider
rtk env MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp pytest -q tests/plot/test_plot_helper_contracts.py tests/plot/test_field_plots.py tests/plot/test_gwexpy_plot_smoke.py tests/plot/test_defaults.py tests/plot/test_init_helpers.py -p no:cacheprovider
rtk ruff check tests/plot/test_plot_helper_contracts.py
rtk ruff format tests/plot/test_plot_helper_contracts.py
rtk ruff format --check tests/plot/test_plot_helper_contracts.py
rtk git diff --check
```
