# Plot Helper Contract Audit

Date: 2026-04-28
Issue: #283
Branch: `codex/wave3-283-plot-helper-contracts`
Mode: contract tests plus narrow runtime hardening for dynamic labels and
FieldPlot colorbar exposure.

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

- Unit conversions; labels reflect existing metadata units only.
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

The helper creates a colorbar label from the field name and unit without
rendering empty brackets for unitless data. Empty labels are omitted rather than
passed as empty strings to Matplotlib. `FieldPlot.last_field_colorbar` now
exposes the most recent scalar colorbar handle returned by `Plot.colorbar()` so
tests and callers can inspect the label without relying on backend-specific
figure axes ordering. Repeated scalar plot calls intentionally replace this
attribute with the latest scalar colorbar.

`FieldPlot.add_vector` uses the same bracket-safe axis label formatter for
vector slices.

### plot_gauch_dashboard

The current GauCh dashboard builds three structural panels:

- `GauCh p-value Map`
- `GauCh KS Statistic Map`
- `Time Series`

The frequency panels derive their y-label unit from the displayed
spectrogram frequency metadata, for example `Frequency [Hz]` or
`Frequency [kHz]`, without converting data. The time-series panel uses
`Time [s]` and formats the amplitude label from the input series unit,
omitting brackets for unitless series. Colorbar labels are still treated as
backend figure details for the GauCh dashboard.

### PairPlot

The current `PairPlot` corner mode for two dictionary-keyed equal-length
series returns a `(2, 2)` axes grid. The upper-right axes is invisible, the
bottom-left cell labels the x/y axes as `H1` and `L1`, and the bottom-right
diagonal cell labels the x-axis as `L1`.

PR #341 (`a69a537`) separately merged the PairPlot mismatch policy: inputs
that differ in length, sample rate, or time span are rejected instead of being
silently aligned or truncated. This pass did not reopen that policy.

### GeoMap

When the optional PyGMT backend is unavailable, `GeoMap()` currently raises
`ImportError` with both a feature-specific message and an install hint:

- `pygmt is required for GeoMap`
- `pip install pygmt`

## Deferred Follow-ups

- GauCh colorbar exposure remains backend-specific; decide separately whether
  dashboard colorbars need stable public handles.
- GauCh Rayleigh branch: add a separate structural test for `rayleigh_spec`
  once the intended Rayleigh title and colorbar contract is accepted.
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
