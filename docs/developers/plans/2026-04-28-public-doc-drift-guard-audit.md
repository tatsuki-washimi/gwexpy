# Public Doc Drift Guard Audit

Date: 2026-04-28
Issue: #275
Branch: `codex/wave4-275-doc-drift-guards`

## Scope

This slice is limited to public tutorial notebook source and the focused
notebook-quality regression test. It does not change runtime `gwexpy` code,
notebook outputs, physics algorithms, statistical thresholds, data models, or
I/O behavior.

Files in scope:

- `tests/docs/test_tutorial_notebook_quality.py`
- `docs/web/en/user_guide/tutorials/advanced_correlation.ipynb`
- `docs/web/en/user_guide/tutorials/case_gbd_format.ipynb`
- `docs/web/en/user_guide/tutorials/rayleigh_gauch_tutorial.ipynb`
- `docs/web/en/user_guide/tutorials/time_frequency_analysis_comparison.ipynb`

## Changes

The four notebooks now pass explicit Matplotlib mappables to colorbars instead
of rediscovering a plotted object from `plt.gca()` state:

- `advanced_correlation.ipynb` assigns the partial-correlation heatmap to `im`
  and passes `mappable=im`.
- `time_frequency_analysis_comparison.ipynb` keeps the existing `im`
  assignments and passes them directly to `plt.colorbar`.
- `case_gbd_format.ipynb` replaces the `sg.plot()` wrapper/colorbar lookup with
  explicit `fig, ax`, `ax.pcolormesh(...)`, labels, title, colorbar, and
  `plt.tight_layout()`.
- `rayleigh_gauch_tutorial.ipynb` replaces three `get_children()` colorbar
  lookups with explicit `ax.pcolormesh(...)` plots for the Rayleigh p-value map,
  GauCh p-value map, and Student-t nu map.

`tests/docs/test_tutorial_notebook_quality.py` now asserts that these public
tutorial notebooks do not reintroduce the stale `plt.gca().get_images()`,
`plt.gca().collections[-1]`, `plt.gca().get_children()`, or `get_clim` lookup
patterns. The explicit-mappable guard is semantic rather than snippet-based: it
parses notebook code cells, tolerates harmless whitespace and line-wrapping
changes, and verifies that colorbar calls use `mappable=` with variables assigned
from `imshow` or `pcolormesh`. Review follow-up tightened the helper to validate
in notebook execution order instead of collecting mappable names globally: each
`colorbar(...)` call is checked against the currently valid mappable assignments,
and reassignment, augmented assignment, or loop targets invalidate stale names.
Synthetic helper regression checks cover colorbar-before-assignment,
colorbar-after-reassignment, and the valid `pcolormesh`-then-colorbar case.

## Verification

The focused regression test was added first and failed on the stale notebook
patterns. After updating the notebooks, the focused test suite passed:

```text
rtk env MPLCONFIGDIR=/tmp/matplotlib XDG_CACHE_HOME=/tmp pytest -q tests/docs/test_tutorial_notebook_quality.py -p no:cacheprovider
43 passed
```

Additional checks recorded in the manifest:

- `rtk ruff check tests/docs/test_tutorial_notebook_quality.py`
- `rtk ruff format --check tests/docs/test_tutorial_notebook_quality.py`
- YAML parse check for `audit-manifest-275-public-doc-drift.yaml`
- `rtk git diff --check`

Review-fix verification replaced brittle exact notebook source assertions with
the semantic/normalized guard above and re-ran the same focused pytest, Ruff,
manifest parse, and diff whitespace checks before amending the commit.

Second review-fix verification changed the semantic guard from global two-pass
collection to execution-order validation. The regression checks failed first on
the old implementation with 2 failures and 44 passes, then passed after the
ordered visitor change with 46 passes before the final verification run.

## Deferred Follow-Ups

- The collapsed AM-FM/HHT section in
  `time_frequency_analysis_comparison.ipynb` remains deferred because this slice
  only addresses stale colorbar mappables.
- Remaining stale mappable child lookups in
  `advanced_spectrogram_processing.ipynb` remain deferred to a later issue slice.
- Public EN/JA tutorial structure and missing JA-local notebook references remain
  outside this docs/test-only slice.
