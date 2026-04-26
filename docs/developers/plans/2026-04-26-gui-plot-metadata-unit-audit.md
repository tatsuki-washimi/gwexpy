# GUI and Plotting Metadata/Unit Audit

Date: 2026-04-26
Issue: #252 "Audit GUI and plotting metadata/unit behavior"
Scope: entry point inventory and test-gap planning only. No behavior changes.

## Constraints

- PR #256, #257, #258, and #259 have merged. This review pass updates only the audit inventory and does not change runtime behavior.
- The #243 audit plan is merged/closed. The #244/#246 matrix metadata contracts still matter for later behavior changes; this audit records current behavior and avoids defining new runtime behavior.
- Main worktree had uncommitted changes, so the original document was prepared in a separate worktree at `/tmp/gwexpy-issue252`; this review pass was prepared in `/tmp/gwexpy-pr261-address-review`.
- GUI execution, NDS access, pyautogui smoke, and physics choices for complex/phase display are marked as human-review items.

## Inputs Reviewed

- `.agent/AGENTS.md`
- `docs/developers/plans/` on `origin/main`
- Issue #252 body
- `gwexpy/plot/`, including `gwexpy/plot/gauch_dashboard.py`
- `gwexpy/gui/`
- `gwexpy/fields/*visualization*` behavior in `gwexpy/fields/scalar.py`, `gwexpy/fields/vector.py`, `gwexpy/fields/tensor.py`, and `gwexpy/fields/base.py`
- `gwexpy/types/series_matrix_visualization.py`
- `gwexpy/types/hht_spectrogram.py` and `TimeSeries.hht(output="spectrogram")`
- `TimeSeries`, `FrequencySeries`, and `Spectrogram` plotting helpers
- `tests/fields/test_scalarfield_visualization.py`
- `tests/plot/`
- `tests/timeseries/test_hht.py` and `tests/timeseries/test_hht_extended.py`
- `tests/gui/`

No existing Issue #252-specific plan was found in `docs/developers/plans/` on `origin/main`. Existing plans mention visual documentation and interactive plotting at a high level, but they do not define metadata/unit display contracts for plotting or GUI entry points.

## Inventory

| Entry point | Source | Axis unit labels | Colorbar/value unit | Name/channel/metadata display | Complex/phase label | Current automated coverage | Gaps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Generic container `.plot()` | `gwexpy/types/mixin/_plot_mixin.py` | Delegates to `gwexpy.plot.Plot` and GWpy | Delegates to `Plot` and GWpy | Delegates to `Plot`; no direct contract here | Delegates to lower layers | Smoke coverage in `tests/plot/test_gwexpy_plot_smoke.py` | Needs contract-level tests for name/channel/metadata once #244/#246 settle. |
| `TimeSeries.plot()` | `gwexpy/timeseries/timeseries.py` via `PlotMixin` | `Plot` infers labels from x index and unit helpers | No colorbar | Series `name` may be used by GWpy legends, but this repo does not define a channel display contract | No explicit phase label | Smoke/default-helper tests only | Add headless tests for existing xlabel/ylabel/legend/title behavior. |
| `FrequencySeries.plot()` | `gwexpy/frequencyseries/frequencyseries.py` via `PlotMixin` | `determine_xlabel` defaults frequency x-axis to `Frequency [unit]` | No colorbar | No explicit channel contract | No explicit phase label | `tests/plot/test_defaults.py` covers helper defaults | Add end-to-end plot label tests using real `FrequencySeries`. |
| `Spectrogram.plot()` | `gwexpy/spectrogram/spectrogram.py` via `PlotMixin` | `Plot` defaults y-axis from frequency axis and unit | `determine_clabel` uses spectrogram unit | No explicit channel contract | Scale heuristics inspect unit/name for phase-like data, but label text is not explicit | Helper tests cover `determine_ylabel` and `determine_clabel` | Add headless tests for actual `Plot(..., method="pcolormesh")` label and colorbar behavior. |
| `Spectrogram.imshow()` / `pcolormesh()` | `gwexpy/spectrogram/spectrogram.py` | Inherited from GWpy | Inherited from GWpy | Inherited from GWpy | Inherited from GWpy | Not directly asserted in this repo | Inventory only until upstream/GWpy behavior is pinned. |
| `HHTSpectrogram.plot()` from `TimeSeries.hht(output="spectrogram")` | `gwexpy/types/hht_spectrogram.py`, returned by `gwexpy/timeseries/_spectral_special.py` | Repo-owned fallback labels are `Time [{self.xunit}]` and `Frequency [{self.yunit}]`; defaults also set `xscale="auto-gps"` and `yscale="log"` | Repo-owned colorbar label infers `Power [unit]` for squared units, `Amplitude [unit]` otherwise, and `Amplitude/Power [unit]` for non-string unit fallback | `TimeSeries.hht()` preserves `name`, `channel`, and `epoch` on the returned object, but `plot()` does not display name/channel metadata | No explicit phase label; default log normalization assumes positive amplitude/power-like values | HHT tests cover spectrogram construction and options, but not plot labels/colorbars | Add headless tests for HHT x/y labels, x/y scales, colorbar label for `weight="ia"` and `weight="ia2"`, and preservation of caller-provided axis labels. |
| `SeriesMatrixVisualizationMixin.step()` | `gwexpy/types/series_matrix_visualization.py` | Delegates to `.plot(drawstyle=...)` | No colorbar unless underlying type has one | Matrix row/column metadata handled in `gwexpy.plot._init_helpers` | No explicit phase label | Not directly covered | Add tests for row/column label/title behavior after metadata contract lands. |
| `Plot` facade | `gwexpy/plot/plot.py`, `gwexpy/plot/_init_helpers.py`, `gwexpy/plot/defaults.py` | Defaults are computed by `determine_xlabel` and `determine_ylabel`, then applied without overwriting explicit labels | Spectrogram colorbar helpers use unit-derived labels | Matrix overlays display row metadata names as y-labels and column metadata names as titles. Channel is not displayed by repo-owned helpers. Unnamed matrix values may be assigned `"{row} / {col}"` during expansion. | `_is_linear_unit_or_name` uses unit/name heuristics for linear scale, not user-visible labels | `tests/plot/test_defaults.py`, `tests/plot/test_init_helpers.py`, smoke tests | Add tests that distinguish unit labels, row/column metadata labels, and channel absence/presence. |
| `Plot.plot_summary()` | `gwexpy/plot/plot.py` | Summary frequency axis uses `Frequency [Hz]`; spectrogram axis uses `Frequency [Hz]` | Spectrogram colorbar label uses `sg.unit.to_string("latex_inline")` | Matrix row keys or item names are used in titles | Coherence/ASD/PSD title type is inferred from unit/name, but phase-specific labels are not explicit | Not specifically label-tested | Add focused headless tests for titles, summary axes, and colorbar label. |
| `plot_gauch_dashboard()` | `gwexpy/plot/gauch_dashboard.py` | GauCh/Rayleigh panels use `Frequency [Hz]`; the time-series panel uses `Time [s]` and `Amplitude [{ts.unit}]` | Colorbars are explicit: `-log10(p-value)`, `Rayleigh Statistic`, or `KS Statistic (Dn)` | Titles are explicit dashboard labels; `ts.name`/channel metadata are not displayed | Not applicable; dashboard values are p-value/statistic/time-series amplitude views | Tutorial references only; no direct automated dashboard label tests found | Add headless tests for panel titles, axis labels, colorbar labels, and Rayleigh-vs-GauCh fallback paths. |
| `FieldBase.plot()` / `animate()` | `gwexpy/fields/base.py` | Delegates scalar frames to `FieldPlot.add_scalar` | Delegates scalar frames to `FieldPlot.add_scalar` | Animation title shows sliced coordinate only | No complex/phase mode control here | `tests/plot/test_field_plots.py` smoke tests animation setup | Add label assertions; GUI-like animation review should remain headless unless interactive behavior is under test. |
| `FieldPlot.add_scalar()` | `gwexpy/plot/field.py` | X/Y labels include coordinate names and units | Colorbar label includes scalar field name and unit when present | Field name appears only in colorbar label; channel is not applicable | No explicit complex/phase mode | Smoke tests in `tests/plot/test_field_plots.py` | Add direct assertions for xlabel, ylabel, and colorbar label. |
| `FieldPlot.add_vector()` | `gwexpy/plot/field.py` | X/Y labels include coordinate names and units | No scalar colorbar or vector component unit label | Channel is not applicable | No explicit phase label | Smoke tests only | Decide whether vector component units should be visible before changing behavior. |
| `VectorField.plot_magnitude()` | `gwexpy/fields/vector.py` | Delegates to scalar plot | Scalar colorbar via norm field | Norm field name/unit are used through scalar path | No explicit phase label | Smoke coverage | Add label tests for magnitude colorbar. |
| `VectorField.quiver()` / `streamline()` | `gwexpy/fields/vector.py` | Delegates to `FieldPlot.add_vector` | No value-unit label | No metadata display beyond coordinate axes | No explicit phase label | Smoke coverage | Document or test current absence of vector value-unit labels. |
| `VectorField.plot()` | `gwexpy/fields/vector.py` | Scalar background and vector overlay share coordinate axes | Norm background colorbar label is `Magnitude [unit]` | No channel metadata | No explicit phase label | Smoke coverage | Add one label test for norm + quiver combined plot. |
| `TensorField.plot_components()` | `gwexpy/fields/tensor.py` | Delegates each component to `FieldPlot.add_scalar` | Each component gets a scalar colorbar | Component labels use `T_{ij}`; no channel metadata | No explicit phase label | Smoke coverage | Add label tests for component titles/colorbars if this grid remains supported. |
| `ScalarField.extract_points()` | `gwexpy/fields/scalar.py` | Returned `TimeSeries` uses axis-0 index and preserved value unit | No colorbar | Generated point names include coordinates | No explicit phase label | `tests/fields/test_scalarfield_visualization.py` covers units and point names | Consider future cleanup for generated `y=`/`z=` labels versus axis names; behavior change should wait. |
| `ScalarField.extract_profile()` | `gwexpy/fields/scalar.py` | Returns coordinate index and values with units | No colorbar | No name/channel display | No explicit phase label | Existing tests cover units | Good candidate for more label tests through plotting helpers. |
| `ScalarField.slice_map2d()` | `gwexpy/fields/scalar.py` | Preserves coordinate axes and unit in sliced field | No colorbar by itself | Preserves field structure | No explicit phase label | Existing tests cover basic slicing | Covered enough for inventory; plot helpers need more tests. |
| `ScalarField.plot_map2d()` | `gwexpy/fields/scalar.py` | X/Y labels include selected coordinate names and units | Colorbar label is `"{mode} [{self.unit}]"` | Optional user title only; field name/channel are not automatic | Modes include `real`, `imag`, `abs`, `angle`, and `power`, but colorbar uses original field unit for all modes | Existing tests assert axis labels and colorbar presence | Add colorbar label tests. Human review needed before changing angle/power unit semantics. |
| `ScalarField.plot_timeseries_points()` | `gwexpy/fields/scalar.py` | X label includes time axis name/unit; y label is `[unit]` only | No colorbar | Line labels use generated point names | No explicit phase label | Basic plotting tests | Add y-label and legend tests; decide whether field name belongs in y-label later. |
| `ScalarField.plot_profile()` | `gwexpy/fields/scalar.py` | X label includes profile axis name/unit; y label uses mode and transformed value unit | No colorbar | No field name/channel display | Mode label is visible in y-label; transformed units depend on `select_value` | Basic plotting tests | Add tests for complex modes, especially `angle` and `power`. |
| `ScalarField.plot_time_space_map()` | `gwexpy/fields/scalar.py` | X label is selected space axis; y label is time axis | Colorbar label is `"{mode} [{self.unit}]"` | No automatic name/channel display | Same mode/unit gap as `plot_map2d` | Basic plotting tests | Add colorbar label tests; human review before changing mode unit semantics. |
| `BifrequencyMap.plot()` | `gwexpy/frequencyseries/bifrequencymap.py` | X/Y labels include frequency units | Colorbar label uses name and unit | Name is used in colorbar; channel is not applicable | No explicit phase label | Unit math/slicing tests, but not plot labels | Add headless plot label tests. Current code may fail when `name is None` and `unit` is truthy. |
| `BifrequencyMap.plot_lines()` | `gwexpy/frequencyseries/bifrequencymap.py` | X label includes selected frequency/difference unit | Colorbar labels iteration axis unit; y label uses name/unit | Name appears in y label | No explicit phase label | Not plot-label tested | Add tests for name/unit fallback behavior. |
| GUI compute engine | `gwexpy/gui/engine.py` | Returns raw arrays/dicts without unit fields | Value unit is stripped before rendering | Channel names drive input mapping, but result payloads do not carry channel/name metadata | Phase/dB semantics are not represented in payload metadata | GUI renderer tests use mocked plot items | Future behavior PR should define structured result metadata after #244/#246. |
| GUI streaming accumulator | `gwexpy/gui/streaming.py` | Returns raw arrays/dicts without unit fields | Value unit is stripped before rendering | Channel/name metadata not carried in result payloads | Phase/dB semantics are not represented in payload metadata | GUI tests cover render paths, not metadata | Same metadata-payload gap as non-streaming engine. |
| GUI plot renderer | `gwexpy/gui/ui/plot_renderer.py` | Updates bottom time label only; no y-axis/value labels from data metadata | Applies display conversion but does not update value-unit/colorbar labels | Renderer receives no name/channel metadata from engine results | `Phase` display returns degrees for complex data and zeros for real data, but no label says degrees | `tests/gui/test_plot_renderer.py` covers dB conversion, relative time, log-y rect, and bottom time label | Add tests for current absence of y/value labels, then later metadata-driven label tests. |
| GUI graph panel and tabs | `gwexpy/gui/ui/graph_panel.py`, `gwexpy/gui/ui/_tab_builders.py`, `gwexpy/gui/ui/tabs.py` | Default labels are generic `Time` and `Signal`; user can manually override axis labels | No `ColorBarItem` is created for spectrogram images | Unit combos exist in UI, but x/y combos are not wired to renderer metadata | Display combo has `Magnitude`, `Phase`, `dB`; labels do not reflect conversion | Requires GUI/manual smoke for full interaction | Manual smoke should verify label override, display mode, spectrogram visibility, and lack/presence of colorbar. |
| GUI normalization helper | `gwexpy/gui/plotting/normalize.py` | Converts series-like inputs to raw x/y arrays | Strips value unit | Strips name/channel | No complex/phase metadata | Not central in current tests | If retained, document that it is array normalization only, not metadata normalization. |
| Ancillary plot helpers | `gwexpy/plot/skymap.py`, `gwexpy/plot/pairplot.py`, `gwexpy/plot/geomap.py` | Domain-specific labels or caller-provided labels | Domain-specific colorbars | Not tied to GW container name/channel contracts | Not applicable | Limited smoke/coverage | Treat as out of scope unless Issue #252 is expanded to all plot modules. |

## Test Coverage Split

### Headless Matplotlib Coverage

These areas are suitable for automated headless tests:

- `gwexpy.plot.defaults` helper behavior for x/y/colorbar labels.
- `gwexpy.plot.Plot` defaults for `TimeSeries`, `FrequencySeries`, `Spectrogram`, `SpectrogramMatrix`, and series matrices.
- `FieldPlot.add_scalar`, `VectorField.plot`, and `TensorField.plot_components` axis/colorbar labels.
- `ScalarField.plot_map2d`, `plot_profile`, `plot_timeseries_points`, and `plot_time_space_map` labels.
- `BifrequencyMap.plot` and `plot_lines` labels and no-name fallback behavior.
- `HHTSpectrogram.plot` labels, log/auto-gps scales, colorbar labels, and explicit-label preservation.
- `plot_gauch_dashboard` panel titles, axis labels, colorbar labels, and Rayleigh-vs-GauCh branch behavior.

Existing headless tests already cover smoke behavior and some helper-level units, but they do not consistently assert visible labels, metadata display, or absence of channel labels.

### GUI Automated Coverage

These areas can be tested with mocked or offscreen GUI components:

- `PlotRenderer._apply_unit_conversion` conversion output for `None`, `Magnitude`, `Phase`, and `dB`.
- Bottom-axis label update for absolute and relative time.
- Current result payload shape from `Engine.compute` and `SpectralAccumulator.get_results`, including the current absence of unit/name/channel metadata.
- Spectrogram image rendering shape and log-y rectangle handling.

Existing GUI tests focus on rendering mechanics and range stability. They do not assert metadata/unit display contracts because the current result payloads do not carry enough metadata to label those displays.

### Manual GUI Smoke

These require a human or prepared GUI environment:

- `tests/run_gui_tests.sh` with an available display or Xvfb.
- `tests/run_gui_nds_tests.sh`, which requires NDS configuration/network and should not run as part of this audit.
- `tests/gui/integration/test_pyautogui_smoke.py`, which requires a real display and pyautogui-capable desktop session.
- Visual confirmation of spectrogram colorbar expectations, display-unit controls, manual axis-label overrides, and multi-graph layout.

## Human Review Items

- Complex/phase display semantics: `ScalarField` map helpers expose `angle` and `power`, but some colorbar labels use the original field unit rather than the transformed unit. Changing this needs physics and metadata-contract review.
- GUI `Phase` display currently converts complex values to degrees and real values to zeros. Labeling or changing this behavior needs human review.
- GUI dB factor selection depends on graph type and data shape. Any label contract should be reviewed with domain expectations before behavior changes.
- GUI/NDS/pyautogui smoke requires a prepared local environment and should not be inferred from headless tests.

## Safe Validation Performed

Command:

```bash
conda run -n gwexpy pytest --collect-only -q tests/fields/test_scalarfield_visualization.py tests/plot/test_field_plots.py tests/plot/test_gwexpy_plot_smoke.py tests/gui/test_plot_renderer.py tests/gui/test_visual_stability.py
```

Result: 41 tests collected.

Command:

```bash
conda run -n gwexpy pytest tests/plot/test_field_plots.py tests/fields/test_scalarfield_visualization.py tests/gui/test_plot_renderer.py -q
```

Result: 40 passed.

## Prioritized Small PR Candidates

1. Document-only audit PR: land this inventory and test-gap document for Issue #252. This remains safe after the #243 audit-plan merge because it does not change behavior; #244/#246 remain behavior-contract dependencies.
2. Headless label tests for existing Matplotlib contracts: add assertions for `FieldPlot.add_scalar`, `ScalarField.plot_map2d`, `ScalarField.plot_profile`, `ScalarField.plot_time_space_map`, `BifrequencyMap.plot`, `HHTSpectrogram.plot`, and `plot_gauch_dashboard`. Keep expected values aligned to current behavior.
3. Plot facade metadata tests: add focused tests for `TimeSeries`, `FrequencySeries`, `Spectrogram`, and matrix row/column labels through `gwexpy.plot.Plot`. Delay any new metadata expectations until the #244/#246 contracts are accepted.
4. GUI payload contract tests: assert current engine/streaming result payloads and renderer label behavior, especially the current absence of unit/name/channel metadata. This gives a safe baseline before later behavior work.
5. GUI manual smoke checklist PR: document which GUI checks are headless-safe, which require Xvfb, and which require a real display or NDS access.
6. Post-contract behavior PR: after #244/#246, add structured GUI result metadata such as x unit, y unit, value unit, display unit, name, and channel; then update renderer labels and tests.
7. Physics-reviewed complex/phase PR: after human review, decide whether angle/power map colorbar labels should use transformed units and whether GUI phase should label degrees explicitly.
