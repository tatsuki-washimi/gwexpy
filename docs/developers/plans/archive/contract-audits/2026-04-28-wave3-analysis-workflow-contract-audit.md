# Wave 3 Analysis Workflow Contract Audit

Date: 2026-04-28
Issue: #284, "Audit analysis workflow contracts for Bruco coupling and response"
Mode: audit-first; docs and regression tests only.

## Scope For This Slice

This first #284 slice records current result-object and lightweight helper
contracts without changing runtime behavior:

- `BrucoResult` metadata copying, env-driven auto block-size selection,
  deterministic Top-N ranking, channel labels, long-form export ranks,
  amplitude-coherence display, and projection values.
- `CouplingResult` frequency-axis identity, units, valid-mask storage,
  `fftlength` / `overlap` metadata, and `CouplingResultCollection` summary CSV
  behavior.
- `ResponseFunctionResult` plotting order for unsorted injected frequencies,
  non-mutating plot behavior, spectrogram frequency/unit metadata, and channel
  labels.

The tests are concentrated in
`tests/analysis/test_analysis_workflow_contracts.py` so later #284 work can
update a focused contract baseline before any physics or statistics policy
changes.

## Contracts Recorded

### Bruco Result Object

- `BrucoResult` copies the caller-provided metadata mapping during
  construction. Later mutations to the original mapping do not change
  `result.metadata`.
- `block_size="auto"` delegates to the same environment-budget rule used by
  `_resolve_block_size("auto", ...)`; the current rule uses
  `GWEXPY_BRUCO_BLOCK_BYTES`, subtracts `top_n` columns from the budget, and
  clamps to the configured min/max bounds.
- `update_batch()` stores squared coherence in descending Top-N order per
  frequency bin and keeps channel names in the same rank order.
- `to_dataframe(asd=True)` exports one-based ranks, numeric frequency values,
  channel labels, amplitude coherence (`sqrt(coherence)`), and ASD projection
  values computed from `sqrt(target_psd * coherence)`.

### Coupling Result Object

- `CouplingResult.frequencies` returns the same x-axis object as `result.cf`.
- CF units and PSD units are retained on the stored `FrequencySeries` objects.
- `valid_mask`, `witness_name`, `target_name`, `fftlength`, and `overlap` are
  stored as provided by the caller.
- `CouplingResultCollection.to_summary_csv()` writes one row per frequency bin
  for `CouplingResult` values and ignores non-`CouplingResult` entries in the
  mapping.
- Summary CSV `inj_asd` and `bkg_asd` columns are derived from the witness
  injection/background PSDs via square root of the absolute PSD values.

### Response Function Result Object

- `ResponseFunctionResult.plot()` sorts the displayed transfer-function points
  by `injected_freqs`.
- Plotting does not mutate `injected_freqs` or `coupling_factors`.
- Stored spectrogram units, frequency axes, and witness/target labels remain
  available on the result object after plotting.

## Deferred Behavior Changes

These are intentionally not included in this docs/test-only slice:

- Bruco coherence normalization, Welch scaling, ranking tie policy, mixed-rate
  resampling/interpolation policy, process-parallel execution behavior, and
  large-memory block-size defaults beyond the narrow env contract above.
- Strong validation or structured recording of Bruco analysis parameters such
  as `fftlength`, `overlap`, PSD method, window, stride, resampling policy,
  frequency alignment, and target/auxiliary sample rates.
- Response/coupling excess-power definitions, bin selection around injection
  frequency, upper-limit policy, interpolation/alignment behavior, NaN handling,
  and normalization conventions.
- Threshold strategy defaults, percentile correction factor policy, low-`n_avg`
  warning policy, skipped background-row policy beyond existing tests, and
  statistical interpretation of upper limits.
- Optional dependency behavior for SciPy/joblib beyond existing import and
  parallel-path tests.
- Public API stability classification for analysis workflows versus internal
  helpers.

Any follow-up that changes normalization, excess-power definitions, frequency
alignment, thresholds, or statistical handling should be split into a dedicated
PR and flagged for human physics/statistics review.

## Verification

Focused contract check:

```bash
rtk pytest -q tests/analysis/test_analysis_workflow_contracts.py
```

Related analysis regression checks:

```bash
rtk pytest -q \
  tests/analysis/test_analysis_workflow_contracts.py \
  tests/analysis/test_bruco_result.py \
  tests/analysis/test_bruco_logic.py \
  tests/analysis/test_coupling.py \
  tests/analysis/test_stats_and_export.py \
  tests/analysis/test_response.py \
  tests/analysis/test_response_compat.py
```

Changed-file hygiene:

```bash
rtk ruff check tests/analysis/test_analysis_workflow_contracts.py
rtk ruff format --check tests/analysis/test_analysis_workflow_contracts.py
rtk python -c "import yaml; yaml.safe_load(open('docs/developers/plans/audit-manifest-284-analysis-contracts.yaml', encoding='utf-8'))"
rtk git diff --check
```
