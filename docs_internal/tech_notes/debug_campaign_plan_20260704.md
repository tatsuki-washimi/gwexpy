# GWexpy Debug Campaign Plan (v0.1.9-v0.2.0)

**Date**: 2026-07-04  
**Status**: DRAFT / P4-0 blocked in current execution environment  
**Author**: ChatGPT (GitHub connector)  
**Scope**: GUI and CLI are explicitly out of scope. No bug fix implementation, issue operation, or CHANGELOG edit is included in this document-only campaign step.

## 0. Current Execution Verification

This document records the approved debug campaign plan and the attempted P4-0 coverage recalculation.

### 0.1 P4-0 Attempt Result

P4-0 could not be completed in this session because the local execution environment could not obtain a checkout of the repository.

Attempted command:

```bash
cd /mnt/data && git clone --depth 1 https://github.com/tatsuki-washimi/gwexpy.git
```

Observed result:

```text
Cloning into 'gwexpy'...
fatal: unable to access 'https://github.com/tatsuki-washimi/gwexpy.git/': Could not resolve host: github.com
```

Additional local import check:

```text
ModuleNotFoundError: No module named 'gwexpy'
```

Therefore, no `pytest --cov` command was run, and no new module-level coverage numbers should be treated as measured by this execution. The Phase 4 priorities below keep the pre-existing analysis as provisional and mark coverage-dependent ordering as pending P4-0 rerun in an environment with repository checkout and dependencies available.

### 0.2 P4-0 Rerun Commands

Run these commands from a clean checkout with development dependencies installed:

```bash
pytest tests/frequencyseries --cov=gwexpy.frequencyseries --cov-report=term-missing
pytest tests/fields --cov=gwexpy.fields --cov-report=term-missing
pytest tests/analysis --cov=gwexpy.analysis --cov-report=term-missing
pytest tests/timeseries --cov=gwexpy.timeseries --cov-report=term-missing
pytest tests/types --cov=gwexpy.types --cov-report=term-missing
pytest tests/histogram --cov=gwexpy.histogram --cov-report=term-missing
pytest tests/table --cov=gwexpy.table --cov-report=term-missing
```

If `tests/timeseries` is too slow, split by file group while keeping a combined `--cov=gwexpy.timeseries` target.

### 0.3 Verification for This Save Step

- New documentation file only: `docs_internal/tech_notes/debug_campaign_plan_20260704.md`.
- No bug fixes, issue closing/opening, or CHANGELOG changes were performed.
- GitHub branch used for the save step: `docs/debug-campaign-plan-20260704`.

## 1. Context

The preceding history analysis of issues, pull requests, and releases found that GWexpy bug fixing has progressed in waves:

1. I/O and interoperability fixes.
2. Attribute deep-copy and metadata-preservation fixes.
3. Numerical robustness fixes in statistics, spectral analysis, and fitting.
4. Systematic audit-driven hardening.

This campaign uses that trajectory to identify likely remaining bugs and under-validated areas. GUI and CLI are excluded by user decision.

Three parallel investigations were already performed:

- Test-density measurement.
- Residual scan for known bug patterns.
- Audit-document and open-issue review.

A Plan agent also performed code back-checks. The user decision for this step is: save the plan only, do not start implementation or issue operations. P4-0 coverage recalculation was approved as the only pre-save measurement, but it was blocked in this execution environment as recorded in Section 0.

## 2. Facts Established by Investigation

### 2.1 Known Unfixed Bugs / Open Issues

| ID | Content | State |
| :--- | :--- | :--- |
| #481 | GWF read-path gap zero-pad conflicts with v0.1.8 default NaN policy. Existing location checked at `gwexpy/timeseries/_gwf_io.py:219`. | Fix direction decided. |
| #466 (P2 x2) | `fit_series` sigma crop bin-boundary mismatch `ValueError`, plus `run_mcmc` crash when `n_walkers < 2 * ndim`. | Fix direction decided. |
| #465 (P3 x2) | `student_t_indicator` STFT: crash when `stride > fftlength`, loss of GPS epoch in `out_times`, DC/Nyquist bias, and divergence from gwpy defaults. | Not started. |
| #464 (P3 x2) | Rayleigh/GauCh Monte Carlo uses unseeded RNG, thread-unsafe cache, and missing metadata. | Not started. |
| #451 | `TimeSeries.rms()` breaks gwpy compatibility. | Fixed for v0.2.0. |
| #444 | FrequencySeries collection registry fallback is inconsistent. | Decision pending, v0.2.x. |
| SP3/SP9 | `spectral/estimation.py:136` object-dtype `TypeError`; `:481-490` `mean_avg > 0` shortcut invalid for dB data. | Issue not yet filed. |

### 2.2 Tracker Hygiene

- #460 has already been fixed by #473 and released in v0.1.7, but remains open.
- #461 checklist is stale. All items remain unchecked even though themes A-G are effectively consumed. The remaining themes are H/I/J, corresponding to #464, #465, and #466.

### 2.3 Newly Suspected Bugs from Pattern Scan

These are code-existing suspicions, but not yet dynamically verified.

| ID | Location | Suspicion |
| :--- | :--- | :--- |
| C1 | `gwexpy/histogram/io/_hdf5.py:52, 110-120` | `Channel` degrades to `str` on round trip, losing `sample_rate` / `unit`. Same class as prior bug pattern d. |
| C2 | `gwexpy/fields/scalar.py:1593-1595` | `zscore` silently converts `std == 0` degeneracy to `0.0`. Additional coupled case: one NaN in baseline silently turns all output into `0.0`, because `mean/std` are not `nanmean`-style. |
| C3 | `gwexpy/analysis/coupling.py:324-329` | Unit conversion failure is only `logger.debug` and silently falls back to dimensionless. |
| C4 | `gwexpy/histogram/_core.py:134` | `copy(deep=True)` passes `channel` by reference. Same pattern exists at `_core.py:222` (`crop`) and `_rebin.py:176`; treat as a three-location set. |
| C5 | `gwexpy/analysis/bruco.py:374` | External coherence NaNs become zero without warning. Semantically defensible, but warning/docstring behavior needs decision. |

### 2.4 Dangerous Defaults / `eps` Migration State

Measured migration state from the prior audit:

- Migrated: `signal/preprocessing/whitening.py`, `timeseries/preprocess.py`, `timeseries/decomposition.py`.
- Not applicable: `noise/magnetic.py`.
- Remaining risky cases:
  - `timeseries/pipeline.py:395`: `WhitenTransform eps=1e-12`.
  - `timeseries/matrix_analysis.py:794,820`: `partial_correlation_matrix eps=1e-8` directly adds `cov + eps * I`. At strain scale, covariance diagonals can be around `1e-42`, so `eps` fully dominates and can collapse `pcorr` toward zero. This is the most dangerous remaining case.
  - `types/time_plane_transform.py:371`: `normalize_per_sigma eps=1e-30`.

### 2.5 Under-Validated Areas from Existing Test-Density Analysis

Because P4-0 was blocked in this session, the following remains provisional until rerun:

- `frequencyseries`: lowest test/source ratio observed previously, around 0.43. `collections.py` (1,108 LOC) and `bifrequencymap.py` (876 LOC) have almost no direct tests. Two tests are gwpy upstream `import *` stubs; only one contract test is present.
- `timeseries/collections.py`: batch operation propagation (`resample`, `filter`, `whiten`) across all elements is not verified.
- Large or weakly covered files/areas: `fields/scalar.py` (2,303 LOC), `analysis/bruco.py` (1,746 LOC), `analysis/threshold.py` (no dedicated test), `table/segment_plot.py` (551 LOC, no dedicated test).
- Numerical robustness sweep has not yet covered: `fields`, `analysis`, `noise`, `histogram`, `table`, `spectrogram`, and core `timeseries`.
- Regression-style test naming is sparse: only four regression-named tests were found in analysis/docs/fitting.

## 3. Phase 0: Tracker Hygiene + Dangerous Defaults Recording

**Timing**: Immediate / half day.  
**Code changes**: None.

### Tasks

1. Close #460 with a comment referencing fix PR #473 and release v0.1.7.
2. Update #461 checklist to reflect reality:
   - A-G consumed.
   - H/I/J remain and map to #464, #465, #466.
3. File a Dangerous Defaults issue for the three remaining cases. One combined issue linked to the #461 tracker is acceptable.

### Completion Criteria

- #460 is closed.
- #461 reflects actual remaining work.
- Issue number assigned for remaining Dangerous Defaults cases.

## 4. Phase 1: Confirmed Bug Fix Wave 1

**Target**: v0.1.9 first half.  
**Parallelism**: Two PRs can run in parallel.

### W1-a: #481 GWF Gap Padding

**Target**: `gwexpy/timeseries/_gwf_io.py`.

Change default gap padding behavior from `0.0` to `np.nan`, aligning GWF `read(gap="pad")` with the v0.1.8 NaN default policy.

RED tests:

1. GWF fixture with a gap: `read(gap="pad")` should produce NaN in the gap region.
2. Explicit `pad=0.0` should keep backward-compatible zero padding.

Gate:

- Regression tests pass.
- CHANGELOG documents the behavior change.

### W1-b: #466 Fitting / MCMC Robustness

**Target**: `gwexpy/fitting/core.py`.

Fix two P2 failures:

1. `fit_series` sigma crop fails at matching bin boundary due to inconsistent cropping.
2. `run_mcmc` crashes when `n_walkers < 2 * ndim`.

RED tests:

1. `x_range` at a bin boundary with sigma must not raise `ValueError`.
2. Multi-parameter model with insufficient walkers must either auto-promote `n_walkers` or raise a clear explicit error.

Gate:

- Both PRs merged.
- Full test suite green.
- v0.1.9 can use W1 as main content.

## 5. Phase 2: P3 Confirmed Bugs + C-Triage

**Target**: v0.1.9 second half through v0.1.10.

### W2-a: #464 Rayleigh/GauCh Monte Carlo Reproducibility

Tasks:

- Add `seed` / `rng` argument using `np.random.default_rng`.
- Make cache thread-safe.
- Record `seed` and `n_monte_carlo` in metadata.

Verification:

- Same seed, same input, two executions produce exactly identical results.

### W2-b: #465 `student_t_indicator` STFT

Highest estimated effort in Phase 2.

Tasks:

- Raise explicit `ValueError` for `stride > fftlength`.
- Restore GPS epoch with `out_times += ts.t0`.
- Exclude DC/Nyquist bins as appropriate.
- Document intentional difference from gwpy defaults.

Verification:

- Fixture with nonzero `t0` proves output time axis preserves epoch.
- `stride > fftlength` test exercises the explicit error path.

### W2-c: SP3/SP9 Spectral Estimation Issues

Tasks:

- File issues for SP3 and SP9.
- Add object-dtype input test.
- Add negative dB spectrum test showing nonstationarity detection is not short-circuited by `mean_avg > 0` logic.

### W2-d: C-Triage

For C1-C5:

1. Spend 30 minutes to 1 hour per suspicion on a minimal reproduction script in scratchpad form.
2. Decide true/false.
3. If true, file an issue with reproduction code.
4. Save outcome, including false positives, as an audit report under `docs_internal/tech_notes`.

Special handling:

- C2 should be staged:
  - v0.1.x: warning addition.
  - v0.2.0: NaN behavior change.
- C4 should be fixed as a three-location set.
- C5 may become `wontfix` if the current behavior is deliberately documented.

## 6. Phase 3: Remaining Dangerous Defaults + v0.2.0 Preparation

### W3-a: Finish Dangerous Defaults Migration

Targets:

- `timeseries/pipeline.py`.
- `timeseries/matrix_analysis.py`.
- `types/time_plane_transform.py`.

Migration design:

- Follow the established `whitening.py` pattern: `eps='auto'` plus `get_safe_epsilon` / `SAFE_FLOOR_STRAIN`.
- For `matrix_analysis.py`, use covariance-scale relative epsilon, such as `trace(cov) / n * rel_eps`, rather than an absolute additive `eps`.

RED/GREEN strategy:

1. RED: strain-scale data around `1e-21` demonstrates current `pcorr` collapse toward zero.
2. GREEN: scale-invariance test modeled on `tests/numerics/test_scale_invariance.py` passes after the relative epsilon change.

### W3-b: v0.2.0 Breaking-Behavior Work

Tasks:

- #451: `rms` compatibility shim with `DeprecationWarning`, then v0.2.0 behavior switch.
- #444: registry fallback decision and implementation.
- C2: switch NaN/degenerate z-score behavior after warning period.
- Align with #413 migration guide and #401 contract audit wave.

## 7. Phase 4: Under-Validated Area Hardening

Coverage-dependent priority remains pending because P4-0 could not be completed in this execution.

### P4-0: Coverage Recalculation

Run the commands in Section 0.2 and update this section with an observed coverage table:

| Module | Command Status | Coverage | Missing Hotspots | Priority Adjustment |
| :--- | :--- | :--- | :--- | :--- |
| `gwexpy.frequencyseries` | Pending rerun | TBD | TBD | TBD |
| `gwexpy.fields` | Pending rerun | TBD | TBD | TBD |
| `gwexpy.analysis` | Pending rerun | TBD | TBD | TBD |
| `gwexpy.timeseries` | Pending rerun | TBD | TBD | TBD |
| `gwexpy.types` | Pending rerun | TBD | TBD | TBD |
| `gwexpy.histogram` | Pending rerun | TBD | TBD | TBD |
| `gwexpy.table` | Pending rerun | TBD | TBD | TBD |

### P4-1: `frequencyseries` Test Expansion

Add direct tests for:

- `frequencyseries/collections.py`.
- `frequencyseries/bifrequencymap.py`.
- Registry/contract behavior beyond the current minimal contract test.

This is a normal test-addition task, not a special audit campaign.

### P4-2: `timeseries/collections` Batch Propagation

Add low-cost parameterized tests verifying all elements receive batch operations:

- `resample`.
- `filter`.
- `whiten`.

### P4-3: Multi-Agent Numerical Robustness Sweep

Targets:

- `fields`.
- `analysis`.
- `noise`.
- `histogram`.
- `table`.
- `spectrogram`.
- core `timeseries`.

Cost is high and requires separate user approval. First wave should be:

1. `fields/scalar.py`.
2. `analysis/bruco.py`.

### P4-4: Minor I/O Contract Tests

Targets:

- `win`.
- `tdms`.
- `ats`.

Add minimal synthetic-fixture round-trip contract tests where possible.

## 8. Release Assignment

Assuming weekly release pace:

| Release | Content |
| :--- | :--- |
| v0.1.9 | Phase 0 + W1-a/W1-b, plus W2-a if it fits. |
| v0.1.10 | W2-b/W2-c/C-triage confirmed fixes + W3-a. |
| v0.1.11 | Overflow + P4-1/P4-2 if approved. |
| v0.2.0 | #451, #444, C2 NaN behavior change, and other behavior changes gated by #413/#401. |

## 9. Risks

- W1-a changes behavior only for `gap="pad"` when `pad` is omitted. Explicit `pad` compatibility tests are mandatory.
- W3-a changes numerical output in `matrix_analysis.py`. Current behavior is likely meaningless at strain scale, but the change must still be documented in CHANGELOG.
- C2 should use a two-stage transition, warning first and NaN behavior later, to avoid an unannounced breaking change.
- Phase 4 ordering remains provisional until P4-0 is rerun in a working checkout.

## 10. Non-Goals for This Step

This save-only step does not:

- Modify implementation code.
- Close, open, or edit GitHub issues.
- Modify CHANGELOG or release metadata.
- Add tests.
- Claim new coverage measurements.
