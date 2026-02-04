# Stat-Info Audit and Implementation Report

**Date**: 2026-02-04
**Author**: Antigravity

## 1. Documentation Cleanup

The `docs/developers/references/data-analysis/stat-info/` directory contained numerous redundant files, primarily meeting slides (KAGRA DetChar, PEM, ML meetings) by Pil-Jong Jung from 2021.

### Deleted Files (Cleaned up)

The following files were identified as superseded intermediate reports and were removed to improve clarity:

- `MLmeeting_20210319.pdf`, `MLmeeting_20210402.pdf`
- `PEM-meeting_20210409.pdf`, `PEM-meeting_20210514.pdf`, `PEM-meeting_20210521.pdf`, `VK_PEM_meeting-20210528.pdf`
- `KAGRA-DetChar_20210121.pdf`, `KAGRA-DetChar_20210426.pdf`, `KAGRA-DetChar_20210607.pdf`
- `KIW8_DET.pdf`, `KAGRAF2F...Subject...pdf`
- `BEACON_BurstCall.pdf`, `NoiseSub_KAGRAF2F.pdf` (Superseded by full versions)
- `ShowDocument.html` (Junk)

### Retained Files

Key scientific papers and definitive reports were kept:

- `washimi_CAGMONpaper.pdf` (Final CAGMon paper)
- `Glitch Catalog...pdf` (Comprehensive catalog)
- `1-s2.0-S0167715298000066-main.pdf` (FastMI paper)
- `AOAS-2018.pdf`, `Abdi-KendallCorrelation2007...pdf` (Statistical methods)
- `ARIMA_DeNoise.pdf` (seqARIMA Noise Subtraction)
- `BEACON.pdf` (Autoregressive Burst Search Pipeline)

## 2. Implementation Audit

### Implemented Methods

`gwexpy` provides a robust statistical toolkit via `StatisticsMixin` (`gwexpy/timeseries/_statistics.py`) and `TimeSeriesMatrixAnalysisMixin` (`gwexpy/timeseries/matrix_analysis.py`).

| Method | Implementation | Status | Note |
| :--- | :--- | :--- | :--- |
| **Pearson Correlation** | `scipy.stats.pearsonr` | ✅ OK | Standard implementation. |
| **Kendall Rank Corr.** | `scipy.stats.kendalltau` | ✅ OK | Matches Abdi 2007 reference. |
| **MIC** | `minepy.MINE` | ✅ OK | Uses standard `minepy`. Efficient enough for pair-wise but slow for many channels. |
| **Distance Correlation** | `dcor.distance_correlation` | ✅ OK | Can detect non-linear dependencies. |
| **Granger Causality** | `statsmodels.tsa.stattools` | ✅ OK | Supports F-test and Chi2-test. |

### Missing / Non-Optimized Features

1. **Partial Correlation**:
    - **Status**: ❌ **Missing**.
    - **Reference**: Often discussed in CAGMon/Graphical Lasso contexts to distinguish direct from indirect coupling.
    - **Recommendation**: Implement `partial_correlation(x, y, z)` using inverse covariance (precision matrix) or linear regression residuals.

2. **FastMI (FFT-based)**:
    - **Status**: ❌ **Missing** (Standard `minepy` is used).
    - **Reference**: `1-s2.0-S0167715298000066-main.pdf`.
    - **Recommendation**: Valid `minepy` is likely sufficient for now, but `FastMI` could offer speedups for large-scale all-to-all comparisons.

3. **CAGMon (Graph Construction)**:
    - **Status**: ⚠️ **Partial**. `correlation_vector` calculates the edges (correlation values), but the **graph clustering and visualization** logic (building the "Association Graph") is absent.
    - **Recommendation**: Create `gwexpy.analysis.cagmon` to orchestrate `correlation_vector` results into a NetworkX graph.

4. **seqARIMA (Sequential ARIMA)**:
    - **Status**: ❌ **Missing**.
    - **Reference**: `ARIMA_DeNoise.pdf`.
    - **Note**: `gwexpy` supports batch ARIMA via `gwexpy.timeseries.arima` (wrapping `statsmodels`), but does *not* support sequential/online updating of AR coefficients (likely via Kalman Filter or RLS) as described in the BEACON/seqARIMA papers level.
    - **Recommendation**: Investigate `statsmodels.tsa.statespace` or custom RLS integration for online noise subtraction.

## 3. Usage & Caveats

- **MIC Performance**: `minepy` can be slow (`O(n^alpha)`). For `TimeSeriesMatrix` with hundreds of channels, `correlation_vector` runs in parallel (`nproc`), which is good, but users should be warned about compute time for high `alpha` or large `n`.
- **Sample Rate**: `_prep_stat_data` automatically resamples data to matching rates using `resample()`. This is convenient but might introduce aliasing artifacts if not careful. Users should ideally pass pre-processed (bandpassed/decimated) data.
