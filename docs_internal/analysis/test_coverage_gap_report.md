# Test Coverage Gap Analysis Report

**Date**: 2026-01-27
**Status**: DRAFT
**Author**: Antigravity (Agent)

## 1. Executive Summary

This report identifies critical gaps in the test coverage of the `gwexpy` repository. While the overall codebase is robust, specific modules, particularly in IO, Collections, and some Signal Processing areas, exhibit low coverage (< 50%).

Understanding these gaps is crucial for the planned "Phase 2" quality improvement effort to reach 90% coverage.

## 2. Methodology

Coverage data was generated using `pytest-cov` and filtered for core `gwexpy` modules (`types`, `timeseries`, `spectral`, `fields`). Files were ranked by the number of missing lines of code.

## 3. High-Priority Coverage Gaps (Top 10)

The following modules represent the largest accumulation of untested code (by line count of missing statements).

| Rank   | Module                                    | Coverage | Missing Lines | Key Functionality at Risk                                                                                                    |
| :----- | :---------------------------------------- | :------- | :------------ | :--------------------------------------------------------------------------------------------------------------------------- |
| **1**  | `gwexpy/timeseries/collections.py`        | 36%      | **441**       | Batch operations on `TimeSeriesList`/`Dict`. Many wrapper methods for signal processing are untested.                        |
| **2**  | `gwexpy/types/seriesmatrix_validation.py` | 44%      | **267**       | Complex validation logic for matrices (unit consistency, shape checks, alignment). Edge cases here can cause subtle bugs.    |
| **3**  | `gwexpy/types/metadata.py`                | 53%      | **216**       | UFunc overrides and arithmetic operator handling for metadata preservation. Very sensitive logic.                            |
| **4**  | `gwexpy/timeseries/pipeline.py`           | 45%      | **191**       | Data processing pipeline nodes. Likely due to complexity of mocking pipeline data flow.                                      |
| **5**  | `gwexpy/timeseries/_resampling.py`        | 51%      | **182**       | Internal resampling implementations (time-domain vs freq-domain). Critical for data integrity.                               |
| **6**  | `gwexpy/timeseries/matrix_core.py`        | 32%      | **169**       | Core matrix manipulation (slicing, stacking). Low coverage suggests basic usage is tested but complex manipulations are not. |
| **7**  | `gwexpy/timeseries/preprocess.py`         | 71%      | **155**       | Imputation, whitening, standardization logic.                                                                                |
| **8**  | `gwexpy/timeseries/hurst.py`              | 0%       | **148**       | **Completely Untested**. Hurst exponent calculation. This is a blind spot.                                                   |
| **9**  | `gwexpy/timeseries/utils.py`              | 39%      | **123**       | Miscellaneous utilities, often error handling or deprecated paths.                                                           |
| **10** | `gwexpy/timeseries/io/win.py`             | 14%      | **120**       | Windows-specific or legacy IO format. Likely hard to test without specific data files.                                       |

## 4. Analysis by Category

### A. Collections (`collections.py`)

- **Issue**: This file is a heavy wrapper, forwarding method calls to elements.
- **Gap**: Methods like `resample`, `filter`, `whiten` called on a List/Dict are not verified to propagate correctly to all elements.
- **Risk**: User types `ts_list.resample(...)` and it silently fails or only processes the first element.

### B. Validation & Metadata (`validation.py`, `metadata.py`)

- **Issue**: These contain defensive code for edge cases (e.g., mismatched units in arithmetic, shape mismatch in broadcasting).
- **Gap**: "Happy path" tests are likely present, but "Unhappy path" (error raising) is missing.
- **Risk**: Invalid operations might not fail gracefully or might produce garbage data.

### C. IO Modules (`io/win.py`, `io/tdms.py`, `io/ats.py`)

- **Issue**: Proprietary or specific file formats.
- **Gap**: Tests require sample data files which might be missing from the repo or hard to generate on the fly.
- **Risk**: Regressions in reading these formats will go unnoticed until a user reports them.

### D. New/Experimental Features (`hurst.py`)

- **Issue**: Likely a recently added feature that was merged without strict test requirements.
- **Gap**: Zero coverage.
- **Risk**: Functionality is completely unverified.

## 5. Tests to Add (Action Plan)

To bridge the gap to 90%, we propose the following test suites:

### Suite 1: Batch Operations (`tests/timeseries/test_collections_batch.py`)

- **Target**: `gwexpy/timeseries/collections.py`
- **Method**: Create `TimeSeriesList` with 3 mock series. Call `resample`, `filter`, `whiten`. Verify all 3 are processed.

### Suite 2: Hurst Exponent (`tests/timeseries/test_hurst.py`)

- **Target**: `gwexpy/timeseries/hurst.py`
- **Method**: Generate synthetic Brownian motion data (known H=0.5). Verify `hurst_exponent` returns value close to 0.5.

### Suite 3: Validation Edge Cases (`tests/types/test_validation_errors.py`)

- **Target**: `gwexpy/types/seriesmatrix_validation.py`
- **Method**:
  - Try to stack matrices with different sample rates -> Expect Error.
  - Try to add matrix with `unit='m'` to matrix with `unit='s'` -> Expect Error.

### Suite 4: Metadata Math (`tests/types/test_metadata_ufuncs.py`)

- **Target**: `gwexpy/types/metadata.py`
- **Method**: Use `pytest.mark.parametrize` to test broad range of ufuncs (`np.sin`, `np.add`, `np.power`) on MetaData objects and verify result types.

## 6. Recommendations

1.  **Immediate Action**: Implement **Suite 2 (Hurst)** as it addresses a 0% coverage module.
2.  **High Value**: Implement **Suite 1 (Collections)** as it covers a large volume of code with relatively simple tests.
3.  **Long Term**: For IO modules, investigate if small "toy" files can be generated or if `mock` can simulate the file read process sufficiently.
