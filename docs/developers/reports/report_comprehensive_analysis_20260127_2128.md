# Comprehensive Quality Analysis and Guidelines Report

**Timestamp**: 2026-01-27 21:28
**Author**: Antigravity (Agent)

This document consolidates the findings from the quality improvement phase (MyPy analysis, coverage analysis) and outlines the resulting developer guidelines.

---

# Part 1: MyPy Error Analysis and Remedy Strategy

## 1. Executive Summary

As of **2026-01-27**, the repository contains **157 MyPy errors** across **31 files** (when running `mypy .`). While the codebase is functional and passes runtime tests, these static analysis errors represent technical debt that hinders developer productivity and hides potential bugs.

### Verification (Current Repo State)

This report was cross-checked against the working tree state on **2026-01-27**:

- `mypy .` -> **Found 157 errors in 31 files** (checked 324 source files)
- `mypy gwexpy` -> **Found 148 errors in 30 files** (checked 311 source files)

Note: the higher count from `mypy .` includes `scripts/` and other non-package modules.

The errors fall into distinct categories, with **Mixin attribute resolution** and **NumPy method signature incompatibilities** being the most prevalent. This document analyzes these errors and provides concrete strategies for their resolution.

## 2. Error Categories Analysis

The 157 errors can be classified into the following 5 main categories:

| Category                          | Count (Approx) | Description                                                                | Typical Error Message                                           |
| :-------------------------------- | :------------- | :------------------------------------------------------------------------- | :-------------------------------------------------------------- |
| **A. Mixin Attribute Access**     | ~40            | Mixin classes accessing `self.value`, `self.unit` etc. without definition. | `"SignalAnalysisMixin" has no attribute "value"`                |
| **B. Signature Incompatibility**  | ~30            | Mixin methods overriding `ndarray` methods with different signatures.      | `Definition of "astype" ... is incompatible with ... "ndarray"` |
| **C. UFunc & Operator Overloads** | ~60            | Complex UFunc type overloading failures in Metadata classes.               | `No overload variant of "__call__" matches ...`                 |
| **D. Dynamic Imports (QtPy)**     | ~10            | `QtCore` members not found due to dynamic import nature of `qtpy`.         | `Name "QtCore.QThread" is not defined`                          |
| **E. Spectrogram Specifics**      | ~15            | Complex mixin injection in Spectrogram causing self-type mismatch.         | `Invalid self argument ... to attribute function`               |

### A. Mixin Attribute Access (High Priority)

**Problem**:
Classes like `SignalAnalysisMixin` assume they will be mixed into a class (e.g., `TimeSeriesMatrix`) that has `value`, `unit`, `name`, and `channel` attributes. However, the Mixin itself does not declare these, leading MyPy to flag them as missing.

**Remedy Strategy**:
Use Python `Protocol` to define the structural subtyping requirements for the Mixin.

### B. Signature Incompatibility (Medium Priority)

**Problem**:
`gwexpy` classes inherit from `numpy.ndarray` via `gwpy`. Mixins often redefine standard methods like `astype`, `reshape`, `min`, `max` to preserve metadata or units. If the signature doesn't exactly match `numpy`, MyPy complains.

**Remedy Strategy**:

1.  **Align Signatures**: Whenever possible, match the downstream library's signature exactly.
2.  **`# type: ignore[override]`**: If the deviation is intentional and safe, this suppression is acceptable.

### C. UFunc & Operator Overloads (Low Priority / Hard)

**Problem**:
`gwexpy/types/metadata.py` attempts to override `__pow__` and `__call__` for custom behavior. NumPy's UFunc typing is extremely complex with many overloads.

**Remedy Strategy**:
Suppress these specific errors with `# type: ignore[override]` or `[misc]` unless deep changes to the implementation are made.

### D. Dynamic Imports (QtPy) (Quick Fix)

**Problem**:
`qtpy` dynamically imports PyQt5 or PySide2. MyPy can't always statically determine the content of `QtCore`.

**Remedy Strategy**:
Explicitly cast or import from the underlying library if possible, or use a stub file for `qtpy`.

### E. Spectrogram Specifics (Long Term)

**Problem**:
The `SpectrogramMatrix` relies on a very dynamic composition of function injection.

**Remedy Strategy**:
Requires a structural refactor to use standard composition or inheritance instead of dynamic method injection.

## 3. Implementation Roadmap

1.  **Phase 1 (Week 1)**: Mixin Protocol Implementation (Target Category A).
2.  **Phase 2 (Week 1-2)**: QtPy and Signature Fixes (Target Category B & D).
3.  **Phase 3 (Week 2)**: Metadata Type Relaxation (Target Category C).

---

# Part 2: Test Coverage Gap Analysis Report

## 1. Executive Summary

This report identifies critical gaps in the test coverage of the `gwexpy` repository. Specific modules in **Collections**, **Matrix infrastructure**, **IO**, and some **analysis/plotting** areas exhibit low coverage (< 50%).

### Method / Reproducibility Notes

- Coverage numbers below are derived from the repository's `.coverage` data by running: `python -m coverage report -m`.
- The "Top 10" list in the next section is **focused on the timeseries/types layer** (where missed lines are both large and high-impact). GUI/plot modules are tracked separately because their tests tend to be heavier and environment-dependent.

## 2. High-Priority Coverage Gaps (Top 10)

| Rank   | Module                                    | Coverage | Missing Lines | Key Functionality at Risk                         |
| :----- | :---------------------------------------- | :------- | :------------ | :------------------------------------------------ |
| **1**  | `gwexpy/timeseries/collections.py`        | 36%      | **441**       | Batch operations on `TimeSeriesList`/`Dict`.      |
| **2**  | `gwexpy/types/seriesmatrix_validation.py` | 44%      | **267**       | Complex validation logic for matrices.            |
| **3**  | `gwexpy/types/metadata.py`                | 53%      | **216**       | UFunc overrides and arithmetic operator handling. |
| **4**  | `gwexpy/timeseries/pipeline.py`           | 45%      | **191**       | Data processing pipeline nodes.                   |
| **5**  | `gwexpy/timeseries/_resampling.py`        | 51%      | **182**       | Internal resampling implementations.              |
| **6**  | `gwexpy/timeseries/matrix_core.py`        | 32%      | **169**       | Core matrix manipulation.                         |
| **7**  | `gwexpy/timeseries/preprocess.py`         | 71%      | **155**       | Imputation, whitening, standardization logic.     |
| **8**  | `gwexpy/timeseries/hurst.py`              | 0%       | **148**       | **Completely Untested**.                          |
| **9**  | `gwexpy/timeseries/utils.py`              | 39%      | **123**       | Miscellaneous utilities.                          |
| **10** | `gwexpy/timeseries/io/win.py`             | 14%      | **120**       | Windows-specific or legacy IO format.             |

### Additional High-Miss Modules (Overall, by Missed Lines)

The following modules have very large absolute missed-line counts and are worth tracking alongside the timeseries/types backlog:

| Rank   | Module                              | Coverage | Missing Lines | Notes |
| :----- | :---------------------------------- | :------- | :------------ | :---- |
| **1**  | `gwexpy/gui/ui/main_window.py`      | 60%      | **372**       | Large surface area; prioritize targeted integration tests. |
| **2**  | `gwexpy/analysis/bruco.py`          | 44%      | **336**       | Core analysis logic; many branches untested. |
| **3**  | `gwexpy/plot/plot.py`               | 45%      | **279**       | Plotting API behavior/edge cases largely unverified. |
| **4**  | `gwexpy/analysis/coupling.py`       | 19%      | **250**       | High-risk analysis; currently very under-tested. |
| **5**  | `gwexpy/gui/ui/channel_browser.py`  | 9%       | **241**       | GUI-heavy; likely needs qtbot-based tests and/or refactor for testability. |
| **6**  | `gwexpy/types/metadata.py`          | 53%      | **216**       | Complex overloads; add targeted operator/ufunc tests. |

## 3. Recommended Test Suites to Add

1.  **Suite 1: Batch Operations** (`tests/timeseries/test_collections_batch.py`) for `gwexpy/timeseries/collections.py`.
2.  **Suite 2: Hurst Exponent** (`tests/timeseries/test_hurst.py`) for `gwexpy/timeseries/hurst.py`.
3.  **Suite 3: Validation Edge Cases** (`tests/types/test_validation_errors.py`) for `gwexpy/types/seriesmatrix_validation.py`.
4.  **Suite 4: Metadata Math** (`tests/types/test_metadata_ufuncs.py`) for `gwexpy/types/metadata.py`.

---

# Part 3: GWExPy Coding Standards & Developer Guidelines (v0.1)

## 1. Python Version & Compatibility

- **Target Versions**: Python 3.9, 3.10, 3.11, 3.12.
- **Syntax Rule**: Code must be syntactically compatible with Python 3.9.

### 1.1 `from __future__ import annotations` (MUST)

All new/modified Python files **MUST** start with `from __future__ import annotations` to enable forward references and modern type syntax in annotations.

- **Current status (gwexpy package)**: 314 / 318 files (98.7%) include the future import.
- **Backlog**: remaining files should be upgraded opportunistically when touched for other reasons.

### 1.2 Union Type Syntax (CRITICAL)

- **In Annotations**: `int | None` is allowed.
- **In Runtime Contexts**: `isinstance(x, int | str)` fails in Python 3.9. Use `typing.Union` or `typing.Optional`.

## 2. Type Safety & MyPy

`gwexpy` aims to continuously expand MyPy coverage while keeping the developer experience practical (some paths may remain excluded due to GUI/test-data complexity).

- Current configuration excludes (at least) `gwexpy/gui/ui`, `gwexpy/gui/reference-dtt`, `gwexpy/gui/test-data`, and `tests/` from MyPy runs (see `pyproject.toml`).

### 2.1 Mixin Classes (Protocol Pattern)

Use **Protocols** to define expected attributes (`value`, `unit`) in Mixin classes instead of assuming they exist.

```python
@runtime_checkable
class HasSeriesData(Protocol):
    value: Any
    unit: Any

class SignalAnalysisMixin:
    def smooth(self: HasSeriesData, width: int) -> Any:
        return self.value
```

### 2.2 `super()` in Mixins

Use `cast` to hint the expected parent interface when calling `super()` in a Mixin.

## 3. Exception Handling

### 3.1 No Bare `except Exception`

Avoid it by default. Prefer specific exceptions; if a broad catch is necessary (e.g., thread shutdown / GUI event loops), it must:

- log via `logger.exception(...)` / `logger.warning(..., exc_info=True)` (or otherwise preserve tracebacks), and
- include a short comment describing why the broad catch is required.

### 3.2 Warnings

Use `warnings.warn` for deprecations or non-fatal data issues.

## 4. Testing Guidelines

### 4.1 Pytest & Coverage

- **Goal**: 90% coverage for new modules.
- **Command**: `pytest --cov=gwexpy tests/`

### 4.2 GUI Testing (Qt)

- **Use `qtbot`**.
- **`waitExposed`**: NEVER use `waitForWindowShown`.
- **Headless**: Tests must pass in Xvfb.

## 5. Documentation Style

- **Docstrings**: NumPy style is preferred.
- **Type Hints**: Redundant in docstrings if present in signature.

---

# Part 4: Parallel Execution Plan (GPT-5.2-Codex vs Claude Opus 4.5)

This section assigns the tasks described in this report to two agents for **concurrent execution**, with a target workload ratio of **GPT:Claude = 2:1**. The assignments are designed to avoid merge conflicts by keeping file ownership disjoint.

## 1. File Ownership Rules (No-Interference Contract)

- Each agent edits only its assigned files (and their direct tests) unless explicitly coordinated.
- If a change would require touching the other agent's owned files, stop and coordinate first.
- Both agents may add new files under `tests/` as long as they do not overlap on the same test file path.

## 2. Claude Opus 4.5 (1/3 workload): MyPy Category A + D Foundation

**Goal**: Reduce MyPy errors driven by missing mixin attributes and QtPy dynamic imports, without modifying matrix-core or metadata-heavy modules.

### Scope

- **Category A (Mixin Attribute Access)**: Introduce `Protocol`-based structural typing for mixins that assume `self.value`, `self.unit`, etc.
- **Category D (Dynamic Imports / QtPy)**: Fix the `QtCore.QThread` name resolution errors in `gwexpy/gui/nds/nds_thread.py`.

### Owned Files (Claude only)

- `gwexpy/gui/nds/nds_thread.py`
- Mixins that are the direct source of Category A errors (protocol + annotation-only changes)

### Must Not Touch (Owned by GPT)

- `gwexpy/types/metadata.py`
- `gwexpy/**/matrix*.py` and other ndarray-override hot spots (Category B work)
- Test files created/owned by GPT in the next section

## 3. GPT-5.2-Codex (2/3 workload): Coverage Suites + MyPy Category B + C

### 3.1 Coverage Expansion (All 4 suites in this report)

**Goal**: Add tests that directly improve coverage for the identified high-miss modules.

**Owned New Test Files (GPT only)**:

- `tests/timeseries/test_collections_batch.py`
- `tests/timeseries/test_hurst.py`
- `tests/types/test_validation_errors.py`
- `tests/types/test_metadata_ufuncs.py`

### 3.2 MyPy Category B (ndarray signature incompatibilities)

**Goal**: Reduce override/signature errors by matching `numpy.ndarray` signatures where feasible and using minimal `# type: ignore[override]` where intentional.

**Owned Files (GPT only)**:

- `gwexpy/timeseries/matrix.py`
- `gwexpy/frequencyseries/matrix.py`
- Other matrix/mixin modules that produce Category B override errors

### 3.3 MyPy Category C (UFunc & operator overloads in metadata)

**Goal**: Reduce the highest-friction ufunc typing failures with pragmatic typing relaxations and targeted ignores, while preserving runtime behavior.

**Owned Files (GPT only)**:

- `gwexpy/types/metadata.py`

## 4. Shared Verification Checklist

- `ruff check .`
- `pytest -q`
- `mypy .` (track total error count; note that `pyproject.toml` excludes some GUI paths)

## 5. Integration Notes

- Prefer small, reviewable PR-sized diffs per category (A/D vs B/C vs tests).
- If Protocol changes (Category A) require test adjustments, keep those adjustments within Claude-owned mixin context; do not modify GPT-owned new tests without coordination.
