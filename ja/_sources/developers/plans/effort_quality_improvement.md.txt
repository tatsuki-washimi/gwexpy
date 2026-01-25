# Estimate Effort: Core Quality Improvement

**Task**: MyPy coverage expansion, Exception auditing, Code modernization (Py>=3.9).

## 1. Task Decomposition & Difficulty

- **Phase 1: MyPy Coverage (SeriesMatrix Core/Indexing)**
  - **Difficulty**: Medium/High.
  - `series_matrix_core` likely contains complex generic handling and mixin dependencies. Type errors here often propagate. Protocol definitions might be needed.
- **Phase 2: Exception Auditing**
  - **Difficulty**: Low/Medium.
  - Requires context reading to identify correct exception types. `skymap.py` might depend on optional libraries (`astropy_healpix`), complicating imports.
- **Phase 3: Python 3.9+ Modernization**
  - **Difficulty**: Low.
  - Mainly deleting dead code blocks. Risk is low if tests are robust.

## 2. Time Estimation (Wall-clock)

- **Phase 1**: 25 mins (Iterative `mypy` runs and fixes).
- **Phase 2**: 15 mins (Audit and logic update).
- **Phase 3**: 10 mins (Search and cleanup).
- **Overhead**: 5 mins (Plan, Report, Commit).
- **Total**: ~55-60 minutes.

## 3. Quota Estimation

- **Rating**: **High**
- **Reason**: `fix_mypy` often requires reading large files multiple times and extensive reasoning about type hierarchies. `SeriesMatrix` core files are central and dense.

## 4. Efficiency Evaluation & ROI

- **ROI**: High.
  - Bringing `SeriesMatrix` core under strict typing is a major stability win.
  - Removing old python support reduces maintenance burden.
- **Alternative**: If Phase 1 proves too difficult within 30 mins, we will fallback to enabling only one module (`indexing` first, then `core`) or using localized ignores.

## 5. Summary

- **Estimated Total Time**: 60 minutes
- **Estimated Quota Consumption**: High
- **Concerns**: Circular dependencies in typing imports for `SeriesMatrix`.
