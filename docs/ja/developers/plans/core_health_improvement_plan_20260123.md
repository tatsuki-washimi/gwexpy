# Core Code Health and Robustness Improvement Plan (2026-01-23 21:23:45)

## 1. Objectives & Goals
Focus on improving the technical quality and robustness of `gwexpy` non-GUI core modules while optimizing resource usage (leveraging Gemini 3 Flash).
*   **Robustness**: Replace loose `except Exception:` with specific exceptions or proper logging.
*   **Completeness**: Address minor `TODO` items in signal and fitting modules.
*   **Modernization**: Remove legacy Python version checks that are no longer needed (Python >= 3.9).

## 2. Detailed Roadmap

### Phase 1: Exception Handling Audit (Non-GUI)
*   **Goal**: Ensure errors are caught precisely and documented.
*   **Tasks**:
    *   Iterate through non-GUI files identified in the repository review:
        *   `gwexpy/fitting/core.py`
        *   `gwexpy/types/series_matrix_validation_mixin.py`
        *   `gwexpy/plot/skymap.py`
        *   `gwexpy/interop/specutils_.py`
    *   Identify the source of potential errors (e.g., `ImportError`, `ValueError`, `KeyError`).
    *   Replace `except Exception:` with specific types or add `logging.exception()` for better traceability.

### Phase 2: Core Module TODOs
*   **Goal**: Clear technical debt in mathematical/signal processing areas.
*   **Tasks**:
    *   Examine `TODO`s in `gwexpy/fitting/core.py`.
    *   Examine `TODO`s in `gwexpy/spectrogram/matrix_core.py`.
    *   Implement straightforward fixes or documented requirements as per the context.

### Phase 3: Python Modernization
*   **Goal**: Simplify code by assuming Python 3.9+.
*   **Tasks**:
    *   Search for `sys.version_info` or `version <` checks.
    *   Remove legacy compatibility shims (e.g., in `gwexpy/types/metadata.py`).
    *   Update any `typing` patterns to modern standards where applicable (e.g., `list[]` instead of `List[]`).

### Phase 4: Verification
*   **Goal**: Ensure no regressions or linting issues.
*   **Tasks**:
    *   Run `ruff check .` and `mypy .`.
    *   Run all core tests (`tests/fitting`, `tests/spectrogram`, `tests/types`).

## 3. Recommended Models & Skills
*   **Model**: `Gemini 3 Flash`
*   **選定理由**: 多くのファイルの精査と微調整が必要なタスクであり、Flashモデルのスピードと高いクオータを活かすことで効率的に進行可能。
*   **Skills**: `grep_search`, `lint`, `test_code`.

## 4. Effort Estimation
*   **Estimated Total Time**: 30 - 45 minutes (AI wall-clock time).
*   **Estimated Quota Consumption**: **Medium** (Multiple file edits, but individual changes are small).
