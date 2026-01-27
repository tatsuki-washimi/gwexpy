---
description: Improve code robustness by refining exception handling
---

# Task: Refactor Broad Exception Handling

The repository review identified several locations using `except Exception:` which suppresses all errors and hinders debugging.

**Target Files**:

- `gwexpy/spectral/estimation.py`
- `gwexpy/analysis/bruco.py`
- `gwexpy/types/seriesmatrix_base.py`

**Objectives**:

1.  Identify the specific exceptions that _should_ be caught (e.g., `ValueError`, `RuntimeError`).
2.  Replace `except Exception:` with specific exception blocks.
3.  If a broad catch is truly necessary (e.g., in valid safeguard logic), ensure the exception is logged using `logging.exception()` or `warnings.warn()`.

**Verification**:

- Run related tests to ensure no functionality is broken.
- Ensure `ruff` passes.
