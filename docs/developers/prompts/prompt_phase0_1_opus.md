# Prompt for Claude Opus 4.5: Phase 0 & 1 (Safety & Architecture)

**Context**:
You are acting as the **Lead Architect and Safety Engineer** for `gwexpy`, a gravitational wave analysis library. We have identified a critical systemic failure where low-amplitude signals ($10^{-21}$ scale) are being destroyed by hardcoded "Death Floats" (e.g., `1e-12`, `1e-20`) and silent failures.

**Your Mission**:
You are responsible for **Phase 0 (Unsilencing)** and **Phase 1 (Core Architecture)**. These are the highest-risk tasks requiring deep reasoning to avoid breaking existing logic while exposing hidden bugs.

---

### Task 1: Phase 0 - "Unsilencing" (The Surgical Removal of Pokemon Handlers)

**Objective**: The codebase contains 17+ locations where `try...except Exception: pass` (or similar) swallows numerical errors. Your job is to remove these safeguards without causing immediate crashes for valid edge cases, but ensuring *invalid* numerical states (Overflow, ZeroDiv) propagate.

**Instructions**:

1. **Analyze Context**: For each target file (search for `except Exception` or refer to the audit report), understand *why* the original author added the try/except block.
2. **Refactor**:
    * **Option A (Ideal)**: Remove the try/block entirely if it serves no logic purpose.
    * **Option B (Specific)**: Change `except Exception` to `except (FileNotFoundError, KeyError)` etc., letting `FloatingPointError` and `ValueError` crash.
    * **Option C (Logging)**: If swallowing is logic-critical (fallback), YOU MUST add `logger.warning(..., exc_info=True)` so it is visible.
3. **Target Files (Partial List)**:
    * `gwexpy/__init__.py`
    * `gwexpy/timeseries/collections.py`
    * `gwexpy/spectrogram/collections.py`
    * `gwexpy/io/dttxml_common.py`

### Task 2: Phase 1 - "Numerics" Core Architecture

**Objective**: Design and implement the single source of truth for numerical stability.

**Instructions**:

1. **Create `gwexpy/numerics/` module**.
2. **Implement `gwexpy/numerics/constants.py`**:
    * Define `EPS_VARIANCE`, `EPS_PSD`, `EPS_COHERENCE`.
    * Values should be dynamic based on `np.finfo(float).eps` or physically meaningful limits ($10^{-50}$), NOT arbitrary `1e-12`.
3. **Implement `gwexpy/numerics/scaling.py`**:
    * Create an `AutoScaler` class/context manager.
    * Logic: Input Data X -> Calculate Scale S = std(X) -> Process (X/S) -> Rescale Output (Y * S).
    * Ensure it handles `S=0` (silence) gracefully using logical fallback (e.g., `1.0`), not explicit `1e-30`.

**Constraint Checklist**:

* [ ] Do NOT change the behavior of valid code. Only expose invalid code.
* [ ] Prioritize "No Silent Failures" over "No Crashes". crashing is better than lying (returning 0.0 for valid data).
* [ ] Write docstrings explaining *why* a specific EPS was chosen.

**Input Resource**:
Please review the `docs/developers/plans/numerical_hardening_plan.md` for the full context and risk inventory.
