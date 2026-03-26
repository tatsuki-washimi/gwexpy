# Prompt for GPT-5.2-Codex: Phase 2 (Math & Algorithms)

**Context**:
You are the **Lead Numerical Analyst**. We are hardening validity `gwexpy` against "Death Floats" and systemic numerical instability.
A comprehensive audit has been completed. You **MUST** review the following analysis documents before correcting any code:

1. **[Dangerous Defaults Analysis](../analysis/phase1_dangerous_defaults.md)**: Details the 7 critical locations where default args destroy signals.
2. **[Risk Summary](../analysis/step1_2_summary.md)**: Overview of the 160+ risks found.

**Prerequisite**:

* Phase 1 (Opus) is creating `gwexpy.numerics`.
* **IF `gwexpy.numerics` exists**: Import values/scalers from there.
* **IF NOT**: Implement "Temporary Local Logic" that mirrors the architecture defined in `phase1_dangerous_defaults.md`. Do NOT block execution.

**Your Mission**:
Fix the **Algorithmic Logic** in the identified modules. Your goal is **Scale Invariance**: `Function(X)` must behave consistently whether `X` is scaled by $1.0$ or $10^{-20}$.

---

### Task 1: Fix Whitening & Decomposition (The "Null" Variance Bug)

**Target**: `gwexpy/signal/preprocessing/whitening.py`, `gwexpy/timeseries/decomposition.py`
**Reference**: `phase1_dangerous_defaults.md`
**Instructions**:

1. **Whitening**: Modify `eps` handling.
    * Change default from `1e-12` to `None` (or `'auto'`).
    * Logic: `eps = gwexpy.numerics.scaling.get_safe_epsilon(X)` (or local equivalent).
    * Constraint: $10^{-42}$ variance data must NOT result in Identity matrix.
2. **ICA**: Fix `tol` and `prewhiten`.
    * Internal Standardization: Ensure data is scaled to unit variance *before* passing to `FastICA`.
    * Variance Check: If input variance is near machine epsilon, raise specific `ValueError` (do not fail silently).

### Task 2: Fix Fitting & MCMC (The "Jump to Infinity" Bug)

**Target**: `gwexpy/fitting/core.py`
**Reference**: Risk ID **F2** in Plan.
**Instructions**:

1. **MCMC Init**: Locate the line `stds = ... + 1e-8`.
    * **DELETE** the absolute offset.
    * **REPLACE** with Relative Jitter: `jitter = max(abs(val) * 1e-4, SAFE_FLOOR_STRAIN)`.
    * Ensure `SAFE_FLOOR_STRAIN` is imported from `gwexpy.numerics.constants` (or defined locally as `1e-50`).

### Task 3: Fix Matrix Math (The "Underflow" Bug)

**Target**: `gwexpy/types/series_matrix_math.py`, `gwexpy/fields/tensor.py`
**Reference**: Risk ID **C4, C5**.
**Instructions**:

1. **Determinant**:
    * Replace `np.linalg.det(A)` with `np.linalg.slogdet(A)`.
    * Handle `(sign, logdet)` correctly to reconstruct value or work in log-domain if possible.
2. **Inverse/Solve**:
    * Wrap calls in a `try...except np.linalg.LinAlgError`.
    * If singular (due to underflow), attempt **Pre-conditioned Solve**: Scale A by $1/\sigma$, solve, then rescale.

---

**Execution Checklist**:

* [ ] **No Magic Numbers**: If you type `1e-X`, you must justify it or replace it with a constant.
* [ ] **Backward Compatibility**: If changing signatures (e.g. default `eps`), use `kwargs` sniffing or deprecation warnings if necessary, but prioritized CORRECTNESS over compatibility for "Fatal" bugs.
* [ ] **Unit Tests**: For every fix, you must explain how you verified that $10^{-21}$ input works.
