

# Prompt for Gemini CLI: Generate Verification Tests

**Context**:
We are hardening `gwexpy` against numerical instability with low-amplitude signals ($10^{-21}$).
We need a **Verification Test Suite** to prove that our fixes (Phase 0-3) actually work.

**Your Mission**:
Create a new test file `tests/numerics/test_scale_invariance.py`.

**Test Requirements**:

1. **Framework**: Use `pytest`.
2. **Helpers**:
    * Create a fixture `check_scale_invariance(func, data, scale_factor=1e-20, strict=True)`.
    * Logic:
        * Result 1: `y1 = func(data)`
        * Result 2: `y2 = func(data * scale)`
        * Validation: `y2` should be `y1 * scale` (linear) OR `y2` should be `y1` (scale-invariant/normalized), depending on the function.
3. **Test Cases (to implement)**:
    * `test_whitening_invariant`: `compute_whitening_matrix` should return same matrix for $X$ and $10^{-20}X$.
    * `test_ica_source_recovery`: `ica_fit` should recover sources mixed at $10^{-21}$ amplitude.
    * `test_hht_vmin`: `hht_spectrogram` should not contain `nan` or empty plots for small data.
    * `test_safe_log`: Verify `safe_log_scale` allows inputs down to $10^{-50}$ without clipping to flat -200dB.
    * `test_filter_stability`: Verify high-pass/band-pass filters don't explode on $10^{-21}$ data.

**Key Constraints**:

* Use `np.random.randn` to generate inputs.
* Assert `np.allclose` with appropriate tolerances.
* The test **MUST FAIL** on the current codebase (if run before fixes) and **PASS** after the fixes in Phase 1/2.
* The file must be valid python code ready to run.

**Output**:
Output the full content of `tests/numerics/test_scale_invariance.py`.
