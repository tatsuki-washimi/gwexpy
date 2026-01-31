# Task Plan: FFT Specifications & Conventions Documentation
**Date**: 2026-01-24
**Author**: Antigravity

## Objectives & Goals
Clarify the FFT implementation details in `gwexpy` to ensure users can correctly interpret physical units, sign conventions, and normalization.

## Detailed Roadmap
### Phase 1: Verification (Research)
1.  Verify the sign convention ($e^{\pm i \omega t}$) and normalization factor ($1/N$ vs $1$) for all FFT methods:
    -   `ScalarField.fft_time`
    -   `ScalarField.ifft_time`
    -   `ScalarField.fft_space` (and related Vector/Tensor methods)
    -   `ScalarField.spectral_density` (Welch's method normalization)
2.  Research GWpy and Matplotlib default behaviors to ensure compatibility.

### Phase 2: Implementation (Documentation)
1.  Create `docs/reference/en/FFT_Conventions.md` containing:
    -   **Temporal FFT**: 1/N normalization, One-sided (rfft).
    -   **Spatial FFT**: Multi-dimensional (fftn), Two-sided, $k = 2\pi f$ (angular wavenumber).
    -   **Sign Conventions**: Explicitly state the transform pairs.
    -   **Normalization**: Behavior of `nfft` and spectral scaling.
    -   **Shift Handling**: Note that zero-frequency is at index 0 (not shifted by default).
2.  Update `docs/reference/en/ScalarField.md` and others to link to this convention page.
3.  Add Japanese version `docs/reference/ja/FFT_Conventions.md`.

### Phase 3: Review & Sync
1.  Run `check_physics` to confirm the documented math matches the code exactly.
2.  Run `sync_docs` to update any outdated docstrings.

## Testing & Verification Plan
-   None (Documentation only, but verified by `check_physics` logic).

## Models, Recommended Skills, and Effort Estimates
-   **Suggested Model**: `Gemini 2.0 Flash` (Documentation-heavy, low reasoning complexity).
-   **Recommended Skills**: `check_physics`, `sync_docs`, `search_web_research`.
-   **Effort Estimate**:
    -   **Estimated Total Time**: 25 minutes
    -   **Estimated Quota Consumption**: Low
    -   **Breakdown**:
        -   Code Review: 5 mins
        -   Writing EN/JA Docs: 15 mins
        -   Linking & Sync: 5 mins
-   **Concerns**: Consistency between `fft_time` (normalized) and `fft_space` (unnormalized) might confuse users if not explained clearly.
