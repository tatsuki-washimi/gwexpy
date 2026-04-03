---
name: phase1_scale_invariance
description: Use when implementing or reviewing numerical algorithms that may break on tiny gravitational-wave scale inputs or rely on hardcoded eps/tol defaults.
---

# Phase 1 Scale Invariance

This skill helps you detect "Death Floats" and ensure numerical robustness against GW strain scales (~1e-21).

## Overview
Standard numerical thresholds (e.g., `1e-12`) often fail for GW strain data. This skill helps you implement scale-aware defaults and verification.

## When to Use
- Implementing whitening, fitting, Matrix SVD, or PCA.
- Adding new `eps`, `tol`, `atol`, or `rtol` parameters.
- Reviewing spectral analysis or filters.
- Encountering "ill-conditioned matrix" or "loss of precision" errors.

## Core Workflow
1. **Inventory**: Find all hardcoded `1e-X` constants in the numerical path.
2. **Normalize**: If an algorithm requires it, normalize input data before processing and rescale outputs.
3. **Relative Jitter**: Add epsilon relative to the data RMS, e.g., `1e-9 * data.rms()`.
4. **Scale-Invariance Test**: Test that `f(X) ≡ f(1e-20 * X)` produce consistent results.
5. **Numerical Hardening**:
   - Check `np.isfinite` before and after operations.
   - Use `slogdet` instead of `det`.

## Preferred Fix Patterns
- **Instead of**: `eps = 1e-12`
- **Use**: `eps = 1e-9 * np.max(np.abs(data))` or `eps = getattr(data, "machine_precision", 1e-12)`
- **Or**: Use a scale-aware helper from `gwexpy.numerics`.

## Verification
- Run a unit test injecting a signal of magnitude `1e-21`.
- Verify the output SNR or fit results are scale-invariant.

## Common Mistakes
- Thinking `1e-12` is "small enough" for GW strain (it's 1,000,000x larger than the signal).
- Only testing with amplitude `1.0` signals.
- Ignoring rounding errors in long signal processing chains.
