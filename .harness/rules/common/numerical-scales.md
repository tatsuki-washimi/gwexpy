# GWexpy Numerical Scale Rules

Gravitational-wave (GW) strain is typically of the order `~1e-21`. Standard machine epsilon (`1e-12` or `1e-16`) is often too coarse for comparing GW signals.

## Forbidden Magic Numbers

- **Hardcoded Constants**: Avoid `eps=1e-12`, `tol=1e-4`, or other magic numbers that ignore the dynamic scale of data.
- **Zero Comparisons**: Never compare strain data directly to zero without a scale-aware epsilon.

## Preferred Sources of Truth

- **`gwexpy.numerics`**: Always use constants from this module for consistent epsilon and tolerance management.
- **`slogdet` and `preconditioned_solve`**: For matrix operations involving high dynamic ranges, use log-scale determinants and preconditioning.

## Scale-Aware Defaults

- **`None` or `'auto'`**: Use these for parameters like `eps` in arguments, and calculate them relative to the input data's peak or RMS value inside the function.
- **Relative Jitter**: Add `1e-6 * max(abs(data))` instead of a fixed small value when regularizing matrices.

## Algorithm-Specific Notes

- **HHT/EMD**: The `eps` for EMD stopping criteria should be `~0.2-0.3` for general cases but adapted for GW signals.
- **STLT**: Ensure `sigma` does not overflow when dealing with tiny strain values; normalize data before processing if necessary.
- **Whitening**: Ensure the noise PSD estimation is robust against local glitches (use `median` or `Welch`).

## Testing Expectations

- **Scale-Invariance Tests**: Verify that `f(X) ≡ f(X * 1e-20)` (or consistent results) for all numerical algorithms.
- **Tiny-Signal Injection**: Test and document behavior with inputs of `1e-21` magnitude.
- **NaN/Inf Guards**: Explicitly check for non-finite values before and after any matrix inversion or fitting.

## Review Triggers

- Any modification of numerical thresholds or convergence criteria.
- New physics-related mathematical implementations.
