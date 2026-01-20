# Field4D Physics Review (2026-01-20)

## Scope

- Targets: `Field4D`, `Field4DList`, `Field4DDict`
- Focus: FFT domain transitions, axis handling, physical unit consistency
- Files reviewed: `gwexpy/types/field4d.py`, `gwexpy/types/field4d_collections.py`, `gwexpy/types/axis.py`, `gwexpy/types/array4d.py`

## Summary

The core FFT workflows are conceptually aligned with GWpy conventions, but there are several physics/maths risks around axis regularity, origin offsets, Nyquist bin handling, and assumptions about real-valued signals. Collection validation is also too weak to guarantee physical alignment across batched fields.

## Findings

- High: `fft_time`/`ifft_time` compute `dt`/`df` from the first two samples without validating axis regularity or size >= 2, so irregular or length-1 axes yield physically incorrect frequency/time axes. (`gwexpy/types/field4d.py:341`, `gwexpy/types/field4d.py:406`)
- Medium: Time and space inverse transforms rebuild axes with a zero origin, ignoring any original offset; this changes the physical coordinate system and can introduce phase errors. (`gwexpy/types/field4d.py:345`, `gwexpy/types/field4d.py:414`, `gwexpy/types/field4d.py:639`)
- Medium: One-sided normalization doubles all non-DC bins; for even-length FFTs the Nyquist bin should not be doubled. (`gwexpy/types/field4d.py:337`)
- Medium: `rfft/irfft` implicitly assume real-valued signals and Hermitian frequency data, but no validation enforces that constraint. (`gwexpy/types/field4d.py:335`, `gwexpy/types/field4d.py:403`)
- Low: Spatial axis monotonicity is not enforced, and `fft_space` uses signed `dx` while `ifft_space` uses `abs(dk)`; descending axes can lead to inconsistent k-axis direction and reconstruction. (`gwexpy/types/axis.py:54`, `gwexpy/types/field4d.py:521`, `gwexpy/types/field4d.py:628`)
- Low: `Field4DList`/`Field4DDict` validation checks only units/domains/names, not the coordinate arrays, so batched FFT operations can mix mismatched samplings. (`gwexpy/types/field4d_collections.py:38`, `gwexpy/types/field4d_collections.py:170`)

## Code Fix Plan

1. Axis validation in `fft_time`/`ifft_time`:
   - Require axis length >= 2 and monotonic/regular spacing for time/frequency axes.
   - Raise a clear `ValueError` if the axis is irregular or too short.
2. Preserve axis origin (offset):
   - Store `axis0_offset` (and spatial offsets where applicable) in metadata during transforms, or preserve original origin where mathematically valid.
   - Apply the stored offset when reconstructing time or real-space axes.
3. Nyquist bin handling:
   - For even `nfft`, apply doubling to bins `1:-1`; for odd, apply to `1:`.
   - Mirror this logic in `ifft_time` when undoing normalization.
4. Real-signal validation:
   - `fft_time`: reject complex-valued input or explicitly switch to full `fft` if complex support is desired.
   - `ifft_time`: validate that input is one-sided/Hermitian (or document and raise when not).
5. Spatial axis monotonicity:
   - Require spatial axes to be strictly monotonic (ascending or descending) and use consistent sign when computing `k` and `x`.
   - Avoid `abs(dk)` unless deliberately discarding direction.
6. Collection validation:
   - When `validate=True`, also compare axis coordinate arrays (within tolerance) to ensure consistent sampling across fields.

## Test Plan

- Irregular axis rejection for `fft_time`/`ifft_time`.
- Non-zero origin preservation for time and space transforms.
- Nyquist bin amplitude correctness for even `nfft`.
- Complex input rejection (or complex-support tests if implemented).
- Descending axis behavior in `fft_space`/`ifft_space`.
- Collection validation failures on mismatched coordinate arrays.

## Notes

- **2026-01-20**: All 6 code fixes implemented:
  1. ✅ Axis validation in `fft_time`/`ifft_time` (length >= 2, regular spacing)
  2. ✅ Axis origin offset preservation (stored in `_axis0_offset` metadata)
  3. ✅ Nyquist bin handling (bins 1:-1 for even nfft)
  4. ✅ Real-signal validation (complex input rejected with `TypeError`)
  5. ✅ Spatial axis monotonicity and consistent signed delta handling
  6. ✅ Collection validation (axis coordinate arrays compared within tolerance)
- Tests and documentation updates are deferred for separate work.
