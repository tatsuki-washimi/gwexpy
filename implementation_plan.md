# Implementation Plan: Strengthen ScalarField Test Coverage

## Objective
Expand and harden the test suite for the `ScalarField` API to ensure correct domain propagation, unit consistency, and metadata integrity under all supported operations.

## Scope of Work

### 1. Domain Propagation Tests (Core Priority)
Verify domain metadata is correctly preserved, transformed, or rejected for:
- Full and inverse FFT.
- Partial FFT (subset of axes).
- Repeated FFT/iFFT cycles.
- Slicing, cropping, and subsetting.
- Chained operations (e.g., slice -> FFT -> slice).
- **Assertions**: Domain type, axis ordering/labels, grid spacing, physical meaning.

### 2. Unit Consistency and Validation
- Propagation through all transforms.
- Scaling and normalization during FFT.
- Exception handling for invalid/inconsistent combinations.
- Negative tests for incorrect unit usage.

### 3. Metadata Integrity
- Persistence of axis descriptors.
- Retention of custom metadata.
- Verification of cloning vs. referencing.
- Equality/comparison semantics.

### 4. Boundary and Edge Cases
- Zero-length or singleton axes.
- Non-contiguous slices.
- Degenerate FFT sizes (length 1, prime sizes).
- NaN/Inf handling.
- Minimal metadata dicts.

### 5. Test Structure
- Update or create files in `tests/fields/`:
    - `test_scalarfield_domain.py`
    - `test_scalarfield_fft.py`
    - `test_scalarfield_units.py`
- Follow pytest-style explicit assertions.

## Quality Gates
- `pytest` passes (no new skips/xfails).
- `mypy .` passes.
- `ruff check` passes.
- Explicit and actionable failure messages.

## Acceptance Criteria
- All major operations covered by positive/negative tests.
- Domain/unit errors caught by tests.
- No references to legacy `Field4D`.
- Documentation of intended physics/math semantics within the tests.
