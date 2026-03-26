# Task Plan: Enhancing Test Coverage for Physical Fields (Vector/Tensor)
**Date**: 2026-01-24
**Author**: Antigravity

## Objectives & Goals
Increase the test coverage and robustness of `VectorField` and `TensorField` classes. Ensure mathematical consistency, unit safety, and proper metadata preservation across a wide range of operations.

## Detailed Roadmap

### Phase 1: Review and Gap Analysis
1.  **Analyze Source Code**: Deep dive into `gwexpy/fields/vector.py` and `gwexpy/fields/tensor.py` to find edge cases and branching logic (e.g., error conditions).
2.  **Define Test Matrix**: Build a list of missing test scenarios:
    - **Complex Numbers**: Verify L2 norm, dot products, and determinants with complex data (likely after FFT).
    - **Mismatched Metadata**: Test behavior when initializing fields with components having different `axis0` or spatial grids.
    - **Unit Propagation**: Detailed checks for output units (e.g., `VectorField[V] . VectorField[V] -> ScalarField[V^2]`).
    - **Higher Rank Tensors**: Verify `det()` and `trace()` for 3x3 matrices.
    - **Component-wise Selection**: Test `isel` and `sel` methods on collections.
    - **Arithmetic Robustness**: Operations between `VectorField` and `ScalarField` (broadcasting).

### Phase 2: Implementation of Unit Tests
1.  **Expand `tests/fields/test_vectorfield.py`**:
    - Add tests for complex component norms.
    - Add tests for coordinate transformation consistency (basis changes if implemented).
    - Add tests for error handling on mismatched units/axes.
2.  **Expand `tests/fields/test_tensorfield.py`**:
    - Add tests for 3x3 determinant.
    - Add tests for matrix inversion (`inv()`) if not already tested.
    - Add tests for anti-symmetrization.
3.  **New Integration Tests**:
    - Test the full pipeline: `ScalarField -> FFT -> VectorField (Grad) -> Norm`.

### Phase 3: Validation and Refinement
1.  Run tests with `pytest --cov=gwexpy` to measure coverage improvement.
2.  Fix any bugs discovered during testing.
3.  Ensure `ruff` and `mypy` compliance for all test code.

## Testing & Verification Plan
- **Primary Tool**: `pytest`.
- **Coverage Goal**: Above 90% for `vector.py` and `tensor.py`.
- **Execution**: `pytest tests/fields/test_vectorfield.py tests/fields/test_tensorfield.py`.

## Models, Recommended Skills, and Effort Estimates
- **Suggested Model**: `Claude 3.5 Sonnet` or `Gemini 2.0 Flash` (Test generation is a strength of modern LLMs).
- **Recommended Skills**: `test_code`, `lint`, `check_physics`.
- **Effort Estimate**:
    - **Estimated Total Time**: 35 minutes
    - **Estimated Quota Consumption**: Medium (Many code edits and test runs).
    - **Breakdown**:
        - Research & Matrix: 5 mins
        - Vector Tests: 10 mins
        - Tensor Tests: 10 mins
        - Refinement & Coverage Check: 10 mins
- **Concerns**: Tensor math (determinants) can become complex for ranks > 2 if not using standard NumPy helpers internally. Must verify how `gwexpy` handles higher dimensions.
