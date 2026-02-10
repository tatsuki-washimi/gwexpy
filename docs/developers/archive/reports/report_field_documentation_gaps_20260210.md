# Field Classes Documentation Gaps - February 10, 2026

## Executive Summary

The Field API (ScalarField, VectorField, TensorField) is a major differentiating feature of gwexpy but has significant documentation gaps that must be addressed before PyPI release. While the classes are fully implemented with comprehensive functionality, the user-facing documentation lacks:

1. **Visual outputs** in existing tutorials (no plots visible)
2. **Complete coverage** of all three Field classes
3. **Prominent visibility** in navigation

## Current State Assessment

### What EXISTS (Code Implementation)

All three Field classes are fully implemented with rich functionality:

**ScalarField** (`gwexpy/fields/scalar.py`)
- 4D spacetime field (time + 3D space)
- Time ↔ Frequency FFT transformations
- Real ↔ K-space transformations on spatial axes
- Signal processing (PSD, cross-correlation, coherence)
- Visualization methods (plot_freq_space, plot_cross_correlation, etc.)
- Batch operations via FieldList/FieldDict

**VectorField** (`gwexpy/fields/vector.py`)
- Collection of ScalarField components
- Norm calculation
- Component-wise operations
- Cartesian basis support

**TensorField** (`gwexpy/fields/tensor.py`)
- Rank-n tensor field representation
- Trace computation (rank-2)
- Index-based component access

### What EXISTS (Documentation)

**English Documentation:**
- `docs/web/en/user_guide/tutorials/field_scalar_intro.ipynb`
  - 42 cells covering comprehensive ScalarField usage
  - Topics: creation, metadata, slicing, FFT (time & space), batch ops, signal processing
  - **CRITICAL ISSUE**: All 21 code cells have ZERO outputs (no plots, no results)
  - File size: 21.4 KB (source only, no embedded outputs)

- `docs/web/en/user_guide/tutorials/field_scalar_signal.md`
  - Additional field signals documentation
  - Markdown format

- `docs/web/en/user_guide/scalarfield_slicing.md`
  - Advanced slicing documentation
  - Listed in main index under "Advanced Topics"

**Japanese Documentation:**
- `docs/web/ja/user_guide/tutorials/field_scalar_intro.ipynb`
  - Matches English structure (42 cells)
  - Also has ZERO outputs
  - File size: 23.8 KB

**Navigation:**
- Tutorials index (`docs/web/en/user_guide/tutorials/index.rst`) includes:
  - Section "III. High-dimensional Fields"
  - Lists: "Scalar Field Basics", "Scalar Field Signals"
- Main index mentions ScalarField in "Advanced Topics"

### What is MISSING

1. **No outputs in ScalarField tutorials**
   - All plots, graphs, and results are missing
   - Users cannot see visualization examples without running notebooks locally
   - Defeats the purpose of online documentation

2. **No VectorField tutorial**
   - Class exists and is functional
   - No usage examples beyond inline docstring
   - No visualization examples

3. **No TensorField tutorial**
   - Class exists and is functional
   - No usage examples beyond inline docstring
   - No explanation of when/why to use it

4. **Limited prominence**
   - Field classes mentioned but not highlighted as key features
   - TimeSeries gets 6+ tutorials, Field gets 2 (both missing outputs)
   - No "Why use Fields?" introduction

## Comparison with TimeSeries Documentation

| Aspect | TimeSeries/Matrix | Field Classes |
|--------|------------------|---------------|
| Quickstart examples | ✅ Yes | ❌ No |
| Intro tutorials | ✅ 6 notebooks | ⚠️ 1 notebook (no outputs) |
| Advanced tutorials | ✅ Multiple | ⚠️ 1 markdown page |
| Visualization examples | ✅ Many plots | ❌ None visible |
| Class coverage | ✅ All classes | ⚠️ Only ScalarField |
| Case studies | ✅ Several | ❌ None |

## Impact on PyPI Release

### User Experience Issues

1. **Discovery Problem**: Users won't understand when/why to use Field classes
2. **Learning Barrier**: No visual examples makes learning difficult
3. **Feature Invisibility**: VectorField and TensorField are "hidden"
4. **Trust Issue**: Lack of documentation suggests experimental/unstable API

### Competitive Disadvantage

- Field API is gwexpy's unique selling point vs gwpy
- Without proper documentation, this advantage is lost
- Users may assume gwexpy is "just gwpy with extras" rather than understanding the paradigm shift

## Minimum Viable Documentation (Pre-Release)

To meet PyPI release quality standards:

### Critical (Must Have)

1. **Execute ScalarField tutorial to add outputs**
   - Requires environment with: numpy, astropy, matplotlib, gwexpy
   - Generate all plots and embed in notebook
   - Ensures users can see results without local execution

2. **Create VectorField basic tutorial**
   - Minimum: Creation, component access, norm calculation, simple visualization
   - Format: Can be markdown with embedded examples or minimal notebook
   - Length: ~15-20 cells or equivalent markdown

3. **Create TensorField basic tutorial**
   - Minimum: Creation, component access, trace calculation
   - Format: Markdown or minimal notebook
   - Length: ~10-15 cells or equivalent markdown

### Important (Should Have)

4. **Add "Why Fields?" introduction**
   - When to use ScalarField vs TimeSeries
   - Physical interpretation of 4D spacetime structure
   - Real-world use cases

5. **Enhance main index visibility**
   - Add Field classes to feature highlights
   - Include visual teaser/example

### Nice to Have (Can Defer)

6. **Field-based case study**
   - Example: Seismic wave propagation, electromagnetic fields, etc.
   - Demonstrates real-world application

7. **Performance comparisons**
   - Show when Field API is more efficient than TimeSeries
   - Benchmark examples

## Recommended Actions

### Immediate (Before PyPI Release)

1. **Set up execution environment** for ScalarField notebook
   ```bash
   pip install gwexpy matplotlib numpy astropy
   jupyter nbconvert --to notebook --execute field_scalar_intro.ipynb
   ```

2. **Create VectorField tutorial** (estimated: 2-3 hours)
   - Template from VectorField docstring examples
   - Add visualization of vector magnitude and components
   - Save as `field_vector_intro.ipynb` or `field_vector_intro.md`

3. **Create TensorField tutorial** (estimated: 1-2 hours)
   - Basic rank-2 tensor example
   - Trace calculation demonstration
   - Save as `field_tensor_intro.ipynb` or `field_tensor_intro.md`

4. **Update navigation**
   - Add new tutorials to `tutorials/index.rst`
   - Update main `index.rst` to feature Fields more prominently

5. **Sync Japanese documentation**
   - Execute Japanese ScalarField notebook
   - Translate/create VectorField and TensorField tutorials

### Post-Release (Can Be Iterative)

- Add advanced Field tutorials
- Create field-based case studies
- Expand visualization examples
- Add performance benchmarks

## Technical Notes

### Why Outputs Are Missing

The ScalarField tutorial notebooks use `nbsphinx_execute = "never"` configuration in `docs/conf.py`. This means:
- Notebooks are included as-is without re-execution
- If notebooks were saved without outputs, they appear output-less in docs
- Solution: Execute notebooks with `jupyter nbconvert --execute` and commit outputs

### Notebook Execution Requirements

To generate outputs, the build/local environment needs:
- Python 3.9+
- gwexpy (with all dependencies)
- numpy, scipy, astropy
- matplotlib
- jupyter, nbformat

Current docs build environment may not have these installed, hence outputs are missing.

## Conclusion

The Field API documentation has **comprehensive content but critical presentation issues**:
- Existing ScalarField tutorial is excellent but invisible (no outputs)
- VectorField and TensorField are undocumented despite being fully implemented
- Overall visibility is too low for a flagship feature

**Recommendation**: Fix these issues before PyPI release. The code is solid, but documentation gaps will prevent users from discovering and adopting this powerful API.

**Priority**: **HIGH** - Field API is a unique gwexpy feature; inadequate documentation undermines its value proposition.

---

**Report Date**: 2026-02-10
**Author**: Claude Code Agent (Documentation Audit)
**Session**: 01PfQdnx74hPaYDTjSC3DkNM
**Branch**: claude/audit-gwexpy-docs-8vipG
