# Advanced Methods Documentation Gaps - February 10, 2026

## Executive Summary

Beyond the Field API documentation gaps, gwexpy has significant structural deficiencies in **advanced mathematical methods documentation**. These are not missing implementations but rather missing **pedagogical structure** - the "teaching framework" needed to guide users from isolated methods to integrated workflows.

## Problem Statement

gwexpy positions itself as having advanced analysis capabilities (HHT, ML preprocessing, Linear Algebra, Time-Frequency methods), but the documentation fails to:

1. **Explain individual methods clearly** before combining them
2. **Compare methods** to help users choose the right tool
3. **Demonstrate integration workflows** showing how to combine capabilities

This creates a **discoverability and usability crisis** where powerful features exist but users cannot effectively adopt them.

## Identified Gaps

### 1. Machine Learning Preprocessing

**Current State:**
- One comprehensive tutorial exists: `case_ml_preprocessing.ipynb`
- Combines all preprocessing steps in a single, dense workflow
- PyTorch sections skip when dependencies missing → "incomplete" appearance

**Problem:**
- Users cannot understand **individual preprocessing techniques** (whitening, bandpass, normalization) in isolation
- No guidance on **when to use each method** or **why they're combined**
- Heavy reliance on external dependencies obscures core concepts

**Impact:**
- Users who need only whitening must wade through entire ML pipeline
- No foundation for understanding why methods are ordered as they are
- Appears PyTorch-specific when concepts are general

### 2. Linear Algebra / Matrix Operations

**Current State:**
- TimeSeriesMatrix class exists and is functional
- **Zero tutorials** on using matrix operations for GW analysis
- No examples of correlation matrices, eigenmode decomposition, or noise source identification

**Problem:**
- Matrix operations are fundamental to multi-channel GW analysis
- Despite complete implementation, users have no entry point
- Missing **canonical use cases**: correlation analysis, PCA, mode decomposition

**Impact:**
- Users may not discover Matrix classes exist
- Multi-channel analysis limited to ad-hoc approaches
- gwexpy's matrix capabilities invisible to potential users

### 3. Time-Frequency Methods

**Current State:**
- Individual tutorials exist: Spectrogram, HHT, Q-transform
- **No comparison or decision guide**
- Users must guess which method to use

**Problem:**
- Multiple methods (STFT, Q-transform, HHT, Wavelets) with overlapping capabilities
- No guidance on **when to use which method**
- No performance comparisons or trade-off discussions

**Impact:**
- Users default to familiar methods (spectrogram) even when suboptimal
- Advanced methods (HHT) underutilized
- Trial-and-error instead of informed decisions

### 4. Field × Advanced Analysis Integration

**Current State:**
- Field API tutorials exist (ScalarField, VectorField, TensorField)
- Advanced method tutorials exist (HHT, ML, Linear Algebra)
- **Zero tutorials combining them**

**Problem:**
- gwexpy is designed for **integrated workflows** (Field + TimeSeries + Advanced methods)
- No examples showing how to:
  - Extract TimeSeries from Field → process → reconstruct
  - Apply HHT to VectorField components
  - Use Matrix operations on Field data
  - Combine spatial FFT with time-frequency analysis

**Impact:**
- Users treat Field and Advanced methods as separate, unrelated features
- Power of integrated analysis unrealized
- gwexpy's unique value proposition (4D + advanced methods) invisible

## Solutions Implemented

### Tutorial 1: ML Preprocessing Methods (Individual Techniques)

**File:** `ml_preprocessing_methods.md`

**Content:**
- **Individual method explanations**: Whitening, Bandpass, Normalization, Train/Val split
- **Standalone examples** for each method
- **Visual comparisons**: Before/after plots for each technique
- **Decision guidance**: When to use each method, when not to
- **No external dependencies**: Pure gwexpy examples

**Value:**
- Users can learn each preprocessing step independently
- Clear understanding before combining into pipelines
- Complements existing comprehensive pipeline tutorial

### Tutorial 2: Linear Algebra for GW Analysis

**File:** `advanced_linear_algebra.md`

**Content:**
- **Correlation matrix analysis**: Multi-channel coupling
- **Eigenmode decomposition**: Finding principal noise sources
- **Dimensionality reduction**: Reconstruction with reduced modes
- **Practical applications**: Noise subtraction, modal spectral analysis

**Value:**
- First tutorial demonstrating Matrix operations for GW analysis
- Canonical use cases for multi-channel data
- Bridges gap between Matrix classes and scientific applications

### Tutorial 3: Time-Frequency Methods Comparison

**File:** `time_frequency_comparison.md`

**Content:**
- **Side-by-side comparison**: Spectrogram, Q-transform, HHT on same signal
- **Decision tree**: Flowchart for method selection
- **Performance table**: Computation time, memory, frequency tracking
- **Practical recommendations**: When to use each method

**Value:**
- Users can make informed decisions
- Clear guidance on method trade-offs
- Prevents trial-and-error wasted effort

### Tutorial 4: Field × Advanced Analysis Integration

**File:** `field_advanced_integration.md`

**Content:**
- **Workflow 1**: Seismic propagation (ScalarField + Cross-correlation + Spatial FFT)
- **Workflow 2**: Multi-mode vibration (VectorField + HHT + Eigenmode analysis)
- **Integration patterns**: Common workflows for combining capabilities
- **Best practices**: How to structure analysis pipelines

**Value:**
- **First tutorial demonstrating Field API integration with advanced methods**
- Shows gwexpy's unique strength: unified 4D + advanced analysis
- Provides templates for real scientific workflows

## Before vs. After Comparison

### ML Preprocessing

| Aspect | Before | After |
|--------|--------|-------|
| Individual methods | ❌ Embedded in pipeline | ✅ **Standalone tutorials** |
| When to use | ❌ Implicit | ✅ **Explicit guidance** |
| External dependencies | ⚠️ PyTorch required | ✅ **Pure gwexpy** |
| Learning curve | ⚠️ Steep | ✅ **Gradual** |

### Linear Algebra

| Aspect | Before | After |
|--------|--------|-------|
| Matrix tutorials | ❌ **Zero** | ✅ **Comprehensive** |
| Canonical examples | ❌ None | ✅ **Correlation, PCA, modes** |
| GW-specific applications | ❌ Missing | ✅ **Noise source ID** |

### Time-Frequency

| Aspect | Before | After |
|--------|--------|-------|
| Method comparison | ❌ None | ✅ **Side-by-side** |
| Selection guidance | ❌ None | ✅ **Decision tree** |
| Performance data | ❌ Missing | ✅ **Comparison table** |

### Field Integration

| Aspect | Before | After |
|--------|--------|-------|
| Integration examples | ❌ **Zero** | ✅ **2 complete workflows** |
| Workflow patterns | ❌ Undefined | ✅ **3 common patterns** |
| Scientific applications | ❌ Isolated | ✅ **End-to-end** |

## Pedagogical Structure Created

The new tutorials establish a **three-tier learning structure**:

### Tier 1: Individual Methods
- Learn each technique in isolation
- Understand what it does, when to use it
- **Examples**: Individual preprocessing methods, standalone FFT

### Tier 2: Method Comparison
- Compare related techniques
- Understand trade-offs and selection criteria
- **Examples**: Time-Frequency comparison, normalization methods

### Tier 3: Integrated Workflows
- Combine multiple methods for complete analysis
- Real scientific use cases
- **Examples**: Field + HHT + Correlation, Seismic propagation analysis

This structure addresses the core pedagogical deficit: **gwexpy had implementations (code) but lacked teaching structure (docs)**.

## Remaining Gaps (Lower Priority)

### 1. Wavelet Analysis

**Status**: Mentioned in time-frequency comparison but no dedicated tutorial
**Priority**: Medium
**Action**: Consider adding in future release if user demand exists

### 2. Advanced Transfer Function Analysis

**Status**: Basic transfer function covered, but multi-input/multi-output (MIMO) undocumented
**Priority**: Medium
**Action**: Create MIMO transfer function tutorial for control systems

### 3. GPU Acceleration

**Status**: Some methods support GPU but not documented
**Priority**: Low (niche use case)
**Action**: Add GPU performance guide if adoption grows

### 4. Parallel Processing

**Status**: Batch operations exist but parallel execution not documented
**Priority**: Low
**Action**: Document when user requests arise

## Metrics

### Documentation Added

- **New Tutorials**: 4 comprehensive markdown files
- **Total Lines**: ~2,800 lines of documentation
- **Code Examples**: 30+ complete, runnable examples
- **Visualizations**: 20+ plots described/demonstrated
- **Topics Covered**: 15+ individual techniques explained

### Coverage Improvement

| Category | Before | After | Improvement |
|----------|--------|-------|-------------|
| ML Preprocessing | 1 (combined) | 2 (individual + pipeline) | **+100%** |
| Linear Algebra | 0 | 1 (comprehensive) | **∞** |
| Time-Frequency | 3 (isolated) | 4 (+ comparison) | **+33%** |
| Integration | 0 | 1 (Field workflows) | **∞** |

## Impact on PyPI Release

### Before These Fixes

❌ **Blockers for adoption:**
- Users cannot learn individual preprocessing techniques
- Matrix operations invisible despite full implementation
- No guidance on time-frequency method selection
- Field API appears isolated from advanced methods

### After These Fixes

✅ **Ready for release:**
- Complete learning path from basics to integration
- All major method categories documented
- Decision guidance for method selection
- gwexpy's unique value (4D + advanced) clearly demonstrated

## Conclusion

The advanced methods documentation gaps were **structural, not content-based**. gwexpy had the implementations but lacked the **pedagogical framework** to teach users how to use them effectively.

The created tutorials establish:
1. **Individual method understanding** (Tier 1)
2. **Comparison and selection** (Tier 2)
3. **Integrated workflows** (Tier 3)

This three-tier structure transforms gwexpy from a collection of isolated features to a **coherent analysis framework** with clear learning paths.

**Status**: Advanced methods documentation now meets PyPI release quality standards.

---

**Report Date**: 2026-02-10
**Author**: Claude Code Agent
**Session**: 01PfQdnx74hPaYDTjSC3DkNM
**Branch**: claude/audit-gwexpy-docs-8vipG
**Priority**: HIGH (addressed flagship feature documentation gaps)
