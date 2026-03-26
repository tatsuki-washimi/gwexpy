# MyPy Error Analysis and Remedy Strategy

**Date**: 2026-01-27
**Status**: DRAFT
**Author**: Antigravity (Agent)

## 1. Executive Summary

As of January 27, 2026, the `gwexpy` codebase contains **157 MyPy errors** across 31 files. While the codebase is functional and passes runtime tests, these static analysis errors represent technical debt that hinders developer productivity and hides potential bugs.

The errors fall into distinct categories, with **Mixin attribute resolution** and **NumPy method signature incompatibilities** being the most prevalent. This document analyzes these errors and provides concrete strategies for their resolution.

## 2. Error Categories Analysis

The 157 errors can be classified into the following 5 main categories:

| Category | Count (Approx) | Description | Typical Error Message |
| :--- | :--- | :--- | :--- |
| **A. Mixin Attribute Access** | ~40 | Mixin classes accessing `self.value`, `self.unit` etc. without definition. | `"SignalAnalysisMixin" has no attribute "value"` |
| **B. Signature Incompatibility** | ~30 | Mixin methods overriding `ndarray` methods with different signatures. | `Definition of "astype" ... is incompatible with ... "ndarray"` |
| **C. UFunc & Operator Overloads** | ~60 | Complex UFunc type overloading failures in Metadata classes. | `No overload variant of "__call__" matches ...` |
| **D. Dynamic Imports (QtPy)** | ~10 | `QtCore` members not found due to dynamic import nature of `qtpy`. | `Name "QtCore.QThread" is not defined` |
| **E. Spectrogram Specifics** | ~15 | Complex mixin injection in Spectrogram causing self-type mismatch. | `Invalid self argument ... to attribute function` |

### A. Mixin Attribute Access (High Priority)

**Problem**:
Classes like `SignalAnalysisMixin` assume they will be mixed into a class (e.g., `TimeSeriesMatrix`) that has `value`, `unit`, `name`, and `channel` attributes. However, the Mixin itself does not declare these, leading MyPy to flag them as missing.

**Locations**:

- `gwexpy/types/mixin/signal_interop.py`
- `gwexpy/types/axis_api.py`

**Remedy Strategy**:
Use Python `Protocol` to define the structural subtyping requirements for the Mixin.

```python
from typing import Protocol, Any

class HasSeriesData(Protocol):
    value: Any
    unit: Any
    name: str | None
    channel: Any

class SignalAnalysisMixin:
    def smooth(self: HasSeriesData, ...) -> Any:
        # Now valid
        return self.value
```

### B. Signature Incompatibility (Medium Priority)

**Problem**:
`gwexpy` classes inherit from `numpy.ndarray` via `gwpy`. Mixins often redefine standard methods like `astype`, `reshape`, `min`, `max` to preserve metadata or units. If the signature doesn't exactly match `numpy` (e.g., different argument names or types), MyPy complains.

**Locations**:

- `gwexpy/timeseries/matrix.py`
- `gwexpy/frequencyseries/matrix.py`

**Remedy Strategy**:

1. **Align Signatures**: Whenever possible, match the downstream library's signature exactly, using `*args, **kwargs` to pass through unused arguments if necessary.
2. **`# type: ignore[override]`**: If the deviation is intentional and safe (e.g., narrowing a return type or stricter argument validation), this suppression is acceptable but should be documented.

### C. UFunc & Operator Overloads (Low Priority / Hard)

**Problem**:
`gwexpy/types/metadata.py` attempts to override `__pow__` and `__call__` for custom behavior. NumPy's UFunc typing is extremely complex with many overloads. MyPy cannot match the custom implementation to any of the expected NumPy overloads.

**Locations**:

- `gwexpy/types/metadata.py`

**Remedy Strategy**:
This often requires very precise typing that mimics NumPy's stubs. Given the complexity, suppressing these specific errors with `# type: ignore[override]` or `[misc]` is often the pragmatic choice unless deep changes to the implementation are made.

### D. Dynamic Imports (QtPy) (Quick Fix)

**Problem**:
`qtpy` dynamically imports PyQt5 or PySide2. MyPy can't always statically determine the content of `QtCore`.

**Locations**:

- `gwexpy/gui/nds/nds_thread.py`
- `gwexpy/gui/data_sources.py`

**Remedy Strategy**:
Explicitly cast or import from the underlying library if possible, or use a stub file for `qtpy`. Alternatively, correct the import to `from qtpy import QtCore` and ensure `qtpy` is installed in the MyPy environment.
*Note*: `gwexpy/gui/nds/nds_thread.py` line 24 shows `QtCore.QThread` is not defined, which suggests `QtCore` might not be imported correctly in that file scope.

### E. Spectrogram Specifics (Long Term)

**Problem**:
The `SpectrogramMatrix` relies on a very dynamic composition of function injection (e.g., `times`, `frequencies` properties acting as proxies). This confuses the static analyzer regarding what `self` is.

**Remedy Strategy**:
This requires a structural refactor to use standard composition or inheritance instead of dynamic method injection. This is a Phase 3 task.

## 3. Implementation Roadmap

### Phase 1: Mixin Protocol Implementation (Week 1)

Targeting Category A (Attribute Access).

- Create `gwexpy/types/protocols.py` to house common protocols (`HasValue`, `HasUnit`, `SeriesLike`).
- Update `SignalAnalysisMixin`, `AxisApiMixin`, etc. to allow `self` to be typed via these protocols.
- **Expected Impact**: ~40 errors resolved.

### Phase 2: QtPy and Signature Fixes (Week 1-2)

Targeting Category B and D.

- Fix imports in GUI files.
- Add `# type: ignore[override]` to `matrix.py` files where signature deviation is intentional.
- **Expected Impact**: ~40 errors resolved.

### Phase 3: Metadata Type Relaxation (Week 2)

Targeting Category C.

- Review `metadata.py`. Simplify type hints where they conflict excessively with NumPy, or suppress specific UFunc errors.
- **Expected Impact**: ~60 errors resolved.

## 4. Conclusion

Achieving 100% strict type safety is feasible but requires adopting `Protocol` patterns for Mixins and being pragmatic about NumPy inheritance. The priority should remain on **Category A**, as these represent actual ambiguities in the code structure that confuse both tools and developers.
