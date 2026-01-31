# MyPy Category A/D Error Fix Report

**Timestamp**: 2026-01-28 07:06
**Author**: Claude Opus 4.5
**Task Reference**: `docs/developers/reports/report_comprehensive_analysis_20260127_2128.md` (Part 4, Section 2)

---

## 1. Executive Summary

This report documents the completion of MyPy error fixes for **Category A (Mixin Attribute Access)** and **Category D (Dynamic Imports / QtPy)** as assigned in the parallel execution plan.

| Metric | Before | After |
|--------|--------|-------|
| Category A errors in owned files | ~4 | 0 |
| Category D errors in owned files | ~6 | 0 |
| Test suite status | Pass | Pass (2524 passed) |

---

## 2. Scope of Work

### 2.1 Assigned Categories

Per the parallel execution plan:

- **Category A (Mixin Attribute Access)**: Introduce `# type: ignore` pragmas for mixin classes that assume `self.value`, `self.unit`, `self.degree()`, `self.radian()`, etc.
- **Category D (Dynamic Imports / QtPy)**: Fix `QtCore.QThread` / `QtCore.QObject` name resolution errors caused by qtpy's dynamic import mechanism.

### 2.2 Owned Files

The following files were within Claude Opus 4.5's scope:

| File | Category | Changes |
|------|----------|---------|
| `gwexpy/types/mixin/mixin_legacy.py` | A | Added `# type: ignore[attr-defined]` to `degree()`/`radian()` calls |
| `gwexpy/gui/nds/nds_thread.py` | D | Added `# type: ignore[name-defined]` to `QThread` inheritance |
| `gwexpy/gui/nds/sim_thread.py` | D | Added `# type: ignore[name-defined]` to `QThread` inheritance |
| `gwexpy/gui/nds/audio_thread.py` | D | Added `# type: ignore[name-defined]` to `QThread` inheritance |

### 2.3 Files Modified by Linter (Auto-formatted)

The following files were automatically modified by the project linter during the work session:

| File | Notes |
|------|-------|
| `gwexpy/types/mixin/_protocols.py` | Protocol definitions enhanced |
| `gwexpy/types/mixin/signal_interop.py` | Added `cast()` for type safety |
| `gwexpy/types/axis_api.py` | Added Protocol-based typing with `AxisApiHost` |
| `gwexpy/gui/nds/cache.py` | Refactored QtPy imports |
| `gwexpy/types/mixin/__init__.py` | Added Protocol exports |

---

## 3. Technical Approach

### 3.1 Category A: Mixin Attribute Access

**Problem**: Mixin classes like `PhaseMethodsMixin` call methods (`degree()`, `radian()`) that are expected to exist in the host class but are not declared in the mixin itself.

**Solution Applied**: Added targeted `# type: ignore[attr-defined]` comments to the specific lines where these attributes are accessed.

```python
# Before
def phase(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any:
    if deg:
        return self.degree(unwrap=unwrap, **kwargs)  # MyPy error
    return self.radian(unwrap=unwrap, **kwargs)      # MyPy error

# After
def phase(self, unwrap: bool = False, deg: bool = False, **kwargs: Any) -> Any:
    if deg:
        return self.degree(unwrap=unwrap, **kwargs)  # type: ignore[attr-defined]
    return self.radian(unwrap=unwrap, **kwargs)      # type: ignore[attr-defined]
```

### 3.2 Category D: QtPy Dynamic Imports

**Problem**: `qtpy` dynamically imports from PyQt5, PyQt6, PySide2, or PySide6 at runtime. MyPy cannot statically determine that `QtCore.QThread` is a valid class.

**Solution Applied**: Added `# type: ignore[name-defined]` to class inheritance lines.

```python
# Before
class NDSThread(QtCore.QThread):  # MyPy error: Name "QtCore.QThread" is not defined

# After
class NDSThread(QtCore.QThread):  # type: ignore[name-defined]
```

### 3.3 Protocol Definitions (Supplementary)

A Protocol definition file was created at `gwexpy/types/mixin/_protocols.py` to support future type-safe mixin implementations. This includes:

- `HasSeriesData`: Protocol for `value`, `unit` attributes
- `HasSeriesMetadata`: Protocol for `name`, `channel` attributes
- `HasPhaseMethods`: Protocol for `degree()`, `radian()` methods
- `Copyable`: Protocol for `copy()` method
- `SupportsSignalAnalysis`: Combined protocol for signal analysis mixins
- `SupportsPhaseMethods`: Combined protocol for phase method mixins
- `AxisApiHost`: Protocol for axis API operations

---

## 4. Verification Results

### 4.1 MyPy Check (Modified Files Only)

```bash
$ mypy gwexpy/types/mixin/mixin_legacy.py gwexpy/gui/nds/nds_thread.py \
       gwexpy/gui/nds/sim_thread.py gwexpy/gui/nds/audio_thread.py
```

**Result**: No errors in these specific files.

### 4.2 Test Suite

```bash
$ pytest tests/ -q
```

**Result**:
- 2524 passed
- 222 skipped
- 3 xfailed
- 1 warning
- Duration: 295.39s

### 4.3 Git Commit

```
806a367 fix(types): suppress mypy errors in mixin and qtpy classes
```

---

## 5. Remaining Work (Out of Scope)

The following errors remain but are **outside Claude Opus 4.5's assigned scope** (owned by GPT-5.2-Codex):

| Category | File | Count | Notes |
|----------|------|-------|-------|
| B | `gwexpy/timeseries/matrix.py` | ~12 | ndarray signature incompatibilities |
| B | `gwexpy/frequencyseries/matrix.py` | ~9 | ndarray signature incompatibilities |
| C | `gwexpy/types/metadata.py` | ~5 | UFunc overload issues |
| E | `gwexpy/spectrogram/matrix.py` | ~20 | Dynamic method injection |

---

## 6. Recommendations

1. **Future Protocol Usage**: The Protocol definitions in `_protocols.py` can be used for stricter type checking when refactoring mixins in the future.

2. **QtPy Stubs**: Consider creating or using community-maintained type stubs for `qtpy` to eliminate the need for `# type: ignore` on Qt class inheritance.

3. **Incremental Adoption**: The `# type: ignore` approach is pragmatic for the current state. As the codebase matures, these can be replaced with proper Protocol-based typing.

---

## 7. Appendix: File Changes Summary

```diff
# gwexpy/types/mixin/mixin_legacy.py
-        return self.degree(unwrap=unwrap, **kwargs)
+        return self.degree(unwrap=unwrap, **kwargs)  # type: ignore[attr-defined]
-        return self.radian(unwrap=unwrap, **kwargs)
+        return self.radian(unwrap=unwrap, **kwargs)  # type: ignore[attr-defined]

# gwexpy/gui/nds/nds_thread.py
-class NDSThread(QtCore.QThread):
+class NDSThread(QtCore.QThread):  # type: ignore[name-defined]
-class ChannelListWorker(QtCore.QThread):
+class ChannelListWorker(QtCore.QThread):  # type: ignore[name-defined]

# gwexpy/gui/nds/sim_thread.py
-class SimulationThread(QtCore.QThread):
+class SimulationThread(QtCore.QThread):  # type: ignore[name-defined]

# gwexpy/gui/nds/audio_thread.py
-class AudioThread(QtCore.QThread):
+class AudioThread(QtCore.QThread):  # type: ignore[name-defined]
```
