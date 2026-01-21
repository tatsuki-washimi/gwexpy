---
name: extend_gwpy
description: GWpy/Astropyのクラスを継承・拡張する際のテンプレートと注意点（__new__処理、Unit伝播、スライス安全化）を提供する
---

# Extend GWpy Class

This skill provides guidelines and patterns for safely inheriting from and extending GWpy/Astropy classes (TimeSeries, FrequencySeries, Spectrogram, etc.).

## Common Pitfalls & Solutions

### 1. `__new__` and Argument Filtering

When extending `__new__` to accept *extra* arguments (e.g. from higher-level APIs like noise generation), you must **filter out** only the arguments that the parent class does NOT accept.

**CRITICAL:** Do NOT remove arguments that the parent class *does* accept (e.g., `df`, `dt`, `unit`).

```python
    def __new__(cls, *args, **kwargs):
        # Good: Filter specific extra params
        kwargs.pop('my_extra_param', None)
        
        # Bad: Filtering 'df' which parent FrequencySeries expects!
        # kwargs.pop('df', None) 
        
        return super().__new__(cls, *args, **kwargs)
```

### 2. Method Overrides & Return Types

When overriding methods that return a new instance (e.g. arithmetic, `spectrogram`), ensure they return **your subclass**, not the parent GWpy class.

```python
    def spectrogram(self, **kwargs):
        # Override to ensure custom Spectrogram class is returned
        from gwexpy.spectrogram import Spectrogram
        return Spectrogram(...)
```

### 3. Mixin Methods & Unit Propagation

When implementing analysis methods (e.g. via Mixins), explicitly set the `unit` of the returned object if the operation changes it (e.g. `radian`, `degree`).

```python
    def radian(self):
        new = self.copy()
        # ... calculate values ...
        new.unit = u.rad  # Explicitly set global unit
        return new
```

### 4. Safe Slicing (`__getitem__`)

GWpy's `Spectrogram` (Array2D) can behave inconsistently when sliced to 1D. It may return a view that acts like a `Series` but retains 2D array behaviors in unexpected ways, causing plotting errors.

**Pattern for `Spectrogram`:**

```python
    def __getitem__(self, item):
        # Safe 1D fallback
        if self.ndim == 1:
            from gwpy.types.series import Series
            return Series.__getitem__(self, item)
        return super().__getitem__(item)

### 5. Multi-dimensional Structure Maintenance (e.g. Field4D)

When dealing with N-dimensional fields where you want to prevent dimension reduction during indexing (e.g., keeping a 4D structure even when selecting a single time/point), convert integer indices to length-1 slices in `__getitem__`.

**Pattern for Structure Maintenance:**

```python
    def _force_ndim_item(self, item):
        # Convert int indices to slice(i, i+1) to keep dimension
        result = []
        for i, idx in enumerate(normalized_item):
            if isinstance(idx, int):
                result.append(slice(idx, idx + 1))
            else:
                result.append(idx)
        return tuple(result)

    def __getitem__(self, item):
        forced_item = self._force_ndim_item(item)
        return self._reconstruct_subclass(super().__getitem__(forced_item))
```

```

## Checklist

- [ ] Does `__new__` pass all required parent arguments?
- [ ] Do returned objects have the correct class (subclass)?
- [ ] Are `unit` and `name` properties propagated or updated correctly?
- [ ] Is `__getitem__` safe for dimensionality reduction?
- [ ] Are `__array_finalize__` hooks handled if adding new persistent attributes?
