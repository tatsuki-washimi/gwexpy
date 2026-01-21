# Extend GWpy Class

This skill provides guidelines and patterns for safely inheriting from and extending GWpy/Astropy classes (TimeSeries, FrequencySeries, Spectrogram, etc.).

## Common Pitfalls & Solutions

### 1. `__new__` and Metadata

GWpy uses `__new__` to preserve units and metadata. When subclassing, ensure metadata is propagated.

```python
class MyTimeSeries(TimeSeries):
    def __new__(cls, data, *args, **kwargs):
        obj = super().__new__(cls, data, *args, **kwargs)
        return obj
```

### 2. Unit Preservation

Always ensure that mathematical operations preserve units.

### 3. Slicing Safety

Override `__getitem__` if special metadata needs to be updated upon slicing.
