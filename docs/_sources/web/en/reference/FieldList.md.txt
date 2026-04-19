# FieldList

<!-- reference-summary:start -->

**Stability:** Stable

## What it is

Use `FieldList` to batch field operations while keeping domain and metadata checks consistent across members.

## Representative Signatures

```python
FieldList([field0, field1, ...])
FieldList.fft_time_all()
```

## Minimal Example

```python
from gwexpy.fields import ScalarField, FieldList
import numpy as np

fields = FieldList([ScalarField(np.random.randn(8, 3, 3, 3)) for _ in range(2)])
fft_fields = fields.fft_time_all()
```

## Related Theory

- [Prerequisites and Conventions](../user_guide/prerequisites_and_conventions.md)
- [Validated Algorithms](../user_guide/validated_algorithms.md)

## Related Tutorials

- [Field API Intro](../user_guide/tutorials/field_scalar_intro.ipynb)
- [Field Advanced Workflow](../user_guide/tutorials/field_advanced_workflow.ipynb)

## API Reference

The detailed generated API continues below on this page.

<!-- reference-summary:end -->


List-like collection of `ScalarField` objects with batch operations.

Provides validation of unit/axis/domain consistency and convenience batch wrappers for field operations.

## Key Features

- **Batch Signal Processing**: `fft_time_all()`, `filter_all()`, `resample_all()`
- **Batch Selection**: `sel_all()`, `isel_all()`
- **Validation**: Ensures all members in the list share the same axis metadata.

## Example

```python
from gwexpy.fields import ScalarField, FieldList
import numpy as np

fields = FieldList([
    ScalarField(np.random.randn(10, 4, 4, 4)),
    ScalarField(np.random.randn(10, 4, 4, 4)),
])

# Batch FFT
fft_fields = fields.fft_time_all()

# Selection on all fields
subset = fields.sel_all(axis1=slice(0, 2))
```
