# FieldDict

<!-- reference-summary:start -->

**Stability:** Experimental

## What it is

Use `FieldDict` to batch field operations while keeping domain and metadata checks consistent across members.

## Representative Signatures

```python
FieldDict({"ch0": field0, "ch1": field1})
FieldDict.filter_all(*args, **kwargs)
```

## Minimal Example

```python
from gwexpy.fields import ScalarField, FieldDict
import numpy as np

fields = FieldDict({"A": ScalarField(np.random.randn(8, 3, 3, 3))})
subset = fields.isel_all(axis0=slice(0, 4))
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


Dict-like collection of `ScalarField` objects with batch operations.

Validates unit/axis/domain consistency and exposes batch methods on stored `ScalarField` values.

## Key Features

- **Batch Signal Processing**: `fft_time_all()`, `filter_all()`, `resample_all()`
- **Batch Selection**: `sel_all()`, `isel_all()`
- **Arithmetic**: Supports scalar multiplication, addition, and subtraction (e.g., `fields * 2`)
- **Validation**: Ensures all fields in the collection share the same axis metadata.

## Example

```python
from gwexpy.fields import ScalarField, FieldDict
import numpy as np

fields = FieldDict({
    "Ex": ScalarField(np.random.randn(10, 4, 4, 4)),
    "Ey": ScalarField(np.random.randn(10, 4, 4, 4)),
})

# Batch FFT
fft_fields = fields.fft_time_all()

# Scalar arithmetic
scaled_fields = fields * 2.5
```
