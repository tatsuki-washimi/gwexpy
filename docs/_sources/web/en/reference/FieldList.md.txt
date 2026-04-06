# FieldList

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
