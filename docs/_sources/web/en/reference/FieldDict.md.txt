# FieldDict

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
