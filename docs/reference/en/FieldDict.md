# FieldDict

Dict-like collection of `ScalarField` objects with batch operations.

Validates unit/axis/domain consistency and exposes batch FFT methods on stored `ScalarField` values.

## Example

```python
from gwexpy.fields import ScalarField, FieldDict
import numpy as np

fields = FieldDict({
    "Ex": ScalarField(np.random.randn(10, 4, 4, 4)),
    "Ey": ScalarField(np.random.randn(10, 4, 4, 4)),
})
fft_fields = fields.fft_time_all()
```
