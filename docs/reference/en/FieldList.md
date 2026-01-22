# FieldList

List-like collection of `ScalarField` objects with batch operations.

Provides validation of unit/axis/domain consistency and convenience batch FFT wrappers.

## Example

```python
from gwexpy.fields import ScalarField, FieldList
import numpy as np

fields = FieldList([
    ScalarField(np.random.randn(10, 4, 4, 4)),
    ScalarField(np.random.randn(10, 4, 4, 4)),
])
fft_fields = fields.fft_time_all()
```
