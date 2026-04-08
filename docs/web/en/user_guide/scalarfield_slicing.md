# ScalarField Slicing Guide

This guide explains how `ScalarField` **always maintains its 4-dimensional structure** during indexing and slicing operations. This behavior differs from standard NumPy or GWpy array slicing and is essential for preserving metadata consistency.

## Why Maintain 4 Dimensions?

A `ScalarField` is not just an array; it is bound to physical domains such as time, frequency, and spatial coordinates (x, y).
Slicing in NumPy typically "squeezes" or reduces the rank of the array. If this happened in `ScalarField`, the relationship between data points and their corresponding physical metadata (e.g., sample rate, grid spacing) would be lost, making subsequent FFTs or domain conversions impossible.

### NumPy vs. GWexpy Behavior Comparison

![4D Slicing comparison between NumPy and GWexpy showing dimension maintenance](/home/washimi/.gemini/antigravity/brain/389da455-0c02-483f-928d-e8f3db2746b8/scalarfield_slicing_4d_maintenance_1775634419729.png)

| Operation | NumPy Behavior | ScalarField Behavior | Reason |
| :--- | :--- | :--- | :--- |
| `field[0]` | Dimension reduced (Rank Loss) | **4D Maintained** | Protects axis metadata |
| `field[:, :, :, 2]` | Becomes 3D | **4D Maintained** | Preserves coordinate mapping |
| `field + 1.0` | Potential overhead | Optimized vector math | Ensures unit consistency |

---

## Advantages of 4D Maintenance

1.  **Metadata Preservation**: Attributes like `axis0_domain` and `space_domain` remain intact. You can call `.fft()` or `.plot_map()` immediately after slicing.
2.  **Safety in Physics Calculations**: Prevents "accidental broadcasting errors" caused by unexpected rank reduction.
3.  **Predictable Pipelines**: Since the shape is always 4D, pipeline components don't need logic to handle variable-rank inputs.

---

## Practical Examples

### 1. Slicing Behavior

```python
from gwexpy.fields import ScalarField
import numpy as np

# (time, freq, x, y) = (100, 50, 10, 10)
field = ScalarField(np.zeros((100, 50, 10, 10)), ...)

# Extract a snapshot at a specific time
snapshot = field[50]
# Shape is (1, 50, 10, 10). Time axis metadata is preserved.

# Extract a spatial plane (x-y plane)
plane = field[:, :, :, 2]
# Shape is (100, 50, 10, 1). Y-axis information is maintained.
```

### 2. Reducing Dimensions (`squeeze`)

If you intentionally need a 1D or 2D array (e.g., for external libraries or simple plotting), explicitly call `.squeeze()`.

```python
# Get time-series at a specific spatial point
point_ts = field[:, 2, 5, 5]      # (100, 1, 1, 1)
actual_ts = point_ts.squeeze()    # (100,) - Compatible with TimeSeries
```

### 3. Broadcasting Considerations

Because `ScalarField` is always 4D, you must reshape NumPy arrays when performing arithmetic operations.

```python
# ❌ Bad: Adding a 1D array directly
field + np.array([1, 2, 3])  # Shape mismatch

# ✅ Good: Reshape to match the 4D target
calibration = np.array([1, 2, 3]).reshape(3, 1, 1, 1) # (freq, 1, 1, 1)
field + calibration
```

---

## FAQ

### Q: Isn't always having 4D inconvenient for 1D calculations?
**A:** `ScalarField` is designed for "fields" that have extent in space and time. For simple, single-channel time-series analysis, we recommend using the `TimeSeries` class.

### Q: What if I index every dimension, like `ScalarField[0, 0, 0, 0]`?
**A:** If all indices are scalars, a standard Python or NumPy scalar value is returned.

## Related Links

- {doc}`tutorials/field_scalar_intro` - Introduction to ScalarField
- {doc}`../reference/api/field` - Field Module API Reference
- {doc}`numerical_stability` - Numerical stability in 4D operations
