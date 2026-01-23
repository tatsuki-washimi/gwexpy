# VectorField

`VectorField` is a vector-valued field represented as a collection of `ScalarField` components.

## Overview

`VectorField` extends `FieldDict` to provide specialized operations for vector fields in physics, such as electromagnetic fields, velocity fields, or displacement fields.

## Key Features

- **Geometric Operations**: `dot()`, `cross()`, `project()`, `norm()`
- **Batch Processing**: `fft_time_all()`, `filter_all()`, `resample_all()`, etc.
- **Arithmetic**: Scalar multiplication, addition, and subtraction
- **Visualization**: `plot()` (Magnitude+Quiver), `quiver()` (arrow plot), `streamline()` (streamline plot)
- **Export**: `to_array()` for NumPy interoperability

## Basic Usage

```python
from gwexpy.fields import ScalarField, VectorField
import numpy as np
from astropy import units as u

# Create component fields
fx = ScalarField(np.random.randn(100, 4, 4, 4), unit=u.V)
fy = ScalarField(np.random.randn(100, 4, 4, 4), unit=u.V)
fz = ScalarField(np.random.randn(100, 4, 4, 4), unit=u.V)

# Create VectorField
vf = VectorField({'x': fx, 'y': fy, 'z': fz})

# Compute magnitude
magnitude = vf.norm()  # Returns ScalarField

# Dot product
dot_result = vf.dot(vf)  # Returns ScalarField with unit V²

# Cross product (3-component vectors only)
v1 = VectorField({'x': fx, 'y': fy, 'z': fz})
v2 = VectorField({'x': fy, 'y': fz, 'z': fx})
cross_result = v1.cross(v2)  # Returns VectorField
```

## Batch Operations

All `ScalarField` operations can be applied to all components at once:

```python
# FFT all components
vf_freq = vf.fft_time_all()

# Filter all components
from gwpy.signal import filter_design
lp = filter_design.lowpass(100, 1000)
vf_filtered = vf.filter_all(lp)

# Resample all components
vf_resampled = vf.resample_all(50)  # 50 Hz
```

## See Also

- [ScalarField](ScalarField.md) - The underlying scalar field class
- [TensorField](TensorField.md) - For tensor-valued fields
- [FieldDict](FieldDict.md) - The base collection class
