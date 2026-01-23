# TensorField

`TensorField` is a tensor-valued field represented as a collection of `ScalarField` components.

## Overview

`TensorField` extends `FieldDict` to represent physical tensor fields such as stress tensors, strain tensors, or metric perturbations. Components are indexed by tuples of integers, e.g., `(i, j)` for rank-2 tensors.

## Key Features

- **Matrix Operations (Rank-2)**: `@` (matrix multiplication), `det()`, `trace()`, `symmetrize()`
- **Multiplication Support**:
    - `TensorField @ VectorField -> VectorField`
    - `TensorField @ TensorField -> TensorField`
- **Batch Processing**: `fft_time_all()`, `filter_all()`, `resample_all()`, etc.
- **Arithmetic**: Scalar multiplication, addition, and subtraction
- **Visualization**: `plot_components()` (Plot all components in grid)
- **Export**: `to_array()` for NumPy interoperability (exports 6D array for rank-2)

## Basic Usage

```python
from gwexpy.fields import ScalarField, TensorField
import numpy as np
from astropy import units as u

# Create component fields for a stress tensor
f_pa = ScalarField(np.random.randn(100, 4, 4, 4), unit=u.Pa)

# Create 2x2 TensorField
tf = TensorField({
    (0, 0): f_pa, (0, 1): f_pa * 0.1,
    (1, 0): f_pa * 0.1, (1, 1): f_pa * 0.8
}, rank=2)

# Compute trace
tr = tf.trace()  # Returns ScalarField (sum of diagonal)

# Compute determinant
d = tf.det()  # Returns ScalarField with unit Pa²

# Symmetrize
tf_sym = tf.symmetrize()
```

## Matrix-Vector Interaction

`TensorField` can operate on `VectorField` objects:

```python
from gwexpy.fields import VectorField

# Create a velocity vector
v = VectorField({'x': ScalarField(np.ones((100, 4, 4, 4)), unit=u.m/u.s), 
                 'y': ScalarField(np.ones((100, 4, 4, 4)), unit=u.m/u.s)})

# Apply tensor to vector (e.g., stress applied to velocity)
result_v = tf @ v  # Returns VectorField
```

## See Also

- [ScalarField](ScalarField.md) - The underlying scalar field class
- [VectorField](VectorField.md) - For vector-valued fields
- [FieldDict](FieldDict.md) - The base collection class
