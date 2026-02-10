# Introduction to TensorField Class

This tutorial introduces the `TensorField` class in `gwexpy`, which represents tensor-valued fields in 4D spacetime.

## What is TensorField?

`TensorField` is a container for managing tensor-valued physical fields where each tensor component is a `ScalarField`. Tensors are mathematical objects that generalize scalars (rank-0), vectors (rank-1) to higher ranks, commonly appearing in relativity, stress/strain analysis, and electromagnetic theory.

**Key Features:**
- **Arbitrary rank**: Support for rank-2, rank-3, and higher tensors
- **Component indexing**: Access components via tuple indices (e.g., `(0, 0)`, `(1, 2)`)
- **Tensor operations**: Trace, contraction, and other tensor algebra
- **Consistent structure**: All components share identical spacetime axes

**Common Use Cases:**
- **Rank-2 tensors**: Stress tensor, strain tensor, metric tensor, Ricci tensor
- **Electromagnetic tensor**: Field strength tensor F_μν
- **Higher ranks**: Riemann curvature tensor (rank-4), etc.

## Basic Usage

### Creating a Rank-2 TensorField

Let's create a simple 2×2 stress tensor field:

```python
import numpy as np
from astropy import units as u
from gwexpy.fields import ScalarField, TensorField

# Setup spacetime grid
nt, nx, ny, nz = 50, 8, 8, 8
t = np.arange(nt) * 0.02 * u.s
x = np.arange(nx) * 1.0 * u.m
y = np.arange(ny) * 1.0 * u.m
z = np.arange(nz) * 1.0 * u.m

# Create a simple stress tensor σ
# σ_00: Normal stress in x-direction
data_00 = np.ones((nt, nx, ny, nz)) * 100  # 100 Pa
field_00 = ScalarField(
    data_00,
    unit=u.Pa,
    axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'],
    axis0_domain='time',
    space_domain='real'
)

# σ_11: Normal stress in y-direction
data_11 = np.ones((nt, nx, ny, nz)) * 80  # 80 Pa
field_11 = ScalarField(
    data_11,
    unit=u.Pa,
    axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'],
    axis0_domain='time',
    space_domain='real'
)

# σ_01 = σ_10: Shear stress (symmetric)
data_01 = np.sin(2 * np.pi * t.value[:, None, None, None]) * 20
field_01 = ScalarField(
    data_01,
    unit=u.Pa,
    axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'],
    axis0_domain='time',
    space_domain='real'
)

# Create TensorField
# Keys are tuples representing tensor indices
stress_tensor = TensorField({
    (0, 0): field_00,
    (0, 1): field_01,
    (1, 0): field_01,  # Symmetric: σ_10 = σ_01
    (1, 1): field_11,
}, rank=2)

print(f"Tensor rank: {stress_tensor.rank}")
print(f"Components: {list(stress_tensor.keys())}")
```

### Accessing Components

Components are accessed using tuple indices:

```python
# Access diagonal components
sigma_xx = stress_tensor[(0, 0)]
sigma_yy = stress_tensor[(1, 1)]

# Access off-diagonal (shear) component
sigma_xy = stress_tensor[(0, 1)]

print(f"σ_xx shape: {sigma_xx.shape}")
print(f"σ_xy unit: {sigma_xy.unit}")
```

### Computing Tensor Trace

For rank-2 tensors, the trace is the sum of diagonal elements:

```python
# Compute trace: Tr(σ) = σ_00 + σ_11
trace = stress_tensor.trace()

print(f"Trace type: {type(trace)}")  # Returns ScalarField
print(f"Trace shape: {trace.shape}")
print(f"Mean trace value: {np.mean(trace.value):.2f} Pa")
# Expected: ~180 Pa (100 + 80)
```

## Practical Example: Electromagnetic Field Tensor

The electromagnetic field tensor F_μν is a rank-2 antisymmetric tensor that unifies electric and magnetic fields in special relativity.

### Mathematical Background

The field tensor is defined as:

```
F_μν = [ 0    -Ex/c  -Ey/c  -Ez/c ]
       [ Ex/c   0     -Bz    By   ]
       [ Ey/c   Bz     0     -Bx  ]
       [ Ez/c  -By     Bx     0   ]
```

where E is the electric field, B is the magnetic field, and c is the speed of light.

### Implementation

```python
from astropy.constants import c

# Create simplified E and B fields
nt, nx, ny, nz = 100, 12, 12, 12
t = np.linspace(0, 1, nt) * u.s
x = np.linspace(-3, 3, nx) * u.m
y = np.linspace(-3, 3, ny) * u.m
z = np.linspace(-3, 3, nz) * u.m

# Plane wave: E_x and B_y components
omega = 2 * np.pi * 5  # 5 Hz
k = omega / c.value  # Wave number

# E-field in x direction
Ex_data = np.sin(omega * t.value[:, None, None, None]) * np.ones((1, nx, ny, nz))
Ex_field = ScalarField(
    Ex_data,
    unit=u.V/u.m,
    axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'],
    axis0_domain='time',
    space_domain='real'
)

# B-field in y direction (phase-matched with E)
By_data = Ex_data / c.value  # B = E/c for EM wave
By_field = ScalarField(
    By_data,
    unit=u.T,  # Tesla
    axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'],
    axis0_domain='time',
    space_domain='real'
)

# Zero fields for other components
zero_field = ScalarField(
    np.zeros((nt, nx, ny, nz)),
    unit=u.V/u.m,
    axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'],
    axis0_domain='time',
    space_domain='real'
)

# Build F_μν (simplified 2D version: only x and y)
# F_01 = -Ex/c, F_10 = Ex/c (antisymmetric)
# Note: In full 4D spacetime, indices run 0-3 (t, x, y, z)
F_field = TensorField({
    (0, 0): zero_field,
    (0, 1): ScalarField(-Ex_field.value / c.value, unit=u.T,
                        axis0=t, axis1=x, axis2=y, axis3=z,
                        axis_names=['t','x','y','z'],
                        axis0_domain='time', space_domain='real'),
    (1, 0): ScalarField(Ex_field.value / c.value, unit=u.T,
                        axis0=t, axis1=x, axis2=y, axis3=z,
                        axis_names=['t','x','y','z'],
                        axis0_domain='time', space_domain='real'),
    (1, 1): zero_field,
}, rank=2)

print(f"EM tensor rank: {F_field.rank}")
print(f"EM tensor components: {list(F_field.keys())}")

# Check trace (should be zero for EM field tensor)
trace_F = F_field.trace()
print(f"EM tensor trace (should be ~0): {np.max(np.abs(trace_F.value)):.2e}")
```

## Higher-Rank Tensors

`TensorField` supports arbitrary rank. For example, the Riemann curvature tensor in general relativity is rank-4:

```python
# Create a rank-4 tensor component (example)
R_0101 = ScalarField(
    np.random.randn(nt, nx, ny, nz),
    unit=u.dimensionless_unscaled,
    axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'],
    axis0_domain='time',
    space_domain='real'
)

# Partial Riemann tensor (just showing structure)
riemann_partial = TensorField({
    (0, 1, 0, 1): R_0101,
    # ... other components ...
}, rank=4)

print(f"Riemann tensor rank: {riemann_partial.rank}")
```

## Transformations and Operations

Like `ScalarField` and `VectorField`, each tensor component can be transformed independently:

### Time-Frequency FFT

```python
# Transform all components to frequency domain
stress_tensor_freq = TensorField({
    key: field.fft_time()
    for key, field in stress_tensor.items()
}, rank=stress_tensor.rank)

print(f"Frequency domain: {stress_tensor_freq[(0,0)].axis0_domain}")
```

### Spatial FFT

```python
# Transform to k-space
stress_tensor_k = TensorField({
    key: field.fft_space()
    for key, field in stress_tensor.items()
}, rank=stress_tensor.rank)

print(f"K-space: {stress_tensor_k[(0,0)].space_domains}")
```

## Copy and Manipulation

```python
# Create a copy
stress_copy = stress_tensor.copy()

# Modify components
stress_copy[(0, 0)] = stress_copy[(0, 0)] * 1.5  # Increase σ_xx by 50%

# Original is unchanged
print(f"Original σ_xx mean: {np.mean(stress_tensor[(0,0)].value):.2f} Pa")
print(f"Modified σ_xx mean: {np.mean(stress_copy[(0,0)].value):.2f} Pa")
```

## Summary

`TensorField` enables sophisticated tensor calculus in spacetime:

- **Flexible rank**: Support for rank-2, rank-3, and higher tensors
- **Component access**: Tuple-based indexing (e.g., `(0, 1)`)
- **Tensor operations**: Trace, contraction, etc.
- **Transformations**: FFT and signal processing on each component
- **Physical applications**: Stress, strain, EM fields, general relativity

### When to Use Each Field Class

| Field Class | Use Case | Example |
|-------------|----------|---------|
| **ScalarField** | Single-valued fields | Temperature, pressure, potential |
| **VectorField** | Directional fields | Velocity, force, E-field, B-field |
| **TensorField** | Multi-component tensors | Stress, strain, metric, curvature |

### Next Steps

- **VectorField**: [Vector Field Introduction](field_vector_intro.md)
- **ScalarField**: [Scalar Field Introduction](field_scalar_intro.ipynb)
- **Advanced topics**: Tensor contraction, Lorentz transformations, etc.
