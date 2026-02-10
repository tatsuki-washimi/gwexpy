# Introduction to VectorField Class

This tutorial introduces the `VectorField` class in `gwexpy`, which represents vector-valued fields in 4D spacetime.

## What is VectorField?

`VectorField` is a specialized container that manages multiple `ScalarField` components as a single vector-valued physical field. Each component (e.g., x, y, z) is a full `ScalarField` with its own spacetime structure, but the `VectorField` ensures all components share identical axes and metadata.

**Key Features:**
- **Component-wise operations**: Apply FFT, filtering, and signal processing to each vector component independently
- **Geometrical consistency**: All components guaranteed to have the same axis structure
- **Vector algebra**: Compute norm, dot products, and other vector operations
- **Flexible basis**: Support for Cartesian and custom coordinate systems

**When to use VectorField:**
- Electromagnetic fields (E-field, B-field)
- Velocity or acceleration fields
- Force or displacement fields
- Any physical quantity with directional components in spacetime

## Basic Usage

### Creating a VectorField

A `VectorField` is constructed from a dictionary of `ScalarField` components:

```python
import numpy as np
from astropy import units as u
from gwexpy.fields import ScalarField, VectorField

# Create sample 4D data for each component
nt, nx, ny, nz = 100, 8, 8, 8
t = np.arange(nt) * 0.01 * u.s
x = np.arange(nx) * 0.5 * u.m
y = np.arange(ny) * 0.5 * u.m
z = np.arange(nz) * 0.5 * u.m

# X component: wave propagating in +x direction
data_x = np.sin(2 * np.pi * (5 * t.value[:, None, None, None] - x.value[None, :, None, None]))
field_x = ScalarField(
    data_x,
    unit=u.V/u.m,
    axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'],
    axis0_domain='time',
    space_domain='real'
)

# Y component: constant background
data_y = np.ones((nt, nx, ny, nz)) * 0.1
field_y = ScalarField(
    data_y,
    unit=u.V/u.m,
    axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'],
    axis0_domain='time',
    space_domain='real'
)

# Z component: small noise
data_z = np.random.randn(nt, nx, ny, nz) * 0.05
field_z = ScalarField(
    data_z,
    unit=u.V/u.m,
    axis0=t, axis1=x, axis2=y, axis3=z,
    axis_names=['t', 'x', 'y', 'z'],
    axis0_domain='time',
    space_domain='real'
)

# Create VectorField
vec_field = VectorField({'x': field_x, 'y': field_y, 'z': field_z})

print(f"VectorField components: {list(vec_field.keys())}")
print(f"Basis: {vec_field.basis}")
```

### Accessing Components

Individual components can be accessed like a dictionary:

```python
# Access individual components
Ex = vec_field['x']
Ey = vec_field['y']
Ez = vec_field['z']

print(f"X component shape: {Ex.shape}")
print(f"Y component shape: {Ey.shape}")
```

### Computing Vector Magnitude

The `norm()` method computes the L2 norm (magnitude) at each spacetime point:

```python
# Compute magnitude: |E| = √(Ex² + Ey² + Ez²)
magnitude = vec_field.norm()

print(f"Magnitude type: {type(magnitude)}")  # Returns a ScalarField
print(f"Magnitude shape: {magnitude.shape}")
print(f"Magnitude unit: {magnitude.unit}")
```

## Transformations

Since each component is a `ScalarField`, you can apply FFT and other transformations to the entire vector field:

### Time-Frequency Transform

```python
# Transform all components to frequency domain
vec_field_freq = VectorField({
    key: field.fft_time()
    for key, field in vec_field.items()
})

print(f"Frequency domain: {vec_field_freq['x'].axis0_domain}")
```

### Spatial FFT (Real → K-space)

```python
# Transform all components to wavenumber space
vec_field_k = VectorField({
    key: field.fft_space()
    for key, field in vec_field.items()
})

print(f"K-space domains: {vec_field_k['x'].space_domains}")
```

## Converting to Array

For numerical analysis, you can convert the VectorField to a single 5D NumPy array:

```python
# Convert to 5D array: (time, x, y, z, components)
array_5d = vec_field.to_array()

print(f"Array shape: {array_5d.shape}")
# Expected: (100, 8, 8, 8, 3) for 3 components
```

The last axis (axis 4) contains the vector components in the order they were defined.

## Practical Example: Electromagnetic Field

Here's a complete example simulating a simple electromagnetic field:

```python
import matplotlib.pyplot as plt

# Create a localized wave packet in E-field
nt, nx, ny, nz = 200, 16, 16, 16
t = np.linspace(0, 1, nt) * u.s
x = np.linspace(-5, 5, nx) * u.m
y = np.linspace(-5, 5, ny) * u.m
z = np.linspace(-5, 5, nz) * u.m

# Gaussian envelope in space, oscillating in time
X, Y, Z = np.meshgrid(x.value, y.value, z.value, indexing='ij')
r = np.sqrt(X**2 + Y**2 + Z**2)
envelope = np.exp(-r**2 / 2)

# Oscillating E-field components
omega = 2 * np.pi * 10  # 10 Hz
Ex_data = envelope[None, :, :, :] * np.sin(omega * t.value[:, None, None, None])
Ey_data = envelope[None, :, :, :] * np.cos(omega * t.value[:, None, None, None]) * 0.5
Ez_data = np.zeros((nt, nx, ny, nz))

# Create ScalarField components
Ex_field = ScalarField(Ex_data, unit=u.V/u.m, axis0=t, axis1=x, axis2=y, axis3=z,
                       axis_names=['t','x','y','z'], axis0_domain='time', space_domain='real')
Ey_field = ScalarField(Ey_data, unit=u.V/u.m, axis0=t, axis1=x, axis2=y, axis3=z,
                       axis_names=['t','x','y','z'], axis0_domain='time', space_domain='real')
Ez_field = ScalarField(Ez_data, unit=u.V/u.m, axis0=t, axis1=x, axis2=y, axis3=z,
                       axis_names=['t','x','y','z'], axis0_domain='time', space_domain='real')

# Create E-field vector
E_field = VectorField({'Ex': Ex_field, 'Ey': Ey_field, 'Ez': Ez_field})

# Compute field magnitude
E_magnitude = E_field.norm()

# Visualize magnitude at a specific time slice (t=0.5s)
time_idx = 100  # Middle of time range
magnitude_slice = E_magnitude[time_idx, :, :, nz//2].value.squeeze()

plt.figure(figsize=(8, 6))
plt.imshow(magnitude_slice.T, origin='lower', extent=[-5, 5, -5, 5], cmap='viridis')
plt.colorbar(label='|E| [V/m]')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Electric Field Magnitude (t=0.5s, z=0 plane)')
plt.show()
```

## Summary

`VectorField` provides a powerful abstraction for vector-valued physical fields:

- **Creation**: Build from ScalarField components
- **Access**: Dictionary-like component access
- **Operations**: Norm, component-wise FFT, etc.
- **Consistency**: Automatic validation of axes and metadata
- **Flexibility**: Works with any coordinate basis

### Next Steps

- **ScalarField basics**: [Scalar Field Introduction](field_scalar_intro.ipynb)
- **TensorField**: [Tensor Field Introduction](field_tensor_intro.md) for rank-2 and higher tensors
- **Advanced signal processing**: Apply PSD, coherence, and other methods to vector components
