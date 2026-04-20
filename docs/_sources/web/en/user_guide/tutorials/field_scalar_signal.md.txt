# Field API: ScalarField Signal Processing

This tutorial explains advanced signal processing techniques using the `ScalarField` class, focusing on domain transformations between spatial and wavenumber (K) domains, and data extraction at specific coordinates.

## 1. Spatial FFT

`ScalarField` supports FFT not only on the time axis (Axis 0) but also on the spatial axes (Axes 1, 2, 3). This allows for the analysis of wave packets and mode structures in physical fields.

```python
# Transform all spatial axes to K-domain using FFT
field_k = field.fft_space()

print(f"Space domains: {field_k.space_domains}")  # {'kx': 'k', 'ky': 'k', 'kz': 'k'}
```

## 2. Wavelength and Wavenumber Analysis

In the K-domain, you can retrieve wavenumbers for each axis and calculate the wavelength corresponding to specific wavenumbers.

```python
from astropy import units as u

# Get wavelengths for a specific spatial axis
wavelengths = field_k.wavelength("kx")
```

## 3. Point Extraction (Time-series at specific coordinates)

You can extract one or more `TimeSeries` objects at arbitrary spatial coordinates $(x, y, z)$ from the 4D field. If the specified coordinates are not on the grid, interpolation is automatically performed.

```python
# Extract a time-series at coordinates (0.5 m, 0.5 m, 0.5 m)
ts_list = field.extract_points([(0.5 * u.m, 0.5 * u.m, 0.5 * u.m)])
ts_at_point = ts_list[0]

ts_at_point.plot()
```

## 4. Slice Extraction (2D Planes)

It is easy to extract 2D cross-sections (Plane2D) by fixing specific axes.

```python
# Extract the plane at z=0
plane_z0 = field.sel(z=0 * u.m)
plane_z0.plot()
```

---

For more details, please refer to [ScalarField Basics](field_scalar_intro.ipynb).
