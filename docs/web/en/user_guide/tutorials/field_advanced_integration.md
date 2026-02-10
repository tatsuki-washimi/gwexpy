# Field API × Advanced Analysis: Integration Workflow

This tutorial demonstrates how to **combine** Field classes (ScalarField, VectorField, TensorField) with advanced analysis methods for end-to-end scientific workflows.

## Why Integration Matters

gwexpy's power comes from **combining** capabilities:
- **Field API**: 4D spacetime structure
- **Advanced methods**: HHT, ML preprocessing, correlation analysis
- **Time series tools**: TimeSeries, Matrix operations

This tutorial shows **complete workflows** rather than isolated examples.

## Workflow 1: Seismic Wave Propagation Analysis

### Scientific Goal

Analyze how seismic noise propagates through a detector array using:
1. ScalarField for spacetime representation
2. Cross-correlation for propagation delays
3. Eigenmode analysis for noise sources

### Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from gwexpy.fields import ScalarField
from gwexpy.timeseries import TimeSeriesMatrix

# Simulation parameters
sample_rate = 256 * u.Hz
duration = 10 * u.s
nx, ny, nz = 20, 20, 1  # 2D spatial grid (z=0 plane)

# Create spatial grid
x = np.linspace(0, 100, nx) * u.m  # 100m × 100m area
y = np.linspace(0, 100, ny) * u.m
z = np.array([0]) * u.m
t = np.arange(int((sample_rate * duration).to_value(u.dimensionless_unscaled))) / sample_rate.value * u.s

nt = len(t)
```

### Step 1: Create ScalarField with Propagating Wave

```python
# Simulate seismic wave from point source
source_pos = (30.0, 50.0)  # (x, y) in meters
source_freq = 5.0  # Hz
v_propagation = 300.0  # m/s

# Initialize field
data = np.zeros((nt, nx, ny, nz))

# Add noise
np.random.seed(42)
data += np.random.randn(nt, nx, ny, nz) * 0.1

# Add propagating wave
for i, xi in enumerate(x.value):
    for j, yj in enumerate(y.value):
        # Distance from source
        r = np.sqrt((xi - source_pos[0])**2 + (yj - source_pos[1])**2)

        # Time delay
        delay = r / v_propagation

        # Amplitude decay (1/r)
        amp = 1.0 / (1.0 + r/10)

        # Signal: delayed sine wave
        signal = amp * np.sin(2 * np.pi * source_freq * (t.value - delay))
        data[:, i, j, 0] += signal

# Create ScalarField
field = ScalarField(
    data,
    unit=u.m/u.s,  # Velocity
    axis0=t,
    axis1=x,
    axis2=y,
    axis3=z,
    axis_names=['t', 'x', 'y', 'z'],
    axis0_domain='time',
    space_domain='real',
    name='Seismic Velocity'
)

print(f"Field shape: {field.shape}")
print(f"Duration: {duration}, Spatial extent: {x[-1]} × {y[-1]}")
```

### Step 2: Extract TimeSeries at Multiple Locations

```python
# Extract time series at 5 spatial points
locations = [
    (10.0*u.m, 50.0*u.m, 0.0*u.m),  # Far west
    (30.0*u.m, 50.0*u.m, 0.0*u.m),  # Source
    (50.0*u.m, 50.0*u.m, 0.0*u.m),  # East
    (70.0*u.m, 50.0*u.m, 0.0*u.m),  # Far east
    (30.0*u.m, 80.0*u.m, 0.0*u.m),  # North
]

timeseries_list = []
for loc in locations:
    # Use .at() method to extract at specific spatial point
    ts = field.at(x=loc[0], y=loc[1], z=loc[2])
    ts.name = f"x={loc[0].value:.0f}m, y={loc[1].value:.0f}m"
    timeseries_list.append(ts)

# Create TimeSeriesMatrix for correlation analysis
from gwexpy.timeseries import TimeSeries
tsm = TimeSeriesMatrix.from_list([ts.squeeze() for ts in timeseries_list])

print(f"Extracted {len(timeseries_list)} time series")
```

### Step 3: Cross-Correlation Analysis

```python
# Compute correlation matrix
corr_matrix = tsm.correlation_matrix()

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Correlation matrix
im = axes[0].imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_title('Spatial Correlation Matrix')
axes[0].set_xlabel('Location Index')
axes[0].set_ylabel('Location Index')
for i in range(len(locations)):
    for j in range(len(locations)):
        text = axes[0].text(j, i, f'{corr_matrix[i,j]:.2f}',
                           ha="center", va="center", fontsize=9)
fig.colorbar(im, ax=axes[0])

# Time series overlay
for i, ts in enumerate(timeseries_list):
    axes[1].plot(ts.times.value, ts.value.squeeze() + i*2,
                label=ts.name, alpha=0.7)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Velocity (offset for clarity)')
axes[1].set_title('Time Series at Different Locations')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Expected Observation:**
- Highest correlation between locations along the wave path
- Time delays visible in time series overlay

### Step 4: PSD Comparison Across Space

```python
# Use ScalarField's PSD method to compute spatially-varying PSD
psd_west = field.psd(point_or_region=(10.0*u.m, 50.0*u.m, 0.0*u.m), fftlength=2)
psd_source = field.psd(point_or_region=(30.0*u.m, 50.0*u.m, 0.0*u.m), fftlength=2)
psd_east = field.psd(point_or_region=(70.0*u.m, 50.0*u.m, 0.0*u.m), fftlength=2)

# Plot
plt.figure(figsize=(10, 6))
plt.loglog(psd_west.frequencies.value, psd_west.value, label='West (10m)', alpha=0.7)
plt.loglog(psd_source.frequencies.value, psd_source.value, label='Source (30m)', linewidth=2)
plt.loglog(psd_east.frequencies.value, psd_east.value, label='East (70m)', alpha=0.7)
plt.axvline(source_freq, color='r', linestyle='--', label=f'Source freq ({source_freq} Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD [(m/s)²/Hz]')
plt.title('Power Spectral Density vs. Location')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.xlim(1, sample_rate.value/2)
plt.tight_layout()
plt.show()
```

### Step 5: Spatial FFT → K-space Analysis

```python
# Transform to k-space (wavenumber domain)
field_k = field.fft_space(axes=['x', 'y'])

print(f"K-space field shape: {field_k.shape}")
print(f"Spatial domains: {field_k.space_domains}")

# Extract k-space magnitude at specific time
t_idx = len(t) // 2  # Middle of time range
kspace_slice = np.abs(field_k[t_idx, :, :, 0].value.squeeze())

# Get k-space axes
kx = field_k._axis1_index.value
ky = field_k._axis2_index.value

# Plot k-space magnitude
plt.figure(figsize=(10, 8))
plt.pcolormesh(kx, ky, kspace_slice.T, shading='auto', cmap='hot')
plt.colorbar(label='|Amplitude| in k-space')
plt.xlabel('kx (rad/m)')
plt.ylabel('ky (rad/m)')
plt.title(f'K-space Magnitude at t={t[t_idx].value:.2f}s')

# Expected wavelength and wavenumber
wavelength = v_propagation / source_freq  # λ = v/f
k_expected = 2 * np.pi / wavelength
circle = plt.Circle((0, 0), k_expected, fill=False, color='cyan',
                    linestyle='--', linewidth=2, label=f'Expected |k|={k_expected:.3f} rad/m')
ax = plt.gca()
ax.add_patch(circle)
plt.legend()
plt.axis('equal')
plt.tight_layout()
plt.show()
```

**Expected Result:**
- Circular pattern in k-space centered at origin
- Radius corresponds to wavelength λ=v/f

## Workflow 2: Multi-Mode Vibration Analysis with HHT

### Scientific Goal

Analyze multi-mode vibrations using:
1. VectorField for 3D displacement
2. HHT for mode decomposition
3. Eigenmode analysis

### Setup

```python
# Create 3-component displacement field
# Simulating two vibration modes: 5 Hz and 12 Hz

mode1_freq = 5.0  # Hz
mode2_freq = 12.0  # Hz

# Mode 1: X-direction dominant
dx_mode1 = np.sin(2 * np.pi * mode1_freq * t.value)[:, None, None, None] * np.ones((1, nx, ny, nz))
dy_mode1 = np.sin(2 * np.pi * mode1_freq * t.value * 0.3)[:, None, None, None] * np.ones((1, nx, ny, nz))

# Mode 2: Y-direction dominant
dx_mode2 = np.sin(2 * np.pi * mode2_freq * t.value * 0.2)[:, None, None, None] * np.ones((1, nx, ny, nz))
dy_mode2 = np.sin(2 * np.pi * mode2_freq * t.value)[:, None, None, None] * np.ones((1, nx, ny, nz))

# Combined + noise
dx_total = (dx_mode1 * 1.0 + dx_mode2 * 0.5 +
            np.random.randn(nt, nx, ny, nz) * 0.1)
dy_total = (dy_mode1 * 0.3 + dy_mode2 * 1.0 +
            np.random.randn(nt, nx, ny, nz) * 0.1)
dz_total = np.random.randn(nt, nx, ny, nz) * 0.05  # Minimal z-motion

# Create ScalarField components
from gwexpy.fields import VectorField

field_dx = ScalarField(dx_total, unit=u.mm, axis0=t, axis1=x, axis2=y, axis3=z,
                      axis_names=['t','x','y','z'], axis0_domain='time', space_domain='real')
field_dy = ScalarField(dy_total, unit=u.mm, axis0=t, axis1=x, axis2=y, axis3=z,
                      axis_names=['t','x','y','z'], axis0_domain='time', space_domain='real')
field_dz = ScalarField(dz_total, unit=u.mm, axis0=t, axis1=x, axis2=y, axis3=z,
                      axis_names=['t','x','y','z'], axis0_domain='time', space_domain='real')

# Create VectorField
displacement = VectorField({'x': field_dx, 'y': field_dy, 'z': field_dz})

print(f"Displacement vector field components: {list(displacement.keys())}")
```

### Apply HHT to Each Component

```python
# Extract time series at center point
center_x = displacement['x'].at(x=50*u.m, y=50*u.m, z=0*u.m).squeeze()
center_y = displacement['y'].at(x=50*u.m, y=50*u.m, z=0*u.m).squeeze()

# Apply EMD to each component
try:
    imfs_x = center_x.emd(method='emd', max_imf=3)
    imfs_y = center_y.emd(method='emd', max_imf=3)

    print(f"X-component IMFs: {list(imfs_x.keys())}")
    print(f"Y-component IMFs: {list(imfs_y.keys())}")

    # Compute instantaneous frequencies
    inst_freq_x_imf0 = imfs_x['IMF0'].instantaneous_frequency()
    inst_freq_y_imf0 = imfs_y['IMF0'].instantaneous_frequency()

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # X-component
    axes[0, 0].plot(center_x.times.value, center_x.value, linewidth=0.5)
    axes[0, 0].set_ylabel('X Displacement (mm)')
    axes[0, 0].set_title('X-Component Time Series')
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(inst_freq_x_imf0.times.value, inst_freq_x_imf0.value)
    axes[1, 0].axhline(mode1_freq, color='r', linestyle='--', label=f'Mode 1 ({mode1_freq} Hz)')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_title('X-Component Instantaneous Frequency')
    axes[1, 0].set_ylim(0, 20)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Y-component
    axes[0, 1].plot(center_y.times.value, center_y.value, linewidth=0.5, color='orange')
    axes[0, 1].set_ylabel('Y Displacement (mm)')
    axes[0, 1].set_title('Y-Component Time Series')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(inst_freq_y_imf0.times.value, inst_freq_y_imf0.value, color='orange')
    axes[1, 1].axhline(mode2_freq, color='r', linestyle='--', label=f'Mode 2 ({mode2_freq} Hz)')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_title('Y-Component Instantaneous Frequency')
    axes[1, 1].set_ylim(0, 20)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

except ImportError:
    print("PyEMD not installed, skipping HHT analysis")
```

**Expected Result:**
- X-component HHT captures ~5 Hz mode
- Y-component HHT captures ~12 Hz mode
- Demonstrates component-wise advanced analysis on VectorField

## Key Integration Patterns

### Pattern 1: Field → TimeSeries → Advanced Method → Back to Field

```python
# Extract → Process → Reconstruct workflow
# 1. Extract TimeSeries from Field
ts = field.at(x=30*u.m, y=50*u.m, z=0*u.m).squeeze()

# 2. Apply advanced method (e.g., whitening)
from gwexpy.signal.preprocessing.whitening import WhiteningModel
whitener = WhiteningModel(fftlength=2, overlap=1)
whitener.fit(ts)
ts_whitened = whitener.transform(ts)

# 3. Reconstruct Field with processed data (conceptual)
# Note: Full reconstruction requires processing all spatial points
print("Pattern 1: Field → TimeSeries → Processing → Result")
```

### Pattern 2: Field FFT → K-space Analysis → Physical Interpretation

```python
# Transform → Analyze → Interpret workflow
# 1. Transform to k-space
field_k = field.fft_space()

# 2. Extract dominant wavenumber
kx_values = field_k._axis1_index.value
ky_values = field_k._axis2_index.value
# Find peak in k-space
kspace_power = np.sum(np.abs(field_k.value)**2, axis=0)  # Sum over time
peak_idx = np.unravel_index(np.argmax(kspace_power), kspace_power.shape[:-1])
peak_kx = kx_values[peak_idx[0]]
peak_ky = kx_values[peak_idx[1]]

# 3. Physical interpretation
wavelength_x = 2 * np.pi / abs(peak_kx) if peak_kx != 0 else np.inf
print(f"Pattern 2: Dominant wavelength ≈ {wavelength_x:.2f} m")
```

### Pattern 3: Multi-Field Correlation

```python
# Compare multiple fields/components
# 1. Extract components
comp_x = displacement['x'].at(x=50*u.m, y=50*u.m, z=0*u.m).squeeze()
comp_y = displacement['y'].at(x=50*u.m, y=50*u.m, z=0*u.m).squeeze()

# 2. Cross-correlation
xcorr = comp_x.xcorr(comp_y, maxlag=0.5)

# 3. Find phase relationship
peak_idx = np.argmax(np.abs(xcorr.value))
lag_at_peak = xcorr.times.value[peak_idx] - xcorr.t0.value
print(f"Pattern 3: Phase lag between X and Y: {lag_at_peak:.3f}s")
```

## Summary: Integration Benefits

| Standalone | Integrated Workflow | Benefit |
|------------|---------------------|---------|
| ScalarField alone | Field + Cross-correlation | Spatial propagation analysis |
| VectorField alone | Vector + HHT per component | Mode-specific decomposition |
| TimeSeries + FFT | Field + Spatial FFT | Wavevector identification |
| Matrix analysis | Field → Matrix → Eigenmodes | Spatial noise modes |

## Best Practices

1. **Start with Field structure** when data is inherently 4D
2. **Extract TimeSeries** for advanced 1D methods (HHT, ML)
3. **Use Matrix operations** for multi-channel correlation
4. **Transform to k-space** for wave propagation analysis
5. **Combine results** across time, frequency, and wavenumber domains

---

**See Also:**
- [ScalarField Tutorial](field_scalar_intro.ipynb)
- [VectorField Tutorial](field_vector_intro.md)
- [HHT Tutorial](advanced_hht.ipynb)
- [Linear Algebra Tutorial](advanced_linear_algebra.md)
- [TimeSeriesMatrix Tutorial](matrix_timeseries.ipynb)
