# Linear Algebra for Gravitational Wave Analysis

This tutorial demonstrates how to use gwexpy's Matrix classes and linear algebra methods for gravitational wave data analysis.

## Why Linear Algebra in GW Analysis?

Multi-channel gravitational wave data naturally forms matrix structures:
- **Correlation matrices**: Identify noise coupling between channels
- **Eigenmode decomposition**: Find principal noise sources
- **Covariance analysis**: Quantify channel relationships
- **Transfer matrices**: Model system responses

## Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from gwexpy.timeseries import TimeSeriesMatrix, TimeSeries
from gwexpy.noise.wave import sine, gaussian

# Generate multi-channel synthetic data
sample_rate = 1024  # Hz
duration = 10  # seconds
n_channels = 6
```

## 1. Correlation Matrix Analysis

### Creating Multi-Channel Data

```python
# Simulate 6 channels with coupled noise
channels = []
base_noise = gaussian(duration=duration, sample_rate=sample_rate, std=1.0)

for i in range(n_channels):
    # Each channel has:
    # 1. Independent noise
    # 2. Coupling to base noise (different strengths)
    coupling = 0.5 ** i  # Exponential decay
    independent = gaussian(duration=duration, sample_rate=sample_rate, std=0.5)

    channel_data = base_noise * coupling + independent
    channel_data.name = f"Channel {i}"
    channels.append(channel_data)

# Create TimeSeriesMatrix
tsm = TimeSeriesMatrix.from_list(channels)
print(f"Matrix shape: {tsm.shape}")  # (6, 1, 10240)
```

### Computing Correlation Matrix

```python
# Compute correlation matrix
corr_matrix = tsm.correlation_matrix()

print(f"Correlation matrix shape: {corr_matrix.shape}")
print(f"Correlation matrix:\n{corr_matrix}")

# Visualize correlation matrix
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient')
plt.title('Channel Correlation Matrix')
plt.xlabel('Channel Index')
plt.ylabel('Channel Index')
for i in range(n_channels):
    for j in range(n_channels):
        text = plt.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)
plt.tight_layout()
plt.show()
```

**Expected Output:**
- Strong correlation (>0.8) between adjacent channels
- Decreasing correlation with distance due to coupling strength decay
- Diagonal = 1.0 (perfect self-correlation)

## 2. Eigenmode Decomposition

### Finding Principal Components

```python
# Compute covariance matrix and eigenvalues
cov_matrix = tsm.covariance_matrix()
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort by eigenvalue magnitude (descending)
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigenvalues (variance explained by each mode):")
for i, ev in enumerate(eigenvalues):
    percent = 100 * ev / eigenvalues.sum()
    print(f"  Mode {i}: {ev:.4f} ({percent:.1f}%)")
```

**Expected Output:**
```
Mode 0: 5.2341 (67.8%)  # Dominant mode (common noise)
Mode 1: 1.4562 (18.9%)  # Second mode
Mode 2: 0.6234 (8.1%)   # Third mode
...
```

### Visualizing Eigenmodes

```python
# Plot first 3 eigenmodes
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i in range(3):
    axes[i].bar(range(n_channels), eigenvectors[:, i])
    axes[i].set_ylabel(f'Mode {i}\nAmplitude')
    axes[i].set_title(f'Eigenmode {i} (λ={eigenvalues[i]:.3f}, '
                     f'{100*eigenvalues[i]/eigenvalues.sum():.1f}% variance)')
    axes[i].grid(True, alpha=0.3)
    axes[i].axhline(0, color='k', linewidth=0.5)

axes[-1].set_xlabel('Channel Index')
axes[-1].set_xticks(range(n_channels))
plt.tight_layout()
plt.show()
```

**Physical Interpretation:**
- **Mode 0**: All channels positive → common noise source
- **Mode 1**: Mixed signs → differential noise
- **Mode 2**: Higher frequency spatial pattern

## 3. Noise Mode Projection

### Project Data onto Eigenmodes

```python
# Project time series onto eigenmodes
mode_timeseries = []

for i in range(3):  # First 3 modes
    # Project: mode_i(t) = Σ_j eigenvector[j,i] * channel[j](t)
    mode_data = np.zeros(tsm.shape[2])
    for j in range(n_channels):
        mode_data += eigenvectors[j, i] * tsm.value[j, 0, :]

    ts = TimeSeries(
        mode_data,
        t0=tsm.t0,
        dt=tsm.dt,
        unit=tsm.units[0, 0],
        name=f'Mode {i}'
    )
    mode_timeseries.append(ts)

# Plot mode time series
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

for i, (ax, ts) in enumerate(zip(axes, mode_timeseries)):
    ax.plot(ts.times.value, ts.value, linewidth=0.5)
    ax.set_ylabel(f'Mode {i}')
    ax.set_title(f'Eigenmode {i} Time Series '
                f'({100*eigenvalues[i]/eigenvalues.sum():.1f}% variance)')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.show()
```

## 4. Dimensionality Reduction

### Reconstruct Data with Reduced Modes

```python
# Reconstruct using only first k modes
k = 2  # Keep only 2 dominant modes

reconstructed = np.zeros_like(tsm.value)
for mode_idx in range(k):
    for ch_idx in range(n_channels):
        reconstructed[ch_idx, 0, :] += (
            eigenvectors[ch_idx, mode_idx] *
            mode_timeseries[mode_idx].value
        )

# Create reconstructed TimeSeriesMatrix
tsm_reconstructed = TimeSeriesMatrix(
    reconstructed,
    t0=tsm.t0,
    dt=tsm.dt,
    channel_names=[f"Ch{i}_recon" for i in range(n_channels)],
    unit=tsm.units[0, 0]
)

# Compare original vs reconstructed for channel 0
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Original
axes[0].plot(tsm.times.value, tsm.value[0, 0, :], linewidth=0.5, label='Original')
axes[0].set_ylabel('Channel 0\nOriginal')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Reconstructed
axes[1].plot(tsm_reconstructed.times.value, tsm_reconstructed.value[0, 0, :],
            linewidth=0.5, color='orange', label=f'Reconstructed ({k} modes)')
axes[1].set_ylabel('Channel 0\nReconstructed')
axes[1].set_xlabel('Time (s)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Compute reconstruction error
error = np.linalg.norm(tsm.value - tsm_reconstructed.value)
total = np.linalg.norm(tsm.value)
print(f"\nReconstruction error (L2 norm): {error/total:.1%}")
print(f"Variance captured: {100*eigenvalues[:k].sum()/eigenvalues.sum():.1f}%")
```

**Expected Output:**
```
Reconstruction error (L2 norm): ~15-20%
Variance captured: ~85-90%
```

## 5. Applications

### A. Noise Subtraction via Eigenmode Filtering

```python
# Remove dominant mode (common noise)
cleaned = tsm.value.copy()
for ch_idx in range(n_channels):
    cleaned[ch_idx, 0, :] -= (
        eigenvectors[ch_idx, 0] * mode_timeseries[0].value
    )

tsm_cleaned = TimeSeriesMatrix(
    cleaned,
    t0=tsm.t0,
    dt=tsm.dt,
    channel_names=[f"Ch{i}_cleaned" for i in range(n_channels)],
    unit=tsm.units[0, 0]
)

# Compare correlation matrices
corr_original = tsm.correlation_matrix()
corr_cleaned = tsm_cleaned.correlation_matrix()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(corr_original, cmap='RdBu_r', vmin=-1, vmax=1)
axes[0].set_title('Original Correlation')
axes[0].set_xlabel('Channel')
axes[0].set_ylabel('Channel')

im1 = axes[1].imshow(corr_cleaned, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_title('After Removing Mode 0')
axes[1].set_xlabel('Channel')

fig.colorbar(im0, ax=axes[0])
fig.colorbar(im1, ax=axes[1])
plt.tight_layout()
plt.show()
```

### B. Modal Contribution to Specific Frequency

```python
# Compute PSD of each mode
mode_psds = []
for ts in mode_timeseries[:3]:
    psd = ts.psd(fftlength=1)
    mode_psds.append(psd)

# Plot
plt.figure(figsize=(10, 6))
for i, psd in enumerate(mode_psds):
    plt.loglog(psd.frequencies.value, psd.value,
              label=f'Mode {i} ({100*eigenvalues[i]/eigenvalues.sum():.1f}%)',
              alpha=0.7)

plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.title('Power Spectral Density of Principal Eigenmodes')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.xlim(1, sample_rate/2)
plt.tight_layout()
plt.show()
```

## Summary

Linear algebra methods in gwexpy enable:

1. **Correlation Analysis**: Identify channel coupling
2. **Eigenmode Decomposition**: Find principal noise sources
3. **Dimensionality Reduction**: Compress data while preserving variance
4. **Noise Subtraction**: Remove common-mode noise
5. **Modal Spectral Analysis**: Understand frequency content of modes

### Key Methods

| Method | Purpose | Output |
|--------|---------|--------|
| `correlation_matrix()` | Channel correlations | n×n matrix |
| `covariance_matrix()` | Variance-covariance | n×n matrix |
| `np.linalg.eigh()` | Eigendecomposition | eigenvalues, eigenvectors |
| Modal projection | Noise mode extraction | Time series per mode |

### Next Steps

- **Field Applications**: Apply to ScalarField for spatial modes
- **Transfer Function Matrices**: System identification
- **Optimal Filtering**: Wiener filtering using correlation structure

---

**See Also:**
- [TimeSeriesMatrix Tutorial](matrix_timeseries.ipynb)
- [Noise Budget Analysis Example](../examples/case_noise_budget.ipynb)
- [Advanced Correlation Methods](advanced_correlation.ipynb)
