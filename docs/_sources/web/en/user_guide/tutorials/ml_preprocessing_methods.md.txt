# ML Preprocessing Methods - Individual Techniques

This tutorial explains each machine learning preprocessing method available in gwexpy **individually**, before combining them into a pipeline.

## Overview

Machine learning models for gravitational wave analysis require careful data preprocessing:

1. **Whitening**: Remove colored noise → flat spectrum
2. **Bandpass Filtering**: Extract specific frequency bands
3. **Normalization**: Standardize amplitudes across channels
4. **Segmentation**: Split data into train/validation sets

This tutorial demonstrates each method **separately** so you understand:
- What each method does
- When to use it
- How to configure it
- What to expect as output

## Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from gwexpy.timeseries import TimeSeries
from gwexpy.noise.wave import sine, gaussian
from gwexpy.signal.preprocessing import whitening, standardization

# Create sample data: 60 Hz signal + colored noise
sample_rate = 4096  # Hz
duration = 10  # seconds
t = np.arange(0, duration, 1/sample_rate)

# Signal: 60 Hz sine wave
signal = 2.0 * np.sin(2 * np.pi * 60 * t)

# Colored noise (1/f noise + white noise)
freqs = np.fft.rfftfreq(len(t), 1/sample_rate)
noise_fft = np.random.randn(len(freqs)) + 1j * np.random.randn(len(freqs))
noise_fft[1:] /= np.sqrt(freqs[1:])  # 1/f coloring
noise = np.fft.irfft(noise_fft, len(t)).real * 0.5

# Combined data
data = signal + noise
ts = TimeSeries(data, t0=0, dt=1/sample_rate, unit='strain', name='H1:GDS-CALIB_STRAIN')

print(f"Data length: {len(ts)} samples ({duration}s)")
```

## Method 1: Whitening

### What is Whitening?

**Purpose**: Transform colored noise (frequency-dependent PSD) to white noise (flat PSD)

**Why needed**: ML models often assume stationary, white noise. Gravitational wave data has strong colored noise (1/f, violin modes, etc.)

### How Whitening Works

1. Estimate PSD of the data
2. Compute whitening filter: `H(f) = 1 / √PSD(f)`
3. Apply filter in frequency domain
4. Transform back to time domain

### Implementation

```python
from gwexpy.signal.preprocessing.whitening import WhiteningModel

# Create whitening model
whitening_model = WhiteningModel(
    fftlength=4,  # 4-second segments for PSD estimation
    overlap=2,    # 50% overlap
    method='welch'
)

# Fit the model (estimates PSD)
whitening_model.fit(ts)

# Apply whitening
ts_whitened = whitening_model.transform(ts)

print(f"Whitened data: {ts_whitened.name}")
```

### Visualization: Before and After

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# Time domain - before
axes[0, 0].plot(ts.times.value[:1000], ts.value[:1000], linewidth=0.5)
axes[0, 0].set_title('Original Signal (Time Domain)')
axes[0, 0].set_xlabel('Time (s)')
axes[0, 0].set_ylabel('Strain')
axes[0, 0].grid(True, alpha=0.3)

# Time domain - after
axes[0, 1].plot(ts_whitened.times.value[:1000], ts_whitened.value[:1000], linewidth=0.5, color='orange')
axes[0, 1].set_title('Whitened Signal (Time Domain)')
axes[0, 1].set_xlabel('Time (s)')
axes[0, 1].set_ylabel('Whitened Strain')
axes[0, 1].grid(True, alpha=0.3)

# PSD - before
psd_original = ts.psd(fftlength=4)
axes[1, 0].loglog(psd_original.frequencies.value, psd_original.value)
axes[1, 0].set_title('Original PSD (Colored)')
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('PSD')
axes[1, 0].grid(True, which='both', alpha=0.3)
axes[1, 0].set_xlim(10, 2000)

# PSD - after
psd_whitened = ts_whitened.psd(fftlength=4)
axes[1, 1].loglog(psd_whitened.frequencies.value, psd_whitened.value, color='orange')
axes[1, 1].set_title('Whitened PSD (Flat)')
axes[1, 1].set_xlabel('Frequency (Hz)')
axes[1, 1].set_ylabel('PSD')
axes[1, 1].grid(True, which='both', alpha=0.3)
axes[1, 1].set_xlim(10, 2000)

plt.tight_layout()
plt.show()
```

**Expected Result:**
- **Before**: PSD slopes down at high frequencies (colored noise)
- **After**: PSD approximately flat (white noise)

### When to Use

✅ **Use whitening when:**
- Input data has strong colored noise
- ML model assumes white noise
- You want to emphasize features across all frequencies equally

❌ **Don't use whitening when:**
- You specifically want to preserve spectral shape
- Signal is already white
- You're doing frequency-domain analysis (use PSD normalization instead)

## Method 2: Bandpass Filtering

### What is Bandpass Filtering?

**Purpose**: Extract signal in specific frequency range, reject noise outside that range

**Why needed**: Gravitational wave signals often occupy narrow frequency bands (e.g., 20-500 Hz for CBC). Filtering improves SNR.

### Implementation

```python
# Apply bandpass filter: 50-100 Hz
ts_filtered = ts.bandpass(50, 100, order=8)

print(f"Filtered to {50}-{100} Hz band")
```

### Visualization: Frequency Response

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Time domain comparison
axes[0].plot(ts.times.value[:2048], ts.value[:2048],
            label='Original', alpha=0.6, linewidth=0.5)
axes[0].plot(ts_filtered.times.value[:2048], ts_filtered.value[:2048],
            label='Bandpass (50-100 Hz)', linewidth=0.8)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Strain')
axes[0].set_title('Bandpass Filter Effect (Time Domain)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Frequency domain comparison
psd_filtered = ts_filtered.psd(fftlength=2)

axes[1].loglog(psd_original.frequencies.value, psd_original.value,
              label='Original', alpha=0.6)
axes[1].loglog(psd_filtered.frequencies.value, psd_filtered.value,
              label='Bandpass (50-100 Hz)', linewidth=2)
axes[1].axvspan(50, 100, alpha=0.2, color='green', label='Pass band')
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('PSD')
axes[1].set_title('Bandpass Filter Effect (Frequency Domain)')
axes[1].legend()
axes[1].grid(True, which='both', alpha=0.3)
axes[1].set_xlim(10, 500)

plt.tight_layout()
plt.show()
```

**Expected Result:**
- PSD suppressed outside 50-100 Hz
- Signal preserved inside pass band
- Smooth roll-off at edges (determined by filter order)

### Multiple Bands

```python
# Apply multiple bandpass filters and sum
bands = [(30, 40), (50, 70), (100, 150)]
ts_multiband = None

for f_low, f_high in bands:
    ts_band = ts.bandpass(f_low, f_high, order=6)
    if ts_multiband is None:
        ts_multiband = ts_band
    else:
        ts_multiband = ts_multiband + ts_band

print(f"Applied {len(bands)} bandpass filters")
```

### When to Use

✅ **Use bandpass when:**
- Signal has known frequency range
- You want to reject out-of-band noise
- Preprocessing for ML models with limited frequency range

❌ **Don't use bandpass when:**
- Signal frequency is unknown or broadband
- You need full spectrum information
- Filter ringing artifacts are problematic

## Method 3: Normalization/Standardization

### What is Normalization?

**Purpose**: Scale data to consistent range, remove mean

**Why needed**: ML models (especially neural networks) converge faster with normalized inputs

### Methods Available

1. **Z-score normalization**: `(x - mean) / std`
2. **Robust normalization**: `(x - median) / MAD` (robust to outliers)
3. **Min-Max scaling**: `(x - min) / (max - min)`

### Implementation

```python
from gwexpy.timeseries import TimeSeries

# Z-score normalization
ts_zscore = ts.standardize(method='zscore')

# Robust normalization (uses Median Absolute Deviation)
ts_robust = ts.standardize(method='zscore', robust=True)

print(f"Original: mean={ts.mean():.3f}, std={ts.std():.3f}")
print(f"Z-score:  mean={ts_zscore.mean():.3e}, std={ts_zscore.std():.3f}")
print(f"Robust:   median={np.median(ts_robust.value):.3e}, MAD*1.4826={np.median(np.abs(ts_robust.value - np.median(ts_robust.value)))*1.4826:.3f}")
```

**Expected Output:**
```
Original: mean=0.023, std=0.891
Z-score:  mean≈0.0, std≈1.0
Robust:   median≈0.0, MAD*1.4826≈1.0
```

### Visualization: Distribution Comparison

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Original distribution
axes[0].hist(ts.value, bins=100, alpha=0.7, edgecolor='black')
axes[0].set_title(f'Original\nμ={ts.mean():.2f}, σ={ts.std():.2f}')
axes[0].set_xlabel('Strain')
axes[0].set_ylabel('Count')
axes[0].grid(True, alpha=0.3)

# Z-score distribution
axes[1].hist(ts_zscore.value, bins=100, alpha=0.7, color='orange', edgecolor='black')
axes[1].set_title(f'Z-score Normalized\nμ≈0, σ≈1')
axes[1].set_xlabel('Normalized Strain')
axes[1].set_ylabel('Count')
axes[1].grid(True, alpha=0.3)

# Robust distribution
axes[2].hist(ts_robust.value, bins=100, alpha=0.7, color='green', edgecolor='black')
axes[2].set_title(f'Robust Normalized\nMedian≈0, MAD≈1')
axes[2].set_xlabel('Normalized Strain')
axes[2].set_ylabel('Count')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Comparison: Z-score vs Robust (with outliers)

```python
# Add artificial outliers
ts_with_outliers = ts.copy()
ts_with_outliers.value[1000:1010] = 50  # Large spike

# Normalize both ways
ts_zscore_out = ts_with_outliers.standardize(method='zscore')
ts_robust_out = ts_with_outliers.standardize(method='zscore', robust=True)

# Plot comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

time_window = slice(900, 1100)  # Around outlier
times = ts.times.value[time_window]

axes[0].plot(times, ts_with_outliers.value[time_window])
axes[0].set_title('Original (with outliers)')
axes[0].set_ylabel('Strain')
axes[0].grid(True, alpha=0.3)

axes[1].plot(times, ts_zscore_out.value[time_window], color='orange')
axes[1].set_title('Z-score (sensitive to outliers)')
axes[1].set_ylabel('Normalized')
axes[1].grid(True, alpha=0.3)

axes[2].plot(times, ts_robust_out.value[time_window], color='green')
axes[2].set_title('Robust (resistant to outliers)')
axes[2].set_ylabel('Normalized')
axes[2].set_xlabel('Time (s)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Expected Result:**
- Z-score: Entire signal squashed due to large outlier variance
- Robust: Outlier visible but rest of signal maintains structure

### When to Use

✅ **Use z-score when:**
- Data is approximately Gaussian
- No significant outliers
- Standard ML preprocessing

✅ **Use robust when:**
- Data contains glitches/outliers
- Non-Gaussian noise
- Want to preserve signal structure despite artifacts

## Method 4: Train/Validation Split

### What is Segmentation?

**Purpose**: Divide data into non-overlapping training and validation sets

**Why needed**: ML models need separate data for training and performance evaluation

### Time-Ordered Split

```python
# Split: 80% train, 20% validation
train_fraction = 0.8
split_point = int(len(ts) * train_fraction)

ts_train = ts[:split_point]
ts_valid = ts[split_point:]

print(f"Train: {len(ts_train)} samples ({len(ts_train)/sample_rate:.1f}s)")
print(f"Valid: {len(ts_valid)} samples ({len(ts_valid)/sample_rate:.1f}s)")
```

**Expected Output:**
```
Train: 32768 samples (8.0s)
Valid: 8192 samples (2.0s)
```

### Visualization

```python
plt.figure(figsize=(12, 4))
plt.plot(ts.times.value, ts.value, linewidth=0.5, alpha=0.6, label='Full Data')
plt.axvline(ts_train.times.value[-1], color='r', linestyle='--', linewidth=2,
           label='Train/Valid Split')
plt.axvspan(ts.times.value[0], ts_train.times.value[-1], alpha=0.1, color='blue',
           label='Training Set')
plt.axvspan(ts_train.times.value[-1], ts.times.value[-1], alpha=0.1, color='green',
           label='Validation Set')
plt.xlabel('Time (s)')
plt.ylabel('Strain')
plt.title('Train/Validation Split (Time-Ordered)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### When to Use

✅ **Time-ordered split when:**
- Temporal dependencies exist (autocorrelation)
- Testing future prediction capability
- Standard practice for time series

❌ **Random split when:**
- Data is i.i.d. (independent and identically distributed)
- **Warning**: Usually not appropriate for GW data!

## Summary: Decision Matrix

| Method | Input | Output | When to Use |
|--------|-------|--------|-------------|
| **Whitening** | Colored noise | White noise | Always for ML with GW data |
| **Bandpass** | Broadband signal | Narrow-band signal | Known frequency range |
| **Z-score** | Arbitrary scale | Mean=0, Std=1 | Gaussian data, no outliers |
| **Robust** | With outliers | Median=0, MAD=1 | Glitchy data |
| **Train/Val Split** | Full dataset | Train + Valid sets | All supervised learning |

## Combining Methods

See [ML Preprocessing Pipeline](case_ml_preprocessing.ipynb) for a complete workflow combining these methods in the optimal order.

**Recommended order:**
1. **Bandpass** (if applicable) - Removes out-of-band noise first
2. **Whitening** - Flattens spectrum
3. **Normalization** - Standardizes scale
4. **Split** - Creates train/valid sets

---

**See Also:**
- [Complete ML Pipeline](case_ml_preprocessing.ipynb) - End-to-end workflow
- [Numerical Stability Guide](../numerical_stability.md) - Precision considerations
- [Advanced Correlation](advanced_correlation.ipynb) - Feature engineering
