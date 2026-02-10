# Time-Frequency Methods: Comparison and Selection Guide

This tutorial compares different time-frequency analysis methods in gwexpy and helps you choose the right one for your analysis.

## Overview

Gravitational wave signals often have **time-varying frequency content**. Different methods reveal different aspects:

| Method | Best For | Time Resolution | Frequency Resolution |
|--------|----------|-----------------|---------------------|
| **Spectrogram** | General purpose | Good | Good |
| **Q-transform** | Chirps, transients | Adaptive | Adaptive (constant Q) |
| **HHT** | Non-stationary, nonlinear | Excellent | Data-adaptive |
| **STFT** | Stationary segments | Fixed (window) | Fixed (window) |
| **Wavelet** | Multi-scale features | Scale-dependent | Scale-dependent |

## Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from gwexpy.timeseries import TimeSeries
from gwexpy.noise.wave import chirp, gaussian

# Create test signal: chirp + noise
sample_rate = 1024  # Hz
duration = 4  # seconds

# Chirp from 20 Hz to 100 Hz
signal_chirp = chirp(
    duration=duration,
    sample_rate=sample_rate,
    f0=20,  # Start frequency
    f1=100,  # End frequency
    t1=duration
)

# Add Gaussian noise
noise = gaussian(duration=duration, sample_rate=sample_rate, std=0.2)
data = signal_chirp + noise

ts = TimeSeries(data, t0=0, dt=1/sample_rate, unit='strain')
print(f"Data: {len(ts)} samples, {duration}s")
```

## Method 1: Spectrogram (STFT-based)

### What it is

**Short-Time Fourier Transform (STFT)**: Divide signal into windows, compute FFT for each

**Formula**: `S(t, f) = |∫ x(τ) w(τ-t) e^(-2πifτ) dτ|²`

### Implementation

```python
# Create spectrogram with 0.5s windows
spec = ts.spectrogram(fftlength=0.5, overlap=0.25)

print(f"Spectrogram shape: {spec.shape}")  # (time_bins, freq_bins)
print(f"Time resolution: {spec.dt}")
print(f"Frequency resolution: {spec.df}")
```

### Visualization

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Spectrogram
im = axes[0].pcolormesh(
    spec.times.value,
    spec.frequencies.value,
    spec.value.T,
    cmap='viridis',
    shading='auto'
)
axes[0].set_ylim(10, 200)
axes[0].set_ylabel('Frequency (Hz)')
axes[0].set_title('Spectrogram (STFT, window=0.5s)')
fig.colorbar(im, ax=axes[0], label='Power')

# Overlay true chirp frequency
t_true = np.linspace(0, duration, 100)
f_true = 20 + (100 - 20) * t_true / duration
axes[0].plot(t_true, f_true, 'r--', linewidth=2, label='True Frequency')
axes[0].legend()

# Time series for reference
axes[1].plot(ts.times.value, ts.value, linewidth=0.5)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Strain')
axes[1].set_title('Original Signal')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Pros and Cons

✅ **Advantages:**
- Fast computation
- Well-understood, standard method
- Good for stationary or slowly-varying signals

❌ **Disadvantages:**
- Fixed time-frequency resolution (uncertainty principle)
- Poor for rapid frequency changes
- Window length affects both time and frequency resolution

### When to Use

✅ **Use Spectrogram when:**
- Signal is quasi-stationary
- You need standard, well-established method
- Fast computation is important
- Frequency changes are slow relative to window size

## Method 2: Q-transform

### What it is

**Constant-Q transform**: Adaptive time-frequency tiling with constant Q-factor

**Q-factor**: `Q = f / Δf` (ratio of center frequency to bandwidth)

### Implementation

```python
# Q-transform with Q=6
q = 6
qgram = ts.q_transform(qrange=(4, 64), frange=(10, 200))

print(f"Q-transform shape: {qgram.shape}")
```

### Visualization

```python
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Q-transform
axes[0].imshow(
    qgram.value.T,
    extent=[qgram.times.value[0], qgram.times.value[-1],
            qgram.frequencies.value[0], qgram.frequencies.value[-1]],
    aspect='auto',
    origin='lower',
    cmap='viridis',
    interpolation='bilinear'
)
axes[0].set_ylabel('Frequency (Hz)')
axes[0].set_title(f'Q-transform (constant Q)')
axes[0].plot(t_true, f_true, 'r--', linewidth=2, label='True Frequency')
axes[0].legend()

# Spectrogram for comparison
im = axes[1].pcolormesh(
    spec.times.value,
    spec.frequencies.value,
    spec.value.T,
    cmap='viridis',
    shading='auto'
)
axes[1].set_ylim(10, 200)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_title('Spectrogram (for comparison)')
axes[1].plot(t_true, f_true, 'r--', linewidth=2, label='True Frequency')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Pros and Cons

✅ **Advantages:**
- Adaptive resolution: better time resolution at high frequencies
- Natural for chirps (binary coalescence)
- Constant Q matches gravitational wave signals

❌ **Disadvantages:**
- Slower than STFT
- More complex interpretation
- Requires Q-factor tuning

### When to Use

✅ **Use Q-transform when:**
- Analyzing chirps (compact binary coalescence)
- Signal spans wide frequency range
- High-frequency transients need good time resolution
- Standard GW transient analysis

## Method 3: Hilbert-Huang Transform (HHT)

### What it is

**Empirical Mode Decomposition (EMD) + Hilbert Transform**:
1. Decompose signal into Intrinsic Mode Functions (IMFs)
2. Compute instantaneous frequency via Hilbert transform

### Implementation

```python
# Perform EMD
imfs = ts.emd(method='emd', max_imf=5)

print(f"Extracted {len(imfs)} IMFs")

# Plot IMFs
fig, axes = plt.subplots(len(imfs), 1, figsize=(12, 10), sharex=True)

for i, (name, imf) in enumerate(imfs.items()):
    axes[i].plot(imf.times.value, imf.value, linewidth=0.5)
    axes[i].set_ylabel(name)
    axes[i].grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (s)')
axes[0].set_title('Empirical Mode Decomposition (EMD)')
plt.tight_layout()
plt.show()
```

### Instantaneous Frequency

```python
# Compute instantaneous frequency for dominant IMF
imf_main = imfs['IMF0']
inst_freq = imf_main.instantaneous_frequency()

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(imf_main.times.value, imf_main.value, linewidth=0.5)
axes[0].set_ylabel('IMF0 Amplitude')
axes[0].set_title('Dominant Intrinsic Mode Function')
axes[0].grid(True, alpha=0.3)

axes[1].plot(inst_freq.times.value, inst_freq.value, linewidth=1)
axes[1].plot(t_true, f_true, 'r--', linewidth=2, label='True Frequency')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Time (s)')
axes[1].set_title('Instantaneous Frequency (HHT)')
axes[1].set_ylim(0, 200)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Pros and Cons

✅ **Advantages:**
- Data-adaptive (no window choice)
- Excellent time resolution
- Handles nonlinear, non-stationary signals
- Direct instantaneous frequency

❌ **Disadvantages:**
- Computationally expensive
- EMD can have mode mixing
- Less established than STFT/Q-transform

### When to Use

✅ **Use HHT when:**
- Signal is highly non-stationary
- Need precise instantaneous frequency
- Standard methods fail to resolve features
- Analyzing glitches or complex transients

❌ **Don't use when:**
- Signal is stationary (overkill)
- Need fast computation
- Standard spectrograms suffice

## Comparison Example: All Methods on Same Signal

```python
fig = plt.figure(figsize=(14, 10))

# Original signal
ax1 = plt.subplot(4, 1, 1)
ax1.plot(ts.times.value, ts.value, linewidth=0.5, color='black')
ax1.set_ylabel('Strain')
ax1.set_title('Original Signal: Chirp (20→100 Hz) + Noise')
ax1.grid(True, alpha=0.3)

# Spectrogram
ax2 = plt.subplot(4, 1, 2, sharex=ax1)
im2 = ax2.pcolormesh(spec.times.value, spec.frequencies.value, spec.value.T,
                     cmap='viridis', shading='auto')
ax2.plot(t_true, f_true, 'r--', linewidth=1.5, alpha=0.8)
ax2.set_ylim(10, 150)
ax2.set_ylabel('Frequency (Hz)')
ax2.set_title('Spectrogram (window=0.5s)')

# Q-transform
ax3 = plt.subplot(4, 1, 3, sharex=ax1)
ax3.imshow(qgram.value.T,
          extent=[qgram.times.value[0], qgram.times.value[-1],
                  qgram.frequencies.value[0], qgram.frequencies.value[-1]],
          aspect='auto', origin='lower', cmap='viridis', interpolation='bilinear')
ax3.plot(t_true, f_true, 'r--', linewidth=1.5, alpha=0.8)
ax3.set_ylim(10, 150)
ax3.set_ylabel('Frequency (Hz)')
ax3.set_title('Q-transform')

# HHT instantaneous frequency
ax4 = plt.subplot(4, 1, 4, sharex=ax1)
ax4.plot(inst_freq.times.value, inst_freq.value, linewidth=1, label='HHT Inst. Freq')
ax4.plot(t_true, f_true, 'r--', linewidth=2, label='True Frequency')
ax4.set_ylim(0, 150)
ax4.set_ylabel('Frequency (Hz)')
ax4.set_xlabel('Time (s)')
ax4.set_title('HHT Instantaneous Frequency')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Decision Tree

```
Is your signal...

1. Stationary or slowly varying?
   YES → Use **Spectrogram**
   NO → Continue to 2

2. A chirp or rapid transient?
   YES → Use **Q-transform**
   NO → Continue to 3

3. Highly non-stationary or nonlinear?
   YES → Use **HHT**
   NO → Try **Spectrogram** first, then Q-transform

4. Need instantaneous frequency?
   YES → Use **HHT**
   NO → Use **Spectrogram** or **Q-transform**

5. Need fast computation?
   YES → Use **Spectrogram**
   NO → Any method works
```

## Performance Comparison

| Method | Computation Time* | Memory | Frequency Tracking |
|--------|------------------|--------|-------------------|
| Spectrogram | 1× (baseline) | Low | Good for slow changes |
| Q-transform | 5-10× | Medium | Excellent for chirps |
| HHT | 20-50× | High | Excellent for all |

*Approximate, depends on parameters

## Summary Table

| Feature | Spectrogram | Q-transform | HHT |
|---------|------------|-------------|-----|
| **Time-frequency resolution** | Fixed | Adaptive (constant Q) | Data-adaptive |
| **Best signal type** | Stationary | Chirps | Non-stationary |
| **Computational cost** | Low | Medium | High |
| **Frequency tracking** | Good | Excellent | Excellent |
| **Ease of interpretation** | Easy | Medium | Complex |
| **Standard GW use** | General | Transients | Special cases |

## Practical Recommendations

### For Routine Analysis
Start with **Spectrogram** - it's fast, well-understood, and sufficient for most cases.

### For Transient Detection
Use **Q-transform** - it's the standard for gravitational wave burst searches.

### For Detailed Characterization
Use **HHT** when you need precise instantaneous frequency or when other methods fail.

### For Publication
Include both **Spectrogram** (familiar to readers) and method-specific analysis (Q-transform or HHT) to show robustness.

---

**See Also:**
- [Spectrogram Tutorial](intro_spectrogram.ipynb)
- [HHT Tutorial](advanced_hht.ipynb)
- [Q-transform Documentation](../reference/api/qtransform.rst)
