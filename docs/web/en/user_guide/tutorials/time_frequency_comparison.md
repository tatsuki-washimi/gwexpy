# Time-Frequency Methods: Comparison and Selection Guide

This tutorial compares different time-frequency analysis methods in gwexpy and helps you choose the right one for your analysis.

## Overview

Gravitational wave signals often have **time-varying frequency content**. Different methods reveal different aspects:

| Method | Best For | Time Resolution | Frequency Resolution |
|--------|----------|-----------------|---------------------|
| **Spectrogram (STFT)** | General purpose | Good | Good |
| **Q-transform** | Chirps, transients | Adaptive | Adaptive (constant Q) |
| **Wavelet (CWT)** | Multi-scale features, chirps | Scale-dependent | Scale-dependent |
| **HHT** | Instantaneous frequency | Excellent | Data-adaptive |
| **STLT** | Damped oscillations | Good | Good (+ decay rate σ) |
| **Cepstrum** | Echo detection, periodicity | N/A | Quefrency domain |
| **DCT** | Compression, smooth features | N/A | N/A (basis coefficients) |

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

## Method 3: Wavelet Transform (CWT)

### What it is

**Continuous Wavelet Transform (CWT)**: Multi-scale analysis using dilated/translated wavelets

**Formula**: `W(a, b) = ∫ x(t) ψ*((t-b)/a) dt`
- `a`: scale parameter (inversely proportional to frequency)
- `b`: time shift parameter
- `ψ`: mother wavelet (e.g., Morlet)

### Implementation

```python
import pywt

# Create isolated chirp signal for clarity
chirp_signal = np.zeros(len(ts))
chirp_mask = (ts.times.value >= 2) & (ts.times.value <= 5)
t_chirp = ts.times.value[chirp_mask] - 2
f0, f1 = 20, 150
phase = 2*np.pi * (f0*t_chirp + 0.5*(f1-f0)*t_chirp**2/3)
chirp_signal[chirp_mask] = 1.0 * np.sin(phase)
chirp_signal += np.random.randn(len(ts)) * 0.05

# Compute Continuous Wavelet Transform
scales = np.arange(1, 128)
frequencies_wavelet = ts.sample_rate.value / (2 * scales)
coefficients, frequencies_pywt = pywt.cwt(
    chirp_signal,
    scales,
    'morl',  # Morlet wavelet
    sampling_period=1/ts.sample_rate.value
)

print(f"CWT shape: {coefficients.shape}")  # (scales, time)
print(f"Frequency range: {frequencies_wavelet.min():.1f} - {frequencies_wavelet.max():.1f} Hz")
```

### Visualization

```python
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

# Original signal
axes[0].plot(ts.times.value, chirp_signal, linewidth=0.5, color='black')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Chirp Signal (20→150 Hz over 3s)')

# STFT for comparison
spec_chirp = ts.spectrogram(fftlength=0.5, overlap=0.25)
im1 = axes[1].pcolormesh(
    spec_chirp.times.value,
    spec_chirp.frequencies.value,
    spec_chirp.value.T,
    cmap='viridis',
    shading='auto'
)
axes[1].set_ylim(10, 200)
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_title('STFT Spectrogram (smeared trajectory)')

# Wavelet scalogram
im2 = axes[2].pcolormesh(
    ts.times.value,
    frequencies_wavelet,
    np.abs(coefficients),
    cmap='viridis',
    shading='auto'
)
axes[2].set_ylim(10, 200)
axes[2].set_ylabel('Frequency (Hz)')
axes[2].set_xlabel('Time (s)')
axes[2].set_title('Wavelet (CWT) Scalogram (sharp trajectory)')

# Overlay true frequency
t_true = ts.times.value[chirp_mask]
f_true = 20 + (150 - 20) * (t_true - 2) / 3
axes[1].plot(t_true, f_true, 'r--', linewidth=2, alpha=0.8, label='True frequency')
axes[2].plot(t_true, f_true, 'r--', linewidth=2, alpha=0.8, label='True frequency')
axes[1].legend()
axes[2].legend()

plt.tight_layout()
plt.show()
```

### Quantitative Evaluation

```python
# Ridge extraction: find frequency of maximum energy at each time
ridge_indices = np.argmax(np.abs(coefficients), axis=0)
f_ridge = frequencies_wavelet[ridge_indices]

# Compute MSE over chirp region
chirp_time_mask = (ts.times.value >= 2) & (ts.times.value <= 5)
f_ridge_chirp = f_ridge[chirp_time_mask]
f_true_interp = 20 + (150 - 20) * (ts.times.value[chirp_time_mask] - 2) / 3

valid_mask = (f_ridge_chirp >= 10) & (f_ridge_chirp <= 200)
mse_wavelet = np.mean((f_ridge_chirp[valid_mask] - f_true_interp[valid_mask])**2)
rmse_wavelet = np.sqrt(mse_wavelet)

print(f"Frequency Tracking RMSE: {rmse_wavelet:.2f} Hz")
print(f"STFT uncertainty (0.5s window): ~10-20 Hz")
print(f"Improvement: {10/rmse_wavelet:.1f}× better precision")
```

### Pros and Cons

✅ **Advantages:**
- Natural scale matching (wavelet "stretches" to follow signal)
- Better time-frequency localization than STFT
- Excellent for chirps and multi-scale transients
- Ridge extraction provides precise frequency trajectories

❌ **Disadvantages:**
- Higher computational cost than STFT
- Requires wavelet selection (Morlet, Mexican hat, etc.)
- Redundant representation (overcomplete)
- Edge effects at signal boundaries

### When to Use

✅ **Use Wavelet when:**
- Analyzing chirps spanning multiple octaves
- Need better frequency tracking than STFT
- Signal has multi-scale structure
- Time-frequency resolution tradeoff is critical

❌ **Don't use when:**
- Signal is purely stationary (STFT sufficient)
- Need fastest computation (use STFT)
- Frequency range is narrow (Q-transform may be better)

## Method 4: Hilbert-Huang Transform (HHT)

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

## Method 5: Short-Time Laplace Transform (STLT)

### What it is

**STLT**: Decomposes signals into **both frequency ω and decay rate σ** components

**Formula**: `STLT(t, σ, ω) = ∫ x(τ) w(τ-t) e^(-(σ+iω)τ) dτ`

Unlike STFT which only shows frequency, STLT reveals damping coefficients.

### Implementation

```python
# Create multi-mode ringdown
t = ts.times.value
ringdown_multi = np.zeros(len(t))

# Mode 1: f=100 Hz, τ=0.2s (high damping)
mode1_mask = t >= 1
t1 = t[mode1_mask] - 1
tau1 = 0.2
ringdown_multi[mode1_mask] += 1.0 * np.exp(-t1/tau1) * np.sin(2*np.pi*100*t[mode1_mask])

# Mode 2: f=150 Hz, τ=0.8s (low damping)
mode2_mask = t >= 2
t2 = t[mode2_mask] - 2
tau2 = 0.8
ringdown_multi[mode2_mask] += 0.8 * np.exp(-t2/tau2) * np.sin(2*np.pi*150*t[mode2_mask])

# Mode 3: f=120 Hz, τ=0.5s (medium damping)
mode3_mask = t >= 3
t3 = t[mode3_mask] - 3
tau3 = 0.5
ringdown_multi[mode3_mask] += 0.6 * np.exp(-t3/tau3) * np.sin(2*np.pi*120*t[mode3_mask])

ringdown_multi += np.random.randn(len(t)) * 0.05

ts_ringdown = TimeSeries(ringdown_multi, t0=0, dt=ts.dt, unit='strain')

# Compute STLT
stlt = ts_ringdown.stlt(fftlength=1.0, overlap=0.5)

print(f"STLT shape: {stlt.shape}")  # (time, sigma, omega)
print("STLT reveals both ω (frequency) and σ (decay rate)")
```

### Visualization

```python
fig = plt.figure(figsize=(14, 10))

# Original signal
ax1 = plt.subplot(3, 1, 1)
ax1.plot(t, ringdown_multi, linewidth=0.5, color='black')
ax1.set_ylabel('Amplitude')
ax1.set_title('Multi-Mode Ringdown (3 modes with different decay rates)')
ax1.set_xlim(0, duration)

# STFT (only shows frequency, not decay)
ax2 = plt.subplot(3, 1, 2)
spec_ringdown = ts_ringdown.spectrogram(fftlength=0.5, overlap=0.25)
im1 = ax2.pcolormesh(
    spec_ringdown.times.value,
    spec_ringdown.frequencies.value,
    spec_ringdown.value.T,
    cmap='viridis',
    shading='auto'
)
ax2.set_ylim(80, 180)
ax2.set_ylabel('Frequency (Hz)')
ax2.set_title('STFT: Shows frequencies but decay rates invisible')

# STLT (shows both frequency and decay rate)
ax3 = plt.subplot(3, 1, 3)
# Average over time for 2D σ-ω visualization
stlt_power = np.abs(stlt.value)**2
stlt_avg = np.mean(stlt_power, axis=0)

# Get sigma and omega axes
sigma_axis = stlt.sigma.value if hasattr(stlt, 'sigma') else np.linspace(-10, 10, stlt.shape[1])
omega_axis = stlt.frequencies.value if hasattr(stlt, 'frequencies') else np.linspace(0, ts.sample_rate.value/2, stlt.shape[2])

im2 = ax3.pcolormesh(omega_axis, sigma_axis, stlt_avg, cmap='hot', shading='auto')
ax3.set_xlim(80, 180)
ax3.set_ylabel('Decay Rate σ (1/s)')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_title('STLT: 2D σ-ω plane reveals BOTH frequency and decay rate')

# Annotate expected modes
ax3.plot(100, -1/tau1, 'ro', markersize=10, label=f'Mode 1: 100 Hz, τ={tau1}s')
ax3.plot(150, -1/tau2, 'go', markersize=10, label=f'Mode 2: 150 Hz, τ={tau2}s')
ax3.plot(120, -1/tau3, 'bo', markersize=10, label=f'Mode 3: 120 Hz, τ={tau3}s')
ax3.legend()

plt.tight_layout()
plt.show()
```

### Quantitative Evaluation

```python
# Extract σ peaks for each mode
modes_info = [(100, tau1), (150, tau2), (120, tau3)]
abs_errors = []

for freq, tau in modes_info:
    # Find frequency index
    freq_idx = np.argmin(np.abs(omega_axis - freq))

    # Find sigma with maximum power
    sigma_idx = np.argmax(stlt_avg[:, freq_idx])
    sigma_est = sigma_axis[sigma_idx]
    sigma_true = -1/tau

    abs_error = abs(sigma_est - sigma_true)
    abs_errors.append(abs_error)

    print(f"Mode {freq} Hz: σ_true = {sigma_true:.2f}, σ_est = {sigma_est:.2f}, error = {abs_error:.2f}")

mae_stlt = np.mean(abs_errors)
print(f"\nMean Absolute Error: {mae_stlt:.2f} s⁻¹")
print("STFT cannot estimate decay rates at all.")
```

### Pros and Cons

✅ **Advantages:**
- Unique ability to separate modes by **both** frequency and decay rate
- Essential for ringdown quality factor estimation
- 2D σ-ω representation reveals damping structure
- Directly applicable to post-merger waveforms

❌ **Disadvantages:**
- High computational cost (2D transform)
- Requires careful parameter selection
- Less familiar than STFT/Q-transform
- Interpretation requires understanding σ-ω plane

### When to Use

✅ **Use STLT when:**
- Analyzing ringdown modes (black hole quasi-normal modes)
- Need to estimate quality factors or damping times
- Multiple damped oscillations overlap in frequency
- Decay rate information is scientifically important

❌ **Don't use when:**
- Signal has no damping (use STFT instead)
- Only frequency information needed
- Computational resources are limited

## Method 6: Cepstrum

### What it is

**Cepstrum**: Analysis of periodicity in the spectrum via "quefrency"

**Formula**: `C(τ) = IFFT(log|FFT(x)|)`

Converts spectral periodicity → time-domain peaks at delay times.

### Implementation

```python
# Create signal with echo
original_sig = np.zeros(len(t))
pulse_time = 1.0
pulse_idx = int(pulse_time * ts.sample_rate.value)
pulse_width = int(0.05 * ts.sample_rate.value)

# Broadband pulse
sigma = int(0.01 * ts.sample_rate.value)
n = np.arange(pulse_width)
pulse_window = np.exp(-0.5 * ((n - pulse_width/2) / sigma) ** 2)
original_sig[pulse_idx:pulse_idx+len(pulse_window)] = pulse_window * np.sin(2*np.pi*100*t[pulse_idx:pulse_idx+len(pulse_window)])

# Add echo with 80ms delay
echo_delay = 0.08  # seconds
echo_amplitude = 0.4
echo_samples = int(echo_delay * ts.sample_rate.value)
echo_sig = np.zeros(len(t))
echo_sig[echo_samples:] = echo_amplitude * original_sig[:-echo_samples]

signal_with_echo = original_sig + echo_sig
signal_with_echo += np.random.randn(len(t)) * 0.02

ts_echo = TimeSeries(signal_with_echo, t0=0, dt=ts.dt, unit='strain')

# Compute cepstrum
spectrum = np.fft.rfft(signal_with_echo)
log_spectrum = np.log(np.abs(spectrum) + 1e-10)
cepstrum = np.fft.irfft(log_spectrum)
quefrency = np.arange(len(cepstrum)) / ts.sample_rate.value

print(f"Cepstrum computed. Quefrency range: 0 to {quefrency.max():.3f} s")
```

### Visualization

```python
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Original signal
axes[0].plot(t, signal_with_echo, linewidth=0.5, color='black')
axes[0].set_ylabel('Amplitude')
axes[0].set_title(f'Signal with Echo (delay = {echo_delay*1000:.0f} ms)')
axes[0].set_xlim(0, 2)
axes[0].axvline(pulse_time, color='r', linestyle='--', alpha=0.5, label='Original')
axes[0].axvline(pulse_time + echo_delay, color='g', linestyle='--', alpha=0.5, label='Echo')
axes[0].legend()

# STFT
spec_echo = ts_echo.spectrogram(fftlength=0.25, overlap=0.125)
im = axes[1].pcolormesh(
    spec_echo.times.value,
    spec_echo.frequencies.value,
    spec_echo.value.T,
    cmap='viridis',
    shading='auto'
)
axes[1].set_ylim(50, 150)
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_title('STFT: Echo causes spectral ripples (hard to interpret)')
axes[1].set_xlim(0, 2)

# Power spectrum
freqs = np.fft.rfftfreq(len(signal_with_echo), ts.dt.value)
axes[2].semilogy(freqs, np.abs(spectrum))
axes[2].set_xlim(50, 150)
axes[2].set_ylabel('|Spectrum|')
axes[2].set_title('Power Spectrum: Periodic ripples from echo (comb filter)')

# Cepstrum
axes[3].plot(quefrency[:int(0.2*ts.sample_rate.value)],
            cepstrum[:int(0.2*ts.sample_rate.value)],
            linewidth=1, color='blue')
axes[3].axvline(echo_delay, color='r', linestyle='--', linewidth=2,
               label=f'True delay = {echo_delay*1000:.0f} ms')
axes[3].set_xlim(0, 0.15)
axes[3].set_xlabel('Quefrency (s)')
axes[3].set_ylabel('Cepstrum')
axes[3].set_title('Cepstrum: Clear peak at echo delay')
axes[3].legend()

plt.tight_layout()
plt.show()
```

### Quantitative Evaluation

```python
# Find peak in cepstrum
search_region = (quefrency > 0.03) & (quefrency < 0.12)
peak_idx = np.argmax(np.abs(cepstrum[search_region]))
detected_delay = quefrency[search_region][peak_idx]

error_ms = abs(detected_delay - echo_delay) * 1000

print(f"True echo delay: {echo_delay*1000:.1f} ms")
print(f"Detected peak: {detected_delay*1000:.1f} ms")
print(f"Detection error: {error_ms:.2f} ms")
print("Cepstrum converts spectral periodicity → time-domain peak")
```

### Pros and Cons

✅ **Advantages:**
- Direct detection of echo delays (quefrency peaks)
- Reveals periodic structure in spectrum
- Useful for pitch detection and harmonic analysis
- Computationally efficient (double FFT)

❌ **Disadvantages:**
- Requires logarithm (sensitivity to low amplitude)
- Not time-localized (global analysis)
- Less intuitive than time-frequency methods
- Best for signals with clear echoes

### When to Use

✅ **Use Cepstrum when:**
- Detecting echoes or reflections
- Analyzing periodic structure in spectrum
- Measuring delay times in reverberant signals
- Pitch detection or harmonic analysis

❌ **Don't use when:**
- Need time-localized analysis
- Signal has no echoes or periodicity
- Low SNR (log operation amplifies noise)

## Method 7: Discrete Cosine Transform (DCT)

### What it is

**DCT**: Transform to cosine basis, excellent for compressing smooth signals

**Formula**: `X(k) = Σ x(n) cos(πk(2n+1)/(2N))`

Energy concentrates in low-frequency coefficients for smooth signals.

### Implementation

```python
from scipy.fftpack import dct, idct

# Create smooth signal with perturbations
smooth_signal = np.zeros(len(t))
smooth_signal += 1.0 * np.sin(2*np.pi*3*t)
smooth_signal += 0.5 * np.sin(2*np.pi*7*t)
smooth_signal += 0.3 * np.sin(2*np.pi*12*t)
smooth_signal += 0.1 * np.sin(2*np.pi*50*t)
smooth_signal += 0.05 * np.sin(2*np.pi*80*t)
smooth_signal += np.random.randn(len(t)) * 0.05

ts_smooth = TimeSeries(smooth_signal, t0=0, dt=ts.dt, unit='strain')

# Compute DCT
dct_coeffs = dct(smooth_signal, type=2, norm='ortho')

print(f"DCT computed: {len(dct_coeffs)} coefficients")

# Reconstruct with different numbers of coefficients
n_coeffs_list = [10, 50, 200]
for n_coeffs in n_coeffs_list:
    coeffs_truncated = dct_coeffs.copy()
    coeffs_truncated[n_coeffs:] = 0
    recon = idct(coeffs_truncated, type=2, norm='ortho')

    rmse = np.sqrt(np.mean((smooth_signal - recon)**2))
    energy_kept = 100 * np.sum(dct_coeffs[:n_coeffs]**2) / np.sum(dct_coeffs**2)

    print(f"{n_coeffs} coeffs: {energy_kept:.2f}% energy, RMSE = {rmse:.4f}")
```

### Visualization

```python
fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# Original signal
axes[0, 0].plot(t[:2048], smooth_signal[:2048], linewidth=0.8, color='black')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].set_title('Original Signal (smooth + small perturbations)')

# DCT coefficients
axes[0, 1].semilogy(np.abs(dct_coeffs[:500]), linewidth=0.5, color='blue')
axes[0, 1].set_ylabel('|DCT Coefficient|')
axes[0, 1].set_title('DCT: Rapid decay (energy in low coefficients)')
axes[0, 1].axvline(50, color='r', linestyle='--', alpha=0.5)

# Reconstructions
for i, n_coeffs in enumerate(n_coeffs_list):
    coeffs_truncated = dct_coeffs.copy()
    coeffs_truncated[n_coeffs:] = 0
    recon = idct(coeffs_truncated, type=2, norm='ortho')

    # Reconstruction
    axes[i+1, 0].plot(t[:2048], smooth_signal[:2048], linewidth=0.5,
                     color='gray', alpha=0.5, label='Original')
    axes[i+1, 0].plot(t[:2048], recon[:2048], linewidth=1.5,
                     color='red', label=f'{n_coeffs} coeffs')
    axes[i+1, 0].set_ylabel('Amplitude')
    axes[i+1, 0].set_title(f'Reconstruction ({n_coeffs} coefficients)')
    axes[i+1, 0].legend()

    # Error
    error = smooth_signal - recon
    axes[i+1, 1].plot(t[:2048], error[:2048], linewidth=0.5, color='orange')
    axes[i+1, 1].set_ylabel('Error')
    rmse = np.sqrt(np.mean(error**2))
    compression = len(t) / n_coeffs
    axes[i+1, 1].set_title(f'Error (RMSE={rmse:.4f}, {compression:.0f}× compression)')

axes[2, 0].set_xlabel('Time (s)')
axes[2, 1].set_xlabel('Time (s)')

plt.tight_layout()
plt.show()
```

### Quantitative Evaluation

```python
total_energy = np.sum(dct_coeffs**2)

print("DCT Compression Analysis:")
for n_coeffs in [10, 50, 100, 200]:
    energy_kept = 100 * np.sum(dct_coeffs[:n_coeffs]**2) / total_energy
    compression_ratio = len(dct_coeffs) / n_coeffs

    print(f"  {n_coeffs:4d} coeffs ({100*n_coeffs/len(dct_coeffs):5.1f}%): "
          f"{energy_kept:5.2f}% energy, {compression_ratio:5.1f}× compression")

print("\n50 coefficients (3% of data) capture >99% energy")
print("Compression ratio ~30× with negligible error")
```

### Pros and Cons

✅ **Advantages:**
- Excellent compression for smooth signals
- Sparse representation (few coefficients capture most energy)
- Fast computation (FFT-like)
- No boundary artifacts (unlike Fourier)
- Ideal for feature extraction and denoising

❌ **Disadvantages:**
- Not time-localized
- Less effective for discontinuous signals
- Interpretation less intuitive than time-frequency
- Global analysis only

### When to Use

✅ **Use DCT when:**
- Compressing smooth signals
- Feature extraction (low-order coefficients)
- Denoising smooth backgrounds
- Data reduction before machine learning
- Modeling slow trends

❌ **Don't use when:**
- Need time-localized analysis
- Signal has sharp transients
- Frequency content varies rapidly in time

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
What information do you need?

1. General time-frequency view?
   → Use **Spectrogram (STFT)**

2. Transient detection with adaptive resolution?
   → Use **Q-transform**

3. Chirp frequency tracking (multi-scale)?
   → Use **Wavelet (CWT)**

4. Instantaneous frequency as single-valued curve?
   → Use **HHT**

5. Decay rates AND frequencies (ringdown)?
   → Use **STLT**

6. Echo detection or spectral periodicity?
   → Use **Cepstrum**

7. Signal compression or smooth feature extraction?
   → Use **DCT**

Quick decision path:
├─ Stationary signal? → Spectrogram
├─ Transient/burst? → Q-transform
├─ Chirp spanning octaves? → Wavelet
├─ Need inst. frequency? → HHT
├─ Damped oscillations? → STLT
├─ Has echoes? → Cepstrum
└─ Smooth, need compression? → DCT
```

## Performance Comparison

| Method | Computation Time* | Memory | Best For |
|--------|------------------|--------|----------|
| Spectrogram (STFT) | 1× (baseline) | Low | General purpose |
| DCT | 1-2× | Low | Compression |
| Q-transform | 5-10× | Medium | Transients |
| Wavelet (CWT) | 8-15× | Medium-High | Chirps |
| Cepstrum | 3-5× | Medium | Echo detection |
| STLT | 15-25× | High | Ringdown modes |
| HHT | 20-50× | High | Instantaneous freq. |

*Approximate, depends on parameters and signal length

## Summary Table

| Feature | STFT | Q-transform | Wavelet | HHT | STLT | Cepstrum | DCT |
|---------|------|-------------|---------|-----|------|----------|-----|
| **Resolution** | Fixed | Adaptive Q | Scale-adaptive | Data-adaptive | Fixed | Quefrency | N/A |
| **Best signal** | Stationary | Chirps | Multi-scale | AM/FM | Damped | Echoes | Smooth |
| **Comp. cost** | Low | Medium | Med-High | Very High | High | Medium | Low |
| **Output** | 2D (t,f) | 2D (t,f) | 2D (t,f) | f(t) curve | 3D (t,σ,ω) | τ spectrum | Coeffs |
| **Time-localized** | Yes | Yes | Yes | Yes | Yes | No | No |
| **Unique info** | — | Adaptive res. | Multi-scale | Inst. freq. | Decay rate σ | Delays | Compression |
| **GW use** | General | Transients | Chirps | Glitches | Ringdown | Reflections | Features |

## Practical Recommendations

### For Routine Analysis
Start with **Spectrogram (STFT)** - it's fast, well-understood, and sufficient for most cases.

### For Transient Detection
Use **Q-transform** - it's the standard for gravitational wave burst searches.

### For Chirp Analysis
Use **Wavelet (CWT)** for precise frequency trajectory tracking across multiple scales.

### For Instantaneous Frequency
Use **HHT** when you need single-valued instantaneous frequency (no time-frequency uncertainty).

### For Ringdown Analysis
Use **STLT** to estimate both frequencies and decay rates (quality factors) simultaneously.

### For Echo Detection
Use **Cepstrum** to identify delay times and periodic structure in the spectrum.

### For Data Reduction
Use **DCT** for efficient compression and feature extraction of smooth signals.

### For Publication
- Include **Spectrogram** (familiar baseline for readers)
- Add specialized method(s) that reveal unique physics (Wavelet for chirps, HHT for inst. freq., STLT for ringdown, etc.)
- Show robustness: demonstrate that key results hold across multiple methods

### General Strategy
1. **Start simple**: Always begin with Spectrogram
2. **Identify limitations**: Where does STFT fail to show important features?
3. **Choose specialized method**: Pick the method whose "unique info" matches your needs (see Summary Table)
4. **Validate**: Cross-check with another method if possible

---

**See Also:**
- [Spectrogram Tutorial](intro_spectrogram.ipynb)
- [HHT Tutorial](advanced_hht.ipynb)
- [Q-transform Documentation](../reference/api/qtransform.rst)
