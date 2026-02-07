#!/usr/bin/env python3
"""Test new methods added to ScalarField."""

import numpy as np
from astropy import units as u

# Create a simple test field
print("Creating test field...")
nt, nx, ny, nz = 100, 4, 4, 4
times = np.arange(nt) * 0.01 * u.s
x = np.arange(nx) * 1.0 * u.m
data = np.random.randn(nt, nx, ny, nz)

from gwexpy.fields import ScalarField

field = ScalarField(
    data,
    unit=u.m / u.s,
    axis0=times,
    axis1=x,
    axis2=x,
    axis3=x,
    axis_names=["t", "x", "y", "z"],
)

print(f"Field shape: {field.shape}")
print(f"Field unit: {field.unit}")

# Test basic preprocessing
print("\n=== Testing Preprocessing Methods ===")

print("Testing detrend...")
detrended = field.detrend("linear")
print(f"  Detrended shape: {detrended.shape}")

print("Testing taper...")
tapered = field.taper(duration=0.1 * u.s)
print(f"  Tapered shape: {tapered.shape}")

print("Testing crop...")
cropped = field.crop(start=0.2 * u.s, end=0.7 * u.s)
print(f"  Cropped shape: {cropped.shape}")

print("Testing pad...")
padded = field.pad(10)
print(f"  Padded shape: {padded.shape}")

# Test mathematical operations
print("\n=== Testing Mathematical Operations ===")

print("Testing abs...")
abs_field = field.abs()
print(f"  Abs shape: {abs_field.shape}")

print("Testing mean...")
mean_val = field.mean()
print(f"  Global mean: {mean_val}")
time_mean = field.mean(axis=0)
print(f"  Time mean shape: {time_mean.shape}")

print("Testing rms...")
rms_val = field.rms()
print(f"  Global RMS: {rms_val}")

# Test signal processing
print("\n=== Testing Signal Processing Methods ===")

print("Testing whiten...")
try:
    whitened = field.whiten(fftlength=0.2)
    print(f"  Whitened shape: {whitened.shape}")
    print(f"  Whitened unit: {whitened.unit}")
except Exception as e:
    print(f"  Whiten failed: {e}")

print("Testing convolve...")
fir = np.array([1, 2, 1]) / 4.0
convolved = field.convolve(fir, mode="same")
print(f"  Convolved shape: {convolved.shape}")

print("Testing inject...")
signal = ScalarField(
    np.sin(2 * np.pi * 10 * times.value)[:, None, None, None] * np.ones((1, nx, ny, nz)),
    unit=u.m / u.s,
    axis0=times,
    axis1=x,
    axis2=x,
    axis3=x,
    axis_names=["t", "x", "y", "z"],
)
injected = field.inject(signal, alpha=0.5)
print(f"  Injected shape: {injected.shape}")

# Test spectral analysis
print("\n=== Testing Spectral Analysis Methods ===")

# Create a second field for cross-spectral analysis
field2 = ScalarField(
    np.random.randn(nt, nx, ny, nz),
    unit=u.m / u.s,
    axis0=times,
    axis1=x,
    axis2=x,
    axis3=x,
    axis_names=["t", "x", "y", "z"],
)

print("Testing csd...")
try:
    csd_result = field.csd(field2, fftlength=0.2)
    print(f"  CSD shape: {csd_result.shape}")
    print(f"  CSD domain: {csd_result.axis0_domain}")
except Exception as e:
    print(f"  CSD failed: {e}")

print("Testing coherence...")
try:
    coh_result = field.coherence(field2, fftlength=0.2)
    print(f"  Coherence shape: {coh_result.shape}")
    print(f"  Coherence domain: {coh_result.axis0_domain}")
except Exception as e:
    print(f"  Coherence failed: {e}")

print("Testing spectrogram...")
try:
    spec_result = field.spectrogram(stride=0.1, fftlength=0.2)
    print(f"  Spectrogram shape: {spec_result.shape}")
except Exception as e:
    print(f"  Spectrogram failed: {e}")

# Test time series utilities
print("\n=== Testing Time Series Utilities ===")

print("Testing is_compatible...")
compatible = field.is_compatible(field2)
print(f"  Compatible: {compatible}")

print("Testing is_contiguous...")
# Create a contiguous segment
next_times = times[-1] + (np.arange(50) + 1) * 0.01 * u.s
field_next = ScalarField(
    np.random.randn(50, nx, ny, nz),
    unit=u.m / u.s,
    axis0=next_times,
    axis1=x,
    axis2=x,
    axis3=x,
    axis_names=["t", "x", "y", "z"],
)
contiguous = field.is_contiguous(field_next)
print(f"  Contiguous: {contiguous}")

print("Testing append...")
appended = field.append(field_next)
print(f"  Appended shape: {appended.shape}")

# Test FieldDict
print("\n=== Testing FieldDict Methods ===")

from gwexpy.fields import FieldDict

field_dict = FieldDict({
    "x": field,
    "y": field2,
})

print("Testing FieldDict.detrend...")
fd_detrended = field_dict.detrend()
print(f"  Keys: {list(fd_detrended.keys())}")
print(f"  Component 'x' shape: {fd_detrended['x'].shape}")

print("Testing FieldDict.abs...")
fd_abs = field_dict.abs()
print(f"  Keys: {list(fd_abs.keys())}")

print("Testing FieldDict.whiten...")
try:
    fd_whitened = field_dict.whiten(fftlength=0.2)
    print(f"  Keys: {list(fd_whitened.keys())}")
except Exception as e:
    print(f"  Whiten failed: {e}")

print("\n=== All tests completed! ===")
