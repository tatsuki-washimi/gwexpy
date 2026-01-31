"""Verify ScalarField spectral_density physics.

This script checks:
1. Parseval's theorem: Energy in time domain equals energy in frequency domain
2. Unit propagation: PSD units are correct (V^2/Hz for density scaling)
3. Peak detection: Sine wave should have peak at correct frequency
4. Spatial spectral density: Wavenumber spectrum with correct units
5. axis domain updates correctly after transform
"""

import numpy as np
from astropy import units as u

from gwexpy.fields import ScalarField
from gwexpy.fields.signal import spectral_density


def test_parseval_theorem():
    """Verify energy conservation between time and frequency domain."""
    print("Test 1: Parseval's theorem...")
    
    # Create test field with known RMS
    np.random.seed(42)
    shape = (256, 4, 4, 4)
    sample_rate = 100 * u.Hz
    
    field = ScalarField.simulate(
        'gaussian',
        shape=shape,
        sample_rate=sample_rate,
        space_step=0.1*u.m,
        std=1.0,
        seed=42,
        unit=u.V,
    )
    
    # Time-domain energy (RMS^2 = variance for zero-mean)
    time_energy = np.mean(field.value ** 2)
    
    # PSD and frequency-domain energy
    psd_field = field.psd(method='fft', scaling='density')
    
    # Integrate PSD over frequency
    df = psd_field._axis0_index[1] - psd_field._axis0_index[0]
    df_val = df.value
    
    # Average over spatial dimensions
    psd_mean = np.mean(psd_field.value, axis=(1, 2, 3))
    freq_energy = np.sum(psd_mean) * df_val
    
    ratio = freq_energy / time_energy
    print(f"  Time-domain energy: {time_energy:.6f}")
    print(f"  Freq-domain energy: {freq_energy:.6f}")
    print(f"  Ratio: {ratio:.6f}")
    
    # Should be close to 1 (Welch method may differ slightly due to windowing)
    assert 0.8 < ratio < 1.2, f"Parseval mismatch: ratio={ratio}"
    print("  PASSED\n")


def test_unit_propagation():
    """Verify correct unit propagation in PSD."""
    print("Test 2: Unit propagation...")
    
    field = ScalarField.simulate(
        'gaussian',
        shape=(100, 4, 4, 4),
        sample_rate=100*u.Hz,
        space_step=0.1*u.m,
        unit=u.V,
        seed=123,
    )
    
    # PSD with density scaling
    psd_density = field.psd(scaling='density')
    expected_unit = u.V**2 / u.Hz
    assert psd_density.unit.is_equivalent(expected_unit), \
        f"Unit mismatch: got {psd_density.unit}, expected {expected_unit}"
    print(f"  Density scaling unit: {psd_density.unit} (expected {expected_unit})")
    
    # PSD with spectrum scaling
    psd_spectrum = field.psd(scaling='spectrum')
    expected_unit = u.V**2
    assert psd_spectrum.unit.is_equivalent(expected_unit), \
        f"Unit mismatch: got {psd_spectrum.unit}, expected {expected_unit}"
    print(f"  Spectrum scaling unit: {psd_spectrum.unit} (expected {expected_unit})")
    
    print("  PASSED\n")


def test_sine_wave_peak():
    """Verify sine wave produces peak at correct frequency."""
    print("Test 3: Sine wave peak detection...")
    
    # Create sine wave at 25 Hz
    target_freq = 25.0  # Hz
    sample_rate = 100 * u.Hz
    shape = (256, 1, 1, 1)
    
    dt = (1 / sample_rate).to(u.s).value
    times = np.arange(shape[0]) * dt
    sine_data = np.sin(2 * np.pi * target_freq * times)
    sine_data = sine_data[:, np.newaxis, np.newaxis, np.newaxis]
    
    field = ScalarField(
        sine_data,
        unit=u.V,
        axis0=times * u.s,
        axis1=np.array([0]) * u.m,
        axis2=np.array([0]) * u.m,
        axis3=np.array([0]) * u.m,
    )
    
    psd_field = field.psd(method='welch', nperseg=64)
    
    # Find peak frequency
    freqs = psd_field._axis0_index.value
    psd_vals = psd_field.value[:, 0, 0, 0]
    peak_idx = np.argmax(psd_vals)
    peak_freq = freqs[peak_idx]
    
    print(f"  Target frequency: {target_freq} Hz")
    print(f"  Detected peak: {peak_freq:.2f} Hz")
    
    # Should be within 1 frequency bin
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    assert abs(peak_freq - target_freq) < 2 * df, \
        f"Peak mismatch: got {peak_freq}, expected {target_freq}"
    print("  PASSED\n")


def test_domain_update():
    """Verify axis0_domain updates correctly."""
    print("Test 4: Domain update after PSD...")
    
    field = ScalarField.simulate(
        'gaussian',
        shape=(64, 4, 4, 4),
        sample_rate=100*u.Hz,
        space_step=0.1*u.m,
        seed=99,
    )
    
    assert field.axis0_domain == 'time', f"Initial domain wrong: {field.axis0_domain}"
    
    psd_field = field.psd()
    assert psd_field.axis0_domain == 'frequency', \
        f"PSD domain wrong: {psd_field.axis0_domain}"
    
    print(f"  Original domain: time")
    print(f"  After PSD: {psd_field.axis0_domain}")
    print("  PASSED\n")


def test_spatial_spectral_density():
    """Verify spatial wavenumber spectrum."""
    print("Test 5: Spatial spectral density...")
    
    # Create spatial sine pattern along x
    shape = (10, 64, 4, 4)
    dx = 0.1 * u.m  # 0.1 m spacing
    wavelength = 1.0  # 1 m wavelength
    target_k = 1.0 / wavelength  # 1 / m
    
    x_vals = np.arange(shape[1]) * dx.value
    spatial_pattern = np.sin(2 * np.pi * target_k * x_vals)
    
    # Broadcast to 4D
    data = np.zeros(shape)
    data[:, :, :, :] = spatial_pattern[np.newaxis, :, np.newaxis, np.newaxis]
    
    field = ScalarField(
        data,
        unit=u.V,
        axis0=np.arange(shape[0]) * u.s,
        axis1=x_vals * u.m,
        axis2=np.arange(shape[2]) * dx,
        axis3=np.arange(shape[3]) * dx,
    )
    
    # Spatial spectral density along x
    kx_spec = field.spectral_density(axis=1, method='fft')
    
    # Check axis name changed
    assert 'kx' in kx_spec.axis_names, f"Axis name not updated: {kx_spec.axis_names}"
    
    # Check domain updated
    assert kx_spec._space_domains.get('kx') == 'k', \
        f"Space domain not updated: {kx_spec._space_domains}"
    
    # Check unit
    expected_unit = u.V**2 / (1/u.m)
    assert kx_spec.unit.is_equivalent(expected_unit), \
        f"Unit mismatch: {kx_spec.unit} vs {expected_unit}"
    
    # Find peak wavenumber
    k_vals = kx_spec._axis1_index.value
    kx_vals = np.mean(kx_spec.value, axis=(0, 2, 3))
    peak_idx = np.argmax(kx_vals)
    peak_k = k_vals[peak_idx]
    
    print(f"  Target wavenumber: {target_k} 1/m")
    print(f"  Detected peak: {peak_k:.4f} 1/m")
    print(f"  Axis name: {kx_spec.axis_names[1]}")
    print(f"  Space domain: {kx_spec._space_domains}")
    
    # Allow some tolerance
    assert abs(abs(peak_k) - target_k) < 0.2, \
        f"Peak mismatch: got {peak_k}, expected {target_k}"
    print("  PASSED\n")


def main():
    print("=" * 60)
    print("ScalarField Spectral Density Physics Verification")
    print("=" * 60 + "\n")
    
    test_parseval_theorem()
    test_unit_propagation()
    test_sine_wave_peak()
    test_domain_update()
    test_spatial_spectral_density()
    
    print("=" * 60)
    print("ALL PHYSICS CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
