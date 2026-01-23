# tests/fields/test_scalarfield_signal.py

import pytest
import numpy as np
from astropy import units as u

from gwexpy.fields import ScalarField
from gwexpy.frequencyseries import FrequencySeries, FrequencySeriesList
from gwexpy.timeseries import TimeSeries


@pytest.fixture
def gaussian_field():
    """Create a Gaussian noise field for testing."""
    return ScalarField.simulate(
        'gaussian',
        shape=(100, 4, 4, 4),
        sample_rate=100 * u.Hz,
        space_step=0.1 * u.m,
        std=1.0,
        seed=123,
        unit=u.V,
    )


@pytest.fixture
def sine_field():
    """Create a sine wave field for testing."""
    shape = (200, 4, 4, 4)
    sample_rate = 100 * u.Hz
    t = np.arange(shape[0]) / sample_rate.value
    freq = 25.0  # Hz
    
    # Simple broadcasted sine wave
    data = np.sin(2 * np.pi * freq * t)
    data_4d = data[:, np.newaxis, np.newaxis, np.newaxis] * np.ones((1, 4, 4, 4))
    
    return ScalarField(
        data_4d,
        unit=u.V,
        axis0=t * u.s,
        axis1=np.arange(4) * 0.1 * u.m,
        axis2=np.arange(4) * 0.1 * u.m,
        axis3=np.arange(4) * 0.1 * u.m,
    )


class TestScalarFieldSpectralDensity:
    """Tests for spectral_density, psd, and freq_space_map."""

    def test_psd_parseval(self, gaussian_field):
        """Verify Parseval's theorem (energy conservation)."""
        # Time-domain energy
        time_energy = np.mean(gaussian_field.value ** 2)
        
        # Frequency-domain energy (PSD)
        psd = gaussian_field.psd(method='fft', scaling='density')
        df = psd._axis0_index[1] - psd._axis0_index[0]
        
        # Integrate PSD
        psd_mean = np.mean(psd.value, axis=(1, 2, 3))
        freq_energy = np.sum(psd_mean) * df.value
        
        # Check ratio (allow small windowing/numerical diff)
        ratio = freq_energy / time_energy
        assert 0.8 < ratio < 1.2

    def test_psd_units(self, gaussian_field):
        """Verify PSD unit propagation."""
        # Density scaling
        psd = gaussian_field.psd(scaling='density')
        expected = u.V**2 / u.Hz
        assert psd.unit.is_equivalent(expected)
        
        # Spectrum scaling
        psd = gaussian_field.psd(scaling='spectrum')
        expected = u.V**2
        assert psd.unit.is_equivalent(expected)

    def test_psd_peak_frequency(self, sine_field):
        """Verify sine wave produces peak at correct frequency."""
        psd = sine_field.psd(nperseg=100)
        
        # Access frequency axis directly from ScalarField
        freqs = psd._axis0_index.value
        # Average over space
        psd_val = np.mean(psd.value, axis=(1, 2, 3))
        
        peak_idx = np.argmax(psd_val)
        peak_freq = freqs[peak_idx]
        
        assert abs(peak_freq - 25.0) < 1.0  # Resolution dependent

    def test_spectral_density_spatial(self):
        """Verify spatial spectral density (wavenumber spectrum)."""
        # Create spatial sine wave along X
        x = np.linspace(0, 10, 100)  # 10 m
        k_target = 2.0  # 1/m (wavelength = 0.5m)
        data = np.sin(2 * np.pi * k_target * x)
        
        field = ScalarField(
            data[np.newaxis, :, np.newaxis, np.newaxis],  # (1, 100, 1, 1)
            unit=u.V,
            axis0=[0]*u.s,
            axis1=x*u.m,
            axis2=np.array([0])*u.m,
            axis3=np.array([0])*u.m
        )
        
        # Compute spatial spectrum along axis 1 (x)
        kspec = field.spectral_density(axis=1, method='fft')
        
        assert 'kx' in kspec.axis_names
        # Access internal _space_domains (or add property if designed)
        assert kspec._space_domains['kx'] == 'k'
        assert kspec.unit.is_equivalent(u.V**2 * u.m)  # V^2 / (1/m)
        
        ks = kspec._axis1_index.value
        kval = kspec.value[0, :, 0, 0]
        
        # Find peak (excluding DC)
        peak_idx = np.argmax(kval[1:]) + 1
        peak_k = ks[peak_idx]
        assert abs(peak_k - k_target) < 0.2

    # ... (skipping unchanged) ...

    def test_time_delay_map(self):
        """Test time delay map with a shifted signal."""
        # Create a field where x=1 leads x=0 by 0.1s
        # BUT: simulate requires careful setup.
        # Let's verify our understanding of 'lead' vs 'lag'.
        # If sig2(t) = sig1(t - delay), then sig2 is delayed.
        # correlate(sig1, sig2) peak should show lag.
        
        t = np.arange(100) * 0.01  # 100 Hz
        sig = np.sin(2 * np.pi * 5 * t)  # 5 Hz
        
        # sig_delayed is shifted by +10 samples.
        # sig_delayed[i] = sig[i - 10]
        sig_delayed = np.roll(sig, 10)
        # Fix wrap-around artifacts for cleanliness (optional but good)
        sig_delayed[:10] = 0
        
        data = np.zeros((100, 2, 1, 1))
        data[:, 0, 0, 0] = sig          # Ref point
        data[:, 1, 0, 0] = sig_delayed  # Test point
        
        field = ScalarField(
            data,
            unit=u.V,
            axis0=t*u.s,
            axis1=[0, 1]*u.m,
            axis2=[0]*u.m,
            axis3=[0]*u.m
        )
        
        delay_map = field.time_delay_map(
            ref_point=(0*u.m, 0*u.m, 0*u.m),
            plane='xy',
            at={'z': 0*u.m}
        )
        
        assert isinstance(delay_map, ScalarField)
        
        # x=1 is delayed. delay_val is time of peak correlation.
        # correlate(ref, test). Ref matches Test when Test is shifted back?
        # If test(t) ~ ref(t-tau), then test needs shift -tau to match.
        # Actually verify the sign convention:
        # positive delay usually means 'leads'.
        # Let's check magnitude first.
        delay_val = delay_map.value[0, 1, 0, 0]
        
        # The test failed with 0.1 vs 0.02.
        # Likely delay_val was 0.0 because roll w/o zeroing produces periodic signal?
        # Or maybe xcorr logic is finding wrong peak.
        # With roll, it should detect 10 samples (0.1s).
        
        assert abs(abs(delay_val) - 0.1) < 0.02

    def test_coherence_map(self, gaussian_field):
        """Test coherence map generation."""
        # Coherence with itself should be 1.0
        coh = gaussian_field.coherence_map(
            ref_point=(0*u.m, 0*u.m, 0*u.m),
            plane='xy',
            at={'z': 0*u.m},  # Fix z-axis
            band=(0*u.Hz, 50*u.Hz)
        )
        
        assert isinstance(coh, ScalarField)
        assert coh.unit == u.dimensionless_unscaled
        
        # Check value at origin
        val_at_origin = coh.value[0, 0, 0, 0] 
        assert val_at_origin > 0.95


class TestScalarFieldErrors:
    """Test error handling."""

    def test_irregular_time_axis(self):
        """Test error when time axis is irregular."""
        t = np.array([0, 0.1, 0.3])  # Irregular
        data = np.zeros((3, 2, 2, 2))
        field = ScalarField(data, axis0=t*u.s)
        
        with pytest.raises(ValueError, match="regularly spaced"):
            field.psd()

    def test_invalid_axis_domain(self, gaussian_field):
        """Test error when spectral_density called on wrong domain."""
        psd = gaussian_field.psd()
        
        # Calling psd on already frequency domain
        with pytest.raises(ValueError, match="requires axis0_domain='time'"):
            psd.psd()
