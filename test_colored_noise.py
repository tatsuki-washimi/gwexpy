
import numpy as np
import pytest
from astropy import units as u
from gwexpy.noise import from_pygwinc, from_obspy, power_law, white_noise, pink_noise, red_noise

# Test colored noise
def test_power_law():
    frequencies = np.arange(1, 101, 1.0)
    
    # 1. White Noise (exponent=0)
    white = power_law(0.0, amplitude=1e-23, frequencies=frequencies)
    assert np.allclose(white.value, 1e-23)
    
    # 2. Pink Noise (exponent=0.5) => ASD ~ f^-0.5
    pink = power_law(0.5, amplitude=1.0, f_ref=1.0, frequencies=frequencies)
    # Check at f=1 (ref) => 1.0 * (1/1)^0.5 = 1.0
    assert np.isclose(pink.value[0], 1.0) 
    # Check at f=4 => 1.0 * (4/1)^-0.5 = 0.5
    idx_4 = np.where(frequencies == 4.0)[0][0]
    assert np.isclose(pink.value[idx_4], 0.5)

    # 3. Red Noise (exponent=1.0) => ASD ~ f^-1
    red = power_law(1.0, amplitude=1.0, f_ref=1.0, frequencies=frequencies)
    # Check at f=10 => 1.0 * (10/1)^-1 = 0.1
    idx_10 = np.where(frequencies == 10.0)[0][0]
    assert np.isclose(red.value[idx_10], 0.1)

def test_noise_aliases():
    freqs = np.linspace(10, 100, 100)
    amp = 1e-20
    
    wn = white_noise(amp, frequencies=freqs)
    assert np.allclose(wn.value, amp)
    
    pn = pink_noise(amp, frequencies=freqs)
    # Check slope roughly? exponent=0.5
    # ratio of f=100 to f=10 is 10. amplitude ratio should be 10^-0.5 = 1/sqrt(10) ~ 0.316
    ratio_f = freqs[-1] / freqs[0] # 10
    ratio_a = pn.value[-1] / pn.value[0]
    assert np.isclose(ratio_a, ratio_f**(-0.5))
    
    rn = red_noise(amp, frequencies=freqs)
    # ratio should be 10^-1 = 0.1
    ratio_a_red = rn.value[-1] / rn.value[0]
    assert np.isclose(ratio_a_red, ratio_f**(-1.0))
