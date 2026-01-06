
import numpy as np
from astropy import units as u
from gwexpy.frequencyseries import FrequencySeries

def test_calculus():
    # Create a frequency series
    f = np.linspace(0, 10, 11) * u.Hz
    val = np.ones(11) # Constant 1
    fs = FrequencySeries(val, frequencies=f, unit="m", name="test")

    print("Checking differentiate...")
    # Differentiate: d/dt corresponds to * (i 2 pi f)
    # Different from d/df. The method name is differentiate, docstring says "Differentiate ... in frequency domain" 
    # but implementation is X(f) * (i 2 pi f).
    # This corresponds to differentiation in TIME domain: d/dt(x(t)) <-> i*2*pi*f * X(f).
    # If the user meant d/df (differentiation regarding frequency), that's different.
    # The docstring I wrote: "Differentiate the FrequencySeries in the frequency domain. Multiplies by (i * 2 * pi * f)^order."
    # Wait, "Differentiate ... in the frequency domain" is ambiguous. 
    # Usually "differentiation in frequency domain" means d/df X(f) -> -i 2 pi t x(t).
    # But usually when people ask for differentiate on a FrequencySeries which represents a spectrum of a time series,
    # they want the spectrum of the derivative of the time series, i.e. i omega X(f).
    # I should verify this interpretation.
    
    diff_fs = fs.differentiate()
    expected_val = val * (1j * 2 * np.pi * f.value)
    
    assert np.allclose(diff_fs.value, expected_val)
    print("Differentiate value correct.")
    print(f"Differentiation unit: {diff_fs.unit}")
    assert diff_fs.unit == u.m * u.Hz

    print("Checking integrate...")
    # Integrate: int dt corresponds to / (i 2 pi f)
    int_fs = fs.integrate()
    
    # Check f=0 handling (set to 0 in my impl)
    assert int_fs.value[0] == 0j
    
    # Check other values
    with np.errstate(divide='ignore', invalid='ignore'):
         expected_int_val = val * (1j * 2 * np.pi * f.value)**(-1)
    expected_int_val[0] = 0
    
    assert np.allclose(int_fs.value, expected_int_val)
    print("Integrate value correct.")
    print(f"Integration unit: {int_fs.unit}")
    assert int_fs.unit == u.m * u.s

if __name__ == "__main__":
    test_calculus()
