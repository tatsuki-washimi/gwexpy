
import numpy as np
from astropy import units as u
from gwexpy.fields import ScalarField

def verify_noise():
    print("Verifying ScalarField.simulate('gaussian')...")
    # Gaussian
    field_gauss = ScalarField.simulate(
        'gaussian', 
        shape=(10, 5, 5, 5), 
        sample_rate=100*u.Hz, 
        space_step=0.1*u.m,
        std=2.0,
        mean=1.0,
        seed=42
    )
    print(f"Gaussian Shape: {field_gauss.shape}")
    print(f"Gaussian Unit: {field_gauss.unit}")
    assert field_gauss.shape == (10, 5, 5, 5)
    assert field_gauss._axis0_domain == 'time'
    
    # Check stats
    assert np.allclose(np.mean(field_gauss.value), 1.0, atol=0.5) # rough check
    
    print("Verifying ScalarField.simulate('plane_wave')...")
    # Plane wave
    # f=10Hz, kx=1 (1/m)
    field_wave = ScalarField.simulate(
        'plane_wave',
        frequency=10*u.Hz,
        k_vector=(1.0/u.m, 0/u.m, 0/u.m),
        shape=(100, 10, 10, 10),
        sample_rate=100*u.Hz,
        space_step=0.1*u.m
    )
    print(f"Wave Shape: {field_wave.shape}")
    # Check if it oscillates
    assert np.std(field_wave.value) > 0.1
    
    print("ALL CHECKS PASSED")

if __name__ == "__main__":
    verify_noise()
