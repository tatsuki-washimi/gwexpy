import pytest
import numpy as np
from astropy import units as u
from gwexpy.frequencyseries import BifrequencyMap

def test_bifrequencymap_inverse():
    # Create a simple 2x2 invertible matrix
    # [4, 7]
    # [2, 6]
    # Det = 24 - 14 = 10. Inverse should be:
    # 1/10 * [6, -7]
    #        [-2, 4]
    # = [0.6, -0.7]
    #   [-0.2, 0.4]
    
    data = np.array([[4, 7], [2, 6]], dtype=float)
    f2 = u.Quantity([10, 20], unit='Hz') # Rows
    f1 = u.Quantity([100, 200], unit='Hz') # Cols
    
    # Unit: m/V
    unit = u.m / u.V
    
    bfm = BifrequencyMap.from_points(data, f2, f1, unit=unit)
    
    # Calculate inverse
    inv_bfm = bfm.inverse()
    
    # Check values
    expected_inv = np.linalg.inv(data)
    np.testing.assert_allclose(inv_bfm.value, expected_inv)
    
    # Check axes swapped
    # New f2 (Y) should be old f1 (Cols)
    assert np.all(inv_bfm.frequency2 == f1)
    # New f1 (X) should be old f2 (Rows)
    assert np.all(inv_bfm.frequency1 == f2)
    
    # Check unit inverted
    assert inv_bfm.unit == u.V / u.m
    
    # Verify product is identity (Identity matrix)
    # Note: Matrix multiplication A @ A_inv = I
    # But here axes are swapped, so we need to be careful about what multiplication means physically.
    # BifrequencyMap multiply logic is not strictly defined as A * B in the class, 
    # but as propagate (A @ x).
    # If we just do matrix multiplication of values:
    product_val = bfm.value @ inv_bfm.value
    np.testing.assert_allclose(product_val, np.eye(2), atol=1e-10)

def test_bifrequencymap_inverse_rectangular():
    # Test with rectangular matrix (pseudo-inverse)
    # 3x2 matrix
    data = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    f2 = u.Quantity([10, 20, 30], unit='Hz') # Rows
    f1 = u.Quantity([100, 200], unit='Hz') # Cols
    
    bfm = BifrequencyMap.from_points(data, f2, f1)
    
    # Inverse
    inv_bfm = bfm.inverse()
    
    # Check shape: should be (2, 3)
    assert inv_bfm.shape == (2, 3)
    
    # Check axes
    assert len(inv_bfm.frequency2) == 2
    assert len(inv_bfm.frequency1) == 3
    assert np.all(inv_bfm.frequency2 == f1)
    assert np.all(inv_bfm.frequency1 == f2)
    
    # Check pseudo-inverse property: A * A_pinv * A = A
    recon = bfm.value @ inv_bfm.value @ bfm.value
    np.testing.assert_allclose(recon, bfm.value, atol=1e-10)

if __name__ == "__main__":
    # Allow running directly
    test_bifrequencymap_inverse()
    test_bifrequencymap_inverse_rectangular()
    print("All tests passed!")
