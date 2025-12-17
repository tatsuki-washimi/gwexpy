#!/usr/bin/env python
"""
Unit tests for TimeSeriesMatrix to ensure it inherits SeriesMatrix safety fixes properly.
"""

import numpy as np
from astropy import units as u
from gwexpy.timeseries import TimeSeriesMatrix

def test_timeseriesmatrix_inherits_value_stability():
    """Test that TimeSeriesMatrix inherits _value stability from SeriesMatrix."""
    print("=" * 60)
    print("Test 1: TimeSeriesMatrix _value stability")
    print("=" * 60)
    
    data = np.random.randn(2, 3, 50)
    dt = 0.01  # 10 ms
    t0 = 0.0
    
    tsm = TimeSeriesMatrix(data, dt=dt, t0=t0)
    
    # Check that _value exists and is properly used
    assert hasattr(tsm, '_value'), "TimeSeriesMatrix should inherit _value from SeriesMatrix"
    assert tsm._value.shape == (2, 3, 50), f"_value shape incorrect: {tsm._value.shape}"
    assert tsm.N_samples == 50, f"N_samples should be 50, got {tsm.N_samples}"
    
    print(f"✓ _value inherited and stable: {tsm._value.shape}")
    print(f"✓ N_samples preserved: {tsm.N_samples}")
    print("PASS\n")


def test_timeseriesmatrix_xarray_no_double_units():
    """Test that TimeSeriesMatrix xarray doesn't double-apply units."""
    print("=" * 60)
    print("Test 2: TimeSeriesMatrix xarray no double-unit")
    print("=" * 60)
    
    data = np.random.randn(2, 2, 30)
    dt = 0.1  # 100 ms
    t0 = 0.0
    
    tsm = TimeSeriesMatrix(data, dt=dt, t0=t0)
    
    # xarray should have seconds unit
    xarray = tsm.xarray
    print(f"xarray: {xarray[:5]}... (first 5)")
    print(f"xarray unit: {xarray.unit if hasattr(xarray, 'unit') else 'N/A'}")
    
    assert hasattr(xarray, 'unit'), "xarray should be a Quantity"
    assert xarray.unit == u.s, f"xarray unit should be s, got {xarray.unit}"
    
    # duration should also have seconds
    duration = tsm.duration
    print(f"duration: {duration}")
    
    assert hasattr(duration, 'unit'), "duration should be a Quantity"
    assert duration.unit == u.s, f"duration unit should be s, got {duration.unit}"
    
    print("✓ xarray has correct unit (seconds, not seconds²)")
    print("✓ duration has correct unit (seconds, not seconds²)")
    print("PASS\n")


def test_timeseriesmatrix_quantity_default_units():
    """Test that Quantity input to TimeSeriesMatrix sets default units."""
    print("=" * 60)
    print("Test 3: TimeSeriesMatrix Quantity input default units")
    print("=" * 60)
    
    data_values = np.random.randn(2, 2, 25)
    data_quantity = u.Quantity(data_values, 'V')  # Volts
    dt = 0.04
    t0 = 0.0
    
    tsm = TimeSeriesMatrix(data_quantity, dt=dt, t0=t0)
    
    # Check that all cells have Volts unit
    for i in range(2):
        for j in range(2):
            unit = tsm.meta[i, j].unit
            assert unit == u.V, f"Cell ({i},{j}) should have unit V, got {unit}"
    
    print("✓ All cells have Volts unit from input Quantity")
    print("PASS\n")


if __name__ == "__main__":
    try:
        test_timeseriesmatrix_inherits_value_stability()
        test_timeseriesmatrix_xarray_no_double_units()
        test_timeseriesmatrix_quantity_default_units()
        
        print("\n" + "=" * 60)
        print("ALL TimeSeriesMatrix TESTS PASSED ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
