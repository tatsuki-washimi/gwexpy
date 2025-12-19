
import pytest
import numpy as np
from astropy import units as u
from gwexpy.types.array3d import Array3D

def test_sel_irregular_nearest():
    # Irregular freq axis
    freqs = np.array([10, 20, 25, 40, 50]) * u.Hz
    data = np.random.randn(2, 2, 5) # (time, something, freq)
    
    arr = Array3D(data, axis2=freqs, axis_names=("t", "x", "f"))
    
    # Select f = 22 Hz -> expected 20 Hz (diff 2) vs 25 Hz (diff 3). -> 20 Hz (index 1)
    # Actually wait. 22-20=2, 25-22=3. So 20 is closer.
    
    sliced = arr.sel(f=22*u.Hz)
    
    # Result should be 2D (time, x)
    # But sliced axis2 is dropped?
    # sel(scalar) -> dropped axis or kept as scalar dimension?
    # isel returns dropped dimension for scalar int.
    # So sel should too.
    
    # Since we dropped an axis, sliced might be Plane2D.
    # Remaining axes 0, 1.
    assert sliced.shape == (2, 2)
    # Verify value match
    # sliced value should be data[:, :, 1]
    assert np.all(sliced.value == data[:, :, 1])

def test_sel_slice():
    # axis 0 time: 0, 1, 2, 3, 4, 5
    times = np.arange(6) * u.s
    data = np.random.randn(6, 2, 2)
    arr = Array3D(data, axis0=times, axis_names=("t", "x", "y"))
    
    # Slice 1.5s to 4.5s
    # indices:
    # 1.5 -> searchsorted left -> index 2 (val 2)
    # 4.5 -> index 5 (val 5)
    # slice(2, 5) -> indices 2, 3, 4. Values 2, 3, 4. 
    
    sliced = arr.sel(t=slice(1.5*u.s, 4.5*u.s))
    
    assert sliced.shape == (3, 2, 2)
    assert sliced.axes[0].size == 3
    assert sliced.axes[0].index[0] == 2*u.s
    assert sliced.axes[0].index[-1] == 4*u.s

def test_isel_mixed():
    data = np.zeros((4, 4, 4))
    arr = Array3D(data)
    
    # slice axis0, int axis1, slice axis2
    # axis0=slice(0,2), axis1=2, axis2=slice(None)
    
    # Result: 2D (since one int). Shape (2, 4).
    # Remaining axes 0 and 2.
    # Should become Plane2D with axis1->axis0("axis0"), axis2->axis2("axis2")
    
    sliced = arr.isel(axis0=slice(0, 2), axis1=2)
    
    assert sliced.shape == (2, 4)
    # axis_names logic in Array3D.isel for 2D return
    # kept axes: (axis0, slice(0,2)), (axis2, all)
    assert sliced.axis_names == ("axis0", "axis2")
    assert sliced.axis1.name == "axis0"
    assert sliced.axis2.name == "axis2"

def test_sel_nonuniform_axis():
    freqs = np.array([10, 20, 25, 40, 50]) * u.Hz
    data = np.arange(2 * 3 * 5).reshape(2, 3, 5)
    arr = Array3D(data, axis2=freqs, axis_names=("t", "x", "f"))

    sliced = arr.sel(f=22 * u.Hz, method="nearest")

    assert sliced.shape == (2, 3)
    assert np.all(sliced.value == data[:, :, 1])
