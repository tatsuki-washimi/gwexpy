
import pytest
import numpy as np
from astropy import units as u
from gwexpy.types.array3d import Array3D
from gwexpy.types.plane2d import Plane2D

@pytest.fixture
def arr3d():
    # shape (3, 4, 5)
    data = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    axis0 = np.arange(3) * u.s
    axis1 = np.arange(4) * u.m
    axis2 = np.arange(5) * u.Hz
    
    return Array3D(data, 
                   axis0=axis0, axis1=axis1, axis2=axis2,
                   axis_names=("time", "dist", "freq"))

def test_array3d_plane_default(arr3d):
    # drop time (axis0), index 0
    # Remaining axes: dist (1), freq (2). Default order 1, 2.
    
    p = arr3d.plane(drop_axis=0, drop_index=0)
    
    assert isinstance(p, Plane2D)
    assert p.shape == (4, 5)
    
    # Plane2D axis1 -> axis0 of plane -> axis1 of original (dist)
    assert p.axis1.name == "dist"
    assert p.axis2.name == "freq"
    
    assert p.axis1.unit == u.m
    assert p.axis2.unit == u.Hz

def test_array3d_plane_reorder(arr3d):
    # drop time, but want axis1=freq, axis2=dist
    p = arr3d.plane(0, 0, axis1="freq", axis2="dist")
    
    assert isinstance(p, Plane2D)
    # Transposed shape
    assert p.shape == (5, 4)
    
    assert p.axis1.name == "freq"
    assert p.axis2.name == "dist"
    
    assert p.axis1.unit == u.Hz
    assert p.axis2.unit == u.m

def test_array3d_plane_by_name(arr3d):
    # drop "freq" (axis2)
    p = arr3d.plane("freq", 0)
    
    assert p.shape == (3, 4)
    assert p.axis1.name == "time"
    assert p.axis2.name == "dist"

def test_array3d_plane_error(arr3d):
    with pytest.raises(ValueError):
        arr3d.plane(0, 0, axis1="dist", axis2="dist")
        
    with pytest.raises(ValueError):
        # dropping "time", cannot use "time" as Output
        arr3d.plane("time", 0, axis1="time", axis2="dist")
