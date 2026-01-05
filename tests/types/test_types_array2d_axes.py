
import numpy as np
from astropy import units as u

# Try importing Array2D.
# Note: gwexpy.types.array2d.Array2D must be available.
from gwexpy.types.array2d import Array2D

def test_array2d_axes():
    data = np.zeros((10, 5))
    # gwpy Array2D: axis0=y (10), axis1=x (5)
    # xindex size 5, yindex size 10

    arr = Array2D(data,
                  xindex=np.arange(5)*u.m,
                  yindex=np.arange(10)*u.s)

    # Check default names
    assert arr.axes[0].name == "axis0"
    assert arr.axes[1].name == "axis1"

    # Check sizes/units
    assert arr.axes[0].size == 10
    assert arr.axes[0].unit == u.s
    assert arr.axes[1].size == 5
    assert arr.axes[1].unit == u.m

def test_array2d_rename_axes():
    data = np.zeros((10, 5))
    arr = Array2D(data, xindex=np.arange(5), yindex=np.arange(10))

    # Rename
    arr2 = arr.rename_axes({"axis0": "time", "axis1": "space"})

    assert arr2.axes[0].name == "time"
    assert arr2.axes[1].name == "space"

    # Original unchanged
    assert arr.axes[0].name == "axis0"

    # axis descriptor access
    assert arr2.axis("time").size == 10

def test_array2d_swapaxes():
    data = np.random.randn(10, 5)
    arr = Array2D(data, xindex=np.arange(5)*u.m, yindex=np.arange(10)*u.s,
                  axis_names=("time", "space"))

    # swapaxes(0, 1) or T
    # axis0 was time (10), axis1 was space (5)
    # New: axis0 should be space, axis1 should be time

    arr_t = arr.swapaxes(0, 1)

    assert arr_t.shape == (5, 10)
    assert arr_t.axes[0].name == "space"
    assert arr_t.axes[1].name == "time"

    assert arr_t.axes[0].size == 5
    assert arr_t.axes[1].size == 10

    # Check data consistency
    assert arr_t.value[0, 0] == data[0, 0]
    # arr[0, 1] is time=0, space=1. value data[0,1]
    # arr_t[1, 0] should be same value.
    assert arr_t.value[1, 0] == data[0, 1]
