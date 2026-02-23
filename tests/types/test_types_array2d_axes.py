import numpy as np
from astropy import units as u

# Try importing Array2D.
# Note: gwexpy.types.array2d.Array2D must be available.
from gwexpy.types.array2d import Array2D


def test_array2d_axes():
    data = np.zeros((5, 10))
    # gwpy 4.0 Array2D: axis0=x (5), axis1=y (10)

    arr = Array2D(data, xindex=np.arange(5) * u.m, yindex=np.arange(10) * u.s)

    # Check default names
    assert arr.axes[0].name == "axis0"
    assert arr.axes[1].name == "axis1"

    # Check sizes/units
    assert arr.axes[0].size == 5
    assert arr.axes[0].unit == u.m
    assert arr.axes[1].size == 10
    assert arr.axes[1].unit == u.s


def test_array2d_rename_axes():
    data = np.zeros((5, 10))
    arr = Array2D(data, xindex=np.arange(5), yindex=np.arange(10))

    # Rename
    arr2 = arr.rename_axes({"axis0": "space", "axis1": "time"})

    assert arr2.axes[0].name == "space"
    assert arr2.axes[1].name == "time"

    # Original unchanged
    assert arr.axes[0].name == "axis0"

    # axis descriptor access
    assert arr2.axis("space").size == 5


def test_array2d_swapaxes():
    data = np.random.randn(5, 10)
    arr = Array2D(
        data,
        xindex=np.arange(5) * u.m,
        yindex=np.arange(10) * u.s,
        axis_names=("space", "time"),
    )

    # swapaxes(0, 1) or T
    # axis0 was space (5), axis1 was time (10)
    # New: axis0 should be time, axis1 should be space

    arr_t = arr.swapaxes(0, 1)

    assert arr_t.shape == (10, 5)
    assert arr_t.axes[0].name == "time"
    assert arr_t.axes[1].name == "space"

    assert arr_t.axes[0].size == 10
    assert arr_t.axes[1].size == 5

    # Check data consistency
    assert arr_t.value[0, 0] == data[0, 0]
    # arr[1, 0] is space=1, time=0. value data[1,0]
    # arr_t[0, 1] should be same value.
    assert arr_t.value[0, 1] == data[1, 0]
