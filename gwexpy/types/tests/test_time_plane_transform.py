
import numpy as np
from astropy import units as u
from gwexpy.types import TimePlaneTransform, Array3D, Plane2D

def test_basic_construction():
    """
    Test basic construction of TimePlaneTransform.
    """
    val = np.random.rand(3, 2, 4)
    times = [0, 1, 2] * u.s
    ax1 = [10, 20] * u.m
    ax2 = [1, 2, 3, 4] * u.Hz

    # Construct using underlying Array3D
    data3d = Array3D(
        val,
        unit="m",
        axis_names=["time", "distance", "frequency"],
        axis0=times,
        axis1=ax1,
        axis2=ax2
    )

    tpt = TimePlaneTransform(data3d, kind="test_kind", meta={"a": 1})

    assert tpt.value.shape == (3, 2, 4)
    assert tpt.times.shape == (3,)
    # Verify times content
    assert np.allclose(tpt.times.value, times.value)

    assert tpt.axis1.name == "distance"
    assert tpt.axis2.name == "frequency"

    assert tpt.kind == "test_kind"
    assert tpt.meta == {"a": 1}
    assert tpt.unit == u.m
    assert len(tpt.axes) == 3

def test_plane_default_ordering():
    """
    Test plane() extraction with default ordering.
    """
    val = np.zeros((3, 2, 4))
    # Fill with identifiable data
    # dim 0: time (3), dim 1: ax1 (2), dim 2: ax2 (4)
    for t in range(3):
        for i in range(2):
            for j in range(4):
                val[t, i, j] = t*100 + i*10 + j

    times = [0, 1, 2] * u.s
    ax1 = [10, 20] * u.m
    ax2 = [1, 2, 3, 4] * u.Hz

    data3d = Array3D(val, axis_names=["time", "ax1", "ax2"], axis0=times, axis1=ax1, axis2=ax2)
    tpt = TimePlaneTransform(data3d)

    # Drop time at index 1 -> shape (2, 4)
    # Remaining axes indices in Array3D are 1 (ax1) and 2 (ax2).
    # Default order should be ascending index: ax1, ax2
    p = tpt.plane(drop_axis=0, drop_index=1)

    assert isinstance(p, Plane2D)
    assert p.shape == (2, 4)
    assert p.axis1.name == "ax1"
    assert p.axis2.name == "ax2"

    # Check values: at t=1, val = 100 + i*10 + j
    expected = val[1, :, :]
    np.testing.assert_array_equal(p.value, expected)

def test_plane_user_ordering():
    """
    Test plane() extraction with explicit user ordering (transposed).
    """
    val = np.random.rand(3, 2, 4)
    data3d = Array3D(val, axis_names=["time", "ax1", "ax2"])
    tpt = TimePlaneTransform(data3d)

    # Request axis1="ax2", axis2="ax1"
    # Result shape should be (4, 2)
    p = tpt.plane(drop_axis="time", drop_index=0, axis1="ax2", axis2="ax1")

    assert p.shape == (4, 2)
    assert p.axis1.name == "ax2"
    assert p.axis2.name == "ax1"

    # Check values: transpose of (2, 4) -> (4, 2)
    expected = val[0, :, :].T
    np.testing.assert_array_equal(p.value, expected)

def test_at_time_nearest():
    """
    Test at_time method with nearest neighbor.
    """
    val = np.arange(3 * 2 * 2).reshape(3, 2, 2)
    times = [0.0, 1.0, 2.0] * u.s # Regular

    data3d = Array3D(val, axis_names=["time", "a", "b"], axis0=times)
    tpt = TimePlaneTransform(data3d)

    # t = 1.1s -> nearest is 1.0s (index 1)
    p = tpt.at_time(1.1 * u.s)

    assert p.shape == (2, 2)
    expected = val[1, :, :]
    np.testing.assert_array_equal(p.value, expected)

    # t = 0.1 -> nearest 0.0 (index 0)
    p0 = tpt.at_time(0.1 * u.s)
    np.testing.assert_array_equal(p0.value, val[0])

def test_tuple_construction():
    """Test construction from tuple."""
    val = np.ones((2, 2, 2))
    t = [0, 1] * u.s
    a1 = [1, 2] * u.m
    a2 = [3, 4] * u.Hz

    tpt = TimePlaneTransform((val, t, a1, a2))
    assert tpt.shape == (2, 2, 2)
    assert tpt.axis1.unit == u.m
    # Default names
    assert tpt.axes[1].name == "axis1"
    assert tpt.axes[2].name == "axis2"

