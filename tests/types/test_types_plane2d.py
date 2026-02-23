import numpy as np
from astropy import units as u

from gwexpy.types.plane2d import Plane2D


def test_plane2d_init():
    data = np.zeros((10, 10))
    p = Plane2D(
        data,
        axis1_name="real",
        axis2_name="imag",
        yindex=np.arange(10) * u.m,  # axis 1
        xindex=np.arange(10) * u.s,  # axis 0
    )

    assert p.axis1.name == "real"
    assert p.axis2.name == "imag"

    # In GWpy 4.0: axis 0 = xindex, axis 1 = yindex
    assert p.axis1.unit == u.s
    assert p.axis2.unit == u.m

    # Check inheritance of axis_names
    assert p.axis_names == ("real", "imag")


def test_plane2d_swapaxes():
    data = np.arange(6).reshape(2, 3)
    p = Plane2D(data, axis1_name="rows", axis2_name="cols")

    # Current: axis1=rows (size 2), axis2=cols (size 3)
    # p.axis1 size=2, p.axis2 size=3

    p_t = p.swapaxes(0, 1)  # or p.T

    assert p_t.shape == (3, 2)

    # New axis1 (axis0 of new obj) should be old axis2 ("cols")
    assert p_t.axis1.name == "cols"
    assert p_t.axis2.name == "rows"

    assert p_t.axis1.size == 3
    assert p_t.axis2.size == 2


def test_plane2d_accessors():
    p = Plane2D(np.zeros((5, 5)), axis1_name="a", axis2_name="b")

    assert p.axis1 == p.axes[0]
    assert p.axis2 == p.axes[1]


def test_plane2d_swapaxes_metadata():
    data = np.arange(6).reshape(2, 3)
    axis1 = np.array([0.0, 2.0]) * u.s
    axis2 = np.array([10.0, 20.0, 30.0]) * u.m
    # In GWpy 4.0:
    # xindex size must match axis 0 (size 2)
    # yindex size must match axis 1 (size 3)
    p = Plane2D(data, axis1_name="re_s", axis2_name="im_s", xindex=axis1, yindex=axis2)

    s = p.swapaxes(0, 1)

    assert s.axis1.name == "im_s"
    assert s.axis2.name == "re_s"
    assert np.all(s.axis1.index == axis2)
    assert np.all(s.axis2.index == axis1)
    assert np.all(s.value == p.value.T)
