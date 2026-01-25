import numpy as np
from astropy import units as u

from gwexpy.types.plane2d import Plane2D


def test_plane2d_init():
    data = np.zeros((10, 10))
    p = Plane2D(
        data,
        axis1_name="real",
        axis2_name="imag",
        yindex=np.arange(10) * u.m,  # axis0/axis1 -> y/x in Array2D
        xindex=np.arange(10) * u.s,
    )

    assert p.axis1.name == "real"
    assert p.axis2.name == "imag"

    assert p.axis1.unit == u.m
    assert p.axis2.unit == u.s

    # Check inheritance of axis_names
    assert p.axis_names == ("real", "imag")


def test_plane2d_swapaxes():
    data = np.arange(6).reshape(2, 3)
    p = Plane2D(data, axis1_name="rows", axis2_name="cols")

    # Current: axis1=rows, axis2=cols
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

    # Should not expose x/y logic if we really want to hide it,
    # but underlying Array2D might have .x .y properties.
    # The requirement was "Avoid exposing x/y terms in the user-facing API" in this class definition context
    # but we can't delete them from instance easily without breaking things or complex descriptors.
    # Verification that we don't *use* u/v is main point.


def test_plane2d_swapaxes_metadata():
    data = np.arange(6).reshape(2, 3)
    axis1 = np.array([0.0, 2.0]) * u.s
    axis2 = np.array([10.0, 20.0, 30.0]) * u.m
    p = Plane2D(data, axis1_name="re_s", axis2_name="im_s", yindex=axis1, xindex=axis2)

    s = p.swapaxes(0, 1)

    assert s.axis1.name == "im_s"
    assert s.axis2.name == "re_s"
    assert np.all(s.axis1.index == axis2)
    assert np.all(s.axis2.index == axis1)
    assert np.all(s.value == p.value.T)
