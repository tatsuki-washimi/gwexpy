import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField
from gwexpy.fields.vector import VectorField


def test_vectorfield_init():
    data = np.zeros((10, 4, 4, 4))
    fx = ScalarField(data + 1, unit=u.V)
    fy = ScalarField(data + 2, unit=u.V)

    vf = VectorField({"x": fx, "y": fy}, basis="cartesian")
    assert vf.basis == "cartesian"
    assert vf["x"] is fx
    assert vf["y"] is fy


def test_vectorfield_validation():
    data = np.zeros((10, 4, 4, 4))
    fx = ScalarField(data, unit=u.V)
    fy = ScalarField(data, unit=u.m)  # Different unit

    with pytest.raises(ValueError, match="Inconsistent unit"):
        VectorField({"x": fx, "y": fy})


def test_vectorfield_norm():
    data = np.zeros((10, 4, 4, 4))
    # Vector (3, 4) -> norm 5
    fx = ScalarField(data + 3, unit=u.V)
    fy = ScalarField(data + 4, unit=u.V)
    vf = VectorField({"x": fx, "y": fy})

    n = vf.norm()
    assert isinstance(n, ScalarField)
    assert np.all(n.value == 5)
    assert n.unit == u.V


def test_vectorfield_to_array():
    data = np.zeros((10, 4, 4, 4))
    fx = ScalarField(data + 1, unit=u.V)
    fy = ScalarField(data + 2, unit=u.V)
    vf = VectorField({"x": fx, "y": fy})

    arr = vf.to_array()
    assert arr.shape == (10, 4, 4, 4, 2)
    assert np.all(arr[..., 0] == 1)
    assert np.all(arr[..., 1] == 2)


def test_vectorfield_fft():
    # Test batch FFT
    times = np.linspace(0, 1, 10, endpoint=False) * u.s
    data = np.random.randn(10, 4, 4, 4)
    f1 = ScalarField(data, axis0=times, unit=u.V)
    vf = VectorField({"x": f1})

    vf_freq = vf.fft_time_all()
    assert isinstance(vf_freq, VectorField)
    assert "x" in vf_freq
    assert vf_freq["x"].axis0_domain == "frequency"


def test_vectorfield_arithmetic():
    data = np.ones((10, 4, 4, 4))
    f1 = ScalarField(data, unit=u.V)
    vf = VectorField({"x": f1})

    vf2 = vf * 2
    assert np.all(vf2["x"].value == 2)
    assert isinstance(vf2, VectorField)


def test_vectorfield_resample():
    # 100 Hz, 1 second
    times = np.linspace(0, 1, 100, endpoint=False) * u.s
    data = np.random.randn(100, 4, 4, 4)
    f1 = ScalarField(data, axis0=times, unit=u.V)
    vf = VectorField({"x": f1})

    # Resample to 50 Hz
    vf_resampled = vf.resample_all(50)
    assert vf_resampled["x"].shape[0] == 50
    assert isinstance(vf_resampled, VectorField)


def test_vectorfield_filter():
    times = np.linspace(0, 1, 1000, endpoint=False) * u.s
    # Sum of 10Hz and 100Hz
    data = np.sin(2 * np.pi * 10 * times.value) + np.sin(2 * np.pi * 100 * times.value)
    data = data[:, np.newaxis, np.newaxis, np.newaxis] * np.ones((1, 4, 4, 4))

    f1 = ScalarField(data, axis0=times, unit=u.V)
    vf = VectorField({"x": f1})

    # Lowpass at 20Hz
    from gwpy.signal import filter_design

    fs = 1.0 / (times[1] - times[0]).to("s").value
    lp = filter_design.lowpass(20, fs)
    vf_filt = vf.filter_all(lp)

    # 100Hz should be gone
    # Note: filter might have some transients, check middle
    assert np.max(np.abs(vf_filt["x"].value[200:800])) < 1.1  # Should be close to 1
    assert isinstance(vf_filt, VectorField)


def test_vectorfield_dot():
    data_x = np.ones((10, 4, 4, 4))
    data_y = np.ones((10, 4, 4, 4)) * 2
    f_x = ScalarField(data_x, unit=u.V)
    f_y = ScalarField(data_y, unit=u.V)
    vf1 = VectorField({"x": f_x, "y": f_y})
    vf2 = VectorField({"x": f_x, "y": f_y})

    dot12 = vf1.dot(vf2)
    # 1*1 + 2*2 = 5
    assert np.all(dot12.value == 5)
    assert isinstance(dot12, ScalarField)
    assert dot12.unit == u.V**2


def test_vectorfield_cross():
    f_x = ScalarField(np.ones((10, 4, 4, 4)), unit=u.m)  # i
    f_y = ScalarField(np.ones((10, 4, 4, 4)), unit=u.m)  # j
    f_z = ScalarField(np.zeros((10, 4, 4, 4)), unit=u.m)

    v1 = VectorField({"x": f_x, "y": f_z, "z": f_z})  # (1, 0, 0)
    v2 = VectorField({"x": f_z, "y": f_y, "z": f_z})  # (0, 1, 0)

    # i x j = k
    v3 = v1.cross(v2)
    assert np.all(v3["x"].value == 0)
    assert np.all(v3["y"].value == 0)
    assert np.all(v3["z"].value == 1)
    assert isinstance(v3, VectorField)


def test_vectorfield_project():
    f_1 = ScalarField(np.ones((10, 4, 4, 4)), unit=u.m)
    v1 = VectorField({"x": f_1 * 5, "y": f_1 * 0})  # (5, 0)
    v2 = VectorField({"x": f_1, "y": f_1})  # (1, 1) -> norm = sqrt(2)

    # proj = (5*1 + 0*1) / sqrt(2) = 5 / sqrt(2)
    proj = v1.project(v2)
    assert np.allclose(proj.value, 5 / np.sqrt(2))
    assert isinstance(proj, ScalarField)


def test_vectorfield_complex_norm():
    # Test norm of complex data (e.g. after FFT)
    data = np.ones((10, 4, 4, 4), dtype=complex) * (3 + 4j)
    f = ScalarField(data, unit=u.V)
    vf = VectorField({"x": f})

    n = vf.norm()
    # abs(3+4j) = 5
    assert np.allclose(n.value, 5)
    assert n.unit == u.V
    assert n._axis0_index is not None


def test_vectorfield_errors():
    # Empty VectorField
    vf_empty = VectorField({})
    with pytest.raises(ValueError, match="Cannot compute norm"):
        vf_empty.norm()

    with pytest.raises(ValueError, match="Cannot compute dot"):
        vf_empty.dot(vf_empty)

    f = ScalarField(np.ones((10, 4, 4, 4)))
    v = VectorField({"x": f})

    # Dot with non-VectorField
    with pytest.raises(TypeError, match="dot expects VectorField"):
        v.dot(f)

    # Mismatched components
    v2 = VectorField({"y": f})
    with pytest.raises(ValueError, match="Component 'x' missing"):
        v.dot(v2)

    # Cross product dim error
    with pytest.raises(ValueError, match="only defined for 3-component"):
        v.cross(v2)


def test_vectorfield_axis_validation():
    # Different axis0 lengths
    f1 = ScalarField(np.ones((10, 4, 4, 4)))
    f2 = ScalarField(np.ones((11, 4, 4, 4)))

    with pytest.raises(ValueError, match="Axis 0 shape mismatch"):
        VectorField({"x": f1, "y": f2})


def test_vectorfield_norm_complex_components_preserves_metadata():
    axis0 = np.linspace(0, 1, 4, endpoint=False) * u.s
    data_x = np.array([1 + 2j, -2 + 0.5j, 0.3 - 1j, -0.1 + 0.0j])
    data_y = np.array([2 - 1j, 1 + 1j, -0.5 + 0.2j, 0.2 - 0.3j])

    data_x = data_x[:, None, None, None]
    data_y = data_y[:, None, None, None]

    fx = ScalarField(data_x, axis0=axis0, unit=u.V)
    fy = ScalarField(data_y, axis0=axis0, unit=u.V)

    vf = VectorField({"x": fx, "y": fy})
    norm = vf.norm()

    expected = np.sqrt(np.abs(data_x) ** 2 + np.abs(data_y) ** 2)
    assert np.allclose(norm.value, expected)
    assert norm.unit == u.V
    assert norm.axis0_domain == fx.axis0_domain
    assert norm.axis_names == fx.axis_names
    assert norm.space_domains == fx.space_domains


def test_vectorfield_axis_mismatch_raises():
    axis_long = np.linspace(0, 1, 5) * u.s
    axis_short = np.linspace(0, 1, 4) * u.s

    f_long = ScalarField(np.ones((5, 2, 2, 2)), axis0=axis_long, unit=u.m)
    f_short = ScalarField(np.ones((4, 2, 2, 2)), axis0=axis_short, unit=u.m)

    with pytest.raises(ValueError, match="Axis 0 shape mismatch"):
        VectorField({"x": f_long, "y": f_short})
