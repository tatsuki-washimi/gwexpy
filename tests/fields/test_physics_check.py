import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField
from gwexpy.fields.tensor import TensorField
from gwexpy.fields.vector import VectorField


@pytest.fixture
def base_field():
    return ScalarField(np.ones((10, 4, 4, 4)), unit=u.dimensionless_unscaled)


def test_physics_units():
    # VectorField: Dot product should have unit V^2 if V
    f_v = ScalarField(np.ones((10, 4, 4, 4)), unit=u.V)
    vf = VectorField({"x": f_v, "y": f_v * 2})
    dot_result = vf.dot(vf)
    assert dot_result.unit.is_equivalent(u.V**2)

    # VectorField: Norm should preserve unit V
    assert vf.norm().unit.is_equivalent(u.V)

    # VectorField: Cross product unit should be m^2 for (m, m)
    f_m = ScalarField(np.ones((10, 4, 4, 4)), unit=u.m)
    v1 = VectorField({"x": f_m, "y": f_m * 0, "z": f_m * 0})
    v2 = VectorField({"x": f_m * 0, "y": f_m, "z": f_m * 0})
    assert v1.cross(v2)["z"].unit.is_equivalent(u.m**2)

    # TensorField: det() for 2x2 should have unit Pa^2
    f_pa = ScalarField(np.ones((10, 4, 4, 4)), unit=u.Pa)
    tf = TensorField(
        {(0, 0): f_pa * 2, (0, 1): f_pa * 0, (1, 0): f_pa * 0, (1, 1): f_pa * 3}, rank=2
    )
    assert tf.det().unit.is_equivalent(u.Pa**2)
    assert tf.trace().unit.is_equivalent(u.Pa)


def test_math_invariants(base_field):
    # VectorField: |v|^2 == v.dot(v)
    f_x = ScalarField(np.ones((10, 4, 4, 4)), unit=u.V)
    vf = VectorField({"x": f_x * 3, "y": f_x * 8})
    assert np.allclose((vf.norm() ** 2).value, vf.dot(vf).value)

    # VectorField: i x j = k
    vi = VectorField({"x": base_field, "y": base_field * 0, "z": base_field * 0})
    vj = VectorField({"x": base_field * 0, "y": base_field, "z": base_field * 0})
    vk = vi.cross(vj)
    assert np.allclose(vk["x"].value, 0)
    assert np.allclose(vk["y"].value, 0)
    assert np.allclose(vk["z"].value, 1)

    # TensorField: Trace of identity
    ident_2x2 = TensorField(
        {
            (0, 0): base_field,
            (0, 1): base_field * 0,
            (1, 0): base_field * 0,
            (1, 1): base_field.copy(),
        },
        rank=2,
    )
    assert np.isclose(ident_2x2.trace().value.mean(), 2.0)
    assert np.isclose(ident_2x2.det().value.mean(), 1.0)

    # TensorField: Symmetrize antisymmetric
    f_pa = ScalarField(np.ones((10, 4, 4, 4)), unit=u.Pa)
    tf_antisym = TensorField({(0, 1): f_pa, (1, 0): f_pa * -1}, rank=2)
    tf_sym = tf_antisym.symmetrize()
    assert np.isclose(tf_sym[(0, 1)].value.mean(), 0.0)


def test_conservation_laws():
    # Create a time-domain signal and verify energy conservation
    N = 128
    dt = 0.01 * u.s
    times = np.arange(N) * dt
    signal_data = np.random.randn(N, 4, 4, 4)

    sf_time = ScalarField(signal_data, axis0=times, unit=u.V, axis0_domain="time")

    # FFT and IFFT round-trip
    sf_freq = sf_time.fft_time()
    sf_time_reconstructed = sf_freq.ifft_time()

    reconstruction_error = np.max(np.abs(sf_time.value - sf_time_reconstructed.value))
    assert reconstruction_error < 1e-10

    # Variance ratio should be 1.0
    time_variance = np.var(sf_time.value)
    reconstructed_variance = np.var(sf_time_reconstructed.value)
    assert np.isclose(reconstructed_variance / time_variance, 1.0, rtol=1e-6)
