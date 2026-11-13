"""Real-space coordinates after ``fft_space() -> ifft_space()`` (#544)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.fft as sp_fft
from astropy import units as u

from gwexpy.fields import ScalarField

N_T = 3


def _axis(origin, step, n, unit=u.m):
    return (origin + step * np.arange(n)) * unit


def _field(x, y=None, z=None, *, scale=1e-21):
    y = _axis(0.0, 1.0, 4) if y is None else y
    z = _axis(0.0, 1.0, 2) if z is None else z
    rng = np.random.default_rng(544)
    data = rng.normal(size=(N_T, len(x), len(y), len(z))) * scale
    return ScalarField(
        data,
        unit=u.V,
        axis0=np.arange(N_T) * 0.5 * u.s,
        axis1=x,
        axis2=y,
        axis3=z,
        axis_names=["t", "x", "y", "z"],
    )


def _assert_axis(actual, expected):
    assert actual.unit == expected.unit
    np.testing.assert_allclose(
        actual.value, expected.value, rtol=1e-12, atol=1e-12 * abs(expected.value).max()
    )


@pytest.mark.parametrize(
    ("origin", "step"),
    [(0.0, 0.5), (12.5, 0.5), (-3.0, 0.25), (7.0, -0.5), (-2.0, -0.25), (0.0, -1.0)],
    ids=[
        "asc-zero",
        "asc-offset",
        "asc-negative-origin",
        "desc-offset",
        "desc-negative-origin",
        "desc-zero",
    ],
)
@pytest.mark.parametrize("n", [2, 3, 8], ids=["n2", "n3", "n8"])
def test_single_axis_round_trip_restores_coordinates(origin, step, n):
    x = _axis(origin, step, n)
    field = _field(x)

    back = field.fft_space(axes=["x"]).ifft_space(axes=["kx"])

    _assert_axis(back._axis1_index, x)
    assert np.sign(back._axis1_index[1] - back._axis1_index[0]) == np.sign(step)
    np.testing.assert_allclose(back.value.real, field.value, rtol=1e-10, atol=1e-33)
    np.testing.assert_allclose(back.value.imag, 0.0, atol=1e-33)
    assert back.axis_names == field.axis_names
    assert back.unit == field.unit


def test_axis_unit_is_restored():
    x = _axis(2.0, 5.0, 6, unit=u.cm)

    back = _field(x).fft_space(axes=["x"]).ifft_space(axes=["kx"])

    _assert_axis(back._axis1_index, x)


def test_all_axes_are_restored_independently():
    x = _axis(10.0, 0.5, 4)
    y = _axis(-1.0, -0.25, 5)
    z = _axis(3.0, 2.0, 2)
    field = _field(x, y, z)

    back = field.fft_space().ifft_space()

    _assert_axis(back._axis1_index, x)
    _assert_axis(back._axis2_index, y)
    _assert_axis(back._axis3_index, z)
    np.testing.assert_allclose(back.value.real, field.value, rtol=1e-10, atol=1e-33)


def test_axes_transformed_in_separate_calls_are_restored():
    x = _axis(10.0, 0.5, 4)
    y = _axis(-1.0, -0.25, 5)
    field = _field(x, y)

    k_field = field.fft_space(axes=["x"]).fft_space(axes=["y"])
    back = k_field.ifft_space(axes=["ky"]).ifft_space(axes=["kx"])

    _assert_axis(back._axis1_index, x)
    _assert_axis(back._axis2_index, y)


def test_untransformed_axes_and_time_axis_are_untouched():
    x = _axis(10.0, 0.5, 4)
    y = _axis(-1.0, -0.25, 5)
    field = _field(x, y)

    k_field = field.fft_space(axes=["x"])
    back = k_field.ifft_space(axes=["kx"])

    for result in (k_field, back):
        _assert_axis(result._axis2_index, y)
        _assert_axis(result._axis3_index, field._axis3_index)
        np.testing.assert_array_equal(
            result._axis0_index.value, field._axis0_index.value
        )
    assert back.space_domains == field.space_domains


def test_origin_survives_k_space_arithmetic():
    """Filtering in k-space is the reason to transform at all."""
    x = _axis(12.5, 0.5, 8)
    field = _field(x)

    back = (field.fft_space(axes=["x"]) * 2.0).ifft_space(axes=["kx"])

    _assert_axis(back._axis1_index, x)
    np.testing.assert_allclose(back.value.real, 2.0 * field.value, rtol=1e-10)


def test_second_round_trip_uses_the_current_axis():
    x = _axis(12.5, 0.5, 8)
    back = _field(x).fft_space(axes=["x"]).ifft_space(axes=["kx"])
    shifted = ScalarField(
        back.value.real,
        unit=u.V,
        axis0=back._axis0_index,
        axis1=_axis(-4.0, 0.5, 8),
        axis2=back._axis2_index,
        axis3=back._axis3_index,
        axis_names=["t", "x", "y", "z"],
    )

    again = shifted.fft_space(axes=["x"]).ifft_space(axes=["kx"])

    _assert_axis(again._axis1_index, _axis(-4.0, 0.5, 8))


def test_source_field_is_not_modified():
    x = _axis(12.5, -0.5, 8)
    field = _field(x)
    before = field.value.copy()

    field.fft_space(axes=["x"]).ifft_space(axes=["kx"])

    np.testing.assert_array_equal(field.value, before)
    _assert_axis(field._axis1_index, x)


# --- transform definition is unchanged -----------------------------------------


@pytest.mark.parametrize(("origin", "step"), [(0.0, 0.5), (12.5, 0.5), (7.0, -0.5)])
def test_forward_coefficients_and_k_axis_do_not_depend_on_origin(origin, step):
    n = 8
    field = _field(_axis(origin, step, n))

    k_field = field.fft_space(axes=["x"])

    np.testing.assert_array_equal(k_field.value, sp_fft.fftn(field.value, axes=[1]))
    expected_k = 2 * np.pi * np.fft.fftfreq(n, d=abs(step)) * np.sign(step)
    np.testing.assert_allclose(k_field._axis1_index.value, expected_k, rtol=1e-12)
    assert k_field._axis1_index.unit == 1 / u.m


# --- k-space field built directly: no origin to recover ----------------------------


def _k_field(sign):
    n, dx = 8, 0.5
    k = sign * 2 * np.pi * np.fft.fftfreq(n, d=dx) / u.m
    rng = np.random.default_rng(1)
    return ScalarField(
        rng.normal(size=(N_T, n, 4, 2)),
        unit=u.V,
        axis0=np.arange(N_T) * 0.5 * u.s,
        axis1=k,
        axis2=_axis(0.0, 1.0, 4),
        axis3=_axis(0.0, 1.0, 2),
        axis_names=["t", "kx", "y", "z"],
        space_domain={"kx": "k", "y": "real", "z": "real"},
    )


def test_direct_k_field_ascending_fallback_is_unchanged():
    back = _k_field(+1).ifft_space(axes=["kx"])

    np.testing.assert_allclose(back._axis1_index.value, 0.5 * np.arange(8), rtol=1e-12)
    assert back._axis1_index.unit == u.m


def test_direct_k_field_descending_fallback_is_descending():
    """A sign-flipped k axis means a descending x axis; it used to ascend."""
    back = _k_field(-1).ifft_space(axes=["kx"])

    values = back._axis1_index.value
    assert np.all(np.diff(values) < 0)
    np.testing.assert_allclose(values, 0.5 * np.arange(7, -1, -1), rtol=1e-12)
    assert back._axis1_index.unit == u.m


# --- length-changing transforms -----------------------------------------------------


def test_padded_forward_transform_keeps_origin_and_spacing():
    x = _axis(12.5, 0.5, 4)
    field = _field(x)

    back = field.fft_space(axes=["x"], n=[8]).ifft_space(axes=["kx"])

    assert back.shape[1] == 8
    np.testing.assert_allclose(
        back._axis1_index.value, 12.5 + 0.5 * np.arange(8), rtol=1e-12
    )
    np.testing.assert_allclose(back.value.real[:, :4], field.value, rtol=1e-10)


# --- collections --------------------------------------------------------------------


def test_fieldlist_round_trip_restores_each_member():
    from gwexpy.fields import FieldList

    first, second = _axis(12.5, 0.5, 8), _axis(7.0, -0.5, 8)
    fields = FieldList([_field(first), _field(second)])

    back = fields.fft_space_all(axes=["x"]).ifft_space_all(axes=["kx"])

    _assert_axis(back[0]._axis1_index, first)
    _assert_axis(back[1]._axis1_index, second)
