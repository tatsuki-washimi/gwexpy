from typing import Literal, cast

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields.demo import (
    make_demo_scalar_field,
    make_propagating_gaussian,
    make_sinusoidal_wave,
    make_standing_wave,
)
from gwexpy.fields.scalar import ScalarField


def test_make_demo_scalar_field_gaussian_metadata():
    field = make_demo_scalar_field(
        "gaussian", nt=4, nx=3, ny=2, nz=2, dt=0.1 * u.s, dx=0.5 * u.m
    )
    assert isinstance(field, ScalarField)
    assert field.shape == (4, 3, 2, 2)
    assert field.axis0_domain == "time"
    assert field.space_domains == {"x": "real", "y": "real", "z": "real"}

    axis0, axis1, axis2, axis3 = field.axes
    assert axis0.name == "t"
    assert axis1.name == "x"
    assert axis2.name == "y"
    assert axis3.name == "z"
    assert axis0.unit == u.s
    assert axis1.unit == u.m
    assert axis0.size == 4


def test_make_demo_scalar_field_noise_is_reproducible():
    field1 = make_demo_scalar_field(
        "noise", nt=3, nx=2, ny=2, nz=2, seed=123, dt=0.1 * u.s
    )
    field2 = make_demo_scalar_field(
        "noise", nt=3, nx=2, ny=2, nz=2, seed=123, dt=0.1 * u.s
    )
    assert np.allclose(np.asarray(field1), np.asarray(field2))


def test_make_sinusoidal_wave_bounds():
    field = make_sinusoidal_wave(nt=3, nx=3, ny=2, nz=2, dt=0.1 * u.s, dx=0.5 * u.m)
    values = np.asarray(field)
    assert values.max() <= 1.0 + 1e-12
    assert values.min() >= -1.0 - 1e-12


def test_make_standing_wave_returns_field():
    field = make_standing_wave(nt=3, nx=3, ny=3, nz=3, dt=0.1 * u.s, dx=0.5 * u.m)
    assert isinstance(field, ScalarField)
    assert field.axis0_domain == "time"


def test_make_propagating_gaussian_nonnegative():
    field = make_propagating_gaussian(
        nt=3, nx=2, ny=2, nz=2, dt=0.1 * u.s, dx=0.5 * u.m
    )
    values = np.asarray(field)
    assert np.min(values) >= 0.0


def test_make_demo_scalar_field_invalid_pattern_raises():
    bad_pattern = cast(Literal["gaussian", "sine", "standing", "noise"], "unknown")
    with pytest.raises(ValueError):
        make_demo_scalar_field(bad_pattern, nt=2, nx=2, ny=2, nz=2)
