"""Contract tests for physics-facing ScalarField algebra."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.fields import ScalarField


def _field(
    *,
    data_value: float = 1.0,
    axis0=None,
    axis1=None,
    axis0_domain: str = "time",
    space_domain: str | dict[str, str] = "real",
    unit=u.V,
) -> ScalarField:
    shape = (4, 3, 2, 2)
    if axis0 is None:
        axis0 = np.arange(shape[0]) * u.s
    if axis1 is None:
        axis1 = np.arange(shape[1]) * u.m
    return ScalarField(
        np.full(shape, data_value),
        unit=unit,
        axis0=axis0,
        axis1=axis1,
        axis2=np.arange(shape[2]) * u.m,
        axis3=np.arange(shape[3]) * u.m,
        axis_names=["t", "x", "y", "z"],
        axis0_domain=axis0_domain,
        space_domain=space_domain,
    )


def test_scalarfield_addition_preserves_left_grid_for_aligned_fields():
    left = _field(data_value=1.0)
    right = _field(data_value=2.0)

    result = left + right

    assert isinstance(result, ScalarField)
    assert result.unit == u.V
    assert result.axis0_domain == "time"
    np.testing.assert_allclose(result.value, 3.0)
    np.testing.assert_allclose(result._axis1_index.to_value(u.m), [0.0, 1.0, 2.0])


def test_scalarfield_addition_rejects_axis0_domain_mismatch():
    left = _field(axis0_domain="time", axis0=np.arange(4) * u.s)
    right = _field(axis0_domain="frequency", axis0=np.arange(4) * u.Hz)

    with pytest.raises(ValueError, match="Field domain mismatch"):
        left + right


def test_scalarfield_addition_rejects_spatial_domain_mismatch():
    left = _field(space_domain={"x": "real", "y": "real", "z": "real"})
    right = _field(
        axis1=np.arange(3) / u.m,
        space_domain={"x": "k", "y": "real", "z": "real"},
    )

    with pytest.raises(ValueError, match="Field spatial domain mismatch"):
        left + right


def test_scalarfield_addition_rejects_misaligned_spatial_coordinates():
    left = _field(axis1=np.arange(3) * u.m)
    right = _field(axis1=np.array([0.0, 1.5, 3.0]) * u.m)

    with pytest.raises(ValueError, match="Field coordinate mismatch on axis 1"):
        left + right


def test_scalarfield_addition_rejects_large_gps_axis_jitter():
    gps_start = 1_000_000_000.0
    left = _field(axis0=(gps_start + np.arange(4)) * u.s)
    right = _field(axis0=(gps_start + np.arange(4) + 0.5) * u.s)

    with pytest.raises(ValueError, match="Field coordinate mismatch on axis 0"):
        left + right


def test_scalarfield_addition_accepts_equivalent_coordinate_units():
    left = _field(axis1=np.array([0.0, 1.0, 2.0]) * u.m)
    right = _field(axis1=np.array([0.0, 100.0, 200.0]) * u.cm)

    result = left + right

    assert isinstance(result, ScalarField)
    np.testing.assert_allclose(result.value, 2.0)
