"""Contract baseline for ScalarField binary algebra metadata alignment."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from gwexpy.fields import ScalarField


def _make_scalar_field(
    *,
    value: float = 1.0,
    unit=u.V,
    axis0=None,
    axis1=None,
    axis2=None,
    axis3=None,
    axis_names=("t", "x", "y", "z"),
    axis0_domain: str = "time",
) -> ScalarField:
    shape = (2, 2, 2, 2)
    axis0 = np.arange(shape[0]) * u.s if axis0 is None else axis0
    axis1 = np.arange(shape[1]) * u.m if axis1 is None else axis1
    axis2 = np.arange(shape[2]) * u.m if axis2 is None else axis2
    axis3 = np.arange(shape[3]) * u.m if axis3 is None else axis3
    return ScalarField(
        np.full(shape, value, dtype=float),
        unit=unit,
        axis0=axis0,
        axis1=axis1,
        axis2=axis2,
        axis3=axis3,
        axis_names=axis_names,
        axis0_domain=axis0_domain,
        space_domain="real",
    )


def _assert_axis_matches(result: ScalarField, expected: ScalarField, name: str) -> None:
    result_axis = result.axis(name).index
    expected_axis = expected.axis(name).index
    assert result_axis.unit == expected_axis.unit
    assert_allclose(result_axis.value, expected_axis.value)


def test_scalarfield_add_aligned_grid_preserves_metadata_and_values():
    left = _make_scalar_field(value=1.0, unit=u.V)
    right = _make_scalar_field(value=2.0, unit=u.mV)

    result = left + right

    assert isinstance(result, ScalarField)
    assert result.axis0_domain == "time"
    assert result.space_domains == {"x": "real", "y": "real", "z": "real"}
    assert result.unit == u.V
    for axis_name in ("t", "x", "y", "z"):
        _assert_axis_matches(result, left, axis_name)
    assert_allclose(result.value, np.full(left.shape, 1.002))


def test_scalarfield_add_rejects_same_shape_x_coordinate_mismatch():
    left = _make_scalar_field(value=1.0)
    right = _make_scalar_field(value=2.0, axis1=np.array([10.0, 11.0]) * u.m)

    try:
        left + right
    except ValueError as exc:
        assert "axis" in str(exc).lower() or "coordinate" in str(exc).lower()
    else:
        pytest.xfail(
            "Issue #287: ScalarField binary arithmetic should reject same-shape "
            "spatial coordinate grid mismatches before inheriting array arithmetic."
        )


def test_scalarfield_add_rejects_same_shape_axis0_domain_mismatch():
    time_field = _make_scalar_field(value=1.0, unit=u.V)
    frequency_field = _make_scalar_field(
        value=2.0,
        unit=u.V,
        axis0=np.arange(2) * u.Hz,
        axis_names=("f", "x", "y", "z"),
        axis0_domain="frequency",
    )

    try:
        time_field + frequency_field
    except ValueError as exc:
        assert "axis" in str(exc).lower() or "domain" in str(exc).lower()
    else:
        pytest.xfail(
            "Issue #287: ScalarField binary arithmetic should reject same-shape "
            "axis0 domain mismatches even when data units permit arithmetic."
        )
