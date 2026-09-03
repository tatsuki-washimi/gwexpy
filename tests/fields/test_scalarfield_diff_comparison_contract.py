"""Fail-closed contracts for the ``ScalarField.diff(field)`` extension."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from astropy import units as u
from astropy.utils.masked import Masked
from gwpy.types import Array as GWpyArray

from gwexpy.fields import ScalarField

_SHAPE = (4, 3, 2, 2)
_AXIS_NAMES = ("clock", "east", "north", "height")
_LEFT_AXES = (
    np.array([10.0, 11.0, 12.0, 13.0]) * u.s,
    np.array([0.0, 1.0, 2.0]) * u.m,
    np.array([0.0, 0.5]) * u.m,
    np.array([2.0, 5.0]) * u.m,
)
_EQUIVALENT_AXES = (
    np.array([10_000.0, 11_000.0, 12_000.0, 13_000.0]) * u.ms,
    np.array([0.0, 100.0, 200.0]) * u.cm,
    np.array([0.0, 50.0]) * u.cm,
    np.array([200.0, 500.0]) * u.cm,
)


def _field(
    values: float | np.ndarray,
    *,
    unit: u.UnitBase = u.V,
    axes: tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity] = _LEFT_AXES,
    axis0_domain: str = "time",
    space_domain: str | dict[str, str] = "real",
    name: str = "left field",
) -> ScalarField:
    data = (
        np.full(_SHAPE, values)
        if np.asarray(values).ndim == 0
        else np.asarray(values).reshape(_SHAPE)
    )
    result = ScalarField(
        data,
        unit=unit,
        name=name,
        epoch=1_234_567_890.25,
        channel="H1:SCALARFIELD-DIFF",
        axis0=axes[0],
        axis1=axes[1],
        axis2=axes[2],
        axis3=axes[3],
        axis_names=_AXIS_NAMES,
        axis0_domain=axis0_domain,
        space_domain=space_domain,
    )
    result._gwex_comparison_provenance = f"{name}-provenance"
    return result


def _snapshot(field: ScalarField) -> dict[str, Any]:
    return {
        "value": field.value.copy(),
        "unit": field.unit,
        "name": field.name,
        "channel": str(field.channel),
        "epoch": field.epoch,
        "axis_names": field.axis_names,
        "axis0_domain": field.axis0_domain,
        "space_domains": field.space_domains,
        "axes": tuple((axis.index.value.copy(), axis.unit) for axis in field.axes),
        "provenance": field._gwex_comparison_provenance,
    }


def _assert_snapshot(field: ScalarField, expected: dict[str, Any]) -> None:
    np.testing.assert_array_equal(field.value, expected["value"])
    assert field.unit == expected["unit"]
    assert field.name == expected["name"]
    assert str(field.channel) == expected["channel"]
    assert field.epoch == expected["epoch"]
    assert field.axis_names == expected["axis_names"]
    assert field.axis0_domain == expected["axis0_domain"]
    assert field.space_domains == expected["space_domains"]
    assert field._gwex_comparison_provenance == expected["provenance"]
    for axis, (values, unit) in zip(field.axes, expected["axes"], strict=True):
        assert axis.unit == unit
        np.testing.assert_array_equal(axis.index.value, values)


def _assert_left_metadata(result: ScalarField, left: ScalarField) -> None:
    assert result.name == left.name
    assert result.channel == left.channel
    assert result.epoch == left.epoch
    assert result.axis_names == left.axis_names
    assert result.axis0_domain == left.axis0_domain
    assert result.space_domains == left.space_domains
    assert result._gwex_comparison_provenance == left._gwex_comparison_provenance
    for actual, expected in zip(result.axes, left.axes, strict=True):
        assert actual.name == expected.name
        assert actual.unit == expected.unit
        np.testing.assert_array_equal(actual.index.value, expected.index.value)


@pytest.mark.parametrize(
    ("mode", "expected_values", "expected_unit"),
    [
        pytest.param("diff", 1.0, u.V, id="diff"),
        pytest.param("ratio", 2.0, u.dimensionless_unscaled, id="ratio"),
        pytest.param("percent", 100.0, u.percent, id="percent"),
    ],
)
def test_comparison_converts_equivalent_data_units(
    mode: str, expected_values: float, expected_unit: u.UnitBase
) -> None:
    left = _field(2.0, unit=u.V)
    right = _field(1_000.0, unit=u.mV, name="right field")

    result = left.diff(right, mode=mode)

    assert result.unit == expected_unit
    np.testing.assert_allclose(result.value, expected_values)
    if mode == "percent":
        np.testing.assert_allclose(
            result.to_value(u.dimensionless_unscaled),
            1.0,
        )
    _assert_left_metadata(result, left)


@pytest.mark.parametrize("mode", ["diff", "ratio", "percent"])
def test_comparison_rejects_incompatible_data_units(mode: str) -> None:
    left = _field(2.0, unit=u.V)
    right = _field(1.0, unit=u.m, name="right field")

    with pytest.raises(u.UnitConversionError):
        left.diff(right, mode=mode)


@pytest.mark.parametrize("mode", ["diff", "ratio", "percent"])
@pytest.mark.parametrize("axis_number", range(4))
def test_comparison_rejects_each_misaligned_coordinate_axis(
    axis_number: int, mode: str
) -> None:
    right_axes = list(_LEFT_AXES)
    right_axes[axis_number] = right_axes[axis_number].copy()
    right_axes[axis_number][0] += 0.25 * right_axes[axis_number].unit
    left = _field(2.0)
    right = _field(1.0, axes=tuple(right_axes), name="right field")

    with pytest.raises(
        ValueError, match=rf"Field coordinate mismatch on axis {axis_number}"
    ):
        left.diff(right, mode=mode)


@pytest.mark.parametrize("mode", ["diff", "ratio", "percent"])
@pytest.mark.parametrize("converted_axis", range(4))
def test_comparison_accepts_equivalent_units_on_each_coordinate_axis(
    converted_axis: int, mode: str
) -> None:
    right_axes = list(_LEFT_AXES)
    right_axes[converted_axis] = _EQUIVALENT_AXES[converted_axis]
    left = _field(2.0)
    right = _field(1.0, axes=tuple(right_axes), name="right field")

    result = left.diff(right, mode=mode)

    _assert_left_metadata(result, left)


@pytest.mark.parametrize("mode", ["diff", "ratio", "percent"])
@pytest.mark.parametrize("mismatch", ["axis0-domain", "space-domain"])
def test_comparison_rejects_domain_mismatch(mismatch: str, mode: str) -> None:
    left = _field(2.0)
    if mismatch == "axis0-domain":
        axes = (_LEFT_AXES[0].value * u.Hz, *_LEFT_AXES[1:])
        right = _field(
            1.0,
            axes=axes,
            axis0_domain="frequency",
            name="right field",
        )
        message = "Field domain mismatch"
    else:
        axes = (
            _LEFT_AXES[0],
            _LEFT_AXES[1].value / u.m,
            _LEFT_AXES[2].value / u.m,
            _LEFT_AXES[3].value / u.m,
        )
        right = _field(1.0, axes=axes, space_domain="k", name="right field")
        message = "Field spatial domain mismatch"

    with pytest.raises(ValueError, match=message):
        left.diff(right, mode=mode)


@pytest.mark.parametrize(
    ("mode", "operation", "expected_unit"),
    [
        pytest.param("diff", np.subtract, u.V, id="diff"),
        pytest.param("ratio", np.divide, u.dimensionless_unscaled, id="ratio"),
        pytest.param(
            "percent",
            lambda left, right: np.divide(np.subtract(left, right), right) * 100,
            u.percent,
            id="percent",
        ),
    ],
)
def test_comparison_preserves_shape_dtype_metadata_and_inputs(
    mode: str,
    operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
    expected_unit: u.UnitBase,
) -> None:
    left_values = np.linspace(1.0, 4.0, np.prod(_SHAPE), dtype=np.float32)
    right_values = np.linspace(0.5, 2.0, np.prod(_SHAPE), dtype=np.float32)
    left = _field(left_values, name="authoritative left")
    right = _field(right_values, name="unchanged right")
    left_before = _snapshot(left)
    right_before = _snapshot(right)

    result = left.diff(right, mode=mode)
    expected = operation(left_values.reshape(_SHAPE), right_values.reshape(_SHAPE))

    assert isinstance(result, ScalarField)
    assert result.shape == _SHAPE
    assert result.dtype == expected.dtype
    assert result.unit == expected_unit
    np.testing.assert_allclose(result.value, expected, rtol=1e-6, atol=0.0)
    _assert_left_metadata(result, left)
    _assert_snapshot(left, left_before)
    _assert_snapshot(right, right_before)


@pytest.mark.parametrize("mode", ["ratio", "percent"])
def test_zero_denominator_preserves_numpy_nonfinite_masks(mode: str) -> None:
    left_values = np.resize(np.array([0.0, 1.0, -1.0, 4.0]), np.prod(_SHAPE))
    right_values = np.resize(np.array([0.0, 0.0, 0.0, 2.0]), np.prod(_SHAPE))
    left = _field(left_values)
    right = _field(right_values, name="right field")

    with np.errstate(divide="ignore", invalid="ignore"):
        if mode == "ratio":
            expected = np.divide(left_values, right_values).reshape(_SHAPE)
        else:
            expected = (
                np.divide(left_values - right_values, right_values) * 100
            ).reshape(_SHAPE)
        result = left.diff(right, mode=mode)

    np.testing.assert_array_equal(np.isfinite(result.value), np.isfinite(expected))
    np.testing.assert_array_equal(np.isnan(result.value), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(result.value), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(result.value), np.isneginf(expected))
    np.testing.assert_array_equal(
        result.value[np.isfinite(expected)], expected[np.isfinite(expected)]
    )


@pytest.mark.parametrize(
    ("mode", "operation", "expected_unit"),
    [
        pytest.param("diff", np.subtract, u.V, id="diff"),
        pytest.param("ratio", np.divide, u.dimensionless_unscaled, id="ratio"),
        pytest.param(
            "percent",
            lambda left, right: np.divide(np.subtract(left, right), right) * 100,
            u.percent,
            id="percent",
        ),
    ],
)
def test_comparison_preserves_astropy_masked_operands(
    mode: str,
    operation: Callable[[np.ndarray, np.ndarray], np.ndarray],
    expected_unit: u.UnitBase,
) -> None:
    left_values = np.resize(np.array([0.0, 1.0, -1.0, 4.0]), np.prod(_SHAPE))
    right_values = np.resize(np.array([0.0, 0.0, 0.0, 2.0]), np.prod(_SHAPE))
    left_mask = (np.arange(np.prod(_SHAPE)) % 5 == 0).reshape(_SHAPE)
    right_mask = (np.arange(np.prod(_SHAPE)) % 7 == 0).reshape(_SHAPE)
    common = {
        "unit": u.V,
        "axis0": _LEFT_AXES[0],
        "axis1": _LEFT_AXES[1],
        "axis2": _LEFT_AXES[2],
        "axis3": _LEFT_AXES[3],
        "axis_names": _AXIS_NAMES,
    }
    left = ScalarField(Masked(left_values.reshape(_SHAPE), mask=left_mask), **common)
    right = ScalarField(Masked(right_values.reshape(_SHAPE), mask=right_mask), **common)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = left.diff(right, mode=mode)
        expected = operation(left_values.reshape(_SHAPE), right_values.reshape(_SHAPE))

    combined_mask = left_mask | right_mask
    assert result.unit == expected_unit
    np.testing.assert_array_equal(result.mask, combined_mask)
    np.testing.assert_array_equal(
        np.asarray(result.unmasked)[~combined_mask],
        expected[~combined_mask],
    )


def test_comparison_shape_mismatch_is_fail_closed() -> None:
    left = _field(2.0)
    right = ScalarField(np.ones((2, 3, 2, 2)), unit=u.V)

    with pytest.raises(ValueError, match="Shape mismatch"):
        left.diff(right)


@pytest.mark.parametrize(("n", "axis"), [(1, -1), (2, 1), (0, 0)])
def test_numeric_diff_route_still_matches_installed_gwpy(n: int, axis: int) -> None:
    values = np.arange(np.prod(_SHAPE), dtype=np.float64).reshape(_SHAPE)
    actual_input = _field(values)
    expected_input = GWpyArray(values, unit=u.V, name=actual_input.name)

    actual = actual_input.diff(n, axis)
    expected = expected_input.diff(n, axis)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.unit == expected.unit
    np.testing.assert_array_equal(actual.value, expected.value)
