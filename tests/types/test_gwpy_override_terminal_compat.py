"""Differential contracts for audited type-level GWpy overrides."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from astropy import units as u
from gwpy.types import Array as GWpyArray
from gwpy.types import Array2D as GWpyArray2D

from gwexpy.fields import ScalarField
from gwexpy.types import Array, Array2D, Array3D, Array4D, Plane2D


def _exception_class(call: Callable[[], Any]) -> type[BaseException] | None:
    try:
        call()
    except BaseException as exc:  # noqa: BLE001 - exception type is the oracle
        return type(exc)
    return None


def _array2d_pair(
    cls: type[Array2D] | type[Plane2D],
) -> tuple[Array2D | Plane2D, GWpyArray2D]:
    data = np.arange(12, dtype=np.float64).reshape(3, 4)
    common = {
        "unit": u.V,
        "name": "asymmetric",
        "xindex": [10, 20, 40] * u.Hz,
        "yindex": [1, 2, 4, 8] * u.s,
    }
    if cls is Plane2D:
        actual = cls(data.copy(), axis1_name="row", axis2_name="column", **common)
    else:
        actual = cls(data.copy(), axis_names=("row", "column"), **common)
    return actual, GWpyArray2D(data.copy(), **common)


def _assert_array2d_result_matches_gwpy(actual: Any, expected: Any) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.unit == expected.unit
    assert actual.name == expected.name
    np.testing.assert_array_equal(actual.value, expected.value)
    for attr in ("xindex", "yindex"):
        actual_index = getattr(actual, attr)
        expected_index = getattr(expected, attr)
        assert actual_index.unit == expected_index.unit
        np.testing.assert_array_equal(actual_index.value, expected_index.value)


@pytest.mark.parametrize("cls", [Array2D, Plane2D], ids=lambda cls: cls.__name__)
@pytest.mark.parametrize(
    "axes",
    [
        pytest.param((0, 1), id="forward"),
        pytest.param((-1, 0), id="negative"),
        pytest.param((0, 0), id="identity"),
    ],
)
def test_array2d_numeric_swapaxes_matches_gwpy(
    cls: type[Array2D] | type[Plane2D], axes: tuple[int, int]
) -> None:
    actual_input, expected_input = _array2d_pair(cls)

    actual = actual_input.swapaxes(*axes)
    expected = expected_input.swapaxes(*axes)

    _assert_array2d_result_matches_gwpy(actual, expected)
    assert actual.axis_names == ("row", "column")
    assert np.shares_memory(actual_input.value, actual.value) is np.shares_memory(
        expected_input.value, expected.value
    )


@pytest.mark.parametrize("cls", [Array2D, Plane2D], ids=lambda cls: cls.__name__)
@pytest.mark.parametrize(
    ("args", "expected_args"),
    [
        pytest.param((), (), id="default"),
        pytest.param(((1, 0),), ((1, 0),), id="tuple"),
        pytest.param((1, 0), (1, 0), id="separate"),
        pytest.param((0, 1), (0, 1), id="identity"),
    ],
)
def test_array2d_numeric_transpose_matches_gwpy(
    cls: type[Array2D] | type[Plane2D],
    args: tuple[Any, ...],
    expected_args: tuple[Any, ...],
) -> None:
    actual_input, expected_input = _array2d_pair(cls)

    actual = actual_input.transpose(*args)
    expected = expected_input.transpose(*expected_args)

    _assert_array2d_result_matches_gwpy(actual, expected)
    assert actual.axis_names == ("row", "column")
    assert np.shares_memory(actual_input.value, actual.value) is np.shares_memory(
        expected_input.value, expected.value
    )


@pytest.mark.parametrize("cls", [Array2D, Plane2D], ids=lambda cls: cls.__name__)
def test_array2d_numeric_axis_errors_match_gwpy(
    cls: type[Array2D] | type[Plane2D],
) -> None:
    actual_input, expected_input = _array2d_pair(cls)

    assert _exception_class(lambda: actual_input.swapaxes(0, 2)) is _exception_class(
        lambda: expected_input.swapaxes(0, 2)
    )
    assert _exception_class(lambda: actual_input.transpose(0, 0)) is _exception_class(
        lambda: expected_input.transpose(0, 0)
    )


@pytest.mark.parametrize("cls", [Array2D, Plane2D], ids=lambda cls: cls.__name__)
def test_array2d_T_keeps_gwpy_swapped_metadata_contract(
    cls: type[Array2D] | type[Plane2D],
) -> None:
    actual_input, expected_input = _array2d_pair(cls)

    actual = actual_input.T
    expected = expected_input.T

    _assert_array2d_result_matches_gwpy(actual, expected)
    assert actual.axis_names == ("column", "row")
    assert np.shares_memory(actual_input.value, actual.value) is np.shares_memory(
        expected_input.value, expected.value
    )


@pytest.mark.parametrize("cls", [Array2D, Plane2D], ids=lambda cls: cls.__name__)
def test_array2d_named_axis_permutations_remain_explicit_extensions(
    cls: type[Array2D] | type[Plane2D],
) -> None:
    actual_input, expected_input = _array2d_pair(cls)

    swapped = actual_input.swapaxes("row", "column")
    transposed = actual_input.transpose("column", "row")
    listed = actual_input.transpose(["column", "row"])

    for result in (swapped, transposed, listed):
        assert result.axis_names == ("column", "row")
        np.testing.assert_array_equal(result.value, actual_input.value.T)
        np.testing.assert_array_equal(result.xindex.value, [1, 2, 4, 8])
        np.testing.assert_array_equal(result.yindex.value, [10, 20, 40])
    assert (
        _exception_class(lambda: expected_input.swapaxes("row", "column")) is TypeError
    )
    assert (
        _exception_class(lambda: expected_input.transpose("column", "row")) is TypeError
    )


def _scalar_pair() -> tuple[ScalarField, GWpyArray]:
    shape = (2, 3, 4, 5)
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    actual = ScalarField(
        values.copy(),
        unit=u.V,
        name="field",
        axis0=[0, 2] * u.s,
        axis1=[10, 20, 40] * u.m,
        axis2=[1, 2, 5, 9] * u.m,
        axis3=[3, 4, 8, 9, 11] * u.m,
        axis_names=("t", "x", "y", "z"),
    )
    expected = GWpyArray(values.copy(), unit=u.V, name="field")
    return actual, expected


def test_scalarfield_diff_signature_preserves_gwpy_layout() -> None:
    actual = inspect.signature(ScalarField.diff).parameters
    expected = inspect.signature(GWpyArray.diff).parameters

    assert list(actual)[:3] == list(expected)
    for name in expected:
        assert actual[name].kind is expected[name].kind
        assert actual[name].default == expected[name].default
    assert actual["mode"].kind is inspect.Parameter.KEYWORD_ONLY
    assert actual["mode"].default is None


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        pytest.param((), {}, id="default"),
        pytest.param((2, 1), {}, id="n-axis-positional"),
        pytest.param((), {"n": 1, "axis": 2}, id="n-axis-keyword"),
        pytest.param((0, 0), {}, id="zero-order"),
    ],
)
def test_scalarfield_diff_common_route_matches_gwpy(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    actual_input, expected_input = _scalar_pair()

    actual = actual_input.diff(*args, **kwargs)
    expected = expected_input.diff(*args, **kwargs)

    assert isinstance(actual, ScalarField)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.unit == expected.unit
    assert actual.name == expected.name
    np.testing.assert_array_equal(actual.value, expected.value)
    assert actual.axis_names == actual_input.axis_names
    assert tuple(axis.size for axis in actual.axes) == actual.shape
    assert actual.axis0_domain == actual_input.axis0_domain
    assert actual.space_domains == actual_input.space_domains
    n = kwargs.get("n", args[0] if args else 1)
    axis = kwargs.get("axis", args[1] if len(args) > 1 else -1)
    normalized_axis = axis % actual_input.ndim
    for index, (actual_axis, source_axis) in enumerate(
        zip(actual.axes, actual_input.axes, strict=True)
    ):
        expected_index = (
            source_axis.index[n:] if index == normalized_axis else source_axis.index
        )
        assert actual_axis.unit == expected_index.unit
        np.testing.assert_array_equal(actual_axis.index.value, expected_index.value)
    assert np.shares_memory(actual_input.value, actual.value) is np.shares_memory(
        expected_input.value, expected.value
    )


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        pytest.param((1, 9), {}, id="axis-out-of-bounds"),
        pytest.param((-1, -1), {}, id="negative-order"),
        pytest.param((), {"prepend": 0}, id="numpy-prepend-not-exposed"),
        pytest.param((1, -1, 0, 0), {}, id="numpy-prepend-append-positional"),
    ],
)
def test_scalarfield_diff_failure_class_matches_gwpy(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    actual_input, expected_input = _scalar_pair()

    assert _exception_class(lambda: actual_input.diff(*args, **kwargs)) is (
        _exception_class(lambda: expected_input.diff(*args, **kwargs))
    )


def test_scalarfield_diff_field_comparison_requires_explicit_field_type() -> None:
    left, _ = _scalar_pair()
    right = left.copy()
    right.value[:] = 2

    difference = left.diff(right)
    ratio = left.diff(right, mode="ratio")
    positional_ratio = left.diff(right, "ratio")

    np.testing.assert_array_equal(difference.value, left.value - right.value)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.testing.assert_array_equal(ratio.value, left.value / right.value)
    np.testing.assert_array_equal(positional_ratio.value, ratio.value)
    assert difference.unit == u.V
    assert ratio.unit == u.dimensionless_unscaled


def _permutation_sources() -> tuple[Array | Array3D | Array4D | ScalarField, ...]:
    return (
        Array(
            np.arange(24).reshape(2, 3, 4),
            axis_names=("a", "b", "c"),
        ),
        Array3D(
            np.arange(24).reshape(2, 3, 4),
            axis0=[0, 2] * u.s,
            axis1=[10, 20, 40] * u.m,
            axis2=[1, 2, 4, 8] * u.Hz,
            axis_names=("a", "b", "c"),
        ),
        Array4D(
            np.arange(48).reshape(2, 2, 3, 4),
            axis0=[0, 2] * u.s,
            axis1=[10, 20] * u.m,
            axis2=[1, 2, 4] * u.Hz,
            axis3=[3, 5, 8, 13] * u.kg,
            axis_names=("a", "b", "c", "d"),
        ),
        ScalarField(
            np.arange(48).reshape(2, 2, 3, 4),
            axis0=[0, 2] * u.s,
            axis1=[10, 20] * u.m,
            axis2=[1, 2, 4] * u.m,
            axis3=[3, 5, 8, 13] * u.m,
            axis_names=("t", "x", "y", "z"),
        ),
    )


@pytest.mark.parametrize(
    "source", _permutation_sources(), ids=lambda source: type(source).__name__
)
def test_other_axis_api_rows_remain_no_finding(source: Any) -> None:
    order = tuple(range(source.ndim - 1, -1, -1))

    swapped = source.swapaxes(0, source.ndim - 1)
    transposed = source.transpose(order)
    property_transpose = source.T

    np.testing.assert_array_equal(swapped.value, np.swapaxes(source.value, 0, -1))
    np.testing.assert_array_equal(transposed.value, np.transpose(source.value, order))
    np.testing.assert_array_equal(property_transpose.value, transposed.value)
    swapped_names = list(source.axis_names)
    swapped_names[0], swapped_names[-1] = swapped_names[-1], swapped_names[0]
    reversed_names = tuple(reversed(source.axis_names))
    assert swapped.axis_names == tuple(swapped_names)
    assert transposed.axis_names == reversed_names
    assert property_transpose.axis_names == reversed_names


def _assert_axis_order(source: Any, result: Any, order: tuple[int, ...]) -> None:
    expected_names = tuple(source.axis_names[index] for index in order)
    assert result.axis_names == expected_names
    for result_axis, source_index in zip(result.axes, order, strict=True):
        source_axis = source.axes[source_index]
        assert result_axis.unit == source_axis.unit
        np.testing.assert_array_equal(result_axis.index.value, source_axis.index.value)


@pytest.mark.parametrize(
    "source", _permutation_sources(), ids=lambda source: type(source).__name__
)
@pytest.mark.parametrize("case", ["identity", "numpy-integer", "bool", "invalid"])
def test_axis_api_numeric_swapaxes_matches_gwpy(source: Any, case: str) -> None:
    expected_input = GWpyArray(source.value.copy(), unit=source.unit)
    if case == "identity":
        axes = (0, 0)
    elif case == "numpy-integer":
        axes = (np.int64(0), np.int64(source.ndim - 1))
    elif case == "bool":
        axes = (True, False)
    else:
        axes = (0, source.ndim)

    actual_error = _exception_class(lambda: source.swapaxes(*axes))
    expected_error = _exception_class(lambda: expected_input.swapaxes(*axes))
    assert actual_error is expected_error
    if expected_error is not None:
        return

    actual = source.swapaxes(*axes)
    expected = expected_input.swapaxes(*axes)
    np.testing.assert_array_equal(actual.value, expected.value)
    assert np.shares_memory(source.value, actual.value) is np.shares_memory(
        expected_input.value, expected.value
    )
    order = list(range(source.ndim))
    order[int(axes[0])], order[int(axes[1])] = (
        order[int(axes[1])],
        order[int(axes[0])],
    )
    _assert_axis_order(source, actual, tuple(order))


@pytest.mark.parametrize(
    "source", _permutation_sources(), ids=lambda source: type(source).__name__
)
@pytest.mark.parametrize(
    "case", ["identity", "numpy-integer", "bool", "duplicate", "excess", "invalid"]
)
def test_axis_api_numeric_transpose_matches_gwpy(source: Any, case: str) -> None:
    expected_input = GWpyArray(source.value.copy(), unit=source.unit)
    if case == "identity":
        axes = tuple(range(source.ndim))
    elif case == "numpy-integer":
        axes = tuple(np.int64(index) for index in reversed(range(source.ndim)))
    elif case == "bool":
        axes = (True, False, *range(2, source.ndim))
    elif case == "duplicate":
        axes = (0,) * source.ndim
    elif case == "excess":
        axes = (*range(source.ndim), 0)
    else:
        axes = (*range(source.ndim - 1), source.ndim)

    actual_error = _exception_class(lambda: source.transpose(*axes))
    expected_error = _exception_class(lambda: expected_input.transpose(*axes))
    assert actual_error is expected_error
    if expected_error is not None:
        return

    actual = source.transpose(*axes)
    expected = expected_input.transpose(*axes)
    np.testing.assert_array_equal(actual.value, expected.value)
    assert np.shares_memory(source.value, actual.value) is np.shares_memory(
        expected_input.value, expected.value
    )
    _assert_axis_order(source, actual, tuple(int(axis) for axis in axes))
