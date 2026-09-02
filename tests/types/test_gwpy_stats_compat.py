"""Differential contracts for GWpy-compatible shared statistics methods."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from astropy import units as u
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwpy.types.array import Array as GwpyArray
from gwpy.types.array2d import Array2D as GwpyArray2D
from gwpy.types.series import Series as GwpySeries

from gwexpy.fields.scalar import ScalarField
from gwexpy.frequencyseries.frequencyseries import FrequencySeries
from gwexpy.timeseries.timeseries import TimeSeries
from gwexpy.types.array import Array
from gwexpy.types.array2d import Array2D
from gwexpy.types.array3d import Array3D
from gwexpy.types.array4d import Array4D
from gwexpy.types.plane2d import Plane2D
from gwexpy.types.series import Series


@dataclass(frozen=True)
class StatsClassCase:
    """Pair a public GWexpy class with its GWpy behavioral oracle."""

    gwexpy_class: type
    gwpy_class: type
    shape: tuple[int, ...]
    constructor_kwargs: dict[str, Any]


CLASS_CASES = (
    pytest.param(
        StatsClassCase(ScalarField, GwpyArray, (1, 1, 2, 3), {}),
        id="ScalarField",
    ),
    pytest.param(
        StatsClassCase(
            FrequencySeries,
            GwpyFrequencySeries,
            (6,),
            {"f0": 10, "df": 2},
        ),
        id="FrequencySeries",
    ),
    pytest.param(
        StatsClassCase(
            TimeSeries,
            GwpyTimeSeries,
            (6,),
            {"t0": 1000, "dt": 0.5},
        ),
        id="TimeSeries",
    ),
    pytest.param(StatsClassCase(Array, GwpyArray, (2, 3), {}), id="Array"),
    pytest.param(
        StatsClassCase(Array2D, GwpyArray2D, (2, 3), {}),
        id="Array2D",
    ),
    pytest.param(StatsClassCase(Array3D, GwpyArray, (1, 2, 3), {}), id="Array3D"),
    pytest.param(
        StatsClassCase(Array4D, GwpyArray, (1, 1, 2, 3), {}),
        id="Array4D",
    ),
    pytest.param(
        StatsClassCase(Plane2D, GwpyArray2D, (2, 3), {}),
        id="Plane2D",
    ),
    pytest.param(
        StatsClassCase(Series, GwpySeries, (6,), {"x0": 10, "dx": 2}),
        id="Series",
    ),
)

STAT_METHODS = tuple(
    pytest.param(method, id=method)
    for method in ("mean", "std", "var", "min", "max", "median")
)

NONFINITE_CASES = (
    pytest.param("finite", id="finite"),
    pytest.param("nan", id="nan"),
    pytest.param("posinf", id="positive-infinity"),
    pytest.param("neginf", id="negative-infinity"),
)


def _values(case: StatsClassCase, variant: str = "finite") -> np.ndarray:
    size = int(np.prod(case.shape))
    values = np.arange(1, size + 1, dtype=np.float64)
    if variant == "nan":
        values[size // 2] = np.nan
    elif variant == "posinf":
        values[size // 2] = np.inf
    elif variant == "neginf":
        values[size // 2] = -np.inf
    return values.reshape(case.shape)


def _make_pair(
    case: StatsClassCase,
    variant: str = "finite",
) -> tuple[Any, Any]:
    values = _values(case, variant)
    kwargs = {"unit": u.m, **case.constructor_kwargs}
    return (
        case.gwexpy_class(values.copy(), **kwargs),
        case.gwpy_class(values.copy(), **kwargs),
    )


def _where_mask(case: StatsClassCase) -> np.ndarray:
    where = np.ones(case.shape, dtype=bool)
    where[..., 1] = False
    return where


def _assert_numeric_result_equal(actual: Any, expected: Any) -> None:
    actual_values = np.asarray(actual.value)
    expected_values = np.asarray(expected.value)

    assert actual_values.shape == expected_values.shape
    assert actual_values.dtype == expected_values.dtype
    assert actual.unit == expected.unit

    for mask in (np.isnan, np.isposinf, np.isneginf):
        np.testing.assert_array_equal(mask(actual_values), mask(expected_values))

    finite = np.isfinite(expected_values)
    np.testing.assert_array_equal(np.isfinite(actual_values), finite)
    np.testing.assert_allclose(
        actual_values[finite],
        expected_values[finite],
        rtol=0.0,
        atol=0.0,
    )


def _signature_layout(
    callable_: Any,
) -> tuple[tuple[str, inspect._ParameterKind, Any], ...]:
    return tuple(
        (parameter.name, parameter.kind, parameter.default)
        for parameter in inspect.signature(callable_).parameters.values()
    )


@pytest.mark.parametrize("case", CLASS_CASES)
@pytest.mark.parametrize("method_name", STAT_METHODS)
def test_shared_statistics_signature_preserves_parent_calling_form(
    case: StatsClassCase,
    method_name: str,
) -> None:
    actual_layout = _signature_layout(getattr(case.gwexpy_class, method_name))

    if method_name in {"mean", "std", "var"}:
        expected_layout = _signature_layout(getattr(case.gwpy_class, method_name))
        assert actual_layout[:-1] == expected_layout
        assert actual_layout[-1] == (
            "ignore_nan",
            inspect.Parameter.KEYWORD_ONLY,
            False,
        )
    elif method_name == "median":
        expected_layout = _signature_layout(getattr(case.gwpy_class, method_name))
        assert actual_layout == expected_layout
    else:
        no_value = np._NoValue
        assert actual_layout == (
            ("self", inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.empty),
            ("axis", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
            ("out", inspect.Parameter.POSITIONAL_OR_KEYWORD, None),
            ("keepdims", inspect.Parameter.POSITIONAL_OR_KEYWORD, False),
            ("initial", inspect.Parameter.POSITIONAL_OR_KEYWORD, no_value),
            ("where", inspect.Parameter.POSITIONAL_OR_KEYWORD, no_value),
            ("ignore_nan", inspect.Parameter.KEYWORD_ONLY, False),
        )


@pytest.mark.parametrize("case", CLASS_CASES)
@pytest.mark.parametrize("method_name", STAT_METHODS)
@pytest.mark.parametrize("variant", NONFINITE_CASES)
def test_shared_statistics_default_matches_gwpy_nonfinite_behavior(
    case: StatsClassCase,
    method_name: str,
    variant: str,
) -> None:
    actual_input, expected_input = _make_pair(case, variant)

    with np.errstate(all="ignore"):
        actual = getattr(actual_input, method_name)()
        expected = getattr(expected_input, method_name)()

    _assert_numeric_result_equal(actual, expected)


@pytest.mark.parametrize("case", CLASS_CASES)
@pytest.mark.parametrize("method_name", STAT_METHODS)
def test_shared_statistics_axis_and_keepdims_match_gwpy(
    case: StatsClassCase,
    method_name: str,
) -> None:
    actual_input, expected_input = _make_pair(case)

    actual = getattr(actual_input, method_name)(axis=-1, keepdims=True)
    expected = getattr(expected_input, method_name)(axis=-1, keepdims=True)

    _assert_numeric_result_equal(actual, expected)


@pytest.mark.parametrize("case", CLASS_CASES)
@pytest.mark.parametrize("method_name", ["mean", "std", "var"])
def test_shared_statistics_dtype_and_where_match_gwpy(
    case: StatsClassCase,
    method_name: str,
) -> None:
    actual_input, expected_input = _make_pair(case)
    where = _where_mask(case)
    kwargs: dict[str, Any] = {
        "axis": -1,
        "dtype": np.float32,
        "keepdims": True,
        "where": where,
    }
    if method_name in {"std", "var"}:
        kwargs["ddof"] = 1

    actual = getattr(actual_input, method_name)(**kwargs)
    expected = getattr(expected_input, method_name)(**kwargs)

    _assert_numeric_result_equal(actual, expected)


@pytest.mark.parametrize("case", CLASS_CASES)
@pytest.mark.parametrize("method_name", ["min", "max"])
def test_min_max_positional_initial_and_where_match_gwpy(
    case: StatsClassCase,
    method_name: str,
) -> None:
    actual_input, expected_input = _make_pair(case)
    where = _where_mask(case)
    initial = 100 * u.m
    args = (-1, None, True, initial, where)

    actual = getattr(actual_input, method_name)(*args)
    expected = getattr(expected_input, method_name)(*args)

    _assert_numeric_result_equal(actual, expected)


@pytest.mark.parametrize("case", CLASS_CASES)
@pytest.mark.parametrize("method_name", STAT_METHODS)
def test_shared_statistics_out_matches_gwpy(
    case: StatsClassCase,
    method_name: str,
) -> None:
    actual_input, expected_input = _make_pair(case)
    output_shape = case.shape[:-1]
    output_unit = u.m**2 if method_name == "var" else u.m
    actual_out = u.Quantity(np.empty(output_shape), unit=output_unit)
    expected_out = u.Quantity(np.empty(output_shape), unit=output_unit)

    try:
        expected = getattr(expected_input, method_name)(axis=-1, out=expected_out)
    except Exception as expected_error:
        with pytest.raises(type(expected_error)):
            getattr(actual_input, method_name)(axis=-1, out=actual_out)
        return

    actual = getattr(actual_input, method_name)(axis=-1, out=actual_out)

    assert (actual is actual_out) is (expected is expected_out)
    _assert_numeric_result_equal(actual, expected)
    _assert_numeric_result_equal(actual_out, expected_out)


@pytest.mark.parametrize("case", CLASS_CASES)
@pytest.mark.parametrize("method_name", ["mean", "std", "var"])
def test_where_remains_keyword_only_for_quantity_statistics(
    case: StatsClassCase,
    method_name: str,
) -> None:
    actual_input, expected_input = _make_pair(case)
    if method_name == "mean":
        args = (None, None, None, False, True)
    else:
        args = (None, None, None, 0, False, True)

    with pytest.raises(TypeError):
        getattr(expected_input, method_name)(*args)
    with pytest.raises(TypeError):
        getattr(actual_input, method_name)(*args)


@pytest.mark.parametrize("case", CLASS_CASES)
def test_median_rejects_a_second_positional_argument_like_gwpy(
    case: StatsClassCase,
) -> None:
    actual_input, expected_input = _make_pair(case)

    with pytest.raises(TypeError):
        expected_input.median(0, None)
    with pytest.raises(TypeError):
        actual_input.median(0, None)


@pytest.mark.parametrize("case", CLASS_CASES)
@pytest.mark.parametrize("method_name", STAT_METHODS)
def test_explicit_ignore_nan_extension_remains_available(
    case: StatsClassCase,
    method_name: str,
) -> None:
    actual_input, _ = _make_pair(case, "nan")
    function = getattr(np, f"nan{method_name}")
    expected_value = function(_values(case, "nan"))
    expected_unit = u.m**2 if method_name == "var" else u.m

    actual = getattr(actual_input, method_name)(ignore_nan=True)

    assert actual.unit == expected_unit
    np.testing.assert_allclose(actual.value, expected_value, rtol=0.0, atol=0.0)
