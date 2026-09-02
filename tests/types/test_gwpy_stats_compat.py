"""Differential contracts for GWpy-compatible shared statistics methods."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from astropy import units as u
from astropy.units import UnitConversionError
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
from gwexpy.types.seriesmatrix import SeriesMatrix


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


def _assert_numeric_result_equal(
    actual: Any,
    expected: Any,
    *,
    compare_gwpy_axes: bool = True,
) -> None:
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
    if compare_gwpy_axes:
        _assert_gwpy_axis_metadata_equal(actual, expected)


def _assert_gwpy_axis_metadata_equal(actual: Any, expected: Any) -> None:
    for attribute in ("xindex", "yindex"):
        try:
            expected_index = getattr(expected, attribute)
        except Exception as expected_error:
            with pytest.raises(type(expected_error)):
                getattr(actual, attribute)
            continue

        actual_index = getattr(actual, attribute)
        assert actual_index.unit == expected_index.unit
        np.testing.assert_array_equal(actual_index.value, expected_index.value)


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


ARRAY2D_CASES = (
    pytest.param(Array2D, id="Array2D"),
    pytest.param(Plane2D, id="Plane2D"),
)


def _make_array2d_pair(
    gwexpy_class: type,
    *,
    index_route: str,
) -> tuple[Any, GwpyArray2D]:
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    kwargs = {"unit": u.m}
    if index_route == "explicit-physical":
        kwargs.update(
            xindex=[10, 20] * u.s,
            yindex=[1, 2, 4] * u.Hz,
        )
    elif index_route == "explicit-default-valued":
        kwargs.update(
            xindex=np.arange(2),
            yindex=np.arange(3),
        )
    if gwexpy_class is Plane2D:
        actual = gwexpy_class(
            values.copy(),
            axis1_name="time",
            axis2_name="frequency",
            **kwargs,
        )
    else:
        actual = gwexpy_class(
            values.copy(),
            axis_names=("time", "frequency"),
            **kwargs,
        )
    return actual, GwpyArray2D(values.copy(), **kwargs)


@pytest.mark.parametrize("gwexpy_class", ARRAY2D_CASES)
@pytest.mark.parametrize("method_name", STAT_METHODS)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "index_route",
    ["implicit", "explicit-physical", "explicit-default-valued"],
)
def test_array2d_keepdims_axis_metadata_matches_gwpy(
    gwexpy_class: type,
    method_name: str,
    axis: int,
    index_route: str,
) -> None:
    actual_input, expected_input = _make_array2d_pair(
        gwexpy_class,
        index_route=index_route,
    )

    actual = getattr(actual_input, method_name)(axis=axis, keepdims=True)
    expected = getattr(expected_input, method_name)(axis=axis, keepdims=True)

    _assert_numeric_result_equal(actual, expected)


@pytest.mark.parametrize("gwexpy_class", ARRAY2D_CASES)
@pytest.mark.parametrize("method_name", ["min", "max"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "index_route",
    ["implicit", "explicit-physical", "explicit-default-valued"],
)
def test_array2d_min_max_reduced_stale_index_outcome_matches_gwpy(
    gwexpy_class: type,
    method_name: str,
    axis: int,
    index_route: str,
) -> None:
    actual_input, expected_input = _make_array2d_pair(
        gwexpy_class,
        index_route=index_route,
    )

    try:
        expected = getattr(expected_input, method_name)(axis=axis, keepdims=False)
    except Exception as expected_error:
        with pytest.raises(type(expected_error)):
            getattr(actual_input, method_name)(axis=axis, keepdims=False)
        return

    actual = getattr(actual_input, method_name)(axis=axis, keepdims=False)
    _assert_numeric_result_equal(actual, expected)


@dataclass(frozen=True)
class ProjectAxisCase:
    """Define the project-owned coordinate contract for N-D reductions."""

    gwexpy_class: type
    shape: tuple[int, ...]
    constructor_kwargs: dict[str, Any]


PROJECT_AXIS_CASES = (
    pytest.param(
        ProjectAxisCase(
            Array,
            (2, 3, 4),
            {"axis_names": ("time", "distance", "frequency")},
        ),
        id="Array",
    ),
    pytest.param(
        ProjectAxisCase(
            Array3D,
            (2, 3, 4),
            {
                "axis_names": ("time", "distance", "frequency"),
                "axis0": [10, 20] * u.s,
                "axis1": [1, 2, 4] * u.m,
                "axis2": [5, 6, 8, 11] * u.Hz,
            },
        ),
        id="Array3D",
    ),
    pytest.param(
        ProjectAxisCase(
            Array4D,
            (2, 3, 4, 5),
            {
                "axis_names": ("time", "x", "y", "z"),
                "axis0": [10, 20] * u.s,
                "axis1": [1, 2, 4] * u.m,
                "axis2": [5, 6, 8, 11] * u.m,
                "axis3": [0, 2, 4, 8, 16] * u.m,
            },
        ),
        id="Array4D",
    ),
    pytest.param(
        ProjectAxisCase(
            ScalarField,
            (2, 3, 4, 5),
            {
                "axis_names": ("time", "x", "y", "z"),
                "axis0": [10, 20] * u.s,
                "axis1": [1, 2, 4] * u.m,
                "axis2": [5, 6, 8, 11] * u.m,
                "axis3": [0, 2, 4, 8, 16] * u.m,
            },
        ),
        id="ScalarField",
    ),
)

PROJECT_REDUCTION_AXES = (
    pytest.param(0, id="first-axis"),
    pytest.param(-1, id="last-axis"),
    pytest.param((0, 2), id="axis-tuple"),
)


def _make_project_axis_pair(case: ProjectAxisCase) -> tuple[Any, GwpyArray]:
    values = np.arange(np.prod(case.shape), dtype=np.float64).reshape(case.shape)
    actual = case.gwexpy_class(
        values.copy(),
        unit=u.V,
        name="statistics-source",
        **case.constructor_kwargs,
    )
    expected = GwpyArray(values.copy(), unit=u.V, name="statistics-source")
    return actual, expected


def _normalized_axes(axis: int | tuple[int, ...], ndim: int) -> tuple[int, ...]:
    axes = axis if isinstance(axis, tuple) else (axis,)
    return tuple(item if item >= 0 else ndim + item for item in axes)


def _assert_project_reduction_axes(
    result: Any,
    source: Any,
    axis: int | tuple[int, ...],
    keepdims: bool,
) -> None:
    reduced_axes = _normalized_axes(axis, source.ndim)
    source_axes = source.axes

    if keepdims:
        assert type(result) is type(source)
        assert result.axis_names == source.axis_names
        assert len(result.axes) == result.ndim
        for index, (actual_axis, source_axis) in enumerate(
            zip(result.axes, source_axes, strict=True)
        ):
            assert actual_axis.name == source_axis.name
            assert len(actual_axis.index) == result.shape[index]
            if index in reduced_axes:
                assert actual_axis.unit == u.dimensionless_unscaled
                np.testing.assert_array_equal(actual_axis.index.value, [0])
            else:
                assert actual_axis.unit == source_axis.unit
                np.testing.assert_array_equal(
                    actual_axis.index.value,
                    source_axis.index.value,
                )
        return

    surviving_axes = [
        source_axis
        for index, source_axis in enumerate(source_axes)
        if index not in reduced_axes
    ]
    assert result.ndim == len(surviving_axes)
    if type(source) is Array:
        assert type(result) is Array
    elif len(surviving_axes) == 3:
        assert type(result) is Array3D
    elif len(surviving_axes) == 2:
        assert type(result) is Plane2D
    elif len(surviving_axes) == 1:
        assert type(result) is Series
    if not surviving_axes:
        assert type(result) is u.Quantity
        return

    if hasattr(result, "axes"):
        assert result.axis_names == tuple(axis_.name for axis_ in surviving_axes)
        assert len(result.axes) == len(surviving_axes)
        for actual_axis, source_axis in zip(result.axes, surviving_axes, strict=True):
            assert actual_axis.name == source_axis.name
            assert actual_axis.unit == source_axis.unit
            np.testing.assert_array_equal(
                actual_axis.index.value,
                source_axis.index.value,
            )
        return

    assert result.ndim == 1
    assert result.xindex.info.name == surviving_axes[0].name
    assert result.xindex.unit == surviving_axes[0].unit
    np.testing.assert_array_equal(
        result.xindex.value,
        surviving_axes[0].index.value,
    )


@pytest.mark.parametrize("case", PROJECT_AXIS_CASES)
@pytest.mark.parametrize("method_name", STAT_METHODS)
@pytest.mark.parametrize("axis", PROJECT_REDUCTION_AXES)
@pytest.mark.parametrize("keepdims", [False, True], ids=["drop", "keep"])
def test_project_nd_reductions_return_coherent_axis_objects(
    case: ProjectAxisCase,
    method_name: str,
    axis: int | tuple[int, ...],
    keepdims: bool,
) -> None:
    actual_input, expected_input = _make_project_axis_pair(case)

    actual = getattr(actual_input, method_name)(axis=axis, keepdims=keepdims)
    expected = getattr(expected_input, method_name)(axis=axis, keepdims=keepdims)

    _assert_numeric_result_equal(actual, expected, compare_gwpy_axes=False)
    assert actual.name == expected.name
    _assert_project_reduction_axes(actual, actual_input, axis, keepdims)


NAN_ARRAY3D_CASE = ProjectAxisCase(
    Array3D,
    (2, 3, 4),
    {
        "axis_names": ("time", "distance", "frequency"),
        "axis0": [10, 20] * u.s,
        "axis1": [1, 2, 4] * u.m,
        "axis2": [5, 6, 8, 11] * u.Hz,
    },
)


NAN_METADATA_CASES = (
    pytest.param(
        ProjectAxisCase(
            Array,
            (2, 3, 4),
            {"axis_names": ("time", "distance", "frequency")},
        ),
        id="Array",
    ),
    pytest.param(
        ProjectAxisCase(
            Array2D,
            (2, 3),
            {
                "axis_names": ("time", "frequency"),
                "xindex": [10, 20] * u.s,
                "yindex": [1, 2, 4] * u.Hz,
            },
        ),
        id="Array2D",
    ),
    pytest.param(
        ProjectAxisCase(
            Plane2D,
            (2, 3),
            {
                "axis1_name": "time",
                "axis2_name": "frequency",
                "xindex": [10, 20] * u.s,
                "yindex": [1, 2, 4] * u.Hz,
            },
        ),
        id="Plane2D",
    ),
    pytest.param(
        NAN_ARRAY3D_CASE,
        id="Array3D",
    ),
    pytest.param(
        ProjectAxisCase(
            Array4D,
            (2, 3, 4, 5),
            {
                "axis_names": ("time", "x", "y", "z"),
                "axis0": [10, 20] * u.s,
                "axis1": [1, 2, 4] * u.m,
                "axis2": [5, 6, 8, 11] * u.m,
                "axis3": [0, 2, 4, 8, 16] * u.m,
            },
        ),
        id="Array4D",
    ),
    pytest.param(
        ProjectAxisCase(
            ScalarField,
            (2, 3, 4, 5),
            {
                "axis_names": ("time", "x", "y", "z"),
                "axis0": [10, 20] * u.s,
                "axis1": [1, 2, 4] * u.m,
                "axis2": [5, 6, 8, 11] * u.m,
                "axis3": [0, 2, 4, 8, 16] * u.m,
            },
        ),
        id="ScalarField",
    ),
)

NAN_REDUCTION_AXES = (
    pytest.param(0, id="first-axis"),
    pytest.param(-1, id="last-axis"),
    pytest.param((0, 1), id="axis-tuple"),
)


def _make_nan_metadata_source(case: ProjectAxisCase) -> Any:
    values = np.arange(1, np.prod(case.shape) + 1, dtype=np.float64).reshape(case.shape)
    values.reshape(-1)[1] = np.nan
    return case.gwexpy_class(
        values,
        unit=u.V,
        name="nan-statistics-source",
        **case.constructor_kwargs,
    )


def _assert_index_metadata_matches(actual: Any, expected: Any, attribute: str) -> None:
    try:
        expected_index = getattr(expected, attribute)
    except Exception as expected_error:
        with pytest.raises(type(expected_error)):
            getattr(actual, attribute)
        return

    actual_index = getattr(actual, attribute)
    assert actual_index.info.name == expected_index.info.name
    assert actual_index.unit == expected_index.unit
    np.testing.assert_array_equal(actual_index.value, expected_index.value)


def _assert_reduction_metadata_matches(actual: Any, expected: Any) -> None:
    assert type(actual) is type(expected)
    assert getattr(actual, "name", None) == getattr(expected, "name", None)
    assert getattr(actual, "axis_names", None) == getattr(expected, "axis_names", None)

    for attribute in ("xindex", "yindex"):
        _assert_index_metadata_matches(actual, expected, attribute)

    expected_axes = getattr(expected, "axes", ())
    actual_axes = getattr(actual, "axes", ())
    assert len(actual_axes) == len(expected_axes)
    for actual_axis, expected_axis in zip(actual_axes, expected_axes, strict=True):
        assert actual_axis.name == expected_axis.name
        assert actual_axis.unit == expected_axis.unit
        np.testing.assert_array_equal(
            actual_axis.index.value,
            expected_axis.index.value,
        )


@pytest.mark.parametrize("case", NAN_METADATA_CASES)
@pytest.mark.parametrize("method_name", STAT_METHODS)
@pytest.mark.parametrize("axis", NAN_REDUCTION_AXES)
@pytest.mark.parametrize("keepdims", [False, True], ids=["drop", "keep"])
def test_explicit_ignore_nan_reduction_uses_shared_metadata_postprocessor(
    case: ProjectAxisCase,
    method_name: str,
    axis: int | tuple[int, ...],
    keepdims: bool,
) -> None:
    source = _make_nan_metadata_source(case)
    template_method = (
        "mean"
        if isinstance(source, Array2D) and method_name in {"min", "max"}
        else method_name
    )
    metadata_template = getattr(source, template_method)(
        axis=axis,
        keepdims=keepdims,
    )
    if template_method != method_name and getattr(metadata_template, "ndim", 0) == 1:
        metadata_template.name = f"{source.name} {method_name}"
    function = getattr(np, f"nan{method_name}")
    expected_values = function(source.value, axis=axis, keepdims=keepdims)
    expected_unit = u.V**2 if method_name == "var" else u.V

    with np.errstate(all="ignore"):
        actual = getattr(source, method_name)(
            axis=axis,
            keepdims=keepdims,
            ignore_nan=True,
        )

    assert actual.unit == expected_unit
    np.testing.assert_array_equal(actual.value, expected_values)
    _assert_reduction_metadata_matches(actual, metadata_template)


@pytest.mark.parametrize("method_name", STAT_METHODS)
@pytest.mark.parametrize("keepdims", [False, True], ids=["drop", "keep"])
def test_series_matrix_ignore_nan_keeps_raw_ndarray_route(
    method_name: str,
    keepdims: bool,
) -> None:
    values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    values[0, 0, 1] = np.nan
    source = SeriesMatrix(values, xindex=np.arange(values.shape[-1]))
    function = getattr(np, f"nan{method_name}")
    expected = function(values, axis=-1, keepdims=keepdims)

    actual = getattr(source, method_name)(
        axis=-1,
        keepdims=keepdims,
        ignore_nan=True,
    )

    assert type(actual) is np.ndarray
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("method_name", STAT_METHODS)
def test_explicit_ignore_nan_ndarray_out_behavior_is_preserved(
    method_name: str,
) -> None:
    source = _make_nan_metadata_source(NAN_ARRAY3D_CASE)
    output = np.full(source.shape[:-1], -99.0)
    function = getattr(np, f"nan{method_name}")
    expected = function(source.value, axis=-1)

    actual = getattr(source, method_name)(
        axis=-1,
        out=output,
        ignore_nan=True,
    )

    assert actual is not output
    assert type(actual) is u.Quantity
    assert actual.unit == (u.V**2 if method_name == "var" else u.V)
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(actual.value, expected)


@pytest.mark.parametrize("method_name", STAT_METHODS)
def test_explicit_ignore_nan_quantity_out_error_and_mutation_are_preserved(
    method_name: str,
) -> None:
    source = _make_nan_metadata_source(NAN_ARRAY3D_CASE)
    output = u.Quantity(
        np.full(source.shape[:-1], -99.0),
        unit=u.V**2 if method_name == "var" else u.V,
    )
    function = getattr(np, f"nan{method_name}")
    expected = function(source.value, axis=-1)

    with pytest.raises(UnitConversionError):
        getattr(source, method_name)(
            axis=-1,
            out=output,
            ignore_nan=True,
        )

    assert output.unit == u.dimensionless_unscaled
    np.testing.assert_array_equal(output.value, expected)
