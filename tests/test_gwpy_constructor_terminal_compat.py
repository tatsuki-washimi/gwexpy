"""Direct terminal contracts for public GWpy-counterpart constructors."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
from gwpy.plot import Plot as GwpyPlot
from gwpy.timeseries import TimeSeries as GwpyTimeSeries
from gwpy.types import Array as GwpyArray
from gwpy.types import Array2D as GwpyArray2D

import gwexpy
from gwexpy.fields import ScalarField
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.plot import FieldPlot, Plot, SkyMap
from gwexpy.types import Array, Array2D, Array3D, Array4D, Plane2D

gwexpy.register_all()


ARRAY_CONSTRUCTORS = [
    pytest.param(Array, GwpyArray, ("axis_names",), id="Array"),
    pytest.param(Array2D, GwpyArray2D, ("axis_names",), id="Array2D"),
    pytest.param(
        Array3D,
        GwpyArray,
        ("axis0", "axis1", "axis2", "axis_names"),
        id="Array3D",
    ),
    pytest.param(
        Array4D,
        GwpyArray,
        ("axis0", "axis1", "axis2", "axis3", "axis_names"),
        id="Array4D",
    ),
    pytest.param(
        Plane2D,
        GwpyArray2D,
        ("axis1_name", "axis2_name", "axis_names"),
        id="Plane2D",
    ),
    pytest.param(
        ScalarField,
        GwpyArray,
        (
            "axis0",
            "axis1",
            "axis2",
            "axis3",
            "axis_names",
            "axis0_domain",
            "space_domain",
        ),
        id="ScalarField",
    ),
    pytest.param(FrequencySeries, GwpyFrequencySeries, (), id="FrequencySeries"),
]


def _assert_common_parameter_layout(
    actual: Callable[..., Any],
    expected: Callable[..., Any],
    extensions: tuple[str, ...],
) -> None:
    actual_parameters = inspect.signature(actual).parameters
    expected_parameters = inspect.signature(expected).parameters
    common_names = [name for name in actual_parameters if name not in extensions]

    assert common_names == list(expected_parameters)
    for name, expected_parameter in expected_parameters.items():
        actual_parameter = actual_parameters[name]
        assert actual_parameter.kind is expected_parameter.kind
        assert actual_parameter.default == expected_parameter.default
    for name in extensions:
        assert actual_parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


def _exception_class(call: Callable[[], Any]) -> type[BaseException] | None:
    try:
        call()
    except BaseException as exc:  # noqa: BLE001 - exception type is the oracle
        return type(exc)
    return None


def _assert_common_array_metadata(actual: Any, expected: Any) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.unit == expected.unit
    assert actual.name == expected.name
    assert str(actual.channel) == str(expected.channel)
    assert actual.epoch == expected.epoch
    np.testing.assert_array_equal(actual.value, expected.value)


@pytest.mark.parametrize(("actual", "expected", "extensions"), ARRAY_CONSTRUCTORS)
def test_constructor_common_parameter_layout_matches_gwpy(
    actual: type[Any], expected: type[Any], extensions: tuple[str, ...]
) -> None:
    _assert_common_parameter_layout(actual.__new__, expected.__new__, extensions)


@pytest.mark.parametrize(
    ("actual_type", "expected_type", "shape"),
    [
        pytest.param(Array, GwpyArray, (6,), id="Array"),
        pytest.param(Array3D, GwpyArray, (2, 3, 4), id="Array3D"),
        pytest.param(Array4D, GwpyArray, (2, 3, 4, 5), id="Array4D"),
        pytest.param(ScalarField, GwpyArray, (2, 3, 4, 5), id="ScalarField"),
    ],
)
def test_array_family_common_keyword_route_matches_gwpy(
    actual_type: type[Any], expected_type: type[Any], shape: tuple[int, ...]
) -> None:
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    kwargs = {
        "unit": u.m,
        "name": "constructor-source",
        "epoch": 1_234_567_890,
        "channel": "H1:TEST",
        "dtype": np.float32,
        "copy": True,
        "subok": True,
        "order": "C",
        "ndmin": len(shape),
    }

    actual = actual_type(value=values, **kwargs)
    expected = expected_type(value=values, **kwargs)

    _assert_common_array_metadata(actual, expected)


@pytest.mark.parametrize(
    "actual_type", [Array2D, Plane2D], ids=lambda cls: cls.__name__
)
def test_array2d_family_full_positional_prefix_matches_gwpy(
    actual_type: type[Any],
) -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    args = (u.V, 10, 2, None, u.Hz, 1, 0.5, None, u.s)
    kwargs = {"name": "grid", "epoch": 1_234_567_890, "channel": "H1:TEST"}

    actual = actual_type(values, *args, **kwargs)
    expected = GwpyArray2D(values, *args, **kwargs)

    _assert_common_array_metadata(actual, expected)
    for name in ("xindex", "yindex"):
        actual_index = getattr(actual, name)
        expected_index = getattr(expected, name)
        assert actual_index.unit == expected_index.unit
        np.testing.assert_array_equal(actual_index.value, expected_index.value)


@pytest.mark.parametrize(
    "actual_type", [Array2D, Plane2D], ids=lambda cls: cls.__name__
)
def test_array2d_family_explicit_indices_match_gwpy(
    actual_type: type[Any],
) -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    kwargs = {
        "unit": u.V,
        "xindex": np.arange(3) * u.dimensionless_unscaled,
        "yindex": [1, 2, 4, 8] * u.s,
        "name": "explicit-grid",
    }

    actual = actual_type(values, **kwargs)
    expected = GwpyArray2D(values, **kwargs)

    _assert_common_array_metadata(actual, expected)
    for name in ("xindex", "yindex"):
        actual_index = getattr(actual, name)
        expected_index = getattr(expected, name)
        assert actual_index.unit == expected_index.unit
        np.testing.assert_array_equal(actual_index.value, expected_index.value)
        assert f"_{name}" in actual.__dict__


@pytest.mark.parametrize(
    ("actual_type", "expected_type", "shape"),
    [
        pytest.param(Array, GwpyArray, (6,), id="Array"),
        pytest.param(Array2D, GwpyArray2D, (2, 3), id="Array2D"),
        pytest.param(Array3D, GwpyArray, (2, 3, 4), id="Array3D"),
        pytest.param(Array4D, GwpyArray, (2, 3, 4, 5), id="Array4D"),
        pytest.param(Plane2D, GwpyArray2D, (2, 3), id="Plane2D"),
        pytest.param(ScalarField, GwpyArray, (2, 3, 4, 5), id="ScalarField"),
    ],
)
def test_array_family_copy_false_matches_gwpy(
    actual_type: type[Any], expected_type: type[Any], shape: tuple[int, ...]
) -> None:
    actual_source = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    expected_source = actual_source.copy()

    actual = actual_type(actual_source, unit=u.m, copy=False)
    expected = expected_type(expected_source, unit=u.m, copy=False)

    _assert_common_array_metadata(actual, expected)
    assert np.shares_memory(actual_source, actual.value) is np.shares_memory(
        expected_source, expected.value
    )


@pytest.mark.parametrize(
    ("actual_type", "expected_type", "shape"),
    [
        pytest.param(Array, GwpyArray, (6,), id="Array"),
        pytest.param(Array2D, GwpyArray2D, (2, 3), id="Array2D"),
        pytest.param(Array3D, GwpyArray, (2, 3, 4), id="Array3D"),
        pytest.param(Array4D, GwpyArray, (2, 3, 4, 5), id="Array4D"),
        pytest.param(Plane2D, GwpyArray2D, (2, 3), id="Plane2D"),
        pytest.param(ScalarField, GwpyArray, (2, 3, 4, 5), id="ScalarField"),
    ],
)
def test_array_family_common_failure_class_matches_gwpy(
    actual_type: type[Any], expected_type: type[Any], shape: tuple[int, ...]
) -> None:
    values = np.ones(shape) * u.m

    assert _exception_class(lambda: actual_type(values, unit=u.s)) is (
        _exception_class(lambda: expected_type(values, unit=u.s))
    )
    assert _exception_class(lambda: actual_type(values, unsupported=True)) is (
        _exception_class(lambda: expected_type(values, unsupported=True))
    )


@pytest.mark.parametrize(
    "actual_type", [Array, Array3D, Array4D, ScalarField], ids=lambda cls: cls.__name__
)
def test_gwpy_array_keyword_only_binding_is_not_hijacked(
    actual_type: type[Any],
) -> None:
    shape = {
        Array: (6,),
        Array3D: (2, 3, 4),
        Array4D: (2, 3, 4, 5),
        ScalarField: (2, 3, 4, 5),
    }[actual_type]
    values = np.ones(shape)

    assert _exception_class(lambda: actual_type(values, u.m)) is (
        _exception_class(lambda: GwpyArray(values, u.m))
    )


def test_constructor_extensions_are_explicit_and_do_not_retry_parent_forms() -> None:
    array = Array([1, 2], axis_names=("sample",))
    array2d = Array2D(np.ones((2, 3)), axis_names=("row", "column"))
    array3d = Array3D(
        np.ones((2, 3, 4)),
        axis0=[0, 1] * u.s,
        axis_names=("time", "row", "column"),
    )
    array4d = Array4D(
        np.ones((2, 3, 4, 5)),
        axis3=np.arange(5) * u.m,
        axis_names=("time", "x", "y", "z"),
    )
    plane = Plane2D(np.ones((2, 3)), axis1_name="row", axis2_name="column")
    field = ScalarField(
        np.ones((2, 3, 4, 5)),
        axis0=np.arange(2) * u.s,
        axis1=np.arange(3) * u.m,
        axis2=np.arange(4) * u.m,
        axis3=np.arange(5) * u.m,
        axis0_domain="time",
        space_domain="real",
    )

    assert array.axis_names == ("sample",)
    assert array2d.axis_names == ("row", "column")
    assert array3d.axis_names == ("time", "row", "column")
    assert array4d.axis_names == ("time", "x", "y", "z")
    assert plane.axis_names == ("row", "column")
    assert field.axis0_domain == "time"
    assert set(field.space_domains.values()) == {"real"}
    for kwargs in (
        {"axis_names": ("sample",)},
        {"axis0": [0, 1]},
        {"axis0_domain": "time"},
    ):
        assert _exception_class(lambda kwargs=kwargs: GwpyArray([1, 2], **kwargs)) is (
            TypeError
        )


def test_frequencyseries_common_constructor_matches_gwpy() -> None:
    values = np.arange(6, dtype=np.float64)
    expected_values = values.copy()
    kwargs = {
        "unit": u.V,
        "f0": 10 * u.Hz,
        "df": 0.25 * u.Hz,
        "name": "spectrum",
        "epoch": 1_234_567_890,
        "channel": "H1:TEST",
        "copy": False,
    }

    actual = FrequencySeries(values, **kwargs)
    expected = GwpyFrequencySeries(expected_values, **kwargs)

    _assert_common_array_metadata(actual, expected)
    assert actual.f0 == expected.f0
    assert actual.df == expected.df
    np.testing.assert_array_equal(actual.frequencies.value, expected.frequencies.value)
    assert np.shares_memory(values, actual.value) is np.shares_memory(
        expected_values, expected.value
    )


def test_frequencyseries_common_failure_class_matches_gwpy() -> None:
    incompatible = np.arange(4, dtype=float) * u.m

    assert _exception_class(
        lambda: FrequencySeries(incompatible, unit=u.s, df=1 * u.Hz)
    ) is _exception_class(
        lambda: GwpyFrequencySeries(incompatible, unit=u.s, df=1 * u.Hz)
    )
    assert _exception_class(
        lambda: FrequencySeries([1, 2, 3], df=1 * u.Hz, unsupported=True)
    ) is _exception_class(
        lambda: GwpyFrequencySeries([1, 2, 3], df=1 * u.Hz, unsupported=True)
    )


def test_frequencyseries_explicit_frequency_index_matches_gwpy() -> None:
    values = np.arange(4, dtype=np.float64)
    frequencies = [1, 2, 4, 8] * u.Hz

    actual = FrequencySeries(values, unit=u.V, frequencies=frequencies)
    expected = GwpyFrequencySeries(values, unit=u.V, frequencies=frequencies)

    _assert_common_array_metadata(actual, expected)
    assert actual.frequencies.unit == expected.frequencies.unit
    np.testing.assert_array_equal(actual.frequencies.value, expected.frequencies.value)


def test_frequencyseries_explicit_noise_range_extension_is_isolated() -> None:
    actual = FrequencySeries([1, 2, 3], df=1 * u.Hz, fmin=10, fmax=20)

    assert actual.shape == (3,)
    assert (
        _exception_class(
            lambda: GwpyFrequencySeries([1, 2, 3], df=1 * u.Hz, fmin=10, fmax=20)
        )
        is TypeError
    )


@pytest.mark.parametrize("actual_type", [Plot, FieldPlot, SkyMap])
def test_plot_constructor_variadic_binding_matches_gwpy(actual_type: type[Any]) -> None:
    actual_parameters = list(
        inspect.signature(actual_type.__init__).parameters.values()
    )
    expected_parameters = list(inspect.signature(GwpyPlot.__init__).parameters.values())

    assert [parameter.kind for parameter in actual_parameters] == [
        parameter.kind for parameter in expected_parameters
    ]
    assert [parameter.default for parameter in actual_parameters] == [
        parameter.default for parameter in expected_parameters
    ]


@pytest.mark.parametrize("actual_type", [Plot, FieldPlot])
def test_plot_family_data_constructor_matches_gwpy(actual_type: type[Any]) -> None:
    actual_series = GwpyTimeSeries(
        np.arange(6, dtype=np.float64), t0=1_000_000_000, dt=0.25, unit=u.V
    )
    expected_series = actual_series.copy()

    actual = actual_type(actual_series)
    expected = GwpyPlot(expected_series)
    try:
        assert len(actual.axes) == len(expected.axes) == 1
        assert actual.axes[0].name == expected.axes[0].name
        assert len(actual.axes[0].lines) == len(expected.axes[0].lines) == 1
        np.testing.assert_array_equal(
            actual.axes[0].lines[0].get_xdata(),
            expected.axes[0].lines[0].get_xdata(),
        )
        np.testing.assert_array_equal(
            actual.axes[0].lines[0].get_ydata(),
            expected.axes[0].lines[0].get_ydata(),
        )
        if isinstance(actual, FieldPlot):
            assert actual.last_field_colorbar is None
    finally:
        plt.close(actual)
        plt.close(expected)


@pytest.mark.parametrize("actual_type", [Plot, FieldPlot])
@pytest.mark.parametrize(
    "kwargs", [{"method": "does_not_exist"}, {"unsupported_artist_kw": True}]
)
def test_plot_family_failure_class_matches_gwpy(
    actual_type: type[Any], kwargs: dict[str, Any]
) -> None:
    actual_series = GwpyTimeSeries(np.arange(4, dtype=float), t0=0, dt=1)
    expected_series = actual_series.copy()

    assert _exception_class(lambda: actual_type(actual_series, **kwargs)) is (
        _exception_class(lambda: GwpyPlot(expected_series, **kwargs))
    )
