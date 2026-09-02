"""Differential contracts for ``BifrequencyMap`` GWpy name collisions."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u
from gwpy.types import Array2D as GWpyArray2D

from gwexpy.frequencyseries import BifrequencyMap


def _map_pair(name: str | None = "audit-map") -> tuple[BifrequencyMap, GWpyArray2D]:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    kwargs = {
        "unit": u.V,
        "xindex": [10, 20, 40] * u.Hz,
        "yindex": [1, 2, 4, 8] * u.Hz,
        "name": name,
    }
    return BifrequencyMap(values.copy(), **kwargs), GWpyArray2D(values.copy(), **kwargs)


def _assert_array2d_payload_equal(actual: Any, expected: Any) -> None:
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


def _exception_class(call: Callable[[], Any]) -> type[BaseException] | None:
    try:
        call()
    except BaseException as exc:  # noqa: BLE001 - exception type is the oracle
        return type(exc)
    return None


def test_diagonal_signature_keeps_gwpy_layout_before_extensions() -> None:
    parameters = inspect.signature(BifrequencyMap.diagonal).parameters
    assert list(parameters)[:4] == ["self", "offset", "axis1", "axis2"]
    assert parameters["offset"].default == 0
    assert parameters["axis1"].default == 0
    assert parameters["axis2"].default == 1
    for name in ("method", "bins", "absolute"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        pytest.param((), {}, id="default"),
        pytest.param((1,), {}, id="offset"),
        pytest.param((1, 0, 1), {}, id="all-positional"),
        pytest.param((), {"offset": -1, "axis1": 0, "axis2": 1}, id="keywords"),
    ],
)
def test_diagonal_common_numeric_routes_match_gwpy(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    actual_input, expected_input = _map_pair()

    actual = actual_input.diagonal(*args, **kwargs)
    expected = expected_input.diagonal(*args, **kwargs)

    assert isinstance(actual, BifrequencyMap)
    _assert_array2d_payload_equal(actual, expected)
    assert np.shares_memory(actual_input.value, actual.value) is np.shares_memory(
        expected_input.value, expected.value
    )


def test_diagonal_explicit_binned_projection_remains_available() -> None:
    actual_input, _ = _map_pair()

    keyword = actual_input.diagonal(method="mean", bins=3, absolute=True)
    legacy = actual_input.diagonal("mean", 3, True)

    np.testing.assert_allclose(keyword.value, [2.6, 5.0, 9.5])
    np.testing.assert_array_equal(keyword.value, legacy.value)
    np.testing.assert_array_equal(keyword.frequencies, legacy.frequencies)
    assert keyword.unit == u.V
    assert keyword.name == "audit-map (diagonal mean)"

    signed_input = BifrequencyMap(
        np.arange(12, dtype=float).reshape(3, 4),
        xindex=[0, 1, 2] * u.Hz,
        yindex=[-2, 0, 2, 4] * u.Hz,
    )
    signed_keyword = signed_input.diagonal(method="mean", bins=3, absolute=True)
    signed_legacy = signed_input.diagonal("mean", 3, True)
    np.testing.assert_allclose(
        signed_legacy.value, signed_keyword.value, equal_nan=True
    )
    np.testing.assert_array_equal(signed_legacy.frequencies, signed_keyword.frequencies)


@pytest.mark.parametrize(
    "args",
    [
        pytest.param((1, 0, 1, 2), id="excess-positional"),
        pytest.param((0, 0, 5), id="axis-out-of-bounds"),
        pytest.param(("not-a-statistic",), id="invalid-string-extension"),
    ],
)
def test_diagonal_invalid_outcome_is_explicitly_classified(
    args: tuple[Any, ...],
) -> None:
    actual_input, expected_input = _map_pair()
    actual_error = _exception_class(lambda: actual_input.diagonal(*args))
    expected_error = _exception_class(lambda: expected_input.diagonal(*args))

    if isinstance(args[0], str):
        assert actual_error is ValueError
        assert expected_error is TypeError
    else:
        assert actual_error is expected_error


def test_crop_signature_preserves_gwpy_calling_form() -> None:
    parameters = inspect.signature(BifrequencyMap.crop).parameters
    assert list(parameters)[:3] == ["self", "start", "end"]
    assert parameters["start"].default is None
    assert parameters["end"].default is None
    assert parameters["copy"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["copy"].default is False


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        pytest.param((), {}, id="full-view"),
        pytest.param((15, 40), {}, id="positional-bounds"),
        pytest.param((), {"start": 15 * u.Hz, "end": 40 * u.Hz}, id="quantity"),
        pytest.param((10, 41), {"copy": True}, id="copied"),
    ],
)
def test_crop_common_route_matches_gwpy(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    actual_input, expected_input = _map_pair()

    actual = actual_input.crop(*args, **kwargs)
    expected = expected_input.crop(*args, **kwargs)

    assert isinstance(actual, BifrequencyMap)
    _assert_array2d_payload_equal(actual, expected)
    assert (actual is actual_input) is (expected is expected_input)
    assert np.shares_memory(actual_input.value, actual.value) is np.shares_memory(
        expected_input.value, expected.value
    )


def test_crop_explicit_two_axis_extension_preserves_legacy_selection() -> None:
    actual_input, expected_input = _map_pair()

    keyword = actual_input.crop(low=1, high=7, low2=20, high2=40)
    positional = actual_input.crop(1, 7, 20, 40)

    np.testing.assert_array_equal(keyword.value, [[4, 5, 6], [8, 9, 10]])
    np.testing.assert_array_equal(keyword.value, positional.value)
    np.testing.assert_array_equal(keyword.frequency2.value, [20, 40])
    np.testing.assert_array_equal(keyword.frequency1.value, [1, 2, 4])
    assert keyword.frequency2.unit == u.Hz
    assert keyword.frequency1.unit == u.Hz
    assert _exception_class(lambda: expected_input.crop(1, 7, 20, 40)) is TypeError


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        pytest.param((1, 2, 3, 4, 5), {}, id="excess-positional"),
        pytest.param((), {"start": 1, "low": 2}, id="duplicate-bound"),
        pytest.param((1 * u.s, 2 * u.s), {}, id="incompatible-unit"),
    ],
)
def test_crop_invalid_outcome_matches_gwpy_exception_class(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    actual_input, expected_input = _map_pair()

    assert _exception_class(lambda: actual_input.crop(*args, **kwargs)) is (
        _exception_class(lambda: expected_input.crop(*args, **kwargs))
    )


def _artist_payload(plot: Any) -> tuple[np.ndarray, str]:
    axes = plot.axes[0]
    artists = [*axes.images, *axes.collections]
    assert len(artists) == 1
    artist = artists[0]
    return np.asarray(artist.get_array()), artist.get_label()


@pytest.mark.parametrize("name", [None, "audit-map"])
@pytest.mark.parametrize("method", [None, "imshow", "pcolormesh"])
def test_plot_common_routes_match_gwpy_artist_source_and_label(
    name: str | None, method: str | None
) -> None:
    actual_input, expected_input = _map_pair(name)
    actual_kwargs = {} if method is None else {"method": method}

    actual = actual_input.plot(**actual_kwargs)
    expected = expected_input.plot(**actual_kwargs)
    try:
        actual_payload, actual_label = _artist_payload(actual)
        expected_payload, expected_label = _artist_payload(expected)
        np.testing.assert_array_equal(actual_payload, expected_payload)
        assert actual_label == expected_label
    finally:
        plt.close(actual)
        plt.close(expected)


@pytest.mark.parametrize("method", ["not-a-method", None, 1])
def test_plot_invalid_method_exception_class_matches_gwpy(method: Any) -> None:
    actual_input, expected_input = _map_pair()

    assert _exception_class(lambda: actual_input.plot(method)) is (
        _exception_class(lambda: expected_input.plot(method))
    )
