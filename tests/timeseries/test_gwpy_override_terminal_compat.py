"""Terminal differential contracts for core ``TimeSeries`` overrides.

The expected object in every common-path test is built independently with the
GWpy version installed in that qualification environment.  These tests close
the remaining #639 core-series and collection implementation groups without
turning GWexpy-only extensions into GWpy compatibility claims.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pytest
from astropy import units as u
from gwpy.time import LIGOTimeGPS
from gwpy.timeseries import TimeSeries as GWpyTimeSeries
from gwpy.timeseries import TimeSeriesDict as GWpyTimeSeriesDict

from gwexpy.timeseries import TimeSeries, TimeSeriesDict


def _exception_class(call: Callable[[], Any]) -> type[BaseException] | None:
    try:
        call()
    except BaseException as exc:  # noqa: BLE001 - exception class is the oracle
        return type(exc)
    return None


def _assert_parameter_layout(
    actual: Callable[..., Any], expected: Callable[..., Any]
) -> None:
    actual_parameters = inspect.signature(actual).parameters
    expected_parameters = inspect.signature(expected).parameters

    assert list(actual_parameters) == list(expected_parameters)
    for name, expected_parameter in expected_parameters.items():
        actual_parameter = actual_parameters[name]
        assert actual_parameter.kind is expected_parameter.kind
        assert actual_parameter.default == expected_parameter.default


def _assert_quantity_equal(actual: u.Quantity, expected: u.Quantity) -> None:
    assert actual.unit == expected.unit
    np.testing.assert_array_equal(actual.value, expected.value)


def _assert_numeric_payload_equal(actual: Any, expected: Any) -> None:
    actual_values = np.asarray(actual.value)
    expected_values = np.asarray(expected.value)
    assert actual_values.shape == expected_values.shape
    assert actual_values.dtype == expected_values.dtype

    if np.iscomplexobj(expected_values):
        components = (
            (actual_values.real, expected_values.real),
            (actual_values.imag, expected_values.imag),
        )
    else:
        components = ((actual_values, expected_values),)

    for actual_component, expected_component in components:
        for mask in (np.isnan, np.isposinf, np.isneginf):
            np.testing.assert_array_equal(
                mask(actual_component), mask(expected_component)
            )

    finite = np.isfinite(expected_values)
    np.testing.assert_array_equal(np.isfinite(actual_values), finite)
    np.testing.assert_allclose(
        actual_values[finite], expected_values[finite], rtol=0.0, atol=0.0
    )


def _assert_series_equal(actual: Any, expected: Any) -> None:
    _assert_numeric_payload_equal(actual, expected)
    assert isinstance(actual, TimeSeries)
    assert actual.unit == expected.unit
    assert actual.name == expected.name
    assert str(actual.channel) == str(expected.channel)
    _assert_quantity_equal(actual.t0, expected.t0)
    _assert_quantity_equal(actual.x0, expected.x0)
    _assert_quantity_equal(actual.dt, expected.dt)
    _assert_quantity_equal(actual.times, expected.times)


def _assert_spectrogram_equal(actual: Any, expected: Any) -> None:
    _assert_numeric_payload_equal(actual, expected)
    assert actual.unit == expected.unit
    assert actual.name == expected.name
    assert str(actual.channel) == str(expected.channel)
    _assert_quantity_equal(actual.t0, expected.t0)
    _assert_quantity_equal(actual.dt, expected.dt)
    _assert_quantity_equal(actual.f0, expected.f0)
    _assert_quantity_equal(actual.df, expected.df)
    _assert_quantity_equal(actual.times, expected.times)
    _assert_quantity_equal(actual.frequencies, expected.frequencies)


def _series(
    cls: type[GWpyTimeSeries] | type[TimeSeries],
    values: Any,
    *,
    t0: float = 1_234_567_890.125,
    dt: float = 0.25,
) -> GWpyTimeSeries | TimeSeries:
    return cls(
        np.asarray(values, dtype=np.float64),
        t0=t0,
        dt=dt,
        unit=u.V,
        name="terminal-series",
        channel="H1:TERMINAL",
    )


def _spectral_series(
    cls: type[GWpyTimeSeries] | type[TimeSeries],
    *,
    times: np.ndarray[Any, np.dtype[np.float64]] | None = None,
) -> GWpyTimeSeries | TimeSeries:
    rng = np.random.default_rng(639)
    values = rng.standard_normal(128)
    metadata = {
        "unit": u.V,
        "name": "terminal-spectrum",
        "channel": "H1:TERMINAL-SPECTRUM",
    }
    if times is not None:
        return cls(values, times=times, **metadata)
    return cls(values, sample_rate=32, t0=1_234_567_890.125, **metadata)


def _dict_series(
    cls: type[GWpyTimeSeries] | type[TimeSeries],
    key: str,
    *,
    t0: float = 1000.0,
    times: np.ndarray[Any, np.dtype[np.float64]] | None = None,
) -> GWpyTimeSeries | TimeSeries:
    values = np.sin(np.arange(64, dtype=np.float64) * 0.2)
    if key == "second":
        values = values + 1
    metadata = {
        "unit": u.V,
        "name": key,
        "channel": f"H1:{key.upper()}",
    }
    if times is not None:
        return cls(values, times=times, **metadata)
    return cls(values, t0=t0, sample_rate=16, **metadata)


def _series_dict(
    series_cls: type[GWpyTimeSeries] | type[TimeSeries],
    dict_cls: type[GWpyTimeSeriesDict] | type[TimeSeriesDict],
    *,
    times: np.ndarray[Any, np.dtype[np.float64]] | None = None,
) -> GWpyTimeSeriesDict | TimeSeriesDict:
    return dict_cls(
        {key: _dict_series(series_cls, key, times=times) for key in ("first", "second")}
    )


def _assert_dict_equal(actual: TimeSeriesDict, expected: GWpyTimeSeriesDict) -> None:
    assert isinstance(actual, TimeSeriesDict)
    assert list(actual) == list(expected)
    for key in expected:
        _assert_series_equal(actual[key], expected[key])


def test_timeseries_copy_parameter_layout_matches_gwpy() -> None:
    _assert_parameter_layout(TimeSeries.copy, GWpyTimeSeries.copy)


def test_timeseries_crop_parameter_layout_matches_gwpy() -> None:
    _assert_parameter_layout(TimeSeries.crop, GWpyTimeSeries.crop)


def test_timeseries_append_parameter_layout_matches_gwpy() -> None:
    _assert_parameter_layout(TimeSeries.append, GWpyTimeSeries.append)


def test_timeseries_spectrogram_parameter_layout_matches_gwpy() -> None:
    _assert_parameter_layout(TimeSeries.spectrogram, GWpyTimeSeries.spectrogram)


def test_timeseries_spectrogram2_parameter_layout_matches_gwpy() -> None:
    _assert_parameter_layout(TimeSeries.spectrogram2, GWpyTimeSeries.spectrogram2)


def test_timeseriesdict_crop_parameter_layout_matches_gwpy() -> None:
    _assert_parameter_layout(TimeSeriesDict.crop, GWpyTimeSeriesDict.crop)


def test_timeseriesdict_append_parameter_layout_matches_gwpy() -> None:
    _assert_parameter_layout(TimeSeriesDict.append, GWpyTimeSeriesDict.append)


def test_timeseriesdict_resample_parameter_layout_matches_gwpy() -> None:
    _assert_parameter_layout(TimeSeriesDict.resample, GWpyTimeSeriesDict.resample)


def test_timeseriesdict_crop_copy_is_keyword_only_like_gwpy() -> None:
    actual = _series_dict(TimeSeries, TimeSeriesDict)
    expected = _series_dict(GWpyTimeSeries, GWpyTimeSeriesDict)

    assert _exception_class(lambda: actual.crop(None, None, True)) is _exception_class(
        lambda: expected.crop(None, None, True)
    )


@pytest.mark.parametrize("copy", [False, True], ids=["view", "copy"])
@pytest.mark.parametrize(
    ("start", "end"),
    [
        pytest.param(None, None, id="unbounded"),
        pytest.param(1000.5, 1002.5, id="inner"),
    ],
)
def test_timeseriesdict_crop_values_mutation_and_memory_match_gwpy(
    start: float | None, end: float | None, copy: bool
) -> None:
    actual = _series_dict(TimeSeries, TimeSeriesDict)
    expected = _series_dict(GWpyTimeSeries, GWpyTimeSeriesDict)
    actual_sources = dict(actual)
    expected_sources = dict(expected)

    actual_result = actual.crop(start, end, copy=copy)
    expected_result = expected.crop(start, end, copy=copy)

    assert actual_result is actual
    assert expected_result is expected
    _assert_dict_equal(actual, expected)
    for key in expected:
        assert np.shares_memory(actual[key].value, actual_sources[key].value) is (
            np.shares_memory(expected[key].value, expected_sources[key].value)
        )


@pytest.mark.parametrize("order", ["C", "F", "A", "K"])
def test_timeseries_copy_orders_match_gwpy(order: str) -> None:
    actual_input = _series(TimeSeries, np.arange(12))[::2]
    expected_input = _series(GWpyTimeSeries, np.arange(12))[::2]

    actual = actual_input.copy(order=order)
    expected = expected_input.copy(order=order)

    _assert_series_equal(actual, expected)
    assert np.shares_memory(actual_input.value, actual.value) is np.shares_memory(
        expected_input.value, expected.value
    )
    assert actual.flags.c_contiguous is expected.flags.c_contiguous
    assert actual.flags.f_contiguous is expected.flags.f_contiguous


def test_timeseries_copy_preserves_private_exact_time_authority() -> None:
    epoch_ns = 1_234_567_890_123_456_789
    source = TimeSeries(np.arange(8.0), t0_ns=epoch_ns, dt=7 * u.ns)

    copied = source.copy(order="K")

    assert copied.t0_gps_ns == epoch_ns
    assert copied.__dict__["_gwex_dt_gps_ns"] == 7
    assert not np.shares_memory(source.value, copied.value)


@pytest.mark.parametrize(
    ("bounds", "copy"),
    [
        pytest.param("unbounded", False, id="unbounded-view"),
        pytest.param("exact-span", False, id="exact-span-view"),
        pytest.param("inner", False, id="inner-view"),
        pytest.param("one-ulp-around", False, id="one-ulp-around-boundaries"),
        pytest.param("outside", False, id="outside-clamped-view"),
        pytest.param("inner", True, id="inner-copy"),
    ],
)
def test_timeseries_crop_bounds_and_memory_match_gwpy(bounds: str, copy: bool) -> None:
    t0 = 1_234_567_890.1234567
    dt = 1 / 30
    size = 16
    actual_input = _series(TimeSeries, np.arange(size), t0=t0, dt=dt)
    expected_input = _series(GWpyTimeSeries, np.arange(size), t0=t0, dt=dt)

    if bounds == "unbounded":
        start = end = None
    elif bounds == "exact-span":
        start, end = t0, float(t0 + size * dt)
    elif bounds == "inner":
        start, end = float(t0 + 3 * dt), float(t0 + 11 * dt)
    elif bounds == "one-ulp-around":
        start = np.nextafter(float(t0 + 3 * dt), -np.inf)
        end = np.nextafter(float(t0 + 11 * dt), np.inf)
    else:
        start, end = float(t0 - 2 * dt), float(t0 + (size + 2) * dt)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        actual = actual_input.crop(start, end, copy=copy)
        expected = expected_input.crop(start, end, copy=copy)

    _assert_series_equal(actual, expected)
    assert (actual is actual_input) is (expected is expected_input)
    assert np.shares_memory(actual_input.value, actual.value) is np.shares_memory(
        expected_input.value, expected.value
    )


def test_timeseries_crop_copy_is_keyword_only_like_gwpy() -> None:
    actual = _series(TimeSeries, np.arange(8))
    expected = _series(GWpyTimeSeries, np.arange(8))

    assert _exception_class(lambda: actual.crop(None, None, True)) is _exception_class(
        lambda: expected.crop(None, None, True)
    )


def test_timeseries_crop_preserves_private_exact_time_authority() -> None:
    epoch_ns = 1_234_567_890_123_456_789
    source = TimeSeries(np.arange(16.0), t0_ns=epoch_ns, dt=125_000_000 * u.ns)

    cropped = source.crop(source.t0 + 0.5 * u.s, source.t0 + 1.5 * u.s)

    assert cropped.t0_gps_ns == epoch_ns + 500_000_000
    assert cropped.__dict__["_gwex_dt_gps_ns"] == 125_000_000


@pytest.mark.parametrize("inplace", [True, False], ids=["inplace", "copy"])
def test_timeseries_append_contiguous_matches_gwpy(inplace: bool) -> None:
    actual_left = _series(TimeSeries, [1, 2], t0=10, dt=1)
    actual_right = _series(TimeSeries, [3, 4], t0=12, dt=1)
    expected_left = _series(GWpyTimeSeries, [1, 2], t0=10, dt=1)
    expected_right = _series(GWpyTimeSeries, [3, 4], t0=12, dt=1)
    actual_before = actual_left.value.copy()
    expected_before = expected_left.value.copy()

    actual = actual_left.append(actual_right, inplace=inplace)
    expected = expected_left.append(expected_right, inplace=inplace)

    _assert_series_equal(actual, expected)
    assert (actual is actual_left) is (expected is expected_left)
    assert np.shares_memory(actual_left.value, actual.value) is np.shares_memory(
        expected_left.value, expected.value
    )
    np.testing.assert_array_equal(actual_left.value, expected_left.value)
    if not inplace:
        np.testing.assert_array_equal(actual_left.value, actual_before)
        np.testing.assert_array_equal(expected_left.value, expected_before)


def test_timeseries_append_gap_failure_matches_gwpy() -> None:
    actual_left = _series(TimeSeries, [1, 2], t0=10, dt=1)
    actual_right = _series(TimeSeries, [3, 4], t0=14, dt=1)
    expected_left = _series(GWpyTimeSeries, [1, 2], t0=10, dt=1)
    expected_right = _series(GWpyTimeSeries, [3, 4], t0=14, dt=1)

    assert _exception_class(
        lambda: actual_left.append(actual_right, gap="raise")
    ) is _exception_class(lambda: expected_left.append(expected_right, gap="raise"))
    np.testing.assert_array_equal(actual_left.value, expected_left.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"gap": "pad"}, id="zero-pad"),
        pytest.param({"pad": -9}, id="custom-pad"),
    ],
)
def test_timeseries_append_gap_padding_matches_gwpy(kwargs: dict[str, Any]) -> None:
    actual_left = _series(TimeSeries, [1, 2], t0=10, dt=1)
    actual_right = _series(TimeSeries, [3, 4], t0=14, dt=1)
    expected_left = _series(GWpyTimeSeries, [1, 2], t0=10, dt=1)
    expected_right = _series(GWpyTimeSeries, [3, 4], t0=14, dt=1)

    actual = actual_left.append(actual_right, **kwargs)
    expected = expected_left.append(expected_right, **kwargs)

    _assert_series_equal(actual, expected)
    assert actual is actual_left
    assert expected is expected_left


@pytest.mark.parametrize("inplace", [True, False], ids=["inplace", "copy"])
def test_timeseries_append_preserves_private_exact_time_authority(
    inplace: bool,
) -> None:
    epoch_ns = 1_234_567_890_123_456_789
    dt_ns = 125_000_000
    left = TimeSeries(np.arange(8.0), t0_ns=epoch_ns, dt=0.125, unit=u.V)
    right = TimeSeries(
        np.arange(4.0),
        t0_ns=epoch_ns + 8 * dt_ns,
        dt=0.125,
        unit=u.V,
    )

    result = left.append(right, inplace=inplace)

    assert result.t0_gps_ns == epoch_ns
    assert result.__dict__["_gwex_dt_gps_ns"] == dt_ns


def test_timeseries_epoch_alias_getters_match_gwpy() -> None:
    actual = _series(TimeSeries, np.arange(4), t0=1000, dt=0.25)
    expected = _series(GWpyTimeSeries, np.arange(4), t0=1000, dt=0.25)

    _assert_quantity_equal(actual.t0, expected.t0)
    _assert_quantity_equal(actual.x0, expected.x0)
    _assert_quantity_equal(actual.t0, actual.x0)


@pytest.mark.parametrize("attribute", ["t0", "x0"])
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(1234, id="numeric"),
        pytest.param(1234.5 * u.s, id="quantity"),
        pytest.param(LIGOTimeGPS(1234, 500_000_000), id="ligotimegps"),
    ],
)
def test_timeseries_epoch_alias_setters_match_gwpy(attribute: str, value: Any) -> None:
    actual = _series(TimeSeries, np.arange(4), t0=1000, dt=0.25)
    expected = _series(GWpyTimeSeries, np.arange(4), t0=1000, dt=0.25)

    setattr(actual, attribute, value)
    setattr(expected, attribute, value)

    _assert_quantity_equal(actual.t0, expected.t0)
    _assert_quantity_equal(actual.x0, expected.x0)
    _assert_quantity_equal(actual.t0, actual.x0)


@pytest.mark.parametrize("attribute", ["t0", "x0"])
def test_timeseries_epoch_alias_deleters_match_gwpy(attribute: str) -> None:
    actual = _series(TimeSeries, np.arange(4), t0=1000, dt=0.25)
    expected = _series(GWpyTimeSeries, np.arange(4), t0=1000, dt=0.25)

    actual_error = _exception_class(lambda: delattr(actual, attribute))
    expected_error = _exception_class(lambda: delattr(expected, attribute))

    assert actual_error is expected_error
    if expected_error is None:
        _assert_quantity_equal(actual.t0, expected.t0)
        _assert_quantity_equal(actual.x0, expected.x0)


@pytest.mark.parametrize("attribute", ["t0", "x0"])
def test_timeseries_epoch_aliases_synchronize_private_exact_authority(
    attribute: str,
) -> None:
    epoch_ns = 1_234_567_890_123_456_789
    source = TimeSeries(np.arange(4.0), t0_ns=epoch_ns, dt=0.25)

    setattr(source, attribute, LIGOTimeGPS(1234, 500_000_007))

    assert source.t0_gps_ns == 1_234_500_000_007
    _assert_quantity_equal(source.t0, source.x0)
    delattr(source, attribute)
    assert "_gwex_t0_gps_ns" not in source.__dict__


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        pytest.param((2,), {}, id="defaults"),
        pytest.param(
            (1,),
            {
                "fftlength": 0.5,
                "overlap": 0.25,
                "window": "hann",
                "method": "welch",
                "nproc": 1,
            },
            id="explicit",
        ),
    ],
)
def test_timeseries_spectrogram_regular_routes_match_gwpy(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    actual = _spectral_series(TimeSeries).spectrogram(*args, **kwargs)
    expected = _spectral_series(GWpyTimeSeries).spectrogram(*args, **kwargs)

    _assert_spectrogram_equal(actual, expected)


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        pytest.param((1,), {}, id="defaults"),
        pytest.param(
            (0.5,),
            {"overlap": 0.25, "window": "hann", "scaling": "spectrum"},
            id="explicit",
        ),
    ],
)
def test_timeseries_spectrogram2_regular_routes_match_gwpy(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    actual = _spectral_series(TimeSeries).spectrogram2(*args, **kwargs)
    expected = _spectral_series(GWpyTimeSeries).spectrogram2(*args, **kwargs)

    _assert_spectrogram_equal(actual, expected)


@pytest.mark.parametrize("method", ["spectrogram", "spectrogram2"])
@pytest.mark.parametrize("irregularity", ["true", "nearly"])
def test_timeseries_spectrogram_irregular_outcomes_match_gwpy(
    method: str, irregularity: str
) -> None:
    base = 1000.0 + np.arange(128, dtype=np.float64) / 32
    if irregularity == "true":
        times = base + np.where(np.arange(base.size) >= 50, 0.003, 0)
    else:
        times = base.copy()
        times[50] = np.nextafter(times[50], np.inf)
    args = (2,) if method == "spectrogram" else (1,)
    actual_input = _spectral_series(TimeSeries, times=times)
    expected_input = _spectral_series(GWpyTimeSeries, times=times)

    actual_error = _exception_class(lambda: getattr(actual_input, method)(*args))
    expected_error = _exception_class(lambda: getattr(expected_input, method)(*args))

    assert actual_error is expected_error
    if expected_error is None:
        actual = getattr(actual_input, method)(*args)
        expected = getattr(expected_input, method)(*args)
        _assert_spectrogram_equal(actual, expected)


def _append_dicts(
    series_cls: type[GWpyTimeSeries] | type[TimeSeries],
    dict_cls: type[GWpyTimeSeriesDict] | type[TimeSeriesDict],
    *,
    shared_other_t0: float = 1002,
) -> tuple[
    GWpyTimeSeriesDict | TimeSeriesDict,
    GWpyTimeSeriesDict | TimeSeriesDict,
]:
    left = dict_cls(
        {
            "shared": _series(series_cls, [1, 2], t0=1000, dt=1),
            "untouched": _series(series_cls, [9, 10], t0=1000, dt=1),
        }
    )
    right = dict_cls(
        {
            "shared": _series(series_cls, [3, 4], t0=shared_other_t0, dt=1),
            "new": _series(series_cls, [5, 6], t0=1000, dt=1),
        }
    )
    return left, right


@pytest.mark.parametrize("copy", [True, False], ids=["copy-new", "share-new"])
def test_timeseriesdict_append_mapping_matches_gwpy(copy: bool) -> None:
    actual, actual_other = _append_dicts(TimeSeries, TimeSeriesDict)
    expected, expected_other = _append_dicts(GWpyTimeSeries, GWpyTimeSeriesDict)
    actual_existing_ids = {key: id(value) for key, value in actual.items()}
    expected_existing_ids = {key: id(value) for key, value in expected.items()}

    actual_result = actual.append(actual_other, copy=copy)
    expected_result = expected.append(expected_other, copy=copy)

    assert actual_result is actual
    assert expected_result is expected
    _assert_dict_equal(actual, expected)
    assert (id(actual["shared"]) == actual_existing_ids["shared"]) is (
        id(expected["shared"]) == expected_existing_ids["shared"]
    )
    assert (id(actual["untouched"]) == actual_existing_ids["untouched"]) is (
        id(expected["untouched"]) == expected_existing_ids["untouched"]
    )
    assert (actual["new"] is actual_other["new"]) is (
        expected["new"] is expected_other["new"]
    )
    assert np.shares_memory(actual["new"].value, actual_other["new"].value) is (
        np.shares_memory(expected["new"].value, expected_other["new"].value)
    )


def test_timeseriesdict_append_gap_padding_matches_gwpy() -> None:
    actual, actual_other = _append_dicts(
        TimeSeries, TimeSeriesDict, shared_other_t0=1004
    )
    expected, expected_other = _append_dicts(
        GWpyTimeSeries, GWpyTimeSeriesDict, shared_other_t0=1004
    )

    actual_result = actual.append(actual_other, gap="pad", pad=-1)
    expected_result = expected.append(expected_other, gap="pad", pad=-1)

    assert actual_result is actual
    assert expected_result is expected
    _assert_dict_equal(actual, expected)


def test_timeseriesdict_append_gap_failure_matches_gwpy() -> None:
    actual, actual_other = _append_dicts(
        TimeSeries, TimeSeriesDict, shared_other_t0=1004
    )
    expected, expected_other = _append_dicts(
        GWpyTimeSeries, GWpyTimeSeriesDict, shared_other_t0=1004
    )

    actual_error = _exception_class(lambda: actual.append(actual_other, gap="raise"))
    expected_error = _exception_class(
        lambda: expected.append(expected_other, gap="raise")
    )

    assert actual_error is expected_error
    _assert_dict_equal(actual, expected)


def test_timeseriesdict_append_single_series_is_explicit_extension() -> None:
    actual = TimeSeriesDict(
        {
            "first": _series(TimeSeries, [1, 2], t0=0, dt=1),
            "second": _series(TimeSeries, [3, 4], t0=0, dt=1),
        }
    )
    actual_other = _series(TimeSeries, [5, 6], t0=2, dt=1)
    expected = GWpyTimeSeriesDict(
        {
            "first": _series(GWpyTimeSeries, [1, 2], t0=0, dt=1),
            "second": _series(GWpyTimeSeries, [3, 4], t0=0, dt=1),
        }
    )
    expected_other = _series(GWpyTimeSeries, [5, 6], t0=2, dt=1)

    assert _exception_class(lambda: expected.append(expected_other)) is AttributeError
    result = actual.append(actual_other)

    assert result is actual
    np.testing.assert_array_equal(actual["first"].value, [1, 2, 5, 6])
    np.testing.assert_array_equal(actual["second"].value, [3, 4, 5, 6])


@pytest.mark.parametrize(
    "rate",
    [
        pytest.param(8, id="scalar"),
        pytest.param({"first": 8, "second": 4}, id="per-key"),
    ],
)
def test_timeseriesdict_numeric_resample_matches_gwpy(
    rate: int | Mapping[str, int],
) -> None:
    actual = _series_dict(TimeSeries, TimeSeriesDict)
    expected = _series_dict(GWpyTimeSeries, GWpyTimeSeriesDict)
    actual_ids = {key: id(value) for key, value in actual.items()}
    expected_ids = {key: id(value) for key, value in expected.items()}

    actual_result = actual.resample(rate)
    expected_result = expected.resample(rate)

    assert actual_result is actual
    assert expected_result is expected
    _assert_dict_equal(actual, expected)
    for key in expected:
        assert (id(actual[key]) != actual_ids[key]) is (
            id(expected[key]) != expected_ids[key]
        )


@pytest.mark.parametrize("irregularity", ["true", "nearly"])
def test_timeseriesdict_numeric_resample_irregular_outcome_matches_gwpy(
    irregularity: str,
) -> None:
    times = 1000.0 + np.arange(64, dtype=np.float64) / 16
    if irregularity == "true":
        times = times + np.where(np.arange(times.size) >= 20, 0.003, 0)
    else:
        times[20] = np.nextafter(times[20], np.inf)
    actual = _series_dict(TimeSeries, TimeSeriesDict, times=times)
    expected = _series_dict(GWpyTimeSeries, GWpyTimeSeriesDict, times=times)

    actual_error = _exception_class(lambda: actual.resample(8))
    expected_error = _exception_class(lambda: expected.resample(8))

    assert actual_error is expected_error
    if expected_error is None:
        _assert_dict_equal(actual, expected)


@pytest.mark.parametrize(
    "rate",
    [pytest.param("0.125s", id="string"), pytest.param(0.125 * u.s, id="quantity")],
)
def test_timeseriesdict_time_bin_resample_is_explicit_extension(rate: Any) -> None:
    actual = _series_dict(TimeSeries, TimeSeriesDict)
    expected = _series_dict(GWpyTimeSeries, GWpyTimeSeriesDict)

    assert _exception_class(lambda: expected.resample(rate)) is not None
    result = actual.resample(rate)

    assert result is actual
    assert isinstance(result, TimeSeriesDict)
    assert list(result) == ["first", "second"]
    for series in result.values():
        assert series.dt == 0.125 * u.s
        assert series.t0 == 1000 * u.s


def test_timeseriesdict_numeric_resample_preserves_private_exact_authority() -> None:
    epoch_ns = 1_234_567_890_123_456_789
    source = TimeSeriesDict(
        {
            key: TimeSeries(np.arange(16.0), t0_ns=epoch_ns, sample_rate=8, unit=u.V)
            for key in ("first", "second")
        }
    )

    result = source.resample(4)

    assert result is source
    for series in result.values():
        assert series.t0_gps_ns == epoch_ns
        assert series.__dict__["_gwex_dt_gps_ns"] == 250_000_000
