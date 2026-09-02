"""Differential contracts for audited ``TimeSeries`` signal methods.

These tests close the ``#639`` implementation groups for ``heterodyne``,
``demodulate``, ``rms``, and ``resample`` against the GWpy version installed
in each qualification environment.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from astropy import units as u
from gwpy.timeseries import TimeSeries as GWpyTimeSeries

from gwexpy.timeseries import TimeSeries


def _series_pair(
    data: Any,
    *,
    sample_rate: float = 8,
    unit: u.UnitBase = u.V,
) -> tuple[GWpyTimeSeries, TimeSeries]:
    metadata = {
        "sample_rate": sample_rate,
        "t0": 1_234_567_890.125,
        "unit": unit,
        "name": "audit-signal",
        "channel": "H1:AUDIT-SIGNAL",
    }
    return GWpyTimeSeries(data, **metadata), TimeSeries(data, **metadata)


def _assert_numeric_payload_equal(actual: Any, expected: Any) -> None:
    actual_values = np.asarray(actual.value)
    expected_values = np.asarray(expected.value)
    assert actual_values.shape == expected_values.shape
    assert actual_values.dtype == expected_values.dtype
    components = ("real", "imag") if np.iscomplexobj(expected_values) else ("real",)
    for component in components:
        actual_component = getattr(actual_values, component)
        expected_component = getattr(expected_values, component)
        np.testing.assert_array_equal(
            np.isnan(actual_component), np.isnan(expected_component)
        )
        np.testing.assert_array_equal(
            np.isposinf(actual_component), np.isposinf(expected_component)
        )
        np.testing.assert_array_equal(
            np.isneginf(actual_component), np.isneginf(expected_component)
        )
    finite = np.isfinite(expected_values)
    np.testing.assert_allclose(
        actual_values[finite], expected_values[finite], rtol=0.0, atol=0.0
    )


def _assert_series_equal(actual: Any, expected: Any) -> None:
    _assert_numeric_payload_equal(actual, expected)
    assert actual.unit == expected.unit
    assert actual.name == expected.name
    assert str(actual.channel) == str(expected.channel)
    assert actual.t0.to_value(u.s) == expected.t0.to_value(u.s)
    assert actual.dt == expected.dt
    np.testing.assert_allclose(actual.times.value, expected.times.value, rtol=0, atol=0)
    assert actual.times.unit == expected.times.unit


def _exception_class(call: Callable[[], Any]) -> type[BaseException] | None:
    try:
        call()
    except BaseException as exc:  # noqa: BLE001 - the exception type is the oracle
        return type(exc)
    return None


def _assert_exact_time_authority(
    series: TimeSeries,
    *,
    epoch_ns: int,
    dt_ns: int,
) -> None:
    assert series.t0_gps_ns == epoch_ns
    assert series.__dict__["_gwex_dt_gps_ns"] == dt_ns
    if len(series) > 1:
        assert series[1:].t0_gps_ns == epoch_ns + dt_ns


def _assert_parameter_layout(
    ours: Callable[..., Any],
    upstream: Callable[..., Any],
    expected_names: list[str],
) -> None:
    ours_signature = inspect.signature(ours)
    upstream_signature = inspect.signature(upstream)
    assert list(ours_signature.parameters)[: len(expected_names)] == expected_names
    for name in expected_names:
        ours_parameter = ours_signature.parameters[name]
        upstream_parameter = upstream_signature.parameters[name]
        assert ours_parameter.kind is upstream_parameter.kind
        assert ours_parameter.default == upstream_parameter.default


def test_heterodyne_signature_matches_gwpy() -> None:
    _assert_parameter_layout(
        TimeSeries.heterodyne,
        GWpyTimeSeries.heterodyne,
        ["self", "phase", "stride", "singlesided"],
    )
    assert list(inspect.signature(TimeSeries.heterodyne).parameters) == [
        "self",
        "phase",
        "stride",
        "singlesided",
    ]
    assert inspect.signature(TimeSeries.heterodyne) == inspect.signature(
        GWpyTimeSeries.heterodyne
    )


@pytest.mark.parametrize("singlesided", [False, True])
def test_heterodyne_matches_gwpy_values_and_metadata(singlesided: bool) -> None:
    samples = np.arange(40, dtype=float)
    phase = np.linspace(-0.3, 1.7, samples.size)
    expected_input, actual_input = _series_pair(samples)

    expected = expected_input.heterodyne(phase, 2, singlesided=singlesided)
    actual = actual_input.heterodyne(phase, 2, singlesided=singlesided)

    _assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "phase",
    [pytest.param(0.25, id="scalar"), pytest.param(np.zeros((2, 20)), id="2d")],
)
def test_heterodyne_phase_failure_class_matches_gwpy(phase: Any) -> None:
    expected_input, actual_input = _series_pair(np.arange(40, dtype=float))

    assert _exception_class(lambda: actual_input.heterodyne(phase)) is _exception_class(
        lambda: expected_input.heterodyne(phase)
    )


def test_heterodyne_wrong_length_failure_class_matches_gwpy() -> None:
    expected_input, actual_input = _series_pair(np.arange(40, dtype=float))
    phase = np.zeros(5)

    assert _exception_class(lambda: actual_input.heterodyne(phase)) is _exception_class(
        lambda: expected_input.heterodyne(phase)
    )


def test_heterodyne_quantity_stride_is_explicit_extension() -> None:
    expected_input, actual_input = _series_pair(np.arange(40, dtype=float))
    phase = np.linspace(0, 1, 40)

    assert (
        _exception_class(lambda: expected_input.heterodyne(phase, 2 * u.s)) is TypeError
    )
    actual = actual_input.heterodyne(phase, 2 * u.s)
    numeric = actual_input.heterodyne(phase, 2)

    _assert_series_equal(actual, numeric)


def test_heterodyne_dimensionless_quantity_stride_matches_gwpy() -> None:
    phase = np.linspace(0, 1, 40)
    expected_input, actual_input = _series_pair(np.arange(40, dtype=float))

    expected = expected_input.heterodyne(phase, 2 * u.one)
    actual = actual_input.heterodyne(phase, 2 * u.one)

    _assert_series_equal(actual, expected)


def test_heterodyne_non_time_stride_failure_class_matches_gwpy() -> None:
    phase = np.linspace(0, 1, 40)
    expected_input, actual_input = _series_pair(np.arange(40, dtype=float))

    assert _exception_class(
        lambda: actual_input.heterodyne(phase, 2 * u.m)
    ) is _exception_class(lambda: expected_input.heterodyne(phase, 2 * u.m))


def test_heterodyne_preserves_exact_time_authority() -> None:
    epoch_ns = 1_234_567_890_123_456_789
    samples = np.arange(64, dtype=float)
    series = TimeSeries(samples, t0_ns=epoch_ns, dt=0.125)

    result = series.heterodyne(np.linspace(0, 2, samples.size), stride=2)

    _assert_exact_time_authority(result, epoch_ns=epoch_ns, dt_ns=2_000_000_000)


def test_demodulate_signature_matches_gwpy() -> None:
    _assert_parameter_layout(
        TimeSeries.demodulate,
        GWpyTimeSeries.demodulate,
        ["self", "f", "stride", "exp", "deg"],
    )
    assert list(inspect.signature(TimeSeries.demodulate).parameters) == [
        "self",
        "f",
        "stride",
        "exp",
        "deg",
    ]
    assert inspect.signature(TimeSeries.demodulate) == inspect.signature(
        GWpyTimeSeries.demodulate
    )


@pytest.mark.parametrize(
    ("exp", "deg"),
    [
        pytest.param(True, True, id="complex"),
        pytest.param(False, True, id="amplitude-phase-degrees"),
        pytest.param(False, False, id="amplitude-phase-radians"),
    ],
)
def test_demodulate_matches_gwpy_values_and_metadata(exp: bool, deg: bool) -> None:
    time = np.arange(80) / 8
    samples = 2.5 * np.cos(2 * np.pi * 1.25 * time + 0.4)
    expected_input, actual_input = _series_pair(samples)

    expected = expected_input.demodulate(1.25, 2, exp=exp, deg=deg)
    actual = actual_input.demodulate(1.25, 2, exp=exp, deg=deg)

    if exp:
        _assert_series_equal(actual, expected)
    else:
        assert isinstance(actual, tuple)
        assert isinstance(expected, tuple)
        for actual_series, expected_series in zip(actual, expected, strict=True):
            _assert_series_equal(actual_series, expected_series)


def test_demodulate_parent_frequency_quantity_and_time_stride_extension() -> None:
    time = np.arange(80) / 8
    expected_input, actual_input = _series_pair(np.cos(2 * np.pi * 1.25 * time))

    assert (
        _exception_class(
            lambda: expected_input.demodulate(1.25 * u.Hz, 2 * u.s, exp=True)
        )
        is TypeError
    )
    actual = actual_input.demodulate(1.25 * u.Hz, 2 * u.s, exp=True)
    numeric = actual_input.demodulate(1.25, 2, exp=True)

    _assert_series_equal(actual, numeric)


@pytest.mark.parametrize("frequency", [1000 * u.mHz, 1 * u.one, 1 * u.m])
def test_demodulate_quantity_frequency_preserves_gwpy_raw_magnitude(
    frequency: u.Quantity,
) -> None:
    samples = np.sin(np.arange(80) * 0.21) + np.arange(80) * 0.013
    expected_input, actual_input = _series_pair(samples)

    expected = expected_input.demodulate(frequency, 2, exp=True)
    actual = actual_input.demodulate(frequency, 2, exp=True)

    _assert_series_equal(actual, expected)


def test_demodulate_dimensionless_quantity_stride_matches_gwpy() -> None:
    samples = np.sin(np.arange(80) * 0.21) + np.arange(80) * 0.013
    expected_input, actual_input = _series_pair(samples)

    expected = expected_input.demodulate(1.25, 2 * u.one, exp=True)
    actual = actual_input.demodulate(1.25, 2 * u.one, exp=True)

    _assert_series_equal(actual, expected)


def test_demodulate_non_time_stride_failure_class_matches_gwpy() -> None:
    expected_input, actual_input = _series_pair(np.arange(40, dtype=float))

    assert _exception_class(
        lambda: actual_input.demodulate(1.25, 2 * u.m, exp=True)
    ) is _exception_class(lambda: expected_input.demodulate(1.25, 2 * u.m, exp=True))


@pytest.mark.parametrize(
    ("exp", "deg"),
    [
        pytest.param(True, True, id="complex"),
        pytest.param(False, True, id="pair-degrees"),
        pytest.param(False, False, id="pair-radians"),
    ],
)
def test_demodulate_preserves_exact_time_authority(exp: bool, deg: bool) -> None:
    epoch_ns = 1_234_567_890_123_456_789
    samples = np.sin(np.arange(64) * 0.2)
    series = TimeSeries(samples, t0_ns=epoch_ns, dt=0.125)

    result = series.demodulate(1.25, stride=2, exp=exp, deg=deg)
    outputs = (result,) if exp else result

    for output in outputs:
        _assert_exact_time_authority(
            output,
            epoch_ns=epoch_ns,
            dt_ns=2_000_000_000,
        )


def test_rms_signature_keeps_only_keyword_nan_extension() -> None:
    _assert_parameter_layout(
        TimeSeries.rms,
        GWpyTimeSeries.rms,
        ["self", "stride"],
    )
    signature = inspect.signature(TimeSeries.rms)
    assert list(signature.parameters) == ["self", "stride", "ignore_nan"]
    assert signature.parameters["ignore_nan"].kind is inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["ignore_nan"].default is None


@pytest.mark.parametrize("stride", [1, 1.0, 2])
def test_rms_default_matches_gwpy_values_and_metadata(stride: float) -> None:
    samples = np.arange(40, dtype=float)
    samples[3] = np.nan
    expected_input, actual_input = _series_pair(samples)

    expected = expected_input.rms(stride)
    actual = actual_input.rms(stride)

    _assert_series_equal(actual, expected)


@pytest.mark.parametrize("stride", [0, -1, 0.01, 1 * u.s])
def test_rms_failure_class_matches_gwpy(stride: Any) -> None:
    expected_input, actual_input = _series_pair(np.arange(40, dtype=float))

    assert _exception_class(lambda: actual_input.rms(stride)) is _exception_class(
        lambda: expected_input.rms(stride)
    )


def test_rms_default_float32_numerics_match_gwpy() -> None:
    samples = np.full(16, 1e-23, dtype=np.float32)
    expected_input, actual_input = _series_pair(samples, sample_rate=4)

    _assert_series_equal(actual_input.rms(), expected_input.rms())


def test_rms_ignore_nan_true_remains_explicit_extension() -> None:
    samples = np.arange(16, dtype=float)
    samples[2] = np.nan
    _, actual_input = _series_pair(samples, sample_rate=4)

    actual = actual_input.rms(1, ignore_nan=True)

    assert np.isfinite(actual.value[0])
    assert actual.name == "audit-signal 1-second RMS"
    assert actual.unit == u.dimensionless_unscaled
    assert actual.dt == 1 * u.s


@pytest.mark.parametrize("ignore_nan", [None, True])
def test_rms_preserves_exact_time_authority(ignore_nan: bool | None) -> None:
    epoch_ns = 1_234_567_890_123_456_789
    samples = np.arange(64, dtype=float)
    samples[2] = np.nan
    series = TimeSeries(samples, t0_ns=epoch_ns, dt=0.125)
    kwargs = {} if ignore_nan is None else {"ignore_nan": ignore_nan}

    result = series.rms(2, **kwargs)

    _assert_exact_time_authority(result, epoch_ns=epoch_ns, dt_ns=2_000_000_000)


def test_resample_signature_keeps_gwpy_positional_layout() -> None:
    _assert_parameter_layout(
        TimeSeries.resample,
        GWpyTimeSeries.resample,
        ["self", "rate", "window", "ftype", "n"],
    )


@pytest.mark.parametrize("rate", [4, 3, 4 * u.Hz])
def test_resample_numeric_route_matches_gwpy(rate: Any) -> None:
    samples = np.sin(np.arange(80) * 0.2) + np.arange(80) * 0.01
    expected_input, actual_input = _series_pair(samples)

    expected = expected_input.resample(rate)
    actual = actual_input.resample(rate)

    _assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param((3, "hann", "fir", None), id="noninteger-window"),
        pytest.param((4, "hamming", "iir", 2), id="integer-iir-order"),
    ],
)
def test_resample_gwpy_positional_options_match(arguments: tuple[Any, ...]) -> None:
    samples = np.sin(np.arange(80) * 0.2) + np.arange(80) * 0.01
    expected_input, actual_input = _series_pair(samples)

    expected = expected_input.resample(*arguments)
    actual = actual_input.resample(*arguments)

    _assert_series_equal(actual, expected)


def test_resample_numeric_route_does_not_consume_time_bin_keywords() -> None:
    expected_input, actual_input = _series_pair(np.arange(40, dtype=float))

    assert _exception_class(
        lambda: actual_input.resample(4, ignore_nan=True)
    ) is _exception_class(lambda: expected_input.resample(4, ignore_nan=True))


def test_resample_same_rate_warning_and_identity_match_gwpy() -> None:
    expected_input, actual_input = _series_pair(np.arange(40, dtype=float))

    with pytest.warns(UserWarning, match="matches current sample_rate"):
        expected = expected_input.resample(8)
    with pytest.warns(UserWarning, match="matches current sample_rate"):
        actual = actual_input.resample(8)

    assert expected is expected_input
    assert actual is actual_input


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda series: series.heterodyne(
                np.zeros(40),
                np.zeros(40),
                False,
            ),
            id="heterodyne-excess-positional",
        ),
        pytest.param(
            lambda series: series.heterodyne(
                np.zeros(40),
                phase=np.zeros(40),
            ),
            id="heterodyne-duplicate-phase",
        ),
        pytest.param(
            lambda series: series.demodulate(1, 2, False),
            id="demodulate-excess-positional",
        ),
        pytest.param(
            lambda series: series.demodulate(1, f=1),
            id="demodulate-duplicate-frequency",
        ),
        pytest.param(
            lambda series: series.rms(1, False),
            id="rms-excess-positional",
        ),
        pytest.param(
            lambda series: series.rms(1, stride=1),
            id="rms-duplicate-stride",
        ),
        pytest.param(
            lambda series: series.resample(4, "hann", "fir", None, "extra"),
            id="resample-excess-positional",
        ),
        pytest.param(
            lambda series: series.resample(4, "hann", window="hann"),
            id="resample-duplicate-window",
        ),
    ],
)
def test_binding_failure_class_matches_gwpy(call: Callable[[Any], Any]) -> None:
    expected_input, actual_input = _series_pair(np.arange(40, dtype=float))

    assert _exception_class(lambda: call(actual_input)) is _exception_class(
        lambda: call(expected_input)
    )


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda series: series.heterodyne(np.zeros(40), 1),
            id="heterodyne",
        ),
        pytest.param(
            lambda series: series.demodulate(1.25, 1, exp=True),
            id="demodulate",
        ),
        pytest.param(lambda series: series.rms(1), id="rms"),
        pytest.param(lambda series: series.resample(4), id="resample"),
    ],
)
def test_nonfinite_masks_match_gwpy(operation: Callable[[Any], Any]) -> None:
    samples = np.arange(40, dtype=float)
    samples[2] = np.inf
    samples[18] = -np.inf
    expected_input, actual_input = _series_pair(samples)

    with np.errstate(all="ignore"):
        expected = operation(expected_input)
        actual = operation(actual_input)

    _assert_series_equal(actual, expected)


@pytest.mark.parametrize(
    "times",
    [
        pytest.param(
            np.array([0.0, 0.1, 0.23, 0.3, 0.4, 0.5, 0.6, 0.7]),
            id="true-irregular",
        ),
        pytest.param(
            np.array([0.0, 0.1, 0.2, 0.30005, 0.4, 0.5, 0.6, 0.7]),
            id="nearly-irregular",
        ),
    ],
)
def test_resample_irregular_failure_class_matches_gwpy(times: np.ndarray) -> None:
    samples = np.arange(times.size, dtype=float)
    expected_input = GWpyTimeSeries(samples, times=times)
    actual_input = TimeSeries(samples, times=times)

    assert _exception_class(lambda: actual_input.resample(5)) is _exception_class(
        lambda: expected_input.resample(5)
    )


@pytest.mark.parametrize("rule", ["2s", 2 * u.s])
def test_resample_time_bin_extension_remains_explicit(rule: Any) -> None:
    series = TimeSeries(
        np.arange(8, dtype=float),
        sample_rate=1,
        t0=1_234_567_890,
        unit=u.V,
        name="time-bin",
        channel="H1:TIME-BIN",
    )

    result = series.resample(rule)

    np.testing.assert_allclose(result.value, [0.5, 2.5, 4.5, 6.5], rtol=0, atol=0)
    assert result.unit == u.V
    assert result.name == "time-bin"
    assert str(result.channel) == "H1:TIME-BIN"
    assert result.t0.to_value(u.s) == series.t0.to_value(u.s)
    assert result.dt == 2 * u.s
