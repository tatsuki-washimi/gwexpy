"""Differential contracts for the v0.2.3 Phase 3 spectral fixes."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy import units as u
from gwpy.frequencyseries import FrequencySeries as GwpyFrequencySeries
from gwpy.timeseries import TimeSeries as GwpyTimeSeries

from gwexpy.frequencyseries import FrequencySeries
from gwexpy.timeseries import TimeSeries

EXACT_RTOL = 0.0
EXACT_ATOL = 0.0
EXTENSION_RTOL = 1e-12
EXTENSION_ATOL = 0.0


def _assert_numeric_result_equal(actual, expected, *, rtol, atol):
    """Compare values and non-finite masks without hiding NaN/Inf changes."""
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
                mask(actual_component),
                mask(expected_component),
            )

    finite = np.isfinite(expected_values)
    np.testing.assert_array_equal(np.isfinite(actual_values), finite)
    np.testing.assert_allclose(
        actual_values[finite],
        expected_values[finite],
        rtol=rtol,
        atol=atol,
    )


def _assert_axis_equal(actual_axis, expected_axis):
    assert actual_axis.unit == expected_axis.unit
    np.testing.assert_allclose(
        actual_axis.value,
        expected_axis.value,
        rtol=EXACT_RTOL,
        atol=EXACT_ATOL,
    )


def _assert_epoch_equal(actual, expected):
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    np.testing.assert_array_equal(
        (actual.jd1, actual.jd2),
        (expected.jd1, expected.jd2),
    )


def _assert_spectral_result_equal(
    actual, expected, *, rtol=EXACT_RTOL, atol=EXACT_ATOL
):
    _assert_numeric_result_equal(actual, expected, rtol=rtol, atol=atol)
    assert actual.unit == expected.unit
    assert actual.name == expected.name
    assert actual.channel == expected.channel
    _assert_epoch_equal(actual.epoch, expected.epoch)
    _assert_axis_equal(actual.frequencies, expected.frequencies)
    if hasattr(expected, "times"):
        _assert_axis_equal(actual.times, expected.times)


def _csd_pair(cls, other_unit):
    rng = np.random.default_rng(20260902)
    size = 1024
    sample_rate = 128.0
    time = np.arange(size) / sample_rate
    reference = (
        np.sin(2 * np.pi * 8 * time)
        + 0.3 * np.sin(2 * np.pi * 17 * time)
        + 0.05 * rng.standard_normal(size)
    )
    response = (
        0.5 * reference
        + 0.2 * np.sin(2 * np.pi * 23 * time)
        + 0.05 * rng.standard_normal(size)
    )
    return (
        cls(
            reference,
            sample_rate=sample_rate,
            t0=1234567890,
            unit=u.V,
            name="reference",
            channel="H1:REF",
        ),
        cls(
            response,
            sample_rate=sample_rate,
            t0=1234567890,
            unit=other_unit,
            name="response",
            channel="L1:RESP",
        ),
    )


@pytest.mark.parametrize("other_unit", [u.V, u.A], ids=["same-unit", "mixed-unit"])
def test_csd_default_preserves_the_parent_result(other_unit):
    actual_reference, actual_response = _csd_pair(TimeSeries, other_unit)
    expected_reference, expected_response = _csd_pair(GwpyTimeSeries, other_unit)
    kwargs = {"fftlength": 1.0, "overlap": 0.5, "window": "hann"}

    actual = actual_reference.csd(actual_response, **kwargs)
    expected = expected_reference.csd(expected_response, **kwargs)

    _assert_spectral_result_equal(actual, expected)


def _rayleigh_series(cls):
    rng = np.random.default_rng(20260902)
    sample_rate = 32.0
    size = 256
    time = np.arange(size) / sample_rate
    data = rng.standard_normal(size) + 0.25 * np.sin(2 * np.pi * 5 * time)
    return cls(
        data,
        sample_rate=sample_rate,
        t0=1000,
        unit=u.V,
        name="rayleigh-input",
        channel="H1:RAYLEIGH",
    )


@pytest.mark.parametrize(
    ("fftlength", "overlap"),
    [
        pytest.param(0.25, 0.125, id="even-exact-half-overlap"),
        pytest.param(0.25, 0.1875, id="even-three-quarter-overlap"),
        pytest.param(9 / 32, None, id="odd-recommended-overlap"),
    ],
)
def test_rayleigh_spectrogram_default_preserves_the_parent_result(
    fftlength,
    overlap,
):
    actual = _rayleigh_series(TimeSeries).rayleigh_spectrogram(
        2.0,
        fftlength=fftlength,
        overlap=overlap,
        window="hann",
        nproc=1,
    )
    expected = _rayleigh_series(GwpyTimeSeries).rayleigh_spectrogram(
        2.0,
        fftlength=fftlength,
        overlap=overlap,
        window="hann",
        nproc=1,
    )

    _assert_spectral_result_equal(actual, expected)


def test_rayleigh_test_uses_the_private_corrected_segment_route(monkeypatch):
    def reject_public_route(*args, **kwargs):
        raise AssertionError(
            "rayleigh_test must not use the public GWpy-compatible route"
        )

    monkeypatch.setattr(TimeSeries, "rayleigh_spectrogram", reject_public_route)
    series = TimeSeries(
        np.random.default_rng(506).standard_normal(129 * 8),
        sample_rate=129,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = series.rayleigh_test(
            fftlength=1.0,
            stride=2.0,
            overlap=None,
            n_monte_carlo=50,
            seed=1,
        )

    assert result.shape[0] == 4
    assert np.isfinite(result.value[:, 1:-1]).all()


def _transfer_pair(cls, *, response_unit=u.V, different_rates=False, t0=1234567890):
    reference = np.random.default_rng(702).standard_normal(1024)
    if different_rates:
        response = np.random.default_rng(703).standard_normal(512)
        response_rate = 64.0
    else:
        response = 0.5 * reference + np.random.default_rng(703).standard_normal(1024)
        response_rate = 128.0

    return (
        cls(
            reference,
            sample_rate=128.0,
            t0=t0,
            unit=u.V,
            name="reference",
            channel="H1:REF",
        ),
        cls(
            response,
            sample_rate=response_rate,
            t0=t0,
            unit=response_unit,
            name="response",
            channel="L1:RESP",
        ),
    )


@pytest.mark.parametrize("response_unit", [u.V, u.A], ids=["same-unit", "mixed-unit"])
def test_steady_transfer_default_preserves_the_parent_result(response_unit):
    actual_reference, actual_response = _transfer_pair(
        TimeSeries,
        response_unit=response_unit,
    )
    expected_reference, expected_response = _transfer_pair(
        GwpyTimeSeries,
        response_unit=response_unit,
    )
    kwargs = {"fftlength": 1.0, "overlap": 0.5, "window": "hann"}

    actual = actual_reference.transfer_function(actual_response, **kwargs)
    expected = expected_reference.transfer_function(expected_response, **kwargs)

    _assert_spectral_result_equal(actual, expected)


def test_steady_transfer_explicit_none_preserves_parent_different_rate_metadata():
    actual_reference, actual_response = _transfer_pair(
        TimeSeries,
        response_unit=u.A,
        different_rates=True,
        t0=1000,
    )
    expected_reference, expected_response = _transfer_pair(
        GwpyTimeSeries,
        response_unit=u.A,
        different_rates=True,
        t0=1000,
    )
    kwargs = {
        "fftlength": 1.0,
        "overlap": 0.25,
        "window": "hann",
        "average": "mean",
    }

    actual = actual_reference.transfer_function(
        actual_response,
        epsilon=None,
        **kwargs,
    )
    expected = expected_reference.transfer_function(expected_response, **kwargs)

    _assert_spectral_result_equal(actual, expected)


def test_steady_transfer_zero_denominator_preserves_parent_nonfinite_masks():
    actual_reference = TimeSeries(
        np.zeros(1024),
        sample_rate=64,
        t0=1000,
        unit=u.V,
        name="reference",
        channel="H1:REF",
    )
    actual_response = TimeSeries(
        np.zeros(1024),
        sample_rate=64,
        t0=1000,
        unit=u.A,
        name="response",
        channel="L1:RESP",
    )
    expected_reference = GwpyTimeSeries(
        np.zeros(1024),
        sample_rate=64,
        t0=1000,
        unit=u.V,
        name="reference",
        channel="H1:REF",
    )
    expected_response = GwpyTimeSeries(
        np.zeros(1024),
        sample_rate=64,
        t0=1000,
        unit=u.A,
        name="response",
        channel="L1:RESP",
    )
    kwargs = {"fftlength": 1.0, "overlap": 0.0, "window": "hann"}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = actual_reference.transfer_function(actual_response, **kwargs)
        expected = expected_reference.transfer_function(expected_response, **kwargs)

    _assert_spectral_result_equal(actual, expected)


def test_steady_transfer_forwards_window_overlap_and_average_to_parent_semantics():
    actual_reference, actual_response = _transfer_pair(TimeSeries)
    expected_reference, expected_response = _transfer_pair(GwpyTimeSeries)
    kwargs = {
        "fftlength": 0.5,
        "overlap": 0.125,
        "window": "boxcar",
        "average": "median",
    }

    actual = actual_reference.transfer_function(actual_response, **kwargs)
    expected = expected_reference.transfer_function(expected_response, **kwargs)

    _assert_spectral_result_equal(actual, expected)


@pytest.mark.parametrize("method", ["gwpy", "csd_psd", "auto"])
def test_deprecated_steady_transfer_routes_preserve_the_parent_result(method):
    actual_reference, actual_response = _transfer_pair(TimeSeries)
    expected_reference, expected_response = _transfer_pair(GwpyTimeSeries)
    kwargs = {"fftlength": 1.0, "overlap": 0.5, "window": "hann"}

    with pytest.warns(DeprecationWarning):
        actual = actual_reference.transfer_function(
            actual_response,
            method=method,
            **kwargs,
        )
    expected = expected_reference.transfer_function(expected_response, **kwargs)

    _assert_spectral_result_equal(actual, expected)


@pytest.mark.parametrize("epsilon", [0.0, 1e-3], ids=["zero", "positive"])
def test_explicit_steady_epsilon_keeps_the_gwexpy_extension(epsilon):
    reference, response = _transfer_pair(TimeSeries, response_unit=u.A)
    kwargs = {
        "fftlength": 1.0,
        "overlap": 0.5,
        "window": "hann",
        "average": "mean",
    }

    actual = reference.transfer_function(response, epsilon=epsilon, **kwargs)
    csd = reference.csd(response, **kwargs)
    psd = reference.psd(**kwargs)
    size = min(csd.size, psd.size)
    expected_values = csd.value[:size] / (psd.value[:size] + epsilon)

    np.testing.assert_allclose(
        actual.value,
        expected_values,
        rtol=EXTENSION_RTOL,
        atol=EXTENSION_ATOL,
    )
    assert actual.unit == u.A / u.V
    assert actual.name == "response / reference"
    assert actual.channel == reference.channel


SPECTRAL_METHODS = (
    pytest.param("fft", {}, id="fft"),
    pytest.param(
        "psd",
        {"fftlength": 0.25, "overlap": 0.125, "window": "hann"},
        id="psd",
    ),
    pytest.param(
        "asd",
        {"fftlength": 0.25, "overlap": 0.125, "window": "hann"},
        id="asd",
    ),
    pytest.param(
        "csd",
        {"fftlength": 0.25, "overlap": 0.125, "window": "hann"},
        id="csd",
    ),
    pytest.param(
        "coherence",
        {"fftlength": 0.25, "overlap": 0.125, "window": "hann"},
        id="coherence",
    ),
)


def _call_spectral_method(series, method_name, kwargs):
    args = (series,) if method_name in {"csd", "coherence"} else ()
    return getattr(series, method_name)(*args, **kwargs)


def _explicit_time_series(cls, *, perturbation=0.0, axis_unit=u.s):
    values = np.random.default_rng(703).standard_normal(256)
    times_seconds = np.arange(values.size) / 128.0
    axis_values = (times_seconds * u.s).to_value(axis_unit)
    axis_values[128] += (perturbation * u.s).to_value(axis_unit)
    return cls(values, times=axis_values * axis_unit, unit=u.V, name="axis")


@pytest.mark.parametrize(("method_name", "kwargs"), SPECTRAL_METHODS)
def test_explicit_regular_axis_preserves_parent_spectral_result(method_name, kwargs):
    actual_series = _explicit_time_series(TimeSeries)
    expected_series = _explicit_time_series(GwpyTimeSeries)

    actual = _call_spectral_method(actual_series, method_name, kwargs)
    expected = _call_spectral_method(expected_series, method_name, kwargs)

    _assert_spectral_result_equal(actual, expected)


@pytest.mark.parametrize("axis_unit", [u.s, u.ms], ids=["seconds", "milliseconds"])
@pytest.mark.parametrize(("method_name", "kwargs"), SPECTRAL_METHODS)
def test_true_irregular_axis_preserves_parent_failure_class(
    method_name,
    kwargs,
    axis_unit,
):
    perturbation = 1e-3 if axis_unit == u.s else 2e-4
    actual_series = _explicit_time_series(
        TimeSeries,
        perturbation=perturbation,
        axis_unit=axis_unit,
    )
    expected_series = _explicit_time_series(
        GwpyTimeSeries,
        perturbation=perturbation,
        axis_unit=axis_unit,
    )

    with pytest.raises(Exception) as expected_error:
        _call_spectral_method(expected_series, method_name, kwargs)
    with pytest.raises(Exception) as actual_error:
        _call_spectral_method(actual_series, method_name, kwargs)

    assert type(actual_error.value) is type(expected_error.value) is AttributeError


@pytest.mark.parametrize(("method_name", "kwargs"), SPECTRAL_METHODS)
def test_nearly_regular_axis_preserves_parent_failure_class(method_name, kwargs):
    actual_series = _explicit_time_series(TimeSeries, perturbation=1e-6)
    expected_series = _explicit_time_series(GwpyTimeSeries, perturbation=1e-6)

    with pytest.raises(Exception) as expected_error:
        _call_spectral_method(expected_series, method_name, kwargs)
    with pytest.raises(Exception) as actual_error:
        _call_spectral_method(actual_series, method_name, kwargs)

    assert type(actual_error.value) is type(expected_error.value) is AttributeError


def test_transient_fft_retains_the_gwexpy_regularity_guard():
    series = _explicit_time_series(TimeSeries, perturbation=1e-3)

    with pytest.raises(ValueError, match="requires a regular sample rate"):
        series.fft(mode="transient")


def _irregular_frequency_series(cls):
    values = np.fft.rfft(np.random.default_rng(703).standard_normal(256))
    frequencies = np.arange(values.size) * 0.5
    frequencies[64] += 0.2
    return cls(values, frequencies=frequencies * u.Hz, epoch=1000)


@pytest.mark.parametrize("mode", [None, "gwpy"], ids=["default", "explicit-gwpy"])
def test_default_ifft_irregular_axis_preserves_parent_failure_class(mode):
    actual = _irregular_frequency_series(FrequencySeries)
    expected = _irregular_frequency_series(GwpyFrequencySeries)

    with pytest.raises(Exception) as expected_error:
        expected.ifft()
    with pytest.raises(Exception) as actual_error:
        if mode is None:
            actual.ifft()
        else:
            actual.ifft(mode=mode)

    assert type(actual_error.value) is type(expected_error.value) is AttributeError


def test_transient_ifft_retains_the_gwexpy_regularity_guard():
    series = _irregular_frequency_series(FrequencySeries)

    with pytest.raises(ValueError, match="requires a regular frequency grid"):
        series.ifft(mode="transient")


def test_rfft_remains_a_gwexpy_only_surface():
    assert hasattr(TimeSeries, "rfft")
    assert not hasattr(GwpyTimeSeries, "rfft")
