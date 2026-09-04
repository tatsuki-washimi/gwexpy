"""Regression contracts for internal reconstruction on non-second axes."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
from gwexpy.timeseries.decomposition import _apply_scaler
from gwexpy.timeseries.pipeline import WhitenTransform
from gwexpy.timeseries.preprocess import standardize_matrix, standardize_timeseries
from gwexpy.timeseries.rolling import rolling_mean


def _series() -> TimeSeries:
    return TimeSeries(
        np.linspace(1.0, 2.0, 32),
        x0=1 * u.min,
        dt=1 * u.min,
        unit=u.V,
        name="nonsecond-axis",
    )


def _assert_axis(result, source, *, expected_dt=None, expected_unit=None) -> None:
    assert result.x0 == source.x0
    assert result.xunit == source.xunit
    assert result.dt == (source.dt if expected_dt is None else expected_dt)
    assert result.times.unit == source.times.unit
    np.testing.assert_allclose(
        result.times.to_value(source.times.unit),
        source.times.to_value(source.times.unit)[: len(result)],
        rtol=0,
        atol=0,
    )
    if expected_unit is not None:
        assert result.unit == expected_unit


@pytest.mark.parametrize(
    "method",
    [
        "hilbert",
        "instantaneous_phase",
        "radian",
        "degree",
        "instantaneous_frequency",
    ],
)
def test_signal_reconstruction_preserves_nonsecond_axis(method: str) -> None:
    source = _series()

    result = getattr(source, method)()

    _assert_axis(result, source)
    assert np.asarray(result.value).shape == source.value.shape
    assert np.array_equal(np.isnan(result.value), np.isnan(source.value))


def test_rms_extension_preserves_nonsecond_axis_authority() -> None:
    source = _series()

    result = source.rms(120, ignore_nan=True)

    assert result.x0 == 1 * u.min
    assert result.xunit == u.min
    assert result.dt == 2 * u.min
    assert result.times.unit == u.min
    np.testing.assert_allclose(result.times.to_value(u.min), np.arange(1.0, 33.0, 2.0))
    assert result.value.shape == (16,)


def test_rolling_reconstruction_preserves_nonsecond_axis() -> None:
    source = _series()

    result = rolling_mean(source, 3)

    _assert_axis(result, source)
    assert result.value.shape == source.value.shape


def test_standardize_timeseries_integer_reconstruction_preserves_axis() -> None:
    source = TimeSeries(
        np.arange(8, dtype=np.int16),
        x0=1 * u.min,
        dt=1 * u.min,
        unit=u.V,
    )

    result, _ = standardize_timeseries(source)

    _assert_axis(result, source, expected_unit=u.dimensionless_unscaled)
    assert result.value.shape == source.value.shape
    assert np.array_equal(np.isnan(result.value), np.isnan(source.value))


def _matrix(*, dtype=float) -> TimeSeriesMatrix:
    return TimeSeriesMatrix(
        np.arange(8, dtype=dtype).reshape(1, 1, 8),
        x0=1 * u.min,
        dt=1 * u.min,
        xunit=u.min,
    )


def test_timeseriesmatrix_copy_preserves_nonsecond_axis() -> None:
    source = _matrix()

    result = source.copy()

    assert result.x0 == source.x0
    assert result.dt == source.dt
    assert result.xunit == source.xunit
    np.testing.assert_array_equal(result.value, source.value)
    np.testing.assert_allclose(
        result.times.to_value(u.min), source.times.to_value(u.min)
    )


def test_standardize_matrix_integer_reconstruction_preserves_axis() -> None:
    source = _matrix(dtype=np.int16)

    result = standardize_matrix(source)

    assert result.x0 == source.x0
    assert result.dt == source.dt
    assert result.xunit == source.xunit
    np.testing.assert_array_equal(np.isnan(result.value), np.isnan(source.value))


def test_decomposition_scaler_reconstruction_preserves_axis() -> None:
    source = _matrix()

    result = _apply_scaler(
        source,
        {"scaler_stats": {"mean": 0.0, "scale": 1.0}},
    )

    assert result.x0 == source.x0
    assert result.dt == source.dt
    assert result.xunit == source.xunit
    np.testing.assert_array_equal(result.value, source.value)


def test_whiten_transform_matrix_conversion_preserves_axis() -> None:
    source = _series()
    transform = WhitenTransform(multivariate=False)

    result, original = transform._to_matrix(source)

    assert original is None
    assert result.x0 == source.x0
    assert result.dt == source.dt
    assert result.xunit == source.xunit
    np.testing.assert_array_equal(result.value[0, 0], source.value)
