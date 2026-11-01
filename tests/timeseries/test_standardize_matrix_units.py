"""Unit and metadata contracts for matrix standardization (issue #768)."""

import numpy as np
import pytest
from astropy import units as u

from gwexpy.timeseries import TimeSeriesMatrix
from gwexpy.timeseries.preprocess import standardize_matrix


def _matrix(*, dtype=float, units=None):
    data = np.array([[[1, 4, 7]], [[200, 300, 500]]], dtype=dtype)
    matrix = TimeSeriesMatrix(
        data,
        units=units if units is not None else [[u.m], [u.cm]],
        names=[["first"], ["second"]],
        channels=[["A"], ["B"]],
        x0=2 * u.min,
        dt=1 * u.min,
        xunit=u.min,
        name="source",
        attrs={"experiment": ["original"]},
    )
    matrix.meta[0, 0]["custom"] = {"tag": "keep"}
    return matrix


@pytest.mark.parametrize("axis", ["time", "channel"])
@pytest.mark.parametrize("dtype", [float, np.int16])
def test_output_units_and_metadata_are_independent(axis, dtype):
    source = _matrix(dtype=dtype)
    result = standardize_matrix(source, axis=axis)

    assert result.value.dtype.kind == "f"
    assert result.units.shape == source.units.shape
    assert all(unit == u.dimensionless_unscaled for unit in result.units.flat)
    assert list(source.units.flat) == [u.m, u.cm]
    np.testing.assert_array_equal(result.names, source.names)
    np.testing.assert_array_equal(result.channels, source.channels)
    assert result.meta[0, 0]["custom"] == {"tag": "keep"}
    assert result.meta[0, 0] is not source.meta[0, 0]
    assert result.name == source.name
    assert result.attrs == source.attrs
    assert result.attrs is not source.attrs
    assert result.x0 == source.x0
    assert result.dt == source.dt
    assert result.xunit == source.xunit


@pytest.mark.parametrize("method", ["zscore", "robust"])
def test_channel_axis_converts_compatible_units_before_pooling(method):
    source = _matrix()
    converted = source.value.copy()
    converted[1, 0] *= u.cm.to(u.m)
    expected_location = np.mean(converted, axis=(0, 1), keepdims=True)
    if method == "robust":
        expected_location = np.median(converted, axis=(0, 1), keepdims=True)
        scale = 1.4826 * np.median(
            np.abs(converted - expected_location), axis=(0, 1), keepdims=True
        )
    else:
        scale = np.std(converted, axis=(0, 1), keepdims=True)

    result = standardize_matrix(source, axis="channel", method=method)

    np.testing.assert_allclose(result.value, (converted - expected_location) / scale)
    np.testing.assert_array_equal(source.value, _matrix().value)


def test_time_axis_allows_incompatible_element_units():
    source = _matrix(units=[[u.m], [u.s]])

    result = standardize_matrix(source, axis="time")

    np.testing.assert_allclose(result.value.mean(axis=-1), 0, atol=1e-14)
    np.testing.assert_allclose(result.value.std(axis=-1), 1)
    assert all(unit == u.dimensionless_unscaled for unit in result.units.flat)


def test_channel_axis_rejects_incompatible_element_units():
    source = _matrix(units=[[u.m], [u.s]])

    with pytest.raises(ValueError, match="incompatible|equivalent|convert"):
        standardize_matrix(source, axis="channel")

    np.testing.assert_array_equal(source.value, _matrix(units=[[u.m], [u.s]]).value)
    assert list(source.units.flat) == [u.m, u.s]


def test_channel_axis_ac016_fixed_vector():
    """One metre and 200 centimetres become z-scores -1 and +1."""
    source = TimeSeriesMatrix(
        np.array([[[1.0]], [[200.0]]]),
        units=[[u.m], [u.cm]],
        dt=1 * u.s,
    )

    result = standardize_matrix(source, axis="channel", ddof=0)

    np.testing.assert_allclose(result.value[:, 0, 0], [-1.0, 1.0])
    assert all(unit == u.dimensionless_unscaled for unit in result.units.flat)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_channel_conversion_avoids_low_precision_overflow(dtype):
    source = TimeSeriesMatrix(
        np.array([[[0, 1, 2]], [[1000, 2000, 3000]]], dtype=dtype),
        units=[[u.cm], [u.m]],
        dt=1 * u.s,
    )

    result = standardize_matrix(source, axis="channel")

    assert np.isfinite(result.value).all()
    np.testing.assert_allclose(result.value[:, 0, :], [[-1, -1, -1], [1, 1, 1]])


def test_channel_conversion_preserves_complex_values():
    source = TimeSeriesMatrix(
        np.array([[[1 + 1j, 2 + 2j]], [[200 + 300j, 300 + 400j]]], dtype=np.complex64),
        units=[[u.m], [u.cm]],
        dt=1 * u.s,
    )

    result = standardize_matrix(source, axis="channel")

    assert result.value.dtype.kind == "c"
    assert np.isfinite(result.value).all()
    assert np.any(result.value.imag != 0)


def test_channel_axis_empty_matrix_preserves_empty_result():
    source = TimeSeriesMatrix(np.empty((0, 1, 3)), dt=1 * u.s)

    with np.errstate(invalid="ignore"):
        result = standardize_matrix(source, axis="channel")

    assert result.shape == source.shape
    assert result.units.shape == source.units.shape
