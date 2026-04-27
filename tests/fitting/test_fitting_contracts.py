from __future__ import annotations

import numpy as np
import pytest
from gwpy.frequencyseries import FrequencySeries

pytest.importorskip("iminuit")

try:
    from gwexpy.fitting.core import fit_series
except ImportError as exc:  # pragma: no cover - optional dependency gate
    pytest.skip(
        f"gwexpy.fitting optional dependencies unavailable: {exc}",
        allow_module_level=True,
    )


def _linear(x, a, b):
    return a * x + b


def _linear_frequency_series() -> FrequencySeries:
    frequencies = np.linspace(0.0, 10.0, 11)
    values = _linear(frequencies, 2.0, 1.0)
    return FrequencySeries(values, frequencies=frequencies, unit="m", name="linear")


def test_x_range_excludes_upper_boundary_and_preserves_full_plot_data():
    series = _linear_frequency_series()

    result = fit_series(series, _linear, p0={"a": 2.0, "b": 1.0}, x_range=(2.0, 8.0))

    np.testing.assert_array_equal(result.x, np.arange(2.0, 8.0))
    np.testing.assert_allclose(
        result.y,
        _linear(np.arange(2.0, 8.0), 2.0, 1.0),
        rtol=1e-12,
    )
    np.testing.assert_array_equal(result.x_data, series.frequencies.value)
    np.testing.assert_array_equal(result.y_data, series.value)
    assert result.x_fit_range == (2.0, 8.0)
    assert result.x_kind == "frequency"
    assert result.unit == series.unit
    assert result.x_unit == series.xunit


def test_full_length_sigma_with_exact_x_range_boundary_currently_mismatches_crop():
    series = _linear_frequency_series()
    sigma = np.full(len(series), 0.1)

    with pytest.raises(ValueError, match=r"Sigma length mismatch: got 7, expected 6"):
        fit_series(
            series,
            _linear,
            p0={"a": 2.0, "b": 1.0},
            sigma=sigma,
            x_range=(2.0, 8.0),
        )


def test_sigma_zero_nan_and_inf_values_are_preserved_without_validation():
    series = _linear_frequency_series()
    p0 = {"a": 2.0, "b": 1.0}

    zero = np.zeros(len(series))
    result_zero = fit_series(series, _linear, p0=p0, sigma=zero)
    np.testing.assert_array_equal(result_zero.dy, zero)
    assert result_zero.has_dy is True
    assert not result_zero.minuit.valid
    assert np.isnan(result_zero.chi2)

    with_nan = np.full(len(series), 0.1)
    with_nan[3] = np.nan
    result_nan = fit_series(series, _linear, p0=p0, sigma=with_nan)
    np.testing.assert_array_equal(result_nan.dy, with_nan)
    assert np.isnan(result_nan.dy[3])
    assert not result_nan.minuit.valid

    with_inf = np.full(len(series), 0.1)
    with_inf[3] = np.inf
    result_inf = fit_series(series, _linear, p0=p0, sigma=with_inf)
    np.testing.assert_array_equal(result_inf.dy, with_inf)
    assert np.isinf(result_inf.dy[3])
    # Optimizer validity is intentionally unspecified until the sigma policy is resolved.


def test_ndarray_covariance_with_x_range_currently_requires_cropped_shape():
    series = _linear_frequency_series()
    full_cov = np.eye(len(series)) * 0.01

    with pytest.raises(
        ValueError,
        match=r"Covariance matrix shape \(11, 11\) does not match data length 6",
    ):
        fit_series(
            series,
            _linear,
            p0={"a": 2.0, "b": 1.0},
            cov=full_cov,
            x_range=(2.0, 8.0),
        )

    cropped_cov = np.eye(6) * 0.01
    result = fit_series(
        series,
        _linear,
        p0={"a": 2.0, "b": 1.0},
        cov=cropped_cov,
        x_range=(2.0, 8.0),
    )

    np.testing.assert_array_equal(result.x, np.arange(2.0, 8.0))
    assert result.cov.shape == (6, 6)
    assert result.cov_inv.shape == (6, 6)
    assert result.dy.shape == (6,)
    np.testing.assert_allclose(result.dy, 0.1)


def test_singular_covariance_zero_diagonal_uses_pinv_and_keeps_zero_dy():
    series = _linear_frequency_series()
    cov = np.eye(len(series)) * 0.01
    cov[0, 0] = 0.0

    result = fit_series(series, _linear, p0={"a": 2.0, "b": 1.0}, cov=cov)

    assert result.minuit.valid
    assert result.cov.shape == (len(series), len(series))
    assert np.isfinite(result.cov_inv).all()
    assert result.dy[0] == 0.0
    assert result.cost_func.cov_cho is None
