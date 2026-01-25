"""
Tests for Generalized Least Squares (GLS) fitting functionality.
"""

import numpy as np
import pytest

try:
    from gwexpy.fitting import GeneralizedLeastSquares, fit_series
    from gwexpy.fitting.models import power_law
except ImportError as exc:
    pytest.skip(
        f"gwexpy.fitting optional dependencies unavailable: {exc}",
        allow_module_level=True,
    )

from gwexpy.frequencyseries import FrequencySeries


def test_gls_class_basic():
    """Test GeneralizedLeastSquares cost function directly."""

    # Simple linear model
    def linear(x, a, b):
        return a * x + b

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.1, 3.9, 6.1])  # Approx y = 2x + 0

    # Diagonal covariance (uncorrelated errors)
    cov_inv = np.diag([1.0, 1.0, 1.0])

    gls = GeneralizedLeastSquares(x, y, cov_inv, linear)

    # Test ndata
    assert gls.ndata == 3

    # Test chi2 calculation at correct parameters
    chi2 = gls(2.0, 0.0)  # a=2, b=0
    assert chi2 < 0.1  # Should be small for good fit

    # Test chi2 at wrong parameters
    chi2_bad = gls(1.0, 0.0)  # a=1, b=0 is wrong
    assert chi2_bad > chi2


def test_gls_with_diagonal_covariance():
    """GLS with diagonal covariance should behave like weighted least squares."""
    np.random.seed(42)

    frequencies = np.logspace(0, 2, 20)
    true_A, true_alpha = 10.0, -1.5
    y_true = power_law(frequencies, A=true_A, alpha=true_alpha)

    # Add heteroscedastic noise
    errors = 0.1 * y_true
    y = y_true + errors * np.random.normal(size=len(frequencies))

    # Create diagonal covariance
    cov = np.diag(errors**2)

    fs = FrequencySeries(y, frequencies=frequencies)

    result = fit_series(fs, "power_law", cov=cov, p0={"A": 5, "alpha": -1})

    assert result.minuit.valid
    assert np.isclose(result.params["A"], true_A, rtol=0.3)
    assert np.isclose(result.params["alpha"], true_alpha, rtol=0.3)

    # dy should be set from sqrt(diag(cov))
    assert result.dy is not None
    np.testing.assert_allclose(result.dy, errors, rtol=1e-10)


def test_gls_with_correlated_covariance():
    """GLS with off-diagonal covariance elements (correlated errors)."""
    np.random.seed(123)

    # Simple 5-point dataset
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # True model: y = 3*x + 1
    def linear(x, a, b):
        return a * x + b

    y_true = linear(x, a=3.0, b=1.0)

    # Create correlated noise: points closer together are more correlated
    n = len(x)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Correlation decays with distance
            cov[i, j] = 0.1 * np.exp(-0.5 * abs(i - j))

    # Generate correlated noise
    L = np.linalg.cholesky(cov)
    noise = L @ np.random.normal(size=n)
    y = y_true + noise

    # Fit with GLS
    from gwexpy.timeseries import TimeSeries

    ts = TimeSeries(y, times=x)

    result = fit_series(ts, linear, cov=cov, p0={"a": 1, "b": 0})

    assert result.minuit.valid
    # GLS should recover parameters reasonably well
    assert np.isclose(result.params["a"], 3.0, atol=0.5)
    assert np.isclose(result.params["b"], 1.0, atol=1.0)


def test_gls_with_ndarray_covariance():
    """Test GLS with 2D ndarray covariance matrix."""
    np.random.seed(456)

    frequencies = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    y_true = power_law(frequencies, A=5.0, alpha=-2.0)

    # Simple diagonal covariance as ndarray
    errors = np.array([0.1, 0.1, 0.2, 0.3, 0.5])
    cov = np.diag(errors**2)

    y = y_true + errors * np.random.normal(size=len(frequencies))

    fs = FrequencySeries(y, frequencies=frequencies)

    result = fit_series(fs, "power_law", cov=cov, p0={"A": 3, "alpha": -1.5})

    assert result.minuit.valid
    # Should be within reasonable range
    assert 1 < result.params["A"] < 20
    assert -3 < result.params["alpha"] < -1


def test_gls_dimension_mismatch_error():
    """Test that dimension mismatch raises appropriate error."""
    frequencies = np.array([1.0, 2.0, 3.0, 4.0])  # 4 points
    y = np.array([1.0, 2.0, 3.0, 4.0])

    # Wrong size covariance (3x3 instead of 4x4)
    cov = np.eye(3)

    fs = FrequencySeries(y, frequencies=frequencies)

    with pytest.raises(ValueError, match="does not match"):
        fit_series(fs, "power_law", cov=cov, p0={"A": 1, "alpha": -1})


def test_gls_complex_not_supported():
    """Test that GLS with complex data raises NotImplementedError."""
    frequencies = np.array([1.0, 2.0, 3.0])
    y = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    cov = np.eye(3)

    fs = FrequencySeries(y, frequencies=frequencies)

    def complex_model(f, a):
        return a * f * (1 + 1j)

    with pytest.raises(NotImplementedError, match="complex"):
        fit_series(fs, complex_model, cov=cov, p0={"a": 1})


def test_gls_cost_shape_validation():
    """Test GeneralizedLeastSquares validates cov_inv shape."""

    def linear(x, a, b):
        return a * x + b

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])

    # Wrong shape cov_inv
    cov_inv_wrong = np.eye(4)  # 4x4 instead of 3x3

    with pytest.raises(ValueError, match="does not match"):
        GeneralizedLeastSquares(x, y, cov_inv_wrong, linear)


def test_custom_cost_function_callable():
    """Test fit_series with user-defined cost function (simple callable)."""
    from iminuit import Minuit

    from gwexpy.timeseries import TimeSeries

    np.random.seed(789)

    # Data
    x = np.linspace(0, 10, 20)
    y_true = 2.5 * x + 1.0
    y = y_true + 0.1 * np.random.normal(size=len(x))

    ts = TimeSeries(y, times=x)

    def linear_model(t, a, b):
        return a * t + b

    # Custom cost function: simple sum of squared residuals
    class SimpleCost:
        errordef = Minuit.LEAST_SQUARES

        def __init__(self, x, y, model):
            self.x = x
            self.y = y
            self.model = model

        def __call__(self, a, b):
            ym = self.model(self.x, a, b)
            return np.sum((self.y - ym) ** 2)

    custom_cost = SimpleCost(x, y, linear_model)

    result = fit_series(
        ts, linear_model, cost_function=custom_cost, p0={"a": 1, "b": 0}
    )

    assert result.minuit.valid
    assert np.isclose(result.params["a"], 2.5, atol=0.3)
    assert np.isclose(result.params["b"], 1.0, atol=0.5)


def test_custom_cost_function_gls_class():
    """Test fit_series with GeneralizedLeastSquares passed as cost_function."""
    from gwexpy.timeseries import TimeSeries

    np.random.seed(101)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def linear(t, a, b):
        return a * t + b

    y_true = linear(x, a=2.0, b=0.5)
    y = y_true + 0.05 * np.random.normal(size=len(x))

    # Create GLS cost function manually
    cov_inv = np.eye(5) * 100  # High precision (low variance)
    gls_cost = GeneralizedLeastSquares(x, y, cov_inv, linear)

    ts = TimeSeries(y, times=x)

    result = fit_series(ts, linear, cost_function=gls_cost, p0={"a": 1, "b": 0})

    assert result.minuit.valid
    assert np.isclose(result.params["a"], 2.0, atol=0.2)
    assert np.isclose(result.params["b"], 0.5, atol=0.3)


def test_cost_function_priority_over_cov():
    """Test that cost_function takes priority over cov parameter."""
    from iminuit import Minuit

    from gwexpy.timeseries import TimeSeries

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])

    def linear(t, a, b):
        return a * t + b

    # Custom cost that always returns a fixed value (for testing)
    class FixedCost:
        errordef = Minuit.LEAST_SQUARES

        def __call__(self, a, b):
            # Simple chi2
            ym = linear(x, a, b)
            return np.sum((y - ym) ** 2)

    custom_cost = FixedCost()

    # Even with cov provided, cost_function should take priority
    cov = np.eye(3) * 999  # This would create different behavior if used

    ts = TimeSeries(y, times=x)
    result = fit_series(
        ts, linear, cov=cov, cost_function=custom_cost, p0={"a": 1, "b": 0}
    )

    # Should still work - cost_function takes priority
    assert result.minuit.valid


def test_backward_compatibility_no_cost_function():
    """Test that fit_series behaves the same when cost_function=None."""
    np.random.seed(202)

    frequencies = np.logspace(0, 2, 15)
    y = power_law(frequencies, A=8.0, alpha=-1.2) * (
        1 + 0.02 * np.random.normal(size=len(frequencies))
    )

    fs = FrequencySeries(y, frequencies=frequencies)

    # Without cost_function (default behavior)
    result = fit_series(fs, "power_law", p0={"A": 5, "alpha": -1})

    assert result.minuit.valid
    assert np.isclose(result.params["A"], 8.0, rtol=0.3)
    assert np.isclose(result.params["alpha"], -1.2, rtol=0.3)
