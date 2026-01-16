"""
Test fitting semantics and A2 numerical logic for gwexpy.fitting.

These tests verify the core "contract" of the fitting module:
1. Parameter recovery (Round-trip)
2. Unit propagation (Input unit -> Output unit)
3. Output types and access patterns
4. GLS direct solver mechanics
"""

import pytest
import numpy as np
from astropy import units as u
from gwpy.frequencyseries import FrequencySeries
from gwexpy.fitting import Fitter, models, GLS

class TestFittingSemantics:
    """Test A2 semantics of fitting module."""

    @pytest.fixture
    def linear_data(self):
        """Create synthetic data: y = 2x + 1"""
        x = np.linspace(0, 10, 20)
        # Unit: meters
        y_val = 2.0 * x + 1.0
        # Add small noise to allow fitting
        np.random.seed(42)
        y_val += np.random.normal(0, 0.01, size=x.size)
        
        return FrequencySeries(y_val, frequencies=x, unit='m')

    def test_fit_parameter_recovery(self, linear_data):
        """Test that fit recovers slope=2 and intercept=1."""
        # Use new Polynomial model class
        model = models.Polynomial(degree=1)
        fitter = Fitter(model)
        
        # Initial guess
        result = fitter.fit(linear_data, p0={'p0': 0, 'p1': 5})
        
        # Check coefficients (p1 ~ 2, p0 ~ 1)
        # Use .value syntax requested by the task
        p1 = result.params['p1'].value
        p0 = result.params['p0'].value
        
        np.testing.assert_allclose(p1, 2.0, rtol=0.1)
        np.testing.assert_allclose(p0, 1.0, rtol=0.1)

    def test_model_evaluation_units(self, linear_data):
        """Test that evaluating the fitted model returns Quantities with correct units."""
        model = models.Polynomial(degree=1)
        fitter = Fitter(model)
        result = fitter.fit(linear_data)
        
        # Evaluate model on the original x axis (Quantity)
        x_freq = linear_data.frequencies
        y_fit = result.model(x_freq)
        
        # Should retain 'm' unit from input data
        assert y_fit.unit == u.m
        assert isinstance(y_fit, u.Quantity)

    def test_gls_solver_mechanics(self):
        """Test Generalized Least Squares solver internal logic (linear)."""
        # Simple matrix problem: y = Ax
        # y = 1 + 2x
        # X = [1, x]
        x_vals = np.array([1, 2, 3])
        X = np.column_stack([np.ones(3), x_vals]) 
        y = np.array([3, 5, 7]) # Perfect line
        
        # Solve via direct GLS class
        solver = GLS(X, y)
        params = solver.solve()
        
        # Expect intercept=1, slope=2
        np.testing.assert_allclose(params, [1, 2], atol=1e-10)

    def test_gls_with_covariance(self):
        """Test GLS with a custom covariance matrix."""
        X = np.array([[1], [1]])
        y = np.array([10, 20])
        # If we have identity cov, mean is 15.
        # If we have cov = [[1, 0], [0, 100]], mean should be close to 10.
        cov = np.diag([1, 100])
        solver = GLS(X, y, cov=cov)
        params = solver.solve()
        
        # Weighted mean: (10/1 + 20/100) / (1/1 + 1/100) = (10+0.2)/1.01 = 10.2/1.01 ~ 10.099
        expected = (10/1 + 20/100) / (1/1 + 1/100)
        np.testing.assert_allclose(params[0], expected, atol=1e-10)

    def test_parameter_value_float_behavior(self, linear_data):
        """Test that ParameterValue behaves like a float."""
        model = models.Polynomial(degree=1)
        fitter = Fitter(model)
        result = fitter.fit(linear_data)
        
        p0 = result.params['p0']
        # Should support float addition
        sum_val = p0 + 10.0
        assert isinstance(sum_val, float)
        assert sum_val == pytest.approx(p0.value + 10.0)
