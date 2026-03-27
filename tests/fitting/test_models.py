"""
Unit tests for gwexpy/fitting/models.py.

Targets the uncovered lines reported at 60% coverage:
  L13  - Model.__call__ raises NotImplementedError
  L85-86 - landau function
  L94-119 - make_pol_func
  L144-146 - get_model with non-string, non-callable argument
  L153-157 - get_model with polN (N>=10) and unknown name
"""

from __future__ import annotations

import numpy as np
import pytest

from gwexpy.fitting.models import (
    MODELS,
    Model,
    Polynomial,
    damped_oscillation,
    exponential,
    gaussian,
    get_model,
    landau,
    make_pol_func,
    power_law,
)


# ---------------------------------------------------------------------------
# Model base class
# ---------------------------------------------------------------------------


class TestModelBaseClass:
    """Tests for the Model base class."""

    def test_call_raises_not_implemented(self):
        """Model.__call__ must raise NotImplementedError (L13)."""
        m = Model()
        with pytest.raises(NotImplementedError):
            m(0.0)

    def test_call_raises_not_implemented_with_args(self):
        """NotImplementedError regardless of extra positional/keyword args."""
        m = Model()
        with pytest.raises(NotImplementedError):
            m(1.0, 2.0, key="val")


# ---------------------------------------------------------------------------
# Polynomial
# ---------------------------------------------------------------------------


class TestPolynomial:
    """Tests for the Polynomial model class."""

    def test_degree_0_positional(self):
        """Constant polynomial via positional args."""
        pol = Polynomial(0)
        assert pol(5.0, 3.0) == pytest.approx(3.0)

    def test_degree_1_positional(self):
        """Linear polynomial via positional args: p0 + p1*x."""
        pol = Polynomial(1)
        # f(2) = 1 + 2*2 = 5
        assert pol(2.0, 1.0, 2.0) == pytest.approx(5.0)

    def test_degree_2_positional(self):
        """Quadratic polynomial via positional args: p0 + p1*x + p2*x^2."""
        pol = Polynomial(2)
        # f(3) = 1 + 2*3 + 3*9 = 1 + 6 + 27 = 34
        assert pol(3.0, 1.0, 2.0, 3.0) == pytest.approx(34.0)

    def test_degree_1_keyword_args(self):
        """Linear polynomial via keyword args (iminuit path)."""
        pol = Polynomial(1)
        assert pol(2.0, p0=1.0, p1=2.0) == pytest.approx(5.0)

    def test_degree_2_keyword_args(self):
        """Quadratic polynomial via keyword args."""
        pol = Polynomial(2)
        assert pol(3.0, p0=1.0, p1=2.0, p2=3.0) == pytest.approx(34.0)

    def test_numpy_array_input(self):
        """Polynomial accepts numpy arrays."""
        pol = Polynomial(1)
        x = np.array([0.0, 1.0, 2.0])
        result = pol(x, 0.0, 1.0)  # f(x) = x
        np.testing.assert_array_almost_equal(result, x)

    def test_param_names(self):
        """param_names attribute is set correctly."""
        pol = Polynomial(3)
        assert pol.param_names == ["p0", "p1", "p2", "p3"]

    def test_degree_attribute(self):
        pol = Polynomial(4)
        assert pol.degree == 4

    def test_signature_has_correct_parameters(self):
        """__signature__ exposes x, p0, ..., pn for iminuit."""
        from inspect import signature

        pol = Polynomial(2)
        params = list(signature(pol).parameters.keys())
        assert params == ["x", "p0", "p1", "p2"]


# ---------------------------------------------------------------------------
# Standalone model functions
# ---------------------------------------------------------------------------


class TestGaussian:
    def test_peak_at_mu(self):
        """Gaussian peaks at x=mu with value A."""
        assert gaussian(0.0, A=2.0, mu=0.0, sigma=1.0) == pytest.approx(2.0)

    def test_symmetry(self):
        """Gaussian is symmetric around mu."""
        assert gaussian(1.0, 1.0, 0.0, 1.0) == pytest.approx(
            gaussian(-1.0, 1.0, 0.0, 1.0)
        )

    def test_numpy_array(self):
        x = np.array([-1.0, 0.0, 1.0])
        result = gaussian(x, 1.0, 0.0, 1.0)
        assert result.shape == x.shape


class TestExponential:
    def test_at_zero(self):
        """exponential(0, A, tau) == A."""
        assert exponential(0.0, 3.0, 5.0) == pytest.approx(3.0)

    def test_decay(self):
        """Value decreases as x increases."""
        assert exponential(1.0, 1.0, 1.0) < exponential(0.0, 1.0, 1.0)

    def test_numpy_array(self):
        x = np.array([0.0, 1.0, 2.0])
        result = exponential(x, 1.0, 1.0)
        assert result.shape == x.shape


class TestPowerLaw:
    def test_unit_x(self):
        assert power_law(1.0, 5.0, 2.0) == pytest.approx(5.0)

    def test_alpha_zero_is_constant(self):
        assert power_law(10.0, 3.0, 0.0) == pytest.approx(3.0)


class TestDampedOscillation:
    def test_at_zero_with_phi_zero(self):
        """At x=0, sin(0)=0 so result is 0."""
        assert damped_oscillation(0.0, 1.0, 1.0, 1.0, phi=0) == pytest.approx(0.0)

    def test_numpy_array(self):
        x = np.linspace(0, 1, 50)
        result = damped_oscillation(x, 1.0, 1.0, 1.0)
        assert result.shape == x.shape


class TestLandau:
    """Tests for the landau function (L85-86)."""

    def test_returns_scalar_for_scalar_input(self):
        """landau returns a scalar for scalar x."""
        result = landau(0.0, A=1.0, mu=0.0, sigma=1.0)
        assert np.isfinite(result)

    def test_positive_amplitude(self):
        """With A > 0 the function value should be positive."""
        result = landau(0.0, A=2.0, mu=0.0, sigma=1.0)
        assert result > 0.0

    def test_peak_near_mu(self):
        """The Moyal distribution peaks slightly below mu; value at mu is finite."""
        result = landau(0.0, A=1.0, mu=0.0, sigma=1.0)
        assert result == pytest.approx(np.exp(-0.5), rel=1e-6)

    def test_amplitude_scaling(self):
        """Doubling A doubles the output."""
        r1 = landau(1.0, A=1.0, mu=0.0, sigma=1.0)
        r2 = landau(1.0, A=2.0, mu=0.0, sigma=1.0)
        assert r2 == pytest.approx(2 * r1)

    def test_numpy_array_input(self):
        """landau works on numpy arrays (L85-86 via vectorised ops)."""
        x = np.linspace(-5.0, 5.0, 100)
        result = landau(x, A=1.0, mu=0.0, sigma=1.0)
        assert result.shape == x.shape
        assert np.all(np.isfinite(result))

    def test_shift_by_mu(self):
        """Shifting mu shifts the distribution."""
        r_mu0 = landau(2.0, A=1.0, mu=0.0, sigma=1.0)
        r_mu2 = landau(4.0, A=1.0, mu=2.0, sigma=1.0)
        assert r_mu0 == pytest.approx(r_mu2)

    def test_sigma_scaling(self):
        """sigma controls width; scaling x and sigma together preserves value."""
        r1 = landau(1.0, A=1.0, mu=0.0, sigma=1.0)
        r2 = landau(2.0, A=1.0, mu=0.0, sigma=2.0)
        assert r1 == pytest.approx(r2)


# ---------------------------------------------------------------------------
# make_pol_func
# ---------------------------------------------------------------------------


class TestMakePolFunc:
    """Tests for make_pol_func (L94-119)."""

    def test_degree_0_constant(self):
        """Degree-0 polynomial is constant."""
        f = make_pol_func(0)
        assert f(99.0, 7.0) == pytest.approx(7.0)

    def test_degree_0_keyword(self):
        """Degree-0 polynomial via keyword arg."""
        f = make_pol_func(0)
        assert f(99.0, p0=7.0) == pytest.approx(7.0)

    def test_degree_1_positional(self):
        """Linear function via positional args."""
        f = make_pol_func(1)
        # f(2) = 1 + 3*2 = 7
        assert f(2.0, 1.0, 3.0) == pytest.approx(7.0)

    def test_degree_1_keyword(self):
        """Linear function via keyword args."""
        f = make_pol_func(1)
        assert f(2.0, p0=1.0, p1=3.0) == pytest.approx(7.0)

    def test_degree_2_positional(self):
        """Quadratic function via positional args."""
        f = make_pol_func(2)
        # f(3) = 1 + 2*3 + 3*9 = 34
        assert f(3.0, 1.0, 2.0, 3.0) == pytest.approx(34.0)

    def test_degree_2_keyword(self):
        """Quadratic function via keyword args."""
        f = make_pol_func(2)
        assert f(3.0, p0=1.0, p1=2.0, p2=3.0) == pytest.approx(34.0)

    def test_signature_parameters(self):
        """Function signature exposes named parameters for iminuit."""
        from inspect import signature

        f = make_pol_func(2)
        params = list(signature(f).parameters.keys())
        assert params == ["x", "p0", "p1", "p2"]

    def test_docstring_set(self):
        """__doc__ is set by make_pol_func (L118)."""
        f = make_pol_func(3)
        assert f.__doc__ is not None
        assert "3" in f.__doc__

    def test_numpy_array_input(self):
        """make_pol_func result works with numpy arrays."""
        f = make_pol_func(1)
        x = np.array([0.0, 1.0, 2.0])
        result = f(x, 0.0, 1.0)  # f(x) = x
        np.testing.assert_array_almost_equal(result, x)

    def test_large_degree(self):
        """make_pol_func works for high degree."""
        f = make_pol_func(10)
        coeffs = [0.0] * 10 + [1.0]  # p10 = 1, all others 0 => f(2) = 2^10
        assert f(2.0, *coeffs) == pytest.approx(1024.0)


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------


class TestGetModel:
    """Tests for get_model (L139-157)."""

    # --- non-string paths (L143-146) ---

    def test_callable_returned_as_is(self):
        """get_model returns a callable directly (L144-145)."""

        def my_func(x):
            return x

        assert get_model(my_func) is my_func

    def test_lambda_returned_as_is(self):
        """get_model returns a lambda directly."""
        fn = lambda x, a: a * x  # noqa: E731
        assert get_model(fn) is fn

    def test_none_returns_none(self):
        """get_model(None) returns None (L146) because None is not callable."""
        assert get_model(None) is None

    def test_non_callable_non_string_returns_none(self):
        """get_model with an integer (non-string, non-callable) returns None."""
        assert get_model(42) is None  # type: ignore[arg-type]

    # --- MODELS dict lookups ---

    def test_lookup_gaussian(self):
        from gwexpy.fitting.models import gaussian as _gaussian

        assert get_model("gaussian") is _gaussian

    def test_lookup_case_insensitive(self):
        from gwexpy.fitting.models import gaussian as _gaussian

        assert get_model("GAUSSIAN") is _gaussian

    def test_lookup_pol0(self):
        result = get_model("pol0")
        assert isinstance(result, Polynomial)
        assert result.degree == 0

    def test_lookup_pol9(self):
        result = get_model("pol9")
        assert isinstance(result, Polynomial)
        assert result.degree == 9

    # --- polN for N >= 10 (L153-155) ---

    def test_pol10_returns_polynomial_degree_10(self):
        """get_model('pol10') creates Polynomial(10) (L153-155)."""
        result = get_model("pol10")
        assert isinstance(result, Polynomial)
        assert result.degree == 10

    def test_pol15_returns_polynomial_degree_15(self):
        """get_model works for any polN with N>=10."""
        result = get_model("pol15")
        assert isinstance(result, Polynomial)
        assert result.degree == 15

    def test_pol10_case_insensitive(self):
        """polN lookup is also case-insensitive."""
        result = get_model("POL10")
        assert isinstance(result, Polynomial)
        assert result.degree == 10

    # --- unknown name (L157) ---

    def test_unknown_name_raises_value_error(self):
        """get_model raises ValueError for an unrecognised name (L157)."""
        with pytest.raises(ValueError, match="Unknown model name"):
            get_model("unknown_xyz")

    def test_unknown_name_error_message_contains_name(self):
        """ValueError message includes the bad name."""
        with pytest.raises(ValueError, match="bad_model_name"):
            get_model("bad_model_name")

    def test_unknown_non_pol_string_raises(self):
        """A string that starts with 'pol' but has non-digit suffix raises."""
        with pytest.raises(ValueError):
            get_model("polXYZ")

    # --- MODELS dictionary sanity ---

    def test_models_dict_contains_expected_keys(self):
        expected = {"gaus", "gaussian", "exp", "expo", "exponential", "landau",
                    "power_law", "damped_oscillation"}
        assert expected.issubset(set(MODELS.keys()))

    def test_models_dict_contains_pol0_to_pol9(self):
        for i in range(10):
            assert f"pol{i}" in MODELS
