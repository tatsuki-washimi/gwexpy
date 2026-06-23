"""Domain-contract regression tests for gwexpy.fitting.models (issue #455 + G7).

All eight findings in #455 (P2×4 degenerate scale + P3×4) plus the G7 supplement
finding (power_law x<=0 domain): degenerate parameters must raise ValueError/TypeError
immediately rather than silently propagating NaN/Inf through the model evaluation.
"""
from __future__ import annotations

import numpy as np
import pytest

from gwexpy.fitting.models import (
    damped_oscillation,
    exponential,
    gaussian,
    get_model,
    landau,
    lorentzian,
    lorentzian_q,
    power_law,
    voigt,
)

# --- gaussian: sigma=0 (#455 P2) ---------------------------------------------


def test_gaussian_sigma_zero_raises():
    with pytest.raises(ValueError, match="sigma must be non-zero"):
        gaussian(0.0, 1.0, 0.0, 0.0)


def test_gaussian_scalar_sigma_zero_was_zdiv():
    """Scalar sigma=0 previously raised ZeroDivisionError; now ValueError."""
    with pytest.raises(ValueError):
        gaussian(1.0, 1.0, 0.0, 0)


def test_gaussian_valid_inputs_unaffected():
    assert gaussian(0.0, 2.0, 0.0, 1.0) == pytest.approx(2.0)


# --- lorentzian: gamma=0 (#455 P2) -------------------------------------------


def test_lorentzian_gamma_zero_raises():
    with pytest.raises(ValueError, match="gamma must be non-zero"):
        lorentzian(0.0, 1.0, 0.0, 0.0)


def test_lorentzian_gamma_zero_array_was_silent_nan():
    """np.array input with gamma=0 silently returned nan; now ValueError."""
    with pytest.raises(ValueError, match="gamma must be non-zero"):
        lorentzian(np.array([0.0, 1.0]), 1.0, 0.0, 0.0)


def test_lorentzian_valid_inputs_unaffected():
    assert lorentzian(0.0, 3.0, 0.0, 1.0) == pytest.approx(3.0)


# --- voigt: sigma=0 (#455 P2) ------------------------------------------------


def test_voigt_sigma_zero_raises():
    with pytest.raises(ValueError, match="sigma must be non-zero"):
        voigt(0.0, 1.0, 0.0, 0.0, 1.0)


def test_voigt_valid_inputs_unaffected():
    result = voigt(0.0, 1.0, 0.0, 1.0, 1.0)
    assert np.isfinite(result)
    assert result == pytest.approx(1.0)


# --- damped_oscillation: tau=0 (#455 P2) -------------------------------------


def test_damped_oscillation_tau_zero_raises():
    with pytest.raises(ValueError, match="tau must be non-zero"):
        damped_oscillation(1.0, 1.0, 0.0, 1.0)


def test_damped_oscillation_tau_zero_numpy_was_silent():
    """tau=0.0 (NumPy float) silently gave nan/inf; now ValueError."""
    with pytest.raises(ValueError, match="tau must be non-zero"):
        damped_oscillation(1.0, 1.0, np.float64(0.0), 1.0)


def test_damped_oscillation_valid_inputs_unaffected():
    result = damped_oscillation(0.0, 1.0, 1.0, 1.0, phi=0.0)
    assert result == pytest.approx(0.0)


# --- exponential: tau=0 (#455 P3) --------------------------------------------


def test_exponential_tau_zero_raises():
    with pytest.raises(ValueError, match="tau must be non-zero"):
        exponential(1.0, 1.0, 0.0)


def test_exponential_valid_inputs_unaffected():
    assert exponential(0.0, 3.0, 5.0) == pytest.approx(3.0)


# --- landau: sigma=0 (#455 P3) -----------------------------------------------


def test_landau_sigma_zero_raises():
    with pytest.raises(ValueError, match="sigma must be non-zero"):
        landau(0.0, 1.0, 0.0, 0.0)


def test_landau_valid_inputs_unaffected():
    result = landau(0.0, 1.0, 0.0, 1.0)
    assert np.isfinite(result)


# --- lorentzian_q: Q=0 (#455 P3) ---------------------------------------------


def test_lorentzian_q_zero_raises():
    with pytest.raises(ValueError, match="Q must be non-zero"):
        lorentzian_q(1.0, 1.0, 100.0, 0.0)


def test_lorentzian_q_zero_int_was_zdiv():
    """Integer Q=0 previously raised ZeroDivisionError; now ValueError."""
    with pytest.raises(ValueError, match="Q must be non-zero"):
        lorentzian_q(1.0, 1.0, 100.0, 0)


def test_lorentzian_q_valid_inputs_unaffected():
    assert lorentzian_q(100.0, 2.0, 100.0, 10.0) == pytest.approx(2.0)


# --- get_model: non-callable non-string (#455 P3) ----------------------------


def test_get_model_none_raises_type_error():
    with pytest.raises(TypeError, match="name must be a string or callable"):
        get_model(None)  # type: ignore[arg-type]


def test_get_model_int_raises_type_error():
    """Integer name previously returned None silently; now TypeError."""
    with pytest.raises(TypeError, match="name must be a string or callable"):
        get_model(42)  # type: ignore[arg-type]


def test_get_model_list_raises_type_error():
    with pytest.raises(TypeError, match="name must be a string or callable"):
        get_model(["gaussian"])  # type: ignore[arg-type]


# --- power_law: x<=0 domain guard (G7 supplement) ----------------------------


def test_power_law_negative_x_raises():
    """x<0 with non-integer alpha silently returned complex/NaN; now ValueError."""
    with pytest.raises(ValueError, match="power_law requires x > 0"):
        power_law(-1.0, 1.0, 0.5)


def test_power_law_zero_x_raises():
    """x=0 with negative alpha gives Inf; now ValueError."""
    with pytest.raises(ValueError, match="power_law requires x > 0"):
        power_law(0.0, 1.0, -1.0)


def test_power_law_array_with_negative_x_raises():
    """Array containing x<=0 raises ValueError."""
    with pytest.raises(ValueError, match="power_law requires x > 0"):
        power_law(np.array([1.0, -1.0, 2.0]), 1.0, 2.0)


def test_power_law_valid_positive_x_unaffected():
    assert power_law(2.0, 3.0, 2.0) == pytest.approx(12.0)


def test_power_law_array_positive_x_unaffected():
    x = np.array([1.0, 2.0, 4.0])
    result = power_law(x, 1.0, 0.5)
    np.testing.assert_allclose(result, np.array([1.0, np.sqrt(2.0), 2.0]))
