"""Input-contract regression tests for the LSQ cost classes (issue #456).

A zero element in ``dy`` previously divided the residuals to ``inf`` and a
complex ``dy`` made the chi2 complex; either makes Minuit fail (or silently
mis-fit) with no diagnostic. The hardened contract rejects them at
construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwexpy.fitting.core import ComplexLeastSquares, RealLeastSquares


def _real_model(x, a):
    return a * x


def _complex_model(x, a):
    return a * x.astype(complex)


# --- valid baseline ---------------------------------------------------------


def test_real_valid_dy_constructs_and_evaluates():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    cost = RealLeastSquares(x, y, np.array([1.0, 1.0, 1.0]), _real_model)
    assert np.isfinite(cost(1.0))


def test_complex_valid_dy_constructs_and_evaluates():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
    cost = ComplexLeastSquares(x, y, np.array([1.0, 1.0, 1.0]), _complex_model)
    val = cost(1.0)
    assert np.isfinite(val)
    assert not np.iscomplexobj(np.asarray(val))


# --- zero / negative dy -> ValueError ---------------------------------------


@pytest.mark.parametrize("cls,model,y", [
    (RealLeastSquares, _real_model, np.array([1.0, 2.0, 3.0])),
    (ComplexLeastSquares, _complex_model, np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])),
])
def test_zero_dy_rejected(cls, model, y):
    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="strictly positive"):
        cls(x, y, np.array([1.0, 0.0, 1.0]), model)


@pytest.mark.parametrize("cls,model,y", [
    (RealLeastSquares, _real_model, np.array([1.0, 2.0, 3.0])),
    (ComplexLeastSquares, _complex_model, np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])),
])
def test_negative_dy_rejected(cls, model, y):
    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="strictly positive"):
        cls(x, y, np.array([1.0, -1.0, 1.0]), model)


# --- non-finite dy -> ValueError --------------------------------------------


def test_non_finite_dy_rejected():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        RealLeastSquares(x, y, np.array([1.0, np.nan, 1.0]), _real_model)


# --- complex dy on ComplexLeastSquares --------------------------------------


def test_complex_dy_rejected():
    x = np.array([1.0, 2.0])
    y = np.array([1.0 + 0j, 2.0 + 0j])
    with pytest.raises(ValueError, match="real-valued"):
        ComplexLeastSquares(x, y, np.array([1.0 + 0.1j, 1.0 + 0.1j]), _complex_model)


def test_real_complex_dy_rejected():
    # dy is a measurement uncertainty: a non-negligible imaginary part must be
    # rejected for RealLeastSquares too, not silently dropped by astype(float).
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="real-valued"):
        RealLeastSquares(x, y, np.array([1.0 + 0.1j, 1.0]), _real_model)


def test_effectively_real_complex_dy_accepted():
    # An array with a complex dtype but negligible imaginary part is accepted
    # and its real part is used.
    x = np.array([1.0, 2.0])
    y = np.array([1.0 + 0j, 2.0 + 0j])
    dy = np.array([1.0 + 0j, 1.0 + 0j])
    cost = ComplexLeastSquares(x, y, dy, _complex_model)
    assert cost.dy.dtype == np.float64
    assert np.isfinite(cost(1.0))
