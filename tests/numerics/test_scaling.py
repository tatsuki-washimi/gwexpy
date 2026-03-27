"""Tests for gwexpy/numerics/scaling.py"""
from __future__ import annotations

import numpy as np
import pytest

from gwexpy.numerics.scaling import AutoScaler, get_safe_epsilon, safe_epsilon, safe_log_scale


# ---------------------------------------------------------------------------
# safe_epsilon
# ---------------------------------------------------------------------------

def test_safe_epsilon_normal_data():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    eps = safe_epsilon(data)
    assert eps > 0
    assert eps == pytest.approx(np.std(data) * 1e-6, rel=1e-9)


def test_safe_epsilon_zero_data_returns_abs_tol():
    data = np.zeros(10)
    eps = safe_epsilon(data)
    # std=0, so max(abs_tol, 0*rel_tol) = abs_tol
    assert eps > 0


def test_safe_epsilon_custom_rel_tol():
    data = np.array([10.0, 20.0, 30.0])
    eps = safe_epsilon(data, rel_tol=0.1)
    assert eps == pytest.approx(np.std(data) * 0.1)


def test_safe_epsilon_custom_abs_tol():
    data = np.zeros(5)
    eps = safe_epsilon(data, abs_tol=1e-3)
    assert eps == pytest.approx(1e-3)


def test_safe_epsilon_tiny_strain_scale():
    data = np.array([1e-21, 2e-21, 3e-21])
    eps = safe_epsilon(data)
    assert eps > 0


def test_get_safe_epsilon_alias():
    data = np.array([1.0, 2.0, 3.0])
    assert get_safe_epsilon(data) == safe_epsilon(data)


# ---------------------------------------------------------------------------
# AutoScaler — _compute_scale
# ---------------------------------------------------------------------------

def test_autoscaler_scale_normal():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sc = AutoScaler(data)
    assert sc.scale == pytest.approx(np.std(data))


def test_autoscaler_scale_zero_data_fallback():
    data = np.zeros(10)
    sc = AutoScaler(data)
    assert sc.scale == pytest.approx(1.0)


def test_autoscaler_scale_silent_data_fallback():
    data = np.full(5, 1e-100)
    sc = AutoScaler(data)
    # std << SAFE_FLOOR, so falls back to 1.0
    assert sc.scale == pytest.approx(1.0)


def test_autoscaler_custom_eps():
    data = np.array([1.0, 1.0, 1.0])  # std = 0
    sc = AutoScaler(data, eps=1e-3)
    # std=0 < eps=1e-3 → scale=1.0
    assert sc.scale == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AutoScaler — normalize / denormalize
# ---------------------------------------------------------------------------

def test_autoscaler_normalize():
    data = np.array([2.0, 4.0, 6.0])
    sc = AutoScaler(data)
    normed = sc.normalize()
    np.testing.assert_allclose(normed, data / sc.scale)


def test_autoscaler_denormalize():
    data = np.array([1.0, 2.0, 3.0])
    sc = AutoScaler(data)
    normed = sc.normalize()
    recovered = sc.denormalize(normed)
    np.testing.assert_allclose(recovered, data, rtol=1e-10)


def test_autoscaler_roundtrip():
    data = np.array([1e-21, 2e-21, 3e-21])
    sc = AutoScaler(data)
    normed = sc.normalize()
    result = normed * 2
    output = sc.denormalize(result)
    np.testing.assert_allclose(output, data * 2, rtol=1e-10)


# ---------------------------------------------------------------------------
# AutoScaler — context manager
# ---------------------------------------------------------------------------

def test_autoscaler_context_manager():
    data = np.array([1.0, 2.0, 3.0])
    with AutoScaler(data) as sc:
        normed = sc.normalize()
        output = sc.denormalize(normed)
    np.testing.assert_allclose(output, data, rtol=1e-10)


def test_autoscaler_repr():
    data = np.array([1.0, 2.0, 3.0])
    sc = AutoScaler(data)
    r = repr(sc)
    assert r.startswith("AutoScaler(scale=")


def test_autoscaler_scale_property():
    data = np.array([1.0, 2.0, 3.0])
    sc = AutoScaler(data)
    assert isinstance(sc.scale, float)


# ---------------------------------------------------------------------------
# safe_log_scale
# ---------------------------------------------------------------------------

def test_safe_log_scale_positive_data():
    data = np.array([1.0, 10.0, 100.0])
    result = safe_log_scale(data)
    assert result.shape == data.shape
    assert np.all(np.isfinite(result))


def test_safe_log_scale_default_factor():
    data = np.array([100.0])
    result = safe_log_scale(data)
    # factor=10, so result ≈ 10*log10(100) = 20
    assert result[0] == pytest.approx(20.0, abs=1.0)


def test_safe_log_scale_custom_factor():
    data = np.array([100.0])
    result = safe_log_scale(data, factor=20.0)
    assert result[0] == pytest.approx(40.0, abs=1.0)


def test_safe_log_scale_all_inf_returns_neg_inf():
    data = np.array([np.inf, np.inf])
    result = safe_log_scale(data)
    assert np.all(result == -np.inf)


def test_safe_log_scale_mixed_finite_inf():
    data = np.array([1.0, np.inf])
    result = safe_log_scale(data)
    # finite element should give finite result
    assert np.isfinite(result[0])


def test_safe_log_scale_zero_value():
    data = np.array([0.0, 1.0])
    result = safe_log_scale(data)
    assert np.all(np.isfinite(result))


def test_safe_log_scale_tiny_values():
    data = np.array([1e-21, 2e-21, 3e-21])
    result = safe_log_scale(data)
    assert result.shape == data.shape


def test_safe_log_scale_custom_dynamic_range():
    data = np.array([1.0, 0.001])
    result_default = safe_log_scale(data)
    result_narrow = safe_log_scale(data, dynamic_range_db=20.0)
    # narrower dynamic range clips low values more
    assert np.all(np.isfinite(result_narrow))
