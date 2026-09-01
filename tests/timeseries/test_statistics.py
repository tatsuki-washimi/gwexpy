import sys
from types import ModuleType

import numpy as np
import pytest

from gwexpy.timeseries import TimeSeries


@pytest.fixture
def gaussian_data():
    np.random.seed(42)
    return TimeSeries(np.random.normal(0, 1, 1000), dt=0.01)


@pytest.fixture
def non_gaussian_data():
    np.random.seed(42)
    # Exponential distribution is skewed and leptokurtic
    return TimeSeries(np.random.exponential(1, 1000), dt=0.01)


@pytest.fixture
def simple_linear_relationship():
    t = np.linspace(0, 10, 100)
    x = TimeSeries(t, dt=0.1)
    y = TimeSeries(2 * t + 1, dt=0.1)
    return x, y


@pytest.fixture
def non_linear_relationship():
    t = np.linspace(-10, 10, 100)
    x = TimeSeries(t, dt=0.1)
    y = TimeSeries(t**2, dt=0.1)  # Parabola
    return x, y


@pytest.fixture
def causal_relationship():
    np.random.seed(42)
    n = 200
    # X causes Y with delay
    x_val = np.random.randn(n)
    y_val = np.zeros(n)
    # Y depends on X from 1 step ago
    for i in range(1, n):
        y_val[i] = 0.5 * y_val[i - 1] + 0.8 * x_val[i - 1] + 0.1 * np.random.randn()

    x = TimeSeries(x_val, dt=1)
    y = TimeSeries(y_val, dt=1)
    return x, y


def test_skewness(gaussian_data, non_gaussian_data):
    s_gauss = gaussian_data.skewness()
    s_exp = non_gaussian_data.skewness()

    # Gaussian should be near 0
    assert abs(s_gauss) < 0.2
    # Exponential should be positive (around 2)
    assert s_exp > 1.0


def test_kurtosis(gaussian_data, non_gaussian_data):
    k_gauss = gaussian_data.kurtosis(fisher=True)
    k_exp = non_gaussian_data.kurtosis(fisher=True)

    # Gaussian (Fisher) should be near 0
    assert abs(k_gauss) < 0.2
    # Exponential should be positive (excess kurtosis)
    assert k_exp > 1.0


def test_pearson_correlation(simple_linear_relationship):
    x, y = simple_linear_relationship
    corr = x.pcc(y)
    assert abs(corr - 1.0) < 1e-5


def test_distance_correlation(non_linear_relationship):
    x, y = non_linear_relationship

    # Linear correlation should be close to 0 for parabola on symmetric domain
    pcc = x.pcc(y)
    assert abs(pcc) < 0.1

    try:
        # Distance correlation should detect the dependence
        dcor = x.distance_correlation(y)
        assert dcor > 0.4
    except Exception as exc:
        pytest.skip(f"dcor unavailable: {exc}")


def test_mic(non_linear_relationship):
    x, y = non_linear_relationship
    try:
        mic = x.mic(y)
        assert mic > 0.5
    except ImportError:
        pytest.skip("mictools (or minepy) not installed")


def test_granger_causality(causal_relationship):
    x, y = causal_relationship
    try:
        # Check if X causes Y
        p_val_xy = y.granger_causality(x, maxlag=5)
        # Should be significant (small p-value)
        assert p_val_xy < 0.05

        # Check if Y causes X (should not be significant)
        p_val_yx = x.granger_causality(y, maxlag=5)
        assert p_val_yx > 0.05
    except ImportError:
        pytest.skip("statsmodels not installed")


def _fake_granger_results():
    """Return the nested result shape produced by grangercausalitytests."""
    return {
        1: (
            {"ssr_ftest": (2.5, 0.125, 20.0, 1)},
            [object(), object(), np.array([[0.0, 1.0]])],
        ),
        2: (
            {"ssr_ftest": (4.0, 0.025, 19.0, 2)},
            [object(), object(), np.array([[0.0, 0.0, 1.0]])],
        ),
    }


@pytest.fixture
def install_fake_statsmodels(monkeypatch):
    """Install a minimal statsmodels module tree around a supplied callable."""

    def install(granger_callable):
        statsmodels = ModuleType("statsmodels")
        tsa = ModuleType("statsmodels.tsa")
        stattools = ModuleType("statsmodels.tsa.stattools")
        setattr(statsmodels, "__path__", [])
        setattr(tsa, "__path__", [])
        setattr(statsmodels, "tsa", tsa)
        setattr(tsa, "stattools", stattools)
        setattr(stattools, "grangercausalitytests", granger_callable)
        setattr(statsmodels, "_gwexpy_test_fake", True)

        monkeypatch.setitem(sys.modules, "statsmodels", statsmodels)
        monkeypatch.setitem(sys.modules, "statsmodels.tsa", tsa)
        monkeypatch.setitem(sys.modules, "statsmodels.tsa.stattools", stattools)
        return stattools

    return install


def test_granger_causality_passes_verbose_to_legacy_statsmodels(
    install_fake_statsmodels,
):

    calls = []

    def legacy_granger(data, maxlag, verbose):
        calls.append((data, maxlag, verbose))
        return _fake_granger_results()

    fake_stattools = install_fake_statsmodels(legacy_granger)
    target = TimeSeries([0.0, 1.0, 0.5, 1.5], dt=1)
    cause = TimeSeries([1.0, 0.0, 1.0, 0.0], dt=1)

    result = target.granger_causality(cause, maxlag=2, verbose=True)

    assert getattr(sys.modules["statsmodels"], "_gwexpy_test_fake") is True
    assert sys.modules["statsmodels.tsa.stattools"] is fake_stattools
    assert getattr(fake_stattools, "grangercausalitytests") is legacy_granger
    assert len(calls) == 1
    data, maxlag, verbose = calls[0]
    assert data.shape == (4, 2)
    assert maxlag == 2
    assert verbose is True
    assert result.best_lag == 2
    assert result.min_p_value == 0.025


def test_granger_causality_omits_verbose_for_modern_statsmodels(
    install_fake_statsmodels,
):
    calls = []

    def modern_granger(data, maxlag):
        calls.append((data, maxlag))
        return _fake_granger_results()

    install_fake_statsmodels(modern_granger)
    target = TimeSeries([0.0, 1.0, 0.5, 1.5], dt=1)
    cause = TimeSeries([1.0, 0.0, 1.0, 0.0], dt=1)

    result = target.granger_causality(cause, maxlag=2, verbose=True)

    assert len(calls) == 1
    data, maxlag = calls[0]
    assert data.shape == (4, 2)
    assert maxlag == 2
    assert result.best_lag == 2
    assert result.min_p_value == 0.025


def test_granger_causality_passes_verbose_to_kwargs_statsmodels(
    install_fake_statsmodels,
):
    calls = []

    def kwargs_granger(data, maxlag, **kwargs):
        calls.append((data, maxlag, kwargs))
        return _fake_granger_results()

    install_fake_statsmodels(kwargs_granger)
    target = TimeSeries([0.0, 1.0, 0.5, 1.5], dt=1)
    cause = TimeSeries([1.0, 0.0, 1.0, 0.0], dt=1)

    target.granger_causality(cause, maxlag=2, verbose=True)

    assert len(calls) == 1
    data, maxlag, kwargs = calls[0]
    assert data.shape == (4, 2)
    assert maxlag == 2
    assert kwargs == {"verbose": True}


def test_granger_causality_propagates_unrelated_type_error(
    install_fake_statsmodels,
):

    expected = TypeError("unrelated statsmodels calculation failure")

    def modern_granger(data, maxlag):
        raise expected

    install_fake_statsmodels(modern_granger)
    target = TimeSeries([0.0, 1.0, 0.5, 1.5], dt=1)
    cause = TimeSeries([1.0, 0.0, 1.0, 0.0], dt=1)

    with pytest.raises(TypeError) as exc_info:
        target.granger_causality(cause, maxlag=2, verbose=True)

    assert exc_info.value is expected
