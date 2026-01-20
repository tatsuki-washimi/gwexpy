"""
Tests for MCMC functionality with GLS support.
"""
import numpy as np
import pytest

# Skip all tests if emcee is not installed
pytest.importorskip("emcee")
pytest.importorskip("corner")


def test_mcmc_with_gls_covariance():
    """Test MCMC with GLS covariance structure."""
    from gwexpy.fitting import fit_series
    from gwexpy.timeseries import TimeSeries

    np.random.seed(42)

    # Simple linear model
    def linear(t, a, b):
        return a * t + b

    x = np.linspace(0, 10, 10)
    y_true = linear(x, a=2.0, b=1.0)

    # Create correlated covariance
    n = len(x)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = 0.1 * np.exp(-0.3 * abs(i - j))

    # Generate correlated noise
    L = np.linalg.cholesky(cov)
    noise = L @ np.random.normal(size=n)
    y = y_true + noise

    ts = TimeSeries(y, times=x)

    # Fit with GLS
    result = fit_series(ts, linear, cov=cov, p0={"a": 1, "b": 0})

    assert result.minuit.valid
    assert result.cov_inv is not None  # Should have cov_inv stored

    # Run MCMC with short chain for testing
    result.run_mcmc(n_walkers=16, n_steps=200, burn_in=50, progress=False)

    assert result.samples is not None
    assert result.samples.shape[1] == 2  # Two parameters: a, b

    # Check parameter_intervals
    intervals = result.parameter_intervals
    assert "a" in intervals
    assert "b" in intervals
    assert len(intervals["a"]) == 3  # (lower, median, upper)


def test_mcmc_parameter_intervals():
    """Test parameter_intervals property."""
    from gwexpy.fitting import fit_series
    from gwexpy.fitting.models import power_law
    from gwexpy.frequencyseries import FrequencySeries

    np.random.seed(123)

    frequencies = np.logspace(0, 2, 15)
    y_true = power_law(frequencies, A=10.0, alpha=-1.5)
    y = y_true * (1 + 0.05 * np.random.normal(size=len(frequencies)))

    fs = FrequencySeries(y, frequencies=frequencies)
    result = fit_series(fs, "power_law", p0={"A": 5, "alpha": -1})

    # Before MCMC, should raise error
    with pytest.raises(RuntimeError):
        _ = result.parameter_intervals

    # Run MCMC
    result.run_mcmc(n_walkers=16, n_steps=150, burn_in=30, progress=False)

    intervals = result.parameter_intervals

    # Check structure
    for name in ["A", "alpha"]:
        assert name in intervals
        q16, q50, q84 = intervals[name]
        # Lower bound should be less than median, median less than upper
        assert q16 <= q50 <= q84


def test_mcmc_chain_property():
    """Test mcmc_chain property returns full chain."""
    from gwexpy.fitting import fit_series
    from gwexpy.timeseries import TimeSeries

    np.random.seed(456)

    def linear(t, a, b):
        return a * t + b

    x = np.linspace(0, 5, 8)
    y = linear(x, 2.5, 0.5) + 0.1 * np.random.normal(size=len(x))

    ts = TimeSeries(y, times=x)
    result = fit_series(ts, linear, p0={"a": 1, "b": 0})

    # Before MCMC
    assert result.mcmc_chain is None

    # Run MCMC
    n_walkers, n_steps = 12, 100
    result.run_mcmc(n_walkers=n_walkers, n_steps=n_steps, burn_in=20, progress=False)

    chain = result.mcmc_chain
    assert chain is not None
    assert chain.shape == (n_steps, n_walkers, 2)  # (steps, walkers, params)


def test_plot_corner_gls_annotation():
    """Test that plot_corner shows GLS annotation when applicable."""
    import matplotlib.pyplot as plt

    from gwexpy.fitting import fit_series
    from gwexpy.timeseries import TimeSeries

    np.random.seed(789)

    def linear(t, a, b):
        return a * t + b

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = linear(x, 2.0, 1.0) + 0.1 * np.random.normal(size=len(x))
    cov = np.diag([0.01] * 5)

    ts = TimeSeries(y, times=x)
    result = fit_series(ts, linear, cov=cov, p0={"a": 1, "b": 0})
    result.run_mcmc(n_walkers=12, n_steps=100, burn_in=20, progress=False)

    fig = result.plot_corner()

    # Check that figure was created
    assert fig is not None

    # Check for GLS annotation (look in figure text)
    texts = [t.get_text() for t in fig.texts]
    assert any("GLS" in t for t in texts)

    plt.close(fig)


def test_plot_fit_band():
    """Test plot_fit_band method."""
    import matplotlib.pyplot as plt

    from gwexpy.fitting import fit_series
    from gwexpy.timeseries import TimeSeries

    np.random.seed(101)

    def linear(t, a, b):
        return a * t + b

    x = np.linspace(0, 10, 15)
    y = linear(x, 1.5, 2.0) + 0.2 * np.random.normal(size=len(x))
    sigma = 0.2 * np.ones_like(y)

    ts = TimeSeries(y, times=x)
    result = fit_series(ts, linear, sigma=sigma, p0={"a": 1, "b": 1})

    # Before MCMC, should raise error
    with pytest.raises(RuntimeError):
        result.plot_fit_band()

    # Run MCMC
    result.run_mcmc(n_walkers=12, n_steps=100, burn_in=20, progress=False)

    ax = result.plot_fit_band(n_samples=50)

    assert ax is not None

    # Check that data, fit, and band were plotted
    lines = ax.get_lines()
    assert len(lines) >= 1  # At least the fit line

    plt.close('all')


def test_mcmc_with_custom_cost_function_gls():
    """Test MCMC with GeneralizedLeastSquares passed as cost_function."""
    from gwexpy.fitting import GeneralizedLeastSquares, fit_series
    from gwexpy.timeseries import TimeSeries

    np.random.seed(202)

    def linear(t, a, b):
        return a * t + b

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = linear(x, 3.0, 0.5) + 0.05 * np.random.normal(size=len(x))

    cov_inv = np.eye(5) * 100
    gls_cost = GeneralizedLeastSquares(x, y, cov_inv, linear)

    ts = TimeSeries(y, times=x)
    result = fit_series(ts, linear, cost_function=gls_cost, p0={"a": 1, "b": 0})

    # cov_inv should be extracted from cost function
    assert result.cov_inv is not None
    np.testing.assert_array_equal(result.cov_inv, cov_inv)

    # Run MCMC
    result.run_mcmc(n_walkers=12, n_steps=100, burn_in=20, progress=False)

    intervals = result.parameter_intervals
    assert "a" in intervals
    assert "b" in intervals
