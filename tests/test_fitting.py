import numpy as np
from gwexpy.timeseries import TimeSeries
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.fitting import fit_series, enable_fitting_monkeypatch
from gwexpy.fitting.models import gaussian, damped_oscillation, power_law

def test_fit_series_direct():
    # Gaussian test
    t = np.linspace(0, 10, 100)
    y = gaussian(t, A=10, mu=5, sigma=1) + np.random.normal(0, 0.1, len(t))
    ts = TimeSeries(y, times=t)
    
    result = fit_series(ts, "gaus", p0={"A": 8, "mu": 4, "sigma": 1.5})
    assert result.minuit.valid
    assert np.isclose(result.params["mu"], 5, atol=0.5)
    assert "A" in result.params
    assert "sigma" in result.params

def test_fitting_monkeypatch():
    enable_fitting_monkeypatch()
    
    t = np.linspace(0, 1, 1000)
    y = damped_oscillation(t, A=1, tau=0.2, f=50) + np.random.normal(0, 0.05, len(t))
    ts = TimeSeries(y, times=t)
    
    # Check if .fit exists
    assert hasattr(ts, "fit")
    
    # Increase tau and add limits to be more stable
    result = ts.fit("damped_oscillation", p0={"A": 0.8, "tau": 0.2, "f": 45, "phi": 0},
                    limits={"tau": (0.01, 2.0)})
    assert result.minuit.valid
    assert np.isclose(result.params["f"], 50, atol=10)

def test_fitting_missing_p0():
    # Test if default values are correctly used when p0 is incomplete
    t = np.linspace(0, 1, 1000)
    # Target phi=0.5 but we let it start at default 0
    y = damped_oscillation(t, A=1, tau=0.2, f=50, phi=0.5) + np.random.normal(0, 0.05, len(t))
    ts = TimeSeries(y, times=t)
    
    # Missing 'phi' in p0. Damped oscillation has phi=0 as default in models.py.
    # Provide enough parameters to guide convergence but omit 'phi'
    result = ts.fit("damped_oscillation", p0={"A": 1.0, "tau": 0.2, "f": 48},
                    limits={"tau": (0.01, 1.0)})
    assert result.minuit.valid
    assert "phi" in result.params
    # Should be close to 0.5
    assert np.isclose(result.params["phi"], 0.5, atol=0.5)

def test_power_law_fitting():
    frequencies = np.logspace(0, 2, 100)
    y = power_law(frequencies, A=10, alpha=-1.5) * (1 + 0.05 * np.random.normal(size=len(frequencies)))
    fs = FrequencySeries(y, frequencies=frequencies)
    
    result = fit_series(fs, "power_law", p0={"A": 5, "alpha": -1})
    assert result.minuit.valid
    assert np.isclose(result.params["alpha"], -1.5, atol=0.3)

def test_polynomial_fitting():
    x = np.linspace(-5, 5, 100)
    # y = 2 + 3*x + 1*x^2
    y = 2 + 3*x + 1*x**2 + np.random.normal(0, 0.5, len(x))
    ts = TimeSeries(y, times=x)
    
    result = fit_series(ts, "pol2", p0={"p0": 0, "p1": 0, "p2": 0})
    assert result.minuit.valid
    assert np.isclose(result.params["p0"], 2, atol=1.0)
    assert np.isclose(result.params["p1"], 3, atol=1.0)
    assert np.isclose(result.params["p2"], 1, atol=0.5)

def test_fitting_sigma():
    t = np.linspace(0, 10, 20)
    y = gaussian(t, A=10, mu=5, sigma=1) + np.random.normal(0, 0.1, len(t))
    ts = TimeSeries(y, times=t)
    
    # 1. Scalar sigma
    result = ts.fit("gaussian", sigma=0.5, p0={"A": 10, "mu": 5, "sigma": 1})
    assert result.minuit.valid
    assert result.has_dy
    assert np.all(result.dy == 0.5)
    
    # 2. Array sigma with auto-crop
    sigma_full = np.ones(20) * 0.1
    # Crop to middle part
    result_crop = ts.fit("gaussian", x_range=(4, 6), sigma=sigma_full, p0={"A": 10, "mu": 5, "sigma": 1})
    assert result_crop.minuit.valid
    assert result_crop.has_dy
    assert len(result_crop.dy) < 20
    
    # 3. No sigma (checking plotting flag)
    result_no = ts.fit("gaussian", p0={"A": 10, "mu": 5, "sigma": 1})
    assert not result_no.has_dy
    # But internally dy is ones for cost calc consistency
    assert np.all(result_no.dy == 1.0)
