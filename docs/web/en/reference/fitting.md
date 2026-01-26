# gwexpy.fitting

The `gwexpy.fitting` module provides advanced fitting functionality using iminuit.

## Overview

- **Least Squares Fitting**: Supports real and complex data
- **GLS (Generalized Least Squares)**: Fitting with covariance matrix consideration
- **MCMC**: Bayesian estimation using emcee
- **Integrated Pipeline**: One-liner API for bootstrap → GLS → MCMC workflow

---

## Classes and Functions

### Main Functions

| Name | Description |
|------|-------------|
| `fit_series()` | Fit a Series object |
| `fit_bootstrap_spectrum()` | Integrated spectrum analysis pipeline |

### Classes

| Name | Description |
|------|-------------|
| `FitResult` | Class to store fit results |
| `GeneralizedLeastSquares` | GLS cost function class |
| `RealLeastSquares` | Cost function for real data |
| `ComplexLeastSquares` | Cost function for complex data |

---

## fit_series

```python
fit_series(
    series,
    model,
    x_range=None,
    sigma=None,
    cov=None,
    cost_function=None,
    p0=None,
    limits=None,
    fixed=None,
    **kwargs
)
```

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `series` | Series | Data to fit |
| `model` | callable / str | Model function or name ("gaussian", "power_law", etc.) |
| `x_range` | tuple | Fit range (xmin, xmax) |
| `sigma` | array / scalar | Errors (ignored if `cov` is specified) |
| `cov` | BifrequencyMap / ndarray | Covariance matrix (for GLS) |
| `cost_function` | callable | Custom cost function (highest priority) |
| `p0` | dict / list | Initial parameter values |
| `limits` | dict | Parameter limits {"A": (0, 100)} |
| `fixed` | list | Parameter names to fix |

### Returns

`FitResult` object

### Example

```python
from gwexpy.frequencyseries import FrequencySeries
from gwexpy.fitting import fit_series

# Prepare data
fs = FrequencySeries(y, frequencies=frequencies)

# Basic fit
result = fit_series(fs, "power_law", p0={"A": 10, "alpha": -1.5})

# GLS fit (with covariance matrix)
result = fit_series(fs, "power_law", cov=covariance_matrix, p0={"A": 10, "alpha": -1.5})

# Check results
print(result.params)
print(result.errors)
result.plot()
```

---

## fit_bootstrap_spectrum

```python
fit_bootstrap_spectrum(
    data_or_spectrogram,
    model_fn,
    freq_range=None,
    method="median",
    rebin_width=None,
    block_size=None,
    ci=0.68,
    window="hann",
    nperseg=16,
    noverlap=None,
    n_boot=1000,
    initial_params=None,
    bounds=None,
    fixed=None,
    run_mcmc=False,
    mcmc_walkers=32,
    mcmc_steps=5000,
    mcmc_burn_in=500,
    plot=True,
    progress=True,
)
```

### Parameters

| Name | Type | Description |
|------|------|-------------|
| `data_or_spectrogram` | TimeSeries / Spectrogram | Input data |
| `model_fn` | callable | Model function `model(f, *params)` |
| `freq_range` | tuple | Frequency range for fitting |
| `method` | str | Bootstrap averaging method ("median" / "mean") |
| `rebin_width` | float | Frequency rebinning width (Hz) |
| `block_size` | int | Block size for block bootstrap |
| `n_boot` | int | Number of bootstrap iterations |
| `initial_params` | dict | Initial parameters |
| `bounds` | dict | Parameter limits |
| `run_mcmc` | bool | Whether to run MCMC |
| `mcmc_steps` | int | Number of MCMC steps |
| `plot` | bool | Whether to plot results |

### Returns

`FitResult` object (with additional attributes):

- `psd`: Bootstrap PSD
- `cov`: Covariance BifrequencyMap
- `bootstrap_method`: Averaging method used

### Example

```python
from gwexpy.fitting import fit_bootstrap_spectrum

def power_law(f, A, alpha):
    return A * f**alpha

result = fit_bootstrap_spectrum(
    spectrogram,
    model_fn=power_law,
    freq_range=(5, 50),
    rebin_width=0.5,
    block_size=4,
    initial_params={"A": 10, "alpha": -1.5},
    run_mcmc=True,
    mcmc_steps=3000,
)

# Results
print(result.params)
print(result.parameter_intervals)  # MCMC confidence intervals
result.plot_corner()
```

---

## FitResult

Class to store fit results.

### Properties

| Name | Type | Description |
|------|------|-------------|
| `params` | dict | Best-fit parameters |
| `errors` | dict | Parameter errors |
| `chi2` | float | χ² value |
| `ndof` | int | Degrees of freedom |
| `reduced_chi2` | float | Reduced χ² |
| `cov_inv` | ndarray | GLS covariance inverse |
| `parameter_intervals` | dict | MCMC percentiles (16, 50, 84) |
| `mcmc_chain` | ndarray | Full MCMC chain |
| `samples` | ndarray | MCMC samples (after burn-in) |

### Methods

| Name | Description |
|------|-------------|
| `plot()` | Plot data and fit curve |
| `bode_plot()` | Bode plot (for complex data) |
| `run_mcmc()` | Run MCMC |
| `plot_corner()` | Corner plot |
| `plot_fit_band()` | Fit plot with confidence band |

### Example

```python
result = fit_series(fs, "power_law", p0={"A": 10, "alpha": -1.5})

# Basic info
print(result)  # Minuit output
print(f"χ²/dof = {result.reduced_chi2:.2f}")

# Plotting
result.plot()

# MCMC
result.run_mcmc(n_steps=5000, burn_in=500)
print(result.parameter_intervals)
result.plot_corner()
result.plot_fit_band()
```

---

## GeneralizedLeastSquares

GLS cost function using covariance inverse.

```python
class GeneralizedLeastSquares:
    errordef = 1.0  # Minuit.LEAST_SQUARES
    
    def __init__(self, x, y, cov_inv, model):
        ...
    
    def __call__(self, *args) -> float:
        # χ² = r.T @ cov_inv @ r
        ...
    
    @property
    def ndata(self) -> int:
        ...
```

### Example

```python
from gwexpy.fitting import GeneralizedLeastSquares, fit_series
from iminuit import Minuit

# Direct usage
def linear(x, a, b):
    return a * x + b

gls = GeneralizedLeastSquares(x, y, cov_inv, linear)
m = Minuit(gls, a=1, b=0)
m.migrad()

# Via fit_series
result = fit_series(ts, linear, cost_function=gls, p0={"a": 1, "b": 0})
```

---

## Built-in Models

Models available in `gwexpy.fitting.models`:

| Name | Formula | Parameters |
|------|---------|------------|
| `gaussian` / `gaus` | A * exp(-(x-μ)²/(2σ²)) | A, mu, sigma |
| `power_law` | A * x^α | A, alpha |
| `damped_oscillation` | A * exp(-t/τ) * cos(2πft + φ) | A, tau, f, phi |
| `pol0` ~ `pol9` | c₀ + c₁x + c₂x² + ... | p0, p1, ... |
| `lorentzian` | A / ((x-x₀)² + γ²) | A, x0, gamma |
| `exponential` | A * exp(-x/τ) | A, tau |

---

## Dependencies

- `iminuit`: Required
- `emcee`: Required for MCMC functionality
- `corner`: Required for corner plots

---

## Unit & Model Semantics

`gwexpy.fitting` ensures unit consistency between data and models:

* **Unit Propagation**: When a model is evaluated via `result.model(x)`, it automatically respects the units of the input data.
    * If input `x` is in $Hz$, the model evaluates using frequency units.
    * The output `y` retains the unit of the fitted data (e.g., $m$, $V^2/Hz$).
* **Parameter Access**: `result.params['name']` returns an object with `.value` and `.error` attributes, decoupling the numerical value from statistical uncertainty.
