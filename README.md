# gwexpy: GWpy Expansions for Experiments

**gwexpy** is an extension library for **GWpy**, designed to facilitate advanced time-series analysis, matrix operations, and signal processing for experimental physics and gravitational wave data analysis. It builds upon GWpy's core data objects (`TimeSeries`, `FrequencySeries`) and introduces high-level containers and methods for multivariate analysis.

## Key Features

### 1. Advanced Containers
- **`TimeSeriesMatrix`**: A 3D container `(channels, 1, time)` for handling multi-channel time-series data with shared time axes. Supports element-wise operations, slicing, and bulk processing.
- **`FrequencySeriesMatrix`**: A matrix container for frequency-domain data, ideal for representing Transfer Functions, CSD, and Coherence matrices.

### 2. Spectral Analysis Extensions
- **Spectral Matrices**: Easily compute Cross-Spectral Density (CSD) and Coherence matrices for entire collections.
  - `TimeSeriesDict.csd_matrix()` / `coherence_matrix()`
  - `TimeSeriesList.csd_matrix()` / `coherence_matrix()`
- **Extended FFT & Transfer Functions**:
  - `TimeSeries.fft(mode="transient")`: Transient-friendly FFT with padding options (zero, reflect).
  - `TimeSeries.transfer_function()`: Supports 'gwpy' (Welch), 'fft' (Direct ratio), and 'auto' modes.

### 3. Preprocessing & Decomposition
- **Alignment**: `align_timeseries_collection` to synchronize varying start times and sample rates.
- **Imputation**: `impute_timeseries` / `TimeSeries.impute()` for handling gaps (interpolation, padding).
- **Standardization**: Z-score and robust scaling for `TimeSeries` and `TimeSeriesMatrix`.
- **Whitening**: `whiten_channels` (PCA/ZCA) for multivariate whitening.
- **Decomposition**: Integrated **PCA** and **ICA`** methods for `TimeSeriesMatrix`.
  - `mat.pca_fit()`, `mat.ica_fit()`
  - Full support for `inverse_transform` to reconstruct signals in the original domain.

### 4. Advanced Statistics
- **ARIMA**: Forecasting and residual analysis using `statsmodels` backend (`ts.fit_arima()`).
- **Hurst Exponent**: Global and local Hurst exponent estimation (`ts.hurst()`, `ts.local_hurst()`) for analyzing long-term memory.

---

## Installation

```bash
pip install gwexpy
```
*(Note: Requires `gwpy`, `astropy`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`)*

---

## Usage Examples

### 1. TimeSeriesMatrix & Decomposition
```python
import numpy as np
from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
from astropy import units as u

# Create a TimeSeriesMatrix from a list of Series
t = np.linspace(0, 10, 1000) * u.s
s1 = TimeSeries(np.sin(2*np.pi*1*t), times=t, name="Ch1")
s2 = TimeSeries(np.cos(2*np.pi*1*t) + 0.5*np.random.randn(1000), times=t, name="Ch2")
s3 = TimeSeries(np.random.randn(1000), times=t, name="Ch3")

# Initialize Matrix
mat = TimeSeriesMatrix.from_list([s1, s2, s3]) # Shape: (3, 1, 1000)

# Preprocessing
mat_clean = mat.impute(method="interpolate").standardize()

# Principal Component Analysis (PCA)
scores, model = mat_clean.pca(n_components=2, return_model=True)
# scores is a TimeSeriesMatrix of shape (2, 1, 1000)

# Independent Component Analysis (ICA)
sources = mat_clean.ica(n_components=3)
```

### 2. Spectral Matrices (CSD & Coherence)
```python
from gwexpy.timeseries import TimeSeriesDict

data = TimeSeriesDict()
# ... populate data ...

# Compute 4x4 Coherence Matrix (FrequencySeriesMatrix)
coh_mat = data.coherence_matrix(fftlength=2, overlap=1)

# Access Coherence between Ch1 and Ch2
coh_12 = coh_mat["Ch1", "Ch2"] # Returns FrequencySeries
coh_12.plot()
```

### 3. Advanced Time-Series Methods
```python
ts = s2

# Transient FFT with padding
fs = ts.fft(mode="transient", pad_mode="reflect", pad_left=100)

# ARIMA Forecasting
arima_res = ts.fit_arima(order=(1,0,1))
forecast = arima_res.forecast(steps=100)

# Hurst Exponent
H = ts.hurst()
local_H = ts.local_hurst(window=2*u.s)
```

## Compatibility
gwexpy allows seamless conversion between its containers and standard GWpy objects:
- `TimeSeriesMatrix.to_dict()` -> `TimeSeriesDict`
- `TimeSeriesMatrix.to_list()` -> `TimeSeriesList`
- `FrequencySeriesMatrix` elements are standard `gwpy.frequencyseries.FrequencySeries`.

## Contributing
Contributions are welcome! Please open issues or submit PRs for new features or bug fixes.
