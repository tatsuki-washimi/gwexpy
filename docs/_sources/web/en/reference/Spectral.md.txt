# Spectral Estimation

gwexpy enforces strict unit conventions for spectral density and power spectrum estimation.

## Unit Semantics

- **PSD / ASD (Density)**:
  - Default `method='welch'` (or others) returns a **Power Spectral Density**.
  - **Unit**: $Unit^2 / Hz$ (e.g., $V^2 / Hz$).
- **Power Spectrum**:
  - When `scaling='spectrum'` is specified.
  - **Unit**: $Unit^2$ (e.g., $V^2$).
  - This represents the power in each bin, not normalized by bin width.

## Backends

Supported backends (`scipy`, `lal`, `pycbc`) are wrapped to ensure these unit contracts are met regardless of the underlying implementation.

## Key Functions

| Function                            | Description                                             |
| ----------------------------------- | ------------------------------------------------------- |
| `estimate_psd(ts, ...)`             | Wrapper for PSD estimation with NaN rejection           |
| `bootstrap_spectrogram(sgm, ...)`   | Bootstrap resampling for robust ASD/PSD with error bars |
| `calculate_correlation_factor(...)` | Variance inflation factor for Welch overlap correction  |

## Usage Example

```python
from gwexpy.timeseries import TimeSeries
from gwexpy.spectral import estimate_psd

# Create time series data
ts = TimeSeries(data, sample_rate=1024, unit='V')

# Estimate PSD (density normalization)
psd = estimate_psd(ts, fftlength=1.0)
print(psd.unit)  # V^2 / Hz

# Estimate power spectrum
ps = estimate_psd(ts, fftlength=1.0, scaling='spectrum')
print(ps.unit)  # V^2
```

## Notes

- `estimate_psd()` rejects NaN samples because FFT-based averaging propagates NaNs and invalidates the normalization. Callers must pre-clean data.
- `fftlength` must not exceed the data duration; otherwise a `ValueError` is raised.
