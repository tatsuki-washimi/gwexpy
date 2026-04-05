# Quickstart

Create and analyze time series data from multiple channels.

:::{note}
For a more detailed learning path, see [getting_started](getting_started.md).
:::

## Migration from GWpy

To use GWpy code with GWexpy, simply replace the imports:

```python
# GWpy (legacy)
# from gwpy.timeseries import TimeSeries
# from gwpy.frequencyseries import FrequencySeries

# GWexpy (recommended)
from gwexpy.timeseries import TimeSeries, TimeSeriesDict, TimeSeriesMatrix
from gwexpy.frequencyseries import FrequencySeriesMatrix
```

## Generate and Plot Multi-channel Time Series

Generate multiple channels of time series data:

```python
import numpy as np
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

sample_rate = 1024
duration = 64

# Generate white Gaussian noise time series
tsd = TimeSeriesDict({
    "H1:STRAIN": TimeSeries(np.random.randn(sample_rate * duration), dt=1/sample_rate, t0=0),
    "L1:STRAIN": TimeSeries(np.random.randn(sample_rate * duration), dt=1/sample_rate, t0=0),
})

# Plot
plot = tsd.plot()
plot.show()
```

:::{note}
For colored noise (pink, red, etc.), see [Noise Model Guide](tutorials/intro_noise).
:::

## Batch CSD Conversion: Time Series to Frequency Matrix

Transform multiple channels from time domain to frequency domain, computing cross-spectral densities:

```python
# Convert TimeSeriesDict to matrix
ts_matrix = tsd.to_matrix()

# Compute CSD (Welch's method, 50% overlap)
csm = ts_matrix.csd(
    fftlength=4,
    overlap=0.5,
    window='hann'
)

# Plot frequency matrix
freq_plot = csm.plot()
freq_plot.show()

# Analyze specific frequency bins
print(f"Frequency range: {csm.frequencies[0]:.1f} - {csm.frequencies[-1]:.1f} Hz")
print(f"H1-L1 cross-spectrum (10 Hz): {csm['H1:STRAIN', 'L1:STRAIN'].interpolate(10).value:.2e}")
```
