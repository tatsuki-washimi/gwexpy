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

Generate multiple channels of time series data from an experimental noise model:

```python
import numpy as np
from gwexpy.timeseries import TimeSeriesDict
from gwexpy.signal.noise import PowerLawNoise

# Setup noise model (1/f noise: beta=1)
noise_model = PowerLawNoise(beta=1, dt=1/1024)

# Generate multi-channel time series
tsd = TimeSeriesDict()
tsd["H1:STRAIN"] = noise_model.generate(duration=64)  # Hanford
tsd["L1:STRAIN"] = noise_model.generate(duration=64)  # Livingston

# Plot
plot = tsd.plot()
plot.show()
```

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
