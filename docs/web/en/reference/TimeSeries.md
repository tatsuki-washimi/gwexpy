# TimeSeries

<!-- reference-summary:start -->

## What it is

Use `TimeSeries` for a single regularly sampled time-domain channel with GWexpy signal-processing, modeling, and interoperability extensions.

## Representative Signatures

```python
TimeSeries(data, unit=None, t0=None, dt=None, sample_rate=None, times=None, ...)
TimeSeries.fft(fftlength=None, overlap=0, window="hann", ...)
```

## Minimal Example

```python
from gwexpy.timeseries import TimeSeries
import numpy as np

ts = TimeSeries(np.random.randn(1024), sample_rate=1024, unit="strain")
psd = ts.psd(fftlength=1.0)
```

## Related Theory

- [Physics Models](../user_guide/physics_models.md)
- [Validated Algorithms](../user_guide/validated_algorithms.md)
- [FFT_Conventions](FFT_Conventions.md)

## Related Tutorials

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_en.md)
- [Tutorial Index](../user_guide/tutorials/index.rst)
- [Getting Started](../user_guide/getting_started.md)

## API Reference

The detailed generated API continues below on this page.

<!-- reference-summary:end -->


**Inherits from:** `gwpy.timeseries.TimeSeries`

Extended TimeSeries with full gwexpy functionality.

## Key Extensions

### Statistics and Correlation

- **`correlation(other, method="pearson", ...)`**
  Compute correlation with another TimeSeries.
  Methods: `"pearson"`, `"kendall"`, `"mic"`, `"distance"`.
- **`partial_correlation(other, controls=None, ...)`**
  Compute partial correlation controlling for third-party variables.
- **`fastmi(other, grid_size=128)`**
  Compute mutual information using the FastMI (FFT-based) estimator.
- **`granger_causality(other, maxlag=5)`**
  Test for Granger causality between series.

### Signal Processing

- **`hilbert()` / `envelope()`**
  Compute the analytic signal and its amplitude envelope.
- **`mix_down(f0)`**
  Demodulate the signal at a specific carrier frequency.
- **`fft(mode="steady"|"transient", ...)`**
  Enhanced FFT with zero-padding and window management options.

### Modeling and Preprocessing

- **`arima(order=(p,d,q))`**
  Fit an ARIMA time-series model.
- **`impute(method="interpolate")`**
  Handle missing values (NaNs) in the data.
- **`standardize(method="zscore")`**
  Rescale data to zero mean and unit variance.

## Examples

```python
from gwexpy.timeseries import TimeSeries
ts = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)

# Compute nonlinear correlation
mic_score = ts.correlation(other_ts, method="mic")

# Standardize and compute envelope
env = ts.standardize().envelope()
```

## Pickle / shelve portability

:::{warning}
Never unpickle data from untrusted sources. ``pickle``/``shelve`` can execute
arbitrary code on load.

:::
gwexpy pickling prioritizes portability: unpickling returns **GWpy types**
so that loading does not require gwexpy to be installed.
