# TimeSeries

<!-- reference-summary:start -->

**Stability:** Stable

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
- {ref}`Transient FFT validation <validated-en-transient-fft>` - Amplitude-spectrum assumptions for transient FFT mode
- {ref}`ARIMA forecast timing validation <validated-en-arima-forecast>` - GPS timestamp assumptions for forecast extension
- {ref}`MCMC / GLS likelihood validation <validated-en-mcmc-gls>` - Likelihood assumptions when fitting time-series data
- [FFT_Conventions](FFT_Conventions.md)
- [Prerequisites and Conventions](../user_guide/prerequisites_and_conventions.md)

## Related Tutorials

- [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_en.md)
- [Signal Extraction](../user_guide/tutorials/case_signal_extraction.ipynb)
- [Advanced ARIMA](../user_guide/tutorials/advanced_arima.ipynb)

## API Reference

The detailed generated API continues below on this page.

<!-- reference-summary:end -->


**Inherits from:** [`gwpy.timeseries.TimeSeries`](https://gwpy.readthedocs.io/en/latest/reference/gwpy.timeseries.TimeSeries/)

Extended TimeSeries with full gwexpy functionality.

## Physical Context

`TimeSeries` represents a **single-channel time-domain signal**. Use it when each sample corresponds to one physical instant of a measured or simulated quantity such as gravitational-wave strain, ground velocity, voltage, control error, or microphone output.

- **Time-axis semantics**: `t0`, `dt`, `sample_rate`, and `times` define the physical timing of the segment. In particular, `fetch()` and `fetch_open_data()` results are usually passed downstream with GPS timing intact for segment-based or event-synchronized analysis.
- **Unit semantics**: `unit` is not decorative metadata. It affects how filtering, differentiation/integration, fitting, and frequency-domain interpretation are understood. Explicit units such as `strain`, `m/s`, or `V` make later `1/Hz`-style spectra easier to interpret consistently.
- **Regular sampling assumption**: `TimeSeries` assumes regularly sampled data. For irregular events or interval tables, use `SegmentTable`/table objects instead. For multi-channel workflows, prefer `TimeSeriesMatrix` or `TimeSeriesDict`.

## Analysis Notes

### Before FFT or PSD

`fft()`, `psd()`, `asd()`, and `spectrogram()` map a time-domain signal into frequency-domain summaries. The critical assumptions are not just the samples themselves, but also the **window length, overlap, window function, and averaging convention**.

- use `psd()` or `asd()` when you want a representative stationary-noise summary
- use `spectrogram()` or `q_transform()` when transient bursts or chirps matter
- check [FFT_Conventions](FFT_Conventions.md) and the validation pages before comparing amplitudes across methods

### Meaning of preprocessing

`detrend()`, `highpass()`, `whiten()`, `standardize()`, and `impute()` are not only cosmetic transformations. They decide **which physical components are preserved and which systematic effects are suppressed**.

- `detrend()` / `highpass()` remove low-frequency drift or offsets
- `whiten()` is useful before detection-oriented visualization, correlation analysis, or broad-band comparison
- `impute()` fills gaps, but the filled region should not be interpreted as direct physical measurement

### Common misreadings

1. comparing samples without checking `sample_rate` or `t0`
2. treating a preprocessed series as if it were the original raw physical quantity
3. comparing FFT-based outputs without checking units or amplitude conventions
4. expecting a single-channel series to encode multi-channel causality or spatial structure by itself

## Where to go next

- conventions for moving from time to frequency: [FFT_Conventions](FFT_Conventions.md)
- detector data access and direct I/O: [I/O Formats](../user_guide/io_formats.md)
- migration context from GWpy: [GWpy Migration Guide](../user_guide/gwexpy_for_gwpy_users_en.md)
- time-frequency method trade-offs: [Time-Frequency Comparison Guide](../user_guide/tutorials/time_frequency_comparison.md)
- forecasting workflows: [Advanced ARIMA](../user_guide/tutorials/advanced_arima.ipynb)

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
