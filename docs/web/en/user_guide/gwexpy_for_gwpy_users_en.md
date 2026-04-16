# Migration Guide for GWpy Users

GWexpy inherits its core classes (like TimeSeries) from GWpy while significantly enhancing usability for multi-channel analysis and signal processing.
This guide introduces key differences for GWpy users and demonstrates how to simplify your code with GWexpy's new features.

| Feature / Goal | GWpy Style (Traditional) | GWexpy Style (Recommended) |
| --- | --- | --- |
| Managing Multiple Channels | `TimeSeriesDict` | `TimeSeriesMatrix` (Stable) |
| Batch Analysis (ASD/CSD) | Loops or manual Dict manipulation | `.asd()`, `.csd()` (Stable) |
| Portability & Pickle | May not be Pickle-compatible | High portability (Pickle compatible) |
| Advanced Signal Processing | Manual SciPy calls | Built-in `.hht()`, `.arima()`, etc. (Experimental) |
| Spatial/Multi-dimensional Data | No specific classes available | `ScalarField` (Experimental) |

## 1. Channel Management and Batch Analysis

While GWpy uses `TimeSeriesDict` to manage multiple channels, GWexpy recommends `TimeSeriesMatrix`, which treats channel information as a "matrix axis." This allows for batch spectral analysis without writing loops.

### Example: Calculating CSD (Cross-Spectral Density)

**GWpy Style:**

```python
from gwpy.timeseries import TimeSeriesDict
tsd = TimeSeriesDict.read(cache, channels)
# Loops or nested logic required for all channel pairs
```

**GWexpy Style (Stable):**

```python
from gwexpy.timeseries import TimeSeriesDict
tsd = TimeSeriesDict.read(cache, channels)

# Convert to a matrix
matrix = tsd.to_matrix()

# Batch calculate CSD for all channel pairs at once
csm = matrix.csd(fftlength=4)
csm.plot().show()
```

## 2. Seamless Signal Processing Integration

In addition to base GWpy methods, advanced algorithms from SciPy, Statsmodels, and other libraries are mixed directly into the base classes.

### Selected Extension Methods

* **Fitting (Stable)**: The `.fit()` method enables least-squares and MCMC analysis via `iminuit` directly on your data objects.
* **Peak Detection (Stable)**: `.find_peaks()` makes it easy to identify pulse trains or resonances.
* **Instantaneous Frequency (Experimental)**: Simply call `.hht()` for Hilbert-Huang Transform analysis.
* **Statistical Forecasting (Experimental)**: `.arima()` provides forecasting and noise subtraction based on signal autocorrelation.

## 3. Extended I/O Support

In addition to GWpy's standard support for `gwf`, `hdf5`, and `ascii`, GWexpy adds support for formats frequently used in experiments:

* **GBD (GraphTec)**: Automates digital channel normalization and count-to-voltage conversion based on range headers.
* **TDMS (LabVIEW)**: Direct reading of data recorded by National Instruments hardware.
* **WIN (Seis)**: Decoding of the WIN format, the standard for Japanese seismic networks.
* **Zarr / Parquet**: High-speed cloud/disk I/O for large-scale datasets.

## 4. Portability and Compatibility (Pickle)

GWexpy focuses on the "sharing of analysis results." 
When saving objects via `Pickle`, GWexpy uses a **Transparent Pickle** design (Stable). This ensures that even if the recipient doesn't have GWexpy installed, the objects can be restored as base GWpy objects, provided GWpy is available.

:::{important}
Always avoid loading Pickle data from untrusted sources.
:::

## 5. High-dimensional Data (Field API)

For handling spatial distributions (e.g., sensor arrays), you can use `ScalarField`, which extends `TimeSeries`.

* **Domain Transformation (Experimental)**: Use `.fft_space()` for 2D transformations between time-space and frequency-wavenumber domains.
* **Spatial Extraction**: Extract a time series at any arbitrary coordinate, including interpolation, in a single line.

---

## Next Steps

* [Quickstart](quickstart.md) - Run some real code.
* [Getting Started](getting_started.md) - Check the learning roadmap.
* [Reference](../reference/index.rst) - Explore all methods for each class.
