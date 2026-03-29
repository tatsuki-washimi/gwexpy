# GWexpy Features (For GWpy Users)

This document exhaustively lists the features extended and added to GWexpy on top of GWpy.
The features included here become available simply by importing `gwexpy`.

## 1. Enhancements to Existing GWpy Classes (Minor)

These are method additions and functional enhancements to base classes like `TimeSeries`, `FrequencySeries`, and `Spectrogram` for improved usability.

### Base Method Enhancements (TimeSeries / FrequencySeries / Spectrogram)

- **Phase & Angle**:
  - `.phase(unwrap=False)`: Calculate phase
  - `.angle(unwrap=False)`: Alias for `.phase()`
  - `.radian(unwrap=False)`: Phase in radians
  - `.degree(unwrap=False)`: Phase in degrees
  - Support for `np.unwrap`
- **Resampling, Interpolation, & Editing**:
  - `.interpolate(new_times, method='linear')`: Data interpolation
  - `.resample(rate)`: Enhanced resampling using `scipy` or `obspy` (Lanczos) backend
  - `.decimate(factor)`: Decimation
  - `.tail(n)`: Extract the last `n` samples
  - `.crop(start, end)`: (Supports flexible time specification via `gwexpy.time.to_gps`)
  - `.append(other)`: Enhanced concatenation
- **Preprocessing**:
  - `.impute(method=...)`: Missing value imputation (linear interpolation, ffill, bfill). Can specify constraints like `max_gap`.
  - `.standardize()`: Standardization (Z-score normalization)
  - `.whiten(method='pca'|'zca')`: Whitening. Model can be retrieved with `return_model=True`.
- **Peak Detection**:
  - `.find_peaks(...)`: Detect peaks in the signal (wrapper for `scipy.signal.find_peaks`)

### GPS Time Utilities

- Utility functions supporting vector operations and formatting of GPS times (available in `gwexpy.time`, `gwexpy.plot.gps`, etc.).

### Plotting

- **Datetime Axis**: Automatically formats the horizontal axis to datetime format during plotting.
- **Batch Plotting**: Simultaneous plotting of a spectrogram and its time-averaged ASD via `Spectrogram.plot_summary()`, etc.
- **Maps & Sky Maps**: Data plotting on maps using `gwexpy.plot.GeoMap` and `gwexpy.plot.SkyMap`.
- **PairPlot**: Pair plotting (scatter matrix / correlation map) of multivariate data via `gwexpy.plot.PairPlot`.

### Enhancements to Collection Classes (TimeSeriesList / TimeSeriesDict)

Batch processing capabilities have been extended to GWpy's `TimeSeriesList` and `TimeSeriesDict`.

- Batch signal processing: `.fft_time_all()`, `.resample_all(rate)`, `.filter_all(*args)`
- Batch slicing: `.sel_all(...)`, `.isel_all(...)`
- Broadcasting of arithmetic operations

## 2. Introduction of New Classes

New data structures for handling multi-channel data, matrix data, physical fields, etc.

### New Collections

Collection classes for frequency and time-frequency data, which are not standard in GWpy. They serve the same role as `TimeSeriesList` and `TimeSeriesDict` do for `TimeSeries`.

- `FrequencySeriesList`, `FrequencySeriesDict`
- `SpectrogramList`, `SpectrogramDict`

### Matrices (SeriesMatrix)

Multidimensional data structures that hold data as elements of a "matrix" (Row, Col, Axis...).

- `TimeSeriesMatrix`: (Rows, Cols, Time)
- `FrequencySeriesMatrix`: (Rows, Cols, Frequency)
- `SpectrogramMatrix`: (Rows, Cols, Time, Frequency)

**Features & Capabilities**:

- **Matrix Operations**: Element-wise matrix multiplication (`@`), inverse (`.inv()`), determinant (`.det()`), trace (`.trace()`), Schur complement (`.schur()`). Powerful for transfer function analysis of MIMO systems.
- **Statistical Processing**: Batch computation of statistics across the entire sensor array via row/column-wise mean/variance operations (e.g., `.mean(axis)`, `.std(axis)`).
- **Batch Processing**: Batch application of signal processing (filtering, resampling, whitening, etc.) to each element.

### Physical Fields (Fields)

Classes to handle 4-dimensional physical fields (Time/Frequency + 3D Space/Wavenumber).

- `ScalarField` / `VectorField` / `TensorField`

**Features & Capabilities**:

- **Domain Management**: Manages the state of the Time/Frequency axis (`axis0`) and Space/Wavenumber axis (`space_domain`), supporting mutual transformations (`.fft_time`, `.fft_space`).
- **Spatial Extraction**: Extract and interpolate time series data at arbitrary spatial coordinates via `.extract_points()`.
- **Vector & Tensor Analysis**: Physical operations like dot product, cross product, divergence, curl, and coordinate transformations (e.g., Cartesian coordinate system).
- **Simulation**: Generation of noise fields (isotropic noise, plane waves, etc.) in coordination with `gwexpy.noise.field`.

### Other

- `BifrequencyMap`: Map data with two frequency axes (e.g., STLTGram).

## 3. Advanced Analysis Methods (Single Channel)

### Noise Simulation (`gwexpy.noise`)

- `from_asd(asd, ...)`: Time series generation from ASD (Amplitude Spectral Density)
- `colored_noise(psd, ...)`: Generation of colored noise
- `field.simulate(...)`: Generation of noise fields
- `pygwinc` integration: Utilize interferometer noise budget models via `gwexpy.noise.gwinc_`.

### Time Series Analysis (TimeSeries)

- `.hilbert()`: Hilbert transform (Analytic signal)
- `.envelope()`: Envelope (`abs(hilbert)`)
- `.demodulate(f)`: Lock-in amplification & demodulation
- `.arima(order, ...)`: AR/MA/ARMA/ARIMA model analysis (wrapper for `statsmodels`, `pmdarima`)
- `.ar(p)`, `.ma(q)`, `.arma(p, q)`: Shortcuts for AR/MA/ARMA
- `.hurst()`: Calculation of the Hurst exponent
- `.skewness()`: Skewness
- `.kurtosis()`: Kurtosis

### Frequency Spectrum Analysis (FrequencySeries)

- `.differentiate()`, `.integrate()`: Differentiation/Integration in the frequency domain
- `.differentiate_time()`, `.integrate_time()`: Time differentiation/integration in the frequency domain (e.g., Displacement ↔ Velocity ↔ Acceleration conversions)
- `.group_delay()`: Calculation of group delay

### Time-Frequency Analysis (Spectrogram / Special Transforms)

In addition to GWpy's spectrogram and Q-transform, various time-frequency analysis methods are provided.

- `hht()`: Hilbert-Huang Transform (HHT). IMF decomposition via EMD (Empirical Mode Decomposition) and instantaneous frequency analysis.
- `stlt()`: Short-Time Laplace Transform (STLT). Analysis taking the damping constant $\sigma$ into account.
- `cepstrum()`: Cepstrum analysis (quefrency).
- `cwt()`: Continuous Wavelet Transform (wrapper for `pywt`).

### Statistical Estimation & Fitting

- **Fitting (`gwexpy.fitting`)**:
  - `FitResult`, `Model`: Chi-square fitting (real, complex) and MCMC analysis via iminuit.
  - Methods like `.fit()` are also mixed into `TimeSeries` and `FrequencySeries`.
- **Bootstrap**:
  - `bootstrap_spectrogram`: Bootstrap estimation of spectrograms.

## 4. Advanced Analysis Methods (Multi-Channel)

### Component Analysis (`gwexpy.timeseries.decomposition`)

- `PCA`: Principal Component Analysis
- `ICA`: Independent Component Analysis
- `ZCA`: Whitening (used as `.whiten(method='zca')`)

### Correlation & Statistical Analysis

- **Correlation Metrics**:
  - `dCor`: Distance Correlation
  - `MIC`: Maximal Information Coefficient
  - `Pearson`, `Kendall`: Standard correlation coefficients
- **Bruco**:
  - Noise source identification tool via multi-channel coherence search (`gwexpy.analysis.bruco`).

### System Analysis & Control

- **MIMO / State Space Models**:
  - `python-control` integration: Mutual conversion with control system design libraries via `TimeSeries.from_control(response)`, `to_control_frd`, etc.
  - MIMO transfer function & time series analysis using matrix classes (`SeriesMatrix`).
- **Noise Injection Test Analysis**:
  - Experimental data analysis tools for estimating transfer functions and calculating coupling coefficients (`gwexpy.analysis.response`).
- **Matrix Operations**:
  - Basic operations like inverse (`.inv()`), determinant (`.det()`), trace (`.trace()`), and Schur complement (`.schur()`).

## 5. Interoperability with External Tools

Mutual conversion functionalities with external library objects (`.to_*()` / `.from_*()`).

- **Deep Learning / Array Libraries**:
  - `PyTorch` (`Tensor`, `Dataset`/`DataLoader`): `to_torch`, `to_torch_dataset`, etc.
  - `TensorFlow` (`Tensor`): `to_tf`
  - `JAX` (`Array`): `to_jax`
  - `CuPy` (`ndarray`): `to_cupy`
  - `Dask` (`dask.array`): `to_dask`
  - `Zarr`: `to_zarr` / `from_zarr`
- **Data Structures**:
  - `pandas` (`Series`/`DataFrame`): `to_pandas`, `to_pandas_dataframe`
  - `xarray` (`DataArray`): `to_xarray`
  - `polars` (`Series`/`DataFrame`): `to_polars_series`
  - `SQLite`, `JSON`: Lightweight save/load
- **Domain Specific**:
  - `Obspy` (Seismology): `Trace`, `Stream`
  - `MNE` (EEG/MEG): `Raw`, `RawArray` (`to_mne`)
  - `Neo` (Electrophysiology): `AnalogSignal` (`to_neo`)
  - `Librosa` / `Pydub` (Audio): Audio data conversion
  - `SimPEG` (Geophysics): `Data` object
  - `python-control`: `FrequencyResponseData` (FRD), etc.
  - `ROOT` (CERN): `TGraph`, `TH1D`, `TH2D`, `TMultiGraph`
  - `Specutils`: `Spectrum1D` (`to_specutils`)
  - `Pyspeckit`: `Spectrum` (`to_pyspeckit`)
  - `Astropy`: `TimeSeries` (`to_astropy_timeseries`)
  - `Quantities`: `Quantity` (`to_quantity`)

### Data Format I/O Extensions

In addition to the formats natively supported by GWpy, the following are supported (or extended):

- `ATS` (Metronix MT data)
- `GBD` (GRAPHTEC datalogger)
  - Analog channels are converted from `count -> V` using the amp range (e.g., `20mV`) in the header.
  - `Alarm` / `AlarmOut` / `Pulse*` / `Logic*` are handled as digital/status, values normalized to 0/1, and unit set to dimensionless.
  - You can overwrite the channel names to be treated as digital by specifying `digital_channels=[...]` in `TimeSeriesDict.read(..., format="gbd")`.
- `TDMS` (LabVIEW / NI)
- `MiniSEED`, `SAC`, `GSE2` (Seismic data, Obspy integration)
- `WIN` (Seismic data)
  - Implemented decoding for 0.5-byte (4-bit difference) / 3-byte differences, taking into account known issues and caveats in ObsPy.
  - Implementation in gwexpy: `gwexpy/timeseries/io/win.py` (handling lower nibble for 4-bit differences, skipping trailing nibble for odd differences, decoding signed 24-bit differences, etc.).
- `DTTXML` (LIGO Diagnostic Test Tools)
- `SDB` (Davis weather station data)
- `Midas` (PSI/DAQ format)
- `Parquet`, `Feather`, `Pickle` (pandas integration)
- `WAV` (Audio, extended support)
- `ROOT` file (.root)

## Notes on Pickle / shelve

> [!WARNING]
> Do not load untrusted data using `pickle` / `shelve`. Arbitrary code execution can occur during loading.

To prioritize portability, gwexpy's pickle implementation is designed to **return GWpy types when unpickled**
(meaning data can be restored if gwpy is present, even without gwexpy on the reading side).

```python
import pickle
import numpy as np
from gwexpy.timeseries import TimeSeries

ts = TimeSeries(np.arange(10.0), sample_rate=1.0, t0=0, unit="m")
obj = pickle.loads(pickle.dumps(ts))
# obj is gwpy.timeseries.TimeSeries
```

Compatibility notes:

- `TimeSeries` / `FrequencySeries` / `Spectrogram` -> Restores as GWpy objects when unpickled
- `TimeSeriesDict` / `TimeSeriesList` -> Restores as GWpy collections when unpickled
- `FrequencySeriesDict/List` / `SpectrogramDict/List` -> Restores as Python built-in `dict` / `list` (containing GWpy objects) when unpickled
- Unique gwexpy types like the `Matrix` and `Field` families are exempt from this portability contract

What is preserved (best effort):

- Numerical data (`.value`)
- Axis information (`times` / `frequencies`)
- Metadata generally handled by GWpy (`unit`, `name`, `channel`, `epoch`)

What is NOT preserved:

- Internal attributes specific to gwexpy (e.g., `_gwex_*`) and behaviors added by gwexpy

## 6. GUI Applications

- **pyaggui (`gwexpy.gui.pyaggui`)**:
  - A PyQt5-based GUI application modeled after DTT diaggui (C++, ROOT).
