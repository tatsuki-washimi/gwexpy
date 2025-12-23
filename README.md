# gwexpy: GWpy Expansions for Experiments

**gwexpy** is an (unofficial) extension library for **GWpy**, designed to facilitate advanced time-series analysis, matrix operations, and signal processing for experimental physics and gravitational wave data analysis. It builds upon GWpy's core data objects (`TimeSeries`, `FrequencySeries`) and introduces high-level containers and methods for multivariate analysis, vectorized time conversion, auto Series creation, and interoperability with the broader Python scientific ecosystem.

## Key Features

### 1. Advanced Containers
- **`TimeSeriesMatrix`**: A 3D container `(channels, 1, time)` for handling multi-channel time-series data with shared time axes. Supports element-wise operations, slicing, PCA/ICA decomposition, and bulk processing.
- **`FrequencySeriesMatrix`**: A matrix container for frequency-domain data, ideal for representing Transfer Functions, CSD, and Coherence matrices.
- **`TimePlaneTransform`**: A container for 2D time-frequency representations (like STLT output), supporting interpolation along the time axis.
- **Auto-expanding `SeriesMatrix`**: Assigning to new `(row_key, col_key)` adds rows/cols automatically (e.g. `mat["H1", "Strain"] = ts`).

### 2. Spectral Analysis Extensions
- **Spectral Matrices**: Easily compute Cross-Spectral Density (CSD) and Coherence matrices for entire collections.
  - `TimeSeriesDict.csd_matrix()` / `coherence_matrix()`
  - `TimeSeriesList.csd_matrix()` / `coherence_matrix()`
- **Extended FFT & Transfer Functions**:
  - `TimeSeries.fft(mode="transient")`: Transient-friendly FFT with padding options (zero, reflect).
  - `TimeSeries.transfer_function()`: Supports 'gwpy' (Welch), 'fft' (Direct ratio), and 'auto' modes.

### 3. Signal Transforms & Peak Finding
- **Short-Time Local Transform (STLT)**: `ts.stlt()` creates a time-frequency representation tailored for local stationarity analysis.
- **Peak Finding**: `ts.find_peaks()` wraps `scipy.signal.find_peaks` with unit-aware parameters (height, distance, width in physical units).
- **HHT**: Hilbert-Huang Transform support (EMD + Hilbert Spectrum).
- **Other Transforms**: Laplace Transform, DCT, Cepstrum, CWT.

### 4. Preprocessing & Decomposition
- **Alignment**: `align_timeseries_collection` to synchronize varying start times and sample rates.
- **Imputation**: `impute_timeseries` / `TimeSeries.impute()` for handling gaps.
- **Standardization & Whitening**: transformation methods for `TimeSeries` and `TimeSeriesMatrix`.
- **Decomposition**: Integrated **PCA** and **ICA** methods for `TimeSeriesMatrix`.

### 5. Advanced Statistics
- **ARIMA**: Forecasting and residual analysis.
- **Hurst Exponent**: Global and local Hurst exponent estimation.

### 6. Time Utilities & Auto Series
- **Vectorized time conversion**: `gwexpy.time.to_gps/from_gps/tconvert` handle numpy arrays, pandas, ObsPy, and string arrays.
- **Axis → Series helper**: `as_series` converts a 1D axis (`gwpy.types.index.Index` or `astropy.units.Quantity`) into a `TimeSeries` or `FrequencySeries` (identity mapping). Angular frequency inputs (`rad/s`) are treated as frequency axes and converted to Hz for the x-axis.

### 7. Noise & Physics Models
- **`gwexpy.noise`**: Easily fetch standard noise models.
  - `from_pygwinc()`: Detector noise budgets (aLIGO, KAGRA, etc.).
  - `from_obspy()`: Earth noise models (NLNM/NHNM) and Infrasound models with automatic integration and unit conversion (e.g., Acceleration to Displacement).

---

## Installation

```bash
# Standard installation
pip install .

# Install with ALL optional dependencies (Recommended)
pip install ".[all]"

# Install specific features
pip install ".[interop]"     # All interoperability features (torch, jax, etc.)
pip install ".[geophysics]"  # Obspy, MTh5, wintools, etc.
pip install ".[analysis]"    # EMD, Wavelet, Hurst, etc.
pip install ".[gw]"          # pygwinc, dttxml
pip install ".[audio]"       # Librosa, Pydub, Torchaudio
```
### Dependencies

#### Core Dependencies (Required)
These are automatically installed with `pip install .`.

| Package | Purpose |
| :--- | :--- |
| `gwpy` | Base library for gravitational wave and time-series data |
| `astropy` | Physical units, time conversion, and coordinate systems |
| `numpy` | N-dimensional array processing |
| `pandas` | Time-indexed dataframes and table operations |
| `scipy` | Signal processing, peak finding, and interpolation |
| `matplotlib` | Plotting and visualization |

#### Optional Dependencies
Required only for specific submodules or interpolation features.

| Extra Name | Packages | Features |
| :--- | :--- | :--- |
| `[analysis]` | `pyemd`, `hurst`, `pywt`, `librosa`, `obspy` | HHT/EMD, Hurst exponent, Wavelets, advanced audio/seismic signal processing |
| `[stats]` | `statsmodels`, `scikit-learn`, `bottleneck` | ARIMA, ICA/PCA decomposition, fast rolling statistics |
| `[gw]` | `gwinc`, `dttxml` | `gwexpy.noise`, GW noise budgets & `dttxml` reading |
| `[geophysics]` | `obspy`, `mth5`, `mt_metadata`, `mtpy`, `wintools`, `win2ndarray` | Seismic/EM data I/O, noise models |
| `[bio]` | `mne`, `neo` | EEG/MEG & Electrophysiology data I/O |
| `[gpu]` | `cupy`, `torch`, `tensorflow`, `jax` | GPU‑accelerated array support & Deep Learning |
| `[audio]` | `librosa`, `pydub`, `torchaudio` | Audio format export/import (mp3, wav) |
| `[data]` | `xarray`, `h5py`, `netCDF4` | HDF5, netCDF and XArray data structures |
| `[control]` | `control` | Control system analysis (`to_control_frd`) |
| `[interop]` | `torch`, `jax`, `dask`, `polars`, etc. | Bulk install of all interoperability extras |
| `[polars]` | `polars` | Fast DataFrames support (`to_polars`) |
| `[dask]` | `dask` | Parallel array processing |
| `[zarr]` | `zarr` | Chunked, compressed, binary storage |
| `[dev]` | `pytest`, `pytest-cov`, `ruff`, `mypy` | Tooling for development and testing |

To install everything at once, use: `pip install ".[all]"`.


---

## Usage Examples

### 1. Time Conversion & Auto Series
```python
import numpy as np
from astropy import units as u
import pandas as pd
from gwexpy import as_series
from gwexpy.time import to_gps, from_gps

times = pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 00:00:01"])
gps = to_gps(times)  # vectorized -> numpy array
iso = from_gps(gps)  # back to time strings

# Convert axes to Series (values are the axis values, optionally converted)
ts = as_series((1419724818 + np.arange(10)) * u.s, unit="h")  # TimeSeries (values in hours)
fs = as_series(np.arange(5) * u.Hz, unit="mHz")  # FrequencySeries (values in mHz)
```

### 2. Interoperability (Deep Learning & Big Data)
gwexpy provides seamless conversion to/from major data science frameworks:

```python
from gwexpy.interop import to_torch, to_tf, to_jax, to_cupy, to_dask, to_zarr
ts = ... # TimeSeries

# Deep Learning Frameworks
tensor_pt = to_torch(ts)      # PyTorch Tensor
tensor_tf = to_tf(ts)         # TensorFlow Tensor
tensor_jax = to_jax(ts)       # JAX Array
tensor_cupy = to_cupy(ts)     # CuPy Array
df_pl = ts.to_polars()        # Polars DataFrame

# Big Data & Storage
dask_arr = to_dask(ts, chunks=1000)  # Dask Array
to_zarr(ts, store="data.zarr", path="my_array", overwrite=True)  # Save to Zarr

# Audio & Other Domains
# Supports: Librosa (standard numpy), Pydub (AudioSegment), ObsPy (Trace), MNE, Neo, ROOT
from gwexpy.interop import to_pydub, to_obspy_trace, write_root_file
audio_seg = to_pydub(ts)
trace = to_obspy_trace(ts)

# CERN ROOT Interoperability
# Converts to ROOT TGraph/TMultiGraph/TH2D and saves to .root files
tsd.write("data.root")  # Automatic .root detection for Dictionary/List
mg = tsd.to_tmultigraph(name="comparison")
```

### 3. TimeSeriesMatrix & Decomposition
```python
import numpy as np
from gwexpy.timeseries import TimeSeries, TimeSeriesMatrix
from astropy import units as u

# Create a TimeSeriesMatrix from a list of Series
t = np.linspace(0, 10, 1000) * u.s
s1 = TimeSeries(np.sin(2*np.pi*1*t), times=t, name="Ch1")
s2 = TimeSeries(np.cos(2*np.pi*1*t) + 0.5*np.random.randn(1000), times=t, name="Ch2")
mat = TimeSeriesMatrix.from_list([s1, s2])

# Preprocessing & Decomposition
mat_clean = mat.impute(method="interpolate").standardize()
scores, model = mat_clean.pca(n_components=2, return_model=True)
sources = mat_clean.ica(n_components=2)
```

### 4. Peak Finding
```python
# Find peaks with physical unit constraints
peaks, props = s2.find_peaks(
    height=0.5,           # Minimum height (unit of data)
    distance=0.1 * u.s,   # Minimum distance between peaks
    width=0.05 * u.s      # Minimum width
)
peaks.plot(style='o')
```

### 5. Signal Transforms (STLT & HHT)
```python
# Short-Time Local Transform
stlt_res = s1.stlt(window='2s', stride='0.5s') 
# Returns TimePlaneTransform (interpolate-able)
val_at_5s = stlt_res.at_time(5*u.s, method="linear")

# Hilbert-Huang Transform (requires EMD-signal)
try:
    hht_spec = s1.hht(emd_method="eemd", output="spectrogram")
    hht_spec.plot()
except ImportError:
    print("Install EMD-signal for HHT")
```

### 6. Series Fitting
`gwexpy` provides a powerful fitting API based on `iminuit`. Note that the `.fit()` method on `TimeSeries` and `FrequencySeries` is **opt-in** to avoid automatic modification of `gwpy` classes.

Supported built-in models: `gaussian` (or `gaus`), `exponential` (or `exp`), `landau`, `power_law`, `damped_oscillation`, and polynomials (e.g., `pol1`, `pol2`).

```python
from gwexpy.fitting import enable_fitting_monkeypatch
enable_fitting_monkeypatch()

# Fit a TimeSeries to a model
# result = ts.fit('gaussian', p0={'A': 10, 'mu': 5, 'sigma': 1})
# result = ts.fit('damped_oscillation', p0={'A': 1, 'tau': 0.1, 'f': 50})
# print(result.params)
# result.plot()
```

### 7. Torch Dataset Helper
```python
from gwexpy.interop import to_torch_dataset, to_torch_dataloader

ds = to_torch_dataset(s1, window=256, stride=128)
loader = to_torch_dataloader(ds, batch_size=8, shuffle=True)
for batch in loader:
    # batch: (B, C, window)
    pass
```

## Testing
```bash
python -m pytest
```
Some tests are skipped if optional dependencies or network access are unavailable.

## Contributing
Contributions are welcome! Please open issues or submit PRs for new features or bug fixes.
