# gwexpy: GWpy Expansions for Experiments

**gwexpy** is an (unofficial) extension library for [**GWpy**](https://gwpy.github.io/), designed to facilitate advanced time-series analysis, matrix operations, and signal processing for experimental physics and gravitational wave data analysis. It builds upon [GWpy](https://gwpy.github.io/)'s core data objects (`TimeSeries`, `FrequencySeries`) and introduces high-level containers and methods for multivariate analysis, vectorized time conversion, auto Series creation, and interoperability with the broader Python scientific ecosystem.

> ⚠️ **Note on GUI Module**: The `gwexpy.gui` module (pyaggui) is **experimental and under active development**. It is **not stable** and may change significantly. See `gwexpy/gui/README.md` for details.

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
pip install ".[interop]"     # All interoperability features (data + ML frameworks)
pip install ".[fitting]"     # Fitting & MCMC (iminuit, emcee, corner)
pip install ".[geophysics]"  # Obspy, MTh5, wintools, etc.
pip install ".[analysis]"    # EMD, Wavelet, Hurst, etc.
pip install ".[gw]"          # pygwinc, dttxml
pip install ".[audio]"       # Librosa, Pydub, Torchaudio
```

### Dependencies

#### 📦 依存ライブラリ一覧

このリポジトリでは、以下の外部 Python ライブラリを使用しています。

##### ✅ 必須ライブラリ（基本機能に必要）

- [`gwpy`](https://gwpy.github.io/), [`astropy`](https://www.astropy.org/), [`numpy`](https://numpy.org/), [`pandas`](https://pandas.pydata.org/), [`scipy`](https://scipy.org/), [`matplotlib`](https://matplotlib.org/)

##### 🔄 限定的に必要なライブラリ（特定機能に必要）

以下は特定の解析・可視化・IO変換機能を使う場合に限り必要です。 GUI・ROOT連携・Tensor変換等は **opt-in** 構成です。

| カテゴリ | パッケージ名 | 機能 |
| :--- | :--- | :--- |
| 統計・信号解析系 | [`scikit-learn`](https://scikit-learn.org/), [`statsmodels`](https://www.statsmodels.org/), [`pmdarima`](https://alkaline-ml.com/pmdarima/), [`minepy`](https://minepy.readthedocs.io/), [`dcor`](https://dcor.readthedocs.io/) | ARIMA, ICA/PCA, 相関解析, ローリング統計 |
| ベイズ推定・MCMC | [`iminuit`](https://iminuit.readthedocs.io/), [`emcee`](https://emcee.readthedocs.io/), [`corner`](https://corner.readthedocs.io/) | 高度なフィッティング, MCMC, コーナープロット |
| データ形式変換 | [`h5py`](https://www.h5py.org/), [`netCDF4`](https://unidata.github.io/netcdf4-python/), [`xarray`](https://xarray.pydata.org/), [`zarr`](https://zarr.readthedocs.io/), [`polars`](https://www.pola.rs/), [`dask`](https://www.dask.org/), [`torch`](https://pytorch.org/), [`tensorflow`](https://www.tensorflow.org/), [`jax`](https://github.com/google/jax) | 各種データフォーマット(HDF5/Zarr等), 深層学習フレームワーク変換 |
| 入出力変換 | [`neo`](https://neuralensemble.org/neo/), [`mne`](https://mne.tools/), [`control`](https://python-control.readthedocs.io/), [`gwinc`](https://pypi.org/project/gwinc/), [`dttxml`](https://pypi.org/project/dttxml/), [`pydub`](http://pydub.com/), [`ROOT`](https://root.cern/), [`specutils`](https://specutils.readthedocs.io/), [`pyspeckit`](https://pyspeckit.readthedocs.io/) | 生体データ, 制御解析, 音声, ROOT連携, スペクトル解析 |
| 地学・物理解析 | [`obspy`](https://docs.obspy.org/), [`mth5`](https://mth5.readthedocs.io/), [`mtpy`](https://mtpy.readthedocs.io/), [`hurst`](https://github.com/Mottl/hurst), [`hurst_exponent`](https://pypi.org/project/hurst-exponent/), [`exp_hurst`](https://pypi.org/project/exp-hurst/) | 地震・地磁気データ, Hurst指数解析 |
| GUI | [`PyQt5`](https://www.riverbankcomputing.com/software/pyqt/) | 実験的なGUIツール (`gwexpy.gui`) |

特定機能を使わない場合は、これらのパッケージは不要です。 `pip install ".[all]"` で全て一括インストール可能です。

#### Installation Extras

Required for specific submodules or interpolation features.

| Extra Name | Packages |
| :--- | :--- |
| `[analysis]` | `pyemd`, `hurst`, `hurst-exponent`, `exp-hurst`, `pywt`, `librosa`, `obspy` |
| `[stats]` | `statsmodels`, `scikit-learn`, `bottleneck`, `minepy`, `dcor`, `pmdarima` |
| `[gw]` | `gwinc`, `dttxml` |
| `[geophysics]` | `obspy`, `mth5`, `mt_metadata`, `mtpy`, `wintools`, `win2ndarray` |
| `[bio]` | `mne`, `neo` |
| `[fitting]` | `iminuit`, `emcee`, `corner` |
| `[gpu]` | `cupy`, `torch`, `tensorflow`, `jax` |
| `[audio]` | `librosa`, `pydub`, `torchaudio` |
| `[data]` | `xarray`, `h5py`, `netCDF4` |
| `[control]` | `control` |
| `[interop]` | `torch`, `tensorflow`, `jax`, `dask`, `zarr`, `polars`, `xarray`, `h5py`, `netCDF4`, `obspy`, `mne`, `neo`, `control`, `cupy`, `librosa`, `pydub`, `torchaudio`, `specutils`, `pyspeckit` |
| `[gui]` | `PyQt5` |
| `[dev]` | `pytest`, `pytest-cov`, `ruff`, `mypy` |

To install everything at once, use: `pip install ".[all]"`.

---

## Usage Examples

### 1. Time Conversion & Auto Series (Example: [intro_time-operations.ipynb](examples/intro_time-operations.ipynb))

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

### 2. Interoperability (Deep Learning & Big Data) (Example: [intro_Interop.ipynb](examples/intro_Interop.ipynb))

gwexpy provides seamless conversion to/from major data science frameworks:

```python
from gwexpy.interop import to_torch, to_tensorflow, to_jax, to_cupy, to_dask, to_zarr
ts = ... # TimeSeries

# Deep Learning Frameworks
tensor_pt = to_torch(ts)      # PyTorch Tensor
tensor_tf = to_tensorflow(ts)         # TensorFlow Tensor
tensor_jax = to_jax(ts)       # JAX Array
tensor_cupy = to_cupy(ts)     # CuPy Array
df_pl = ts.to_polars()        # Polars DataFrame

# Big Data & Storage
dask_arr = to_dask(ts, chunks=1000)  # Dask Array
to_zarr(ts, store="data.zarr", path="my_array", overwrite=True)  # Save to Zarr

# Audio & Other Domains
# Supports: Librosa (standard numpy), Pydub (AudioSegment), ObsPy (Trace), MNE, Neo, ROOT
from gwexpy.interop import to_pydub, to_obspy, write_root_file
audio_seg = to_pydub(ts)
trace = to_obspy(ts)

# CERN ROOT Interoperability
# Converts to ROOT TGraph/TMultiGraph/TH2D and saves to .root files
# Now highly optimized with vectorization for fast conversion of large arrays
tsd.write("data.root")  # Automatic .root detection for Dictionary/List
mg = tsd.to_tmultigraph(name="comparison")
```

### 3. TimeSeriesMatrix & Decomposition (Example: [intro_TimeSeriesMatrix.ipynb](examples/intro_TimeSeriesMatrix.ipynb))

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

### 4. Peak Finding (Example: [intro_PeakDetection.ipynb](examples/intro_PeakDetection.ipynb))

```python
# Find peaks with physical unit constraints
peaks, props = s2.find_peaks(
    height=0.5,           # Minimum height (unit of data)
    distance=0.1 * u.s,   # Minimum distance between peaks
    width=0.05 * u.s      # Minimum width
)
peaks.plot(style='o')
```

### 5. Signal Transforms (STLT: [tutorial_ShortTimeLaplaceTransformation.ipynb](examples/tutorial_ShortTimeLaplaceTransformation.ipynb) & HHT: [tutorial_HHT_Analysis.ipynb](examples/tutorial_HHT_Analysis.ipynb))

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

### 6. Series Fitting (Example: [intro_Fitting.ipynb](examples/intro_Fitting.ipynb))

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

# MCMC Analysis (requires emcee & corner)
# result_mcmc = ts.fit('gaussian', method='mcmc', nwalkers=32, nsteps=1000)
# result_mcmc.plot_corner()

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
