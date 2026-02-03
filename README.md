# gwexpy: GWpy Expansions for Experiments

[![CI Status](https://github.com/tatsuki-washimi/gwexpy/actions/workflows/test.yml/badge.svg)](https://github.com/tatsuki-washimi/gwexpy/actions/workflows/test.yml)
[![Documentation](https://github.com/tatsuki-washimi/gwexpy/actions/workflows/docs.yml/badge.svg)](https://tatsuki-washimi.github.io/gwexpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**gwexpy** (GWexpy) is an (unofficial) extension library for [**GWpy**](https://gwpy.github.io/), designed to facilitate advanced time-series analysis, matrix operations, and signal processing for experimental physics and gravitational wave data analysis.

## Documentation

> [!IMPORTANT]
> **Full documentation is available in both English and Japanese:**
>
> - **English:** [https://tatsuki-washimi.github.io/gwexpy/docs/web/en/](https://tatsuki-washimi.github.io/gwexpy/docs/web/en/)
> - **日本語:** [https://tatsuki-washimi.github.io/gwexpy/docs/web/ja/](https://tatsuki-washimi.github.io/gwexpy/docs/web/ja/)
>
> **19 comprehensive tutorials** covering everything from basic usage to advanced signal processing techniques are available in both languages.

---

## Key Features

### 🔢 Advanced Data Structures

**Matrix Types** - Multi-channel array operations

- `TimeSeriesMatrix`, `FrequencySeriesMatrix`, `SpectrogramMatrix`: Multi-dimensional data handling (Rows × Cols × Time/Freq)
- MIMO transfer function analysis and sensor array statistics
- Matrix operations: inverse (`.inv()`), determinant (`.det()`), trace (`.trace()`), Schur complement (`.schur()`)

**Physical Fields** - 4D space-time-frequency data

- `ScalarField`, `VectorField`, `TensorField`: 4D physical fields (time/frequency + 3D space/wavenumber)
- Automatic domain management with FFT transformations (`.fft_time()`, `.fft_space()`)
- Spatial extraction and vector/tensor calculus

**Enhanced Collections**

- `FrequencySeriesList/Dict`, `SpectrogramList/Dict`: Collection classes for frequency-domain data
- Batch processing across all elements (filtering, resampling, whitening)

### 🎯 Advanced Analysis

**Time-Frequency Analysis**

- Hilbert-Huang Transform (HHT/EMD): Nonlinear and non-stationary signal decomposition
- Short-Time Laplace Transform (STLT): Damping analysis with sigma-frequency representation
- Continuous Wavelet Transform (CWT), Cepstrum analysis
- Enhanced spectrograms with summary plotting

**Statistical & ML Methods**

- Principal/Independent Component Analysis (PCA/ICA)
- ARIMA/AR/MA time series modeling (statsmodels/pmdarima integration)
- Distance correlation, MIC (Maximal Information Coefficient), Kendall/Pearson correlation
- Bootstrap estimation for spectrograms

**Multi-Channel Analysis**

- **Bruco**: Coherence-based noise hunting tool for identifying coupling channels
- MIMO system identification with `python-control` integration
- Enhanced CSD/Coherence matrix calculations
- Transient-optimized FFT with flexible zero-padding

**Preprocessing & Signal Processing**

- Missing data imputation with customizable gap constraints
- Whitening methods (PCA/ZCA)
- Robust standardization and alignment
- Unit-aware peak detection (scipy.signal.find_peaks wrapper)

### 🔗 Extensive Interoperability

**Machine Learning & Array Libraries** (30+ integrations)

- Deep Learning: PyTorch (Tensor, Dataset/DataLoader), TensorFlow, JAX
- Accelerated Computing: CuPy, Dask, Zarr
- Data Science: pandas, xarray, polars, SQLite, JSON

**Domain-Specific Tools**

- Seismology: ObsPy (Trace/Stream)
- Neuroscience: MNE (EEG/MEG), Neo (Electrophysiology)
- Audio: Librosa, Pydub, Torchaudio
- Control Systems: python-control (FRD, StateSpace)
- Astronomy: Specutils, Pyspeckit, Astropy
- Geophysics: SimPEG
- Particle Physics: CERN ROOT (TGraph, TH1D, TH2D, TMultiGraph)

### 📂 Expanded File Format Support

Beyond standard GWpy formats (`.gwf`, `.hdf5`), gwexpy natively supports:

- **Seismology**: MiniSEED, SAC, WIN, GSE2 (ObsPy integration)
- **Instrumentation**: ATS (Metronix MT), TDMS (LabVIEW/NI), GBD (GRAPHTEC data logger)
- **LIGO Tools**: DTTXML (Diagnostic Test Tools)
- **General**: ROOT (.root), WAV (extended), Parquet, Feather, Pickle
- **Meteorology**: SDB (Davis weather station)

> [!WARNING]
> Never unpickle data from untrusted sources. `pickle`/`shelve` can execute arbitrary code on load.

Pickle portability note: gwexpy objects are pickled so that unpickling returns **GWpy types**
in a GWpy-only environment (gwexpy not required).

Compatibility details:

- `TimeSeries`, `FrequencySeries`, `Spectrogram` → unpickle as the corresponding GWpy objects.
- `TimeSeriesDict`, `TimeSeriesList` → unpickle as the GWpy collection classes.
- `FrequencySeriesDict/List`, `SpectrogramDict/List` → unpickle as built-in `dict`/`list` whose elements are GWpy objects.
- gwexpy-only types such as the matrix/field classes are not covered by this portability contract.

What is preserved (best effort):

- Core numeric data (`.value` arrays) and axis coordinates (`times`, `frequencies`).
- Common GWpy metadata (`unit`, `name`, `channel`, `epoch`).

What is not preserved:

- gwexpy-only attributes (`_gwex_*`) and behavior that only exists in gwexpy subclasses.

### 🔬 Physics Models & Simulation

- **Detector Noise**: `gwinc` integration for gravitational wave detector sensitivity curves
- **Noise Simulation**: Colored noise generation from ASD/PSD specifications
- **Field Simulation**: Isotropic noise fields and plane wave propagation

See the [Features](https://tatsuki-washimi.github.io/gwexpy/docs/web/en/) page for details.

---

## Installation

```bash
# From PyPI (recommended)
pip install gwexpy

# From GitHub (development version)
pip install git+https://github.com/tatsuki-washimi/gwexpy.git

# From a local checkout
pip install .

# With a specific extra (example)
pip install "gwexpy[analysis]"

# With a specific extra from GitHub
pip install "gwexpy[analysis] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
```

> [!IMPORTANT]
> **gwpy Compatibility**: gwexpy v0.1.0b1 requires `gwpy>=3.0.0,<4.0.0`. gwpy 4.0.0 introduced breaking API changes that are not yet supported. If you encounter import errors, please ensure you have gwpy 3.x installed:
>
> ```bash
> pip install "gwpy>=3.0.0,<4.0.0"
> ```

> [!NOTE]
> NDS/frames support (`[gw]` extra) depends on `nds2-client`, which is not published on PyPI.
> Install it via Conda first (e.g., `conda install -c conda-forge nds2-client`) before adding `[gw]`.

For other domain-specific extras (e.g., `[geophysics]`, `[fitting]`, `[analysis]`), see the [Installation Guide](https://tatsuki-washimi.github.io/gwexpy/docs/web/en/guide/installation.html).

---

## Quick Start

### 1. Vectorized Time Conversion & Auto Series

```python
import numpy as np
from astropy import units as u
from gwexpy import as_series
from gwexpy.time import to_gps

# Vectorized GPS conversion
gps = to_gps(["2025-01-01 00:00:00", "2025-01-01 00:00:01"])

# Convert axes to Series with automatic unit handling
ts = as_series((1419724818 + np.arange(10)) * u.s, unit="h")
```

### 2. TimeSeriesMatrix & Decomposition

```python
from gwexpy.timeseries import TimeSeriesMatrix
# Create matrix from multiple series and perform PCA
mat = TimeSeriesMatrix.from_list([series1, series2, series3])
scores = mat.standardize().pca(n_components=2)
```

For more complex examples, browse our **[Tutorials](https://tatsuki-washimi.github.io/gwexpy/docs/web/en/guide/tutorials/)** (available in [English](https://tatsuki-washimi.github.io/gwexpy/docs/web/en/guide/tutorials/) and [日本語](https://tatsuki-washimi.github.io/gwexpy/docs/web/ja/guide/tutorials/)).

---

## Testing

```bash
python -m pytest
```

Note: GUI/IO sample fixtures are stored under `tests/sample-data/` and are not
versioned in git. If you need to run the data-dependent tests locally, place the
sample files under `tests/sample-data/gui/` (see the test paths for exact names).

## Contributing

Contributions are welcome! Please open issues or submit PRs on [GitHub](https://github.com/tatsuki-washimi/gwexpy).
