# gwexpy: GWpy Expansions for Experiments

[![PyPI version](https://badge.fury.io/py/gwexpy.svg)](https://badge.fury.io/py/gwexpy)
[![CI Status](https://github.com/tatsuki-washimi/gwexpy/actions/workflows/test.yml/badge.svg)](https://github.com/tatsuki-washimi/gwexpy/actions/workflows/test.yml)
[![Documentation](https://github.com/tatsuki-washimi/gwexpy/actions/workflows/docs.yml/badge.svg)](https://tatsuki-washimi.github.io/gwexpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

**gwexpy** (GWexpy) is an (unofficial) extension library for [**GWpy**](https://gwpy.github.io/), designed to facilitate advanced time-series analysis, matrix operations, and signal processing for experimental physics and gravitational wave data analysis.

## Documentation

> [!IMPORTANT]
> **Full documentation is available in both English and Japanese:**
>
> - **English:** [https://tatsuki-washimi.github.io/gwexpy/docs/web/en/](https://tatsuki-washimi.github.io/gwexpy/docs/web/en/)
> - **日本語:** [https://tatsuki-washimi.github.io/gwexpy/docs/web/ja/](https://tatsuki-washimi.github.io/gwexpy/docs/web/ja/)
>
> **Over 25 comprehensive tutorials** covering everything from basic usage to advanced signal processing techniques are available in both languages.

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

**Pickle portability note:** `gwexpy` objects are pickled so that unpickling returns valid **`GWpy` types** in a GWpy-only environment (where `gwexpy` is not installed). This preserves core numeric data and standard metadata, but gwexpy-specific attributes or subclasses are omitted. For full details, see the documentation.

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

# With optional extras (install only what you need)
pip install "gwexpy[analysis]"   # Advanced statistics (scikit-learn, statsmodels, pmdarima)
pip install "gwexpy[seismic]"    # Seismic/MT data (ObsPy, mth5, mtpy)
pip install "gwexpy[control]"    # Control systems (python-control)
pip install "gwexpy[audio]"      # Audio file support (pydub, tinytag)
pip install "gwexpy[gw]"         # GW-specific tools (lalsuite, gwosc, gwinc)
pip install "gwexpy[fitting]"    # Curve fitting/MCMC (iminuit, emcee, corner)
pip install "gwexpy[io]"         # Experimental I/O (nptdms for LabVIEW)
pip install "gwexpy[gui]"        # GUI components (experimental, PyQt5-based)

# Multiple extras
pip install "gwexpy[analysis,seismic,control]"
```

> [!IMPORTANT]
> **gwpy Compatibility**: gwexpy requires `gwpy>=4.0.0`. GWpy 4.0.0 introduced breaking API changes (e.g., Python 3.11+ requirement, new I/O registry). If you encounter import errors, please ensure your environment is updated:
>
> ```bash
> pip install "gwpy>=4.0.0,<5.0.0"
> ```

> [!NOTE]
> **Heavy / External Dependencies**:
> Most gravitational wave and data science features are installed by default. However, standard PyPI installation does not include packages requiring specialized C/C++ builds. If you need NDS2 client or frame data support, install them explicitly via Conda (or your preferred package manager):
>
> ```bash
> conda install -c conda-forge nds2-client ldas-tools-framecpp
> ```
>
> Similarly, for machine learning, GPU computing, and advanced physics format operations (e.g., PyTorch, TensorFlow, librosa), please install the respective official packages independently according to your system's hardware configuration.

### Optional Dependencies Summary

The default installation (`pip install gwexpy`) includes core dependencies for time-series analysis, basic signal processing, and standard I/O. Optional extras enable specialized workflows:

| Extra         | Description                  | Key Packages                          |
|---------------|------------------------------|---------------------------------------|
| `[analysis]`  | Advanced statistics & ML     | scikit-learn, statsmodels, pmdarima   |
| `[seismic]`   | Seismic/MT data formats      | ObsPy, mth5, mtpy                     |
| `[control]`   | Control systems              | python-control                        |
| `[audio]`     | Audio file I/O               | pydub, tinytag                        |
| `[gw]`        | GW-specific tools            | lalsuite, gwosc, gwinc                |
| `[fitting]`   | Curve fitting & MCMC         | iminuit, emcee, corner                |
| `[io]`        | Experimental I/O             | nptdms (LabVIEW)                      |
| `[gui]`       | GUI (experimental)           | PyQt5, pyqtgraph                      |

For further configuration (e.g., `[dev]`), see the [Installation Guide](https://tatsuki-washimi.github.io/gwexpy/docs/web/en/user_guide/installation.html).

---

## Quick Start

> **Note:** `import gwexpy` automatically registers all constructors and I/O
> formats.  If you import submodules directly and encounter a `KeyError` from
> `ConverterRegistry`, call `gwexpy.register_all()` to ensure the registry is
> fully populated.

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
from gwexpy.timeseries import TimeSeriesList
# Create matrix from multiple series and perform PCA
mat = TimeSeriesList([series1, series2, series3]).to_matrix()
scores = mat.standardize().pca(n_components=2)
```

For more complex examples, browse our **[Tutorials](https://tatsuki-washimi.github.io/gwexpy/docs/web/en/user_guide/tutorials/)** (available in [English](https://tatsuki-washimi.github.io/gwexpy/docs/web/en/user_guide/tutorials/) and [日本語](https://tatsuki-washimi.github.io/gwexpy/docs/web/ja/user_guide/tutorials/)).

---

## Examples & Notebooks

Executable marimo notebooks demonstrating key workflows are available in the [`notebooks/`](notebooks/) directory:

- **Transfer Function Workflow** ([`01_transfer_function_workflow.py`](notebooks/01_transfer_function_workflow.py)): DTT XML import, transfer function estimation, and Bode plotting
- **Coherence Ranking** ([`02_coherence_ranking.py`](notebooks/02_coherence_ranking.py)): Synthetic multichannel coherence ranking used for the paper figure
- **GWOSC Case Study** ([`03_gwosc_case_study.py`](notebooks/03_gwosc_case_study.py)): Public GWOSC strain data with derived proxy auxiliary channels

The synthetic paper notebook is exercised in CI; the GWOSC case study is documented as an optional network-dependent reproduction path. Reproduction instructions are collected in [`docs/repro/README.md`](docs/repro/README.md).

## Reproducibility

The SoftwareX manuscript and this repository are tied to the archived release DOI `10.5281/zenodo.19059423`. The code record for the submitted version is the GitHub repository plus the archived Zenodo release, while the reproducibility path is provided through executable marimo notebooks in `notebooks/`.

Typical local reproduction flow:

```bash
pip install "gwexpy[analysis,gw]"
python -m marimo run notebooks/02_coherence_ranking.py
python notebooks/02_coherence_ranking.py
```

For the optional public-data example:

```bash
python -m marimo run notebooks/03_gwosc_case_study.py
```

See [`docs/repro/README.md`](docs/repro/README.md) for figure-generation commands, network caveats, and the distinction between CI-covered synthetic workflows and optional GWOSC-backed workflows.

---

## Testing

```bash
python -m pytest
```

Note: GUI/IO sample fixtures are stored under `tests/sample-data/` and are not
versioned in git. If you need to run the data-dependent tests locally, place the
sample files under `tests/sample-data/gui/` (see the test paths for exact names).

## Repository layout

The package source lives in `gwexpy/` (not `src/gwexpy/`).
This flat layout is intentional: it mirrors the convention used by [GWpy](https://github.com/gwpy/gwpy) and other scientific Python packages in the gravitational-wave ecosystem, ensuring that import paths and developer tooling remain consistent with upstream.

## Contributing

Contributions are welcome! Please open issues or submit PRs on [GitHub](https://github.com/tatsuki-washimi/gwexpy).

---

## Support & Contact

For questions, bug reports, or feature requests:

- **GitHub Issues**: [https://github.com/tatsuki-washimi/gwexpy/issues](https://github.com/tatsuki-washimi/gwexpy/issues) (recommended)
- **Discussions**: [https://github.com/tatsuki-washimi/gwexpy/discussions](https://github.com/tatsuki-washimi/gwexpy/discussions) (for general questions)

For academic citations and correspondence, please refer to the [published SoftwareX paper](docs/gwexpy-paper/) (DOI pending).

> [!NOTE]
> To protect against spam, direct email addresses are not listed here. For private inquiries, please use GitHub Discussions or open a confidential issue.
