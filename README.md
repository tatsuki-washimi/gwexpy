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

- **Advanced Containers**: `TimeSeriesMatrix`, `FrequencySeriesMatrix`, and `LaplaceGram` for multivariate analysis.
- **Spectral Analysis**: Enhanced CSD/Coherence matrix calculations and transient-friendly FFT.
- **Signal Processing**: STLT, HHT, Peak Finding (unit-aware), and ARIMA.
- **Preprocessing**: Robust alignment, imputation, and whitening for collections.
- **Physics Models**: Built-in support for `gwinc` (detector noise) and `obspy` (earth noise).
- **Expanded I/O**: Native support for `.gwf`, `.wav`, `.root` (vectorized), `.mseed`, `.win`, `.ats`, `.tdms`, `.gbd`, and more.
- **Interoperability**: Seamless conversion to PyTorch, TensorFlow, JAX, Dask, and Polars.
- **Robust Serialization**: Full Pickle round-trip support for all data objects (e.g., `TimeSeriesMatrix`, `ScalarField`).

See the [Features](https://tatsuki-washimi.github.io/gwexpy/docs/web/en/) page for details.

---

## Installation

```bash
# GWexpy is not published on PyPI yet (recommended)
pip install git+https://github.com/tatsuki-washimi/gwexpy.git

# From a local checkout
pip install .

# With a specific extra (example)
pip install ".[analysis]"

# With a specific extra from GitHub
pip install "gwexpy[analysis] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
```

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

## Contributing

Contributions are welcome! Please open issues or submit PRs on [GitHub](https://github.com/tatsuki-washimi/gwexpy).
