# gwexpy: GWpy Expansions for Experiments

[![CI Status](https://github.com/tatsuki-washimi/gwexpy/actions/workflows/test.yml/badge.svg)](https://github.com/tatsuki-washimi/gwexpy/actions/workflows/test.yml)
[![Documentation](https://github.com/tatsuki-washimi/gwexpy/actions/workflows/docs.yml/badge.svg)](https://tatsuki-washimi.github.io/gwexpy/)
[![PyPI version](https://img.shields.io/pypi/v/gwexpy.svg)](https://pypi.org/project/gwexpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**gwexpy** is an (unofficial) extension library for [**GWpy**](https://gwpy.github.io/), designed to facilitate advanced time-series analysis, matrix operations, and signal processing for experimental physics and gravitational wave data analysis.

> [!IMPORTANT]
> For full documentation, tutorials, and API reference, please visit:
> **[https://tatsuki-washimi.github.io/gwexpy/](https://tatsuki-washimi.github.io/gwexpy/en/index.html)**

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

See the [Features](https://tatsuki-washimi.github.io/gwexpy/en/index.html) page for details.

---

## Installation

```bash
# Recommended: Install with ALL optional dependencies
pip install ".[all]"

# Minimal installation
pip install .
```

For domain-specific extras (e.g., `[gw]`, `[geophysics]`, `[fitting]`), see the [Installation Guide](https://tatsuki-washimi.github.io/gwexpy/en/index.html).

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

For more complex examples, browse our [Tutorials](https://tatsuki-washimi.github.io/gwexpy/en/index.html).

---

## Testing

```bash
python -m pytest
```

## Contributing

Contributions are welcome! Please open issues or submit PRs on [GitHub](https://github.com/tatsuki-washimi/gwexpy).
