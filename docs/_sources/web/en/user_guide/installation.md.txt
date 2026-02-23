# Installation

GWexpy requires Python 3.9+ and depends on GWpy, NumPy, SciPy, and Astropy.

## Basic install

```bash
pip install gwexpy
```

:::{note}
**Development version from GitHub (latest features):**

```bash
pip install git+https://github.com/tatsuki-washimi/gwexpy.git
```

:::

:::{important}
**For Gravitational Wave Data Analysis (LIGO/Virgo/KAGRA, etc.)**

Some dependencies in the `[gw]` extra, such as `nds2-client` and `python-framel`, are **not available on PyPI** due to complex system-level dependencies.
To use these features, we strongly recommend installing them via **Conda (Miniforge/Anaconda)** first:

```bash
# 1. Install external dependencies
conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp

# 2. Install GWexpy with extras
pip install "gwexpy[gw] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
```

:::

## Development install

```bash
pip install -e ".[dev]"
```

## Optional extras

GWexpy exposes optional extras for domain-specific features:

- `.[gw]` for GW data analysis (nds2, frames, noise models)
- `.[stats]` for stats & signal analysis (polars, ARIMA, ICA/PCA)
- `.[fitting]` for advanced fitting (iminuit, emcee, corner)
- `.[astro]` for astroparticle physics tools (specutils, pyspeckit)
- `.[geophysics]` for obspy, mth5, etc.
- `.[audio]` for librosa/pydub helpers
- `.[bio]` for mne/neo/elephant integrations
- `.[interop]` for high-level interoperability (torch, jax, dask, etc.)
- `.[control]` for python-control integration
- `.[plot]` for plotting & mapping (pygmt)
- `.[analysis]` for transforms and time-frequency tools
- `.[gui]` for the experimental Qt GUI
- `.[all]` to install all optional dependencies

Combine extras as needed, for example:

```bash
pip install ".[gw,stats,plot]"
```

You can also install extras directly from GitHub:

```bash
pip install "gwexpy[analysis] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
```

:::{note}
The `[gw]` extra includes dependencies like `nds2-client` which are **not available on PyPI**.
To use these features, you must install dependencies via **Conda** first:

```bash
conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp
pip install ".[gw]"
```

### Maximal Information Coefficient (MIC)

Calculation of MIC requires `minepy`. On Python 3.11+, standard installation via `pip` or `conda` may fail. We provide an automated build script in the repository:

```bash
python scripts/install_minepy.py
```

:::

## Next Steps

Now that GWexpy is installed, learn the basics with:

- [Quickstart](quickstart.md) - Generate and plot time series data
- [Getting Started](getting_started.md) - Complete learning path
