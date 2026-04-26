---
myst:
  html_meta:
    description: "Install GWexpy for Python 3.11+ with minimal, recommended, GW-analysis, or developer setups, including extras guidance and dependency troubleshooting."
---

# Installation Guide

## At a Glance

| Item | Details |
| --- | --- |
| **Page Role** | Guide |
| **Audience** | New users installing GWexpy, users choosing extras, and contributors setting up a local dev environment |
| **Prerequisites** | Python 3.11+, basic `pip` usage, and basic Conda usage when you need GW binary dependencies |
| **Use Cases** | Install the minimal stack, decide whether Conda is required, choose extras, or prepare a contributor setup |
| **Search Keywords** | `install`, `pip`, `conda`, extras, development version, NDS2, FrameLIB |

## On This Page

- [Install Commands](#en-install-command)
- [Installation Steps](#1-installation-steps)
- [Dependency Troubleshooting](#21-dependency-troubleshooting)
- [Optional Dependencies (Extras) Details](#3-optional-dependencies-extras-details)
- [OS-Specific Notes](#4-os-specific-notes)
- [Security Note (Pickle)](#5-security-note-pickle)
- [Next to Read](#next-to-read)

:::{note}
GWexpy is currently in its **Development Version** and is **not yet published on PyPI or Conda**. For now, install it from the GitHub source repository using the commands on this page.
:::

GWexpy supports **Python 3.11 or later**. You can choose from several installation options (extras) depending on your analysis goals.

(en-install-command)=
## Install Commands

| Goal | Installation Command | Features |
| --- | --- | --- |
| Minimal | `pip install git+https://github.com/tatsuki-washimi/gwexpy.git` | Numerical containers and basic arithmetic. Minimal dependencies. |
| **Recommended** | `pip install "gwexpy[analysis,fitting,plotting] @ git+https://github.com/tatsuki-washimi/gwexpy.git"` | Advanced statistics, curve fitting, and mapping features. |
| GW Analysis | `pip install "gwexpy[gw,io] @ git+https://github.com/tatsuki-washimi/gwexpy.git"` | Frame file support, NDS2 access, and official GW tools. |
| Dev / Full | `pip install "gwexpy[all] @ git+https://github.com/tatsuki-washimi/gwexpy.git"` | Enables all optional features. |

## 1. Installation Steps

### Minimal

For users who want to keep dependencies minimal and only use core containers like `ScalarField`.

- Purpose: verify import and core container usage with the smallest dependency set
- Input: Python 3.11+ and `pip`
- Output: a minimal GWexpy environment

If you need NDS2, FrameLIB, `pygmt`, or other binary-heavy tools, skip this section and start with [Conda Environment (Recommended / GW Analysis)](#conda-environment-recommended--gw-analysis).

```bash
pip install git+https://github.com/tatsuki-washimi/gwexpy.git
```

### Conda Environment (Recommended / GW Analysis)

For gravitational-wave analysis (requiring NDS2 or FrameLIB), we strongly recommend using Conda (e.g., Miniforge) to resolve binary dependencies first.

:::{warning}
If you use Conda, avoid running `pip install` directly in `base` or in a shared environment for unrelated work. Create a **dedicated environment for GWexpy** first, then install both the Conda-managed binary dependencies and the `pip` packages there. This keeps binary dependency resolution isolated and reduces the risk of breaking the environment.
:::

- Purpose: build a GW-ready environment with binary dependencies such as NDS2 and FrameLIB
- Input: a shell with Conda available
- Output: a dedicated `gwexpy` environment with `gw` extras enabled

```bash
# 1. Create environment and resolve binary dependencies
conda create -n gwexpy python=3.11
conda activate gwexpy
conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp

# 2. Install GWexpy with analysis/fitting options
pip install "gwexpy[gw,analysis,fitting] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
```

If you see `No module named nds2` or FrameLIB-related import errors, re-run the `conda install -c conda-forge ...` step inside that dedicated environment first. If you do not need NDS2 or FrameLIB, the minimal or recommended installation without the `gw` extras is usually sufficient.

### Developer Mode

For contributors who want to install from source and set up a testing environment. **Conda is optional here.** Use the Conda workflow above when you need to validate `gw`-related binary dependencies; otherwise, a standard virtual environment such as `venv` is fine for documentation work and general development.

- Purpose: install from a local clone for editing and tests
- Input: Git plus a Python virtual environment
- Output: an editable install via `pip install -e`

```bash
git clone https://github.com/tatsuki-washimi/gwexpy.git
cd gwexpy
pip install -e ".[dev,all]"
```

## 2.1. Dependency Troubleshooting

- If you see `No module named nds2`: install `python-nds2-client` in the active dedicated Conda environment.
- If you see FrameLIB / `framecpp`-related errors: reinstall `python-framel` and `ldas-tools-framecpp` in the same environment.
- If the environment has already been mixed with unrelated packages: recreating it with `conda create -n gwexpy python=3.11` is usually safer than trying to repair it in place.
- If you installed from [Quickstart](quickstart.md) and later discover you need GW binary dependencies: return to [Conda Environment (Recommended / GW Analysis)](#conda-environment-recommended--gw-analysis) before adding more packages.

---

## 3. Optional Dependencies (Extras) Details

| Extra Name | Key Packages | Primary Use Cases |
| --- | --- | --- |
| `analysis` | `scikit-learn`, `statsmodels`, `pmdarima` | Noise cancellation, forecasting, and machine learning. |
| `fitting` | `iminuit`, `emcee`, `corner` | Least-squares fitting and MCMC analysis. |
| `gw` | `lalsuite`, `gwosc`, `gwinc`, `ligo.skymap` | Data discovery, sensitivity calculations, and sky mapping. |
| `io` | `nptdms` | Reading LabVIEW TDMS formats. |
| `netcdf4` | `netCDF4`, `xarray` | Reading and writing NetCDF4 time-series files via xarray. |
| `zarr` | `zarr` | Reading and writing Zarr array stores. |
| `plotting` | `pygmt` | High-precision geographic mapping (GeoMap). |
| `audio` | `pydub` | Audio export and processing. |
| `seismic` | `obspy`, `mth5`, `mtpy` | Seismic and magnetotelluric data. |
| `control` | `control` (python-control) | Control systems and transfer functions. |
| `gui` | `PyQt5`, `pyqtgraph` | Graphical interface (prototype stage). Not included in `all`. |

---

## 4. OS-Specific Notes

* **Linux**: Ensure standard build tools (`build-essential`) are installed.
* **macOS (Apple Silicon)**: Using the `conda-forge` channel ensures most binaries run natively on M1/M2/M3.
* **Windows (WSL2)**: We recommend installing GWexpy within a Linux environment on WSL2 rather than on Windows proper.

## 5. Security Note (Pickle)

To facilitate sharing analysis results, GWexpy supports saving and restoring objects via `Pickle`. This uses **Transparent Pickle** technology, allowing objects to be restored as base GWpy objects even if the recipient does not have GWexpy installed.

:::{caution}
**Never load Pickle files from untrusted sources.**
Python's `pickle` module has inherent security risks (Arbitrary Code Execution) when loading data. Always exchange data through trusted channels or consider more secure serialization formats like `HDF5`.

:::
<a id="next-to-read"></a>
<a id="next-steps"></a>

## 6. Next to Read

* [Quickstart](quickstart.md) - confirm import and plotting with the smallest possible example
* [Troubleshooting](troubleshooting.md) - reverse-lookup common install, plotting, and binary dependency issues
* [Getting Started](getting_started.md) - choose the right learning path after installation
* [Prerequisites and Conventions](prerequisites_and_conventions.md) - review FFT, GPS time, and GWpy-compatibility assumptions
