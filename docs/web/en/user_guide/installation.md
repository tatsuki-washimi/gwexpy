# Installation Guide

:::{note}
GWexpy is currently in its **Development Version**. Since it is not yet officially published on PyPI or Conda, we recommend a source installation from GitHub. These commands will be fully operational after the official release.
:::

GWexpy supports **Python 3.11 or later**. You can choose from several installation options (extras) depending on your analysis goals.

| Goal | Installation Command | Features |
| --- | --- | --- |
| Minimal | `pip install git+https://github.com/tatsuki-washimi/gwexpy.git` | Numerical containers and basic arithmetic. Minimal dependencies. |
| **Recommended** | `pip install "gwexpy[analysis,fitting,plotting] @ git+https://github.com/tatsuki-washimi/gwexpy.git"` | Advanced statistics, curve fitting, and mapping features. |
| GW Analysis | `pip install "gwexpy[gw,io] @ git+https://github.com/tatsuki-washimi/gwexpy.git"` | Frame file support, NDS2 access, and official GW tools. |
| Dev / Full | `pip install "gwexpy[all] @ git+https://github.com/tatsuki-washimi/gwexpy.git"` | Enables all optional features. |

## 1. Installation Steps

### Minimal

For users who want to keep dependencies minimal and only use core containers like `ScalarField`.

```bash
pip install git+https://github.com/tatsuki-washimi/gwexpy.git
```

### Conda Environment (Recommended / GW Analysis)

For gravitational-wave analysis (requiring NDS2 or FrameLIB), we strongly recommend using Conda (e.g., Miniforge) to resolve binary dependencies first.

:::{warning}
If you use Conda, avoid running `pip install` directly in `base` or in a shared environment for unrelated work. Create a **dedicated environment for GWexpy** first, then install both the Conda-managed binary dependencies and the `pip` packages there. This keeps binary dependency resolution isolated and reduces the risk of breaking the environment.
:::

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

```bash
git clone https://github.com/tatsuki-washimi/gwexpy.git
cd gwexpy
pip install -e ".[dev,all]"
```

## 2.1. Dependency Troubleshooting

- If you see `No module named nds2`: install `python-nds2-client` in the active dedicated Conda environment.
- If you see FrameLIB / `framecpp`-related errors: reinstall `python-framel` and `ldas-tools-framecpp` in the same environment.
- If the environment has already been mixed with unrelated packages: recreating it with `conda create -n gwexpy python=3.11` is usually safer than trying to repair it in place.

---

## 3. Optional Dependencies (Extras) Details

| Extra Name | Key Packages | Primary Use Cases |
| --- | --- | --- |
| `analysis` | `scikit-learn`, `statsmodels`, `pmdarima` | Noise cancellation, forecasting, and machine learning. |
| `fitting` | `iminuit`, `emcee`, `corner` | Least-squares fitting and MCMC analysis. |
| `gw` | `lalsuite`, `gwosc`, `gwinc`, `ligo.skymap` | Data discovery, sensitivity calculations, and sky mapping. |
| `io` | `nptdms` | Reading LabVIEW TDMS formats. |
| `plotting` | `pygmt` | High-precision geographic mapping (GeoMap). |
| `audio` | `pydub` | Audio export and processing. |
| `seismic` | `obspy`, `mth5`, `mtpy` | Seismic and magnetotelluric data. |
| `control` | `control` (python-control) | Control systems and transfer functions. |
| `gui` | `PyQt5`, `pyqtgraph` | Graphical interface (prototype stage). |

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
## 6. Further Reading

* [Quickstart](quickstart.md) - Get started in 3 lines
* [Getting Started](getting_started.md) - Learning roadmap
