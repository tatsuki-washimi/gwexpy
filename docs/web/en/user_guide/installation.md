# Installation Guide

:::{note}
GWexpy is currently in its **Development Version**. Since it is not yet officially published on PyPI or Conda, we recommend a source installation from GitHub. These commands will be fully operational after the official release.
:::

GWexpy supports **Python 3.11 or later**. You can choose from several installation options (extras) depending on your analysis goals.

.. list-table:: Recommended Installation Options
   :widths: 25 50 25
   :header-rows: 1

   * - Goal
     - Installation Command
     - Features
   * - Minimal
     - `pip install git+https://github.com/tatsuki-washimi/gwexpy.git`
     - Numerical containers and basic arithmetic. Minimal dependencies.
   * - **Recommended**
     - `pip install "gwexpy[analysis,fitting,plotting] @ git+https://github.com/tatsuki-washimi/gwexpy.git"`
     - Advanced statistics, curve fitting, and mapping features.
   * - GW Analysis
     - `pip install "gwexpy[gw,io] @ git+https://github.com/tatsuki-washimi/gwexpy.git"`
     - Frame file support, NDS2 access, and official GW tools.
   * - Dev / Full
     - `pip install "gwexpy[all] @ git+https://github.com/tatsuki-washimi/gwexpy.git"`
     - Enables all optional features.

## 1. Installation Steps

.. tab-set::

   .. tab-item:: Minimal

      For users who want to keep dependencies minimal and only use core containers like `ScalarField`.

      ```bash
      pip install git+https://github.com/tatsuki-washimi/gwexpy.git
      ```

   .. tab-item:: Conda Environment (Recommended / GW Analysis)

      For gravitational-wave analysis (requiring NDS2 or FrameLIB), we strongly recommend using Conda (e.g., Miniforge) to resolve binary dependencies first.

      ```bash
      # 1. Create environment and resolve binary dependencies
      conda create -n gwexpy python=3.11
      conda activate gwexpy
      conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp

      # 2. Install GWexpy with analysis/fitting options
      pip install "gwexpy[gw,analysis,fitting] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
      ```

   .. tab-item:: Developer Mode

      For contributors who want to install from source and set up a testing environment.

      ```bash
      git clone https://github.com/tatsuki-washimi/gwexpy.git
      cd gwexpy
      pip install -e ".[dev,all]"
      ```

---

## 2. Optional Dependencies (Extras) Details

.. list-table:: Extras Overview
   :widths: 20 40 40
   :header-rows: 1

   * - Extra Name
     - Key Packages
     - Primary Use Cases
   * - `analysis`
     - `scikit-learn`, `statsmodels`, `pmdarima`
     - Noise cancellation, forecasting, and machine learning.
   * - `fitting`
     - `iminuit`, `emcee`, `corner`
     - Least-squares fitting and MCMC analysis.
   * - `gw`
     - `lalsuite`, `gwosc`, `gwinc`, `ligo.skymap`
     - Data discovery, sensitivity calculations, and sky mapping.
   * - `io`
     - `nptdms`
     - Reading LabVIEW TDMS formats.
   * - `plotting`
     - `pygmt`
     - High-precision geographic mapping (GeoMap).
   * - `audio`
     - `pydub`
     - Audio export and processing.
   * - `seismic`
     - `obspy`, `mth5`, `mtpy`
     - Seismic and magnetotelluric data.
   * - `control`
     - `control` (python-control)
     - Control systems and transfer functions.
   * - `gui`
     - `PyQt5`, `pyqtgraph`
     - Graphical interface (prototype stage).

---

## 3. OS-Specific Notes

* **Linux**: Ensure standard build tools (`build-essential`) are installed.
* **macOS (Apple Silicon)**: Using the `conda-forge` channel ensures most binaries run natively on M1/M2/M3.
* **Windows (WSL2)**: We recommend installing GWexpy within a Linux environment on WSL2 rather than on Windows proper.

## 4. Security Note (Pickle)

To facilitate sharing analysis results, GWexpy supports saving and restoring objects via `Pickle`. This uses **Transparent Pickle** technology, allowing objects to be restored as base GWpy objects even if the recipient does not have GWexpy installed.

:::{caution}
**Never load Pickle files from untrusted sources.**
Python's `pickle` module has inherent security risks (Arbitrary Code Execution) when loading data. Always exchange data through trusted channels or consider more secure serialization formats like `HDF5`.

:::
## 5. Next Steps

* :doc:`quickstart` - Get started in 3 lines
* :doc:`getting_started` - Learning roadmap
