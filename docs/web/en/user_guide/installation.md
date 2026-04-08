# Installation

GWexpy supports **Python 3.9 or higher**.
Choose the installation method that best fits your analysis goals.

## Supported Environments

.. list-table:: 
   :widths: 30 70

   * - **OS**
     - Linux, macOS, Windows (WSL2 recommended)
   * - **Python**
     - 3.9, 3.10, 3.11, 3.12
   * - **Core Dependencies**
     - GWpy, NumPy, SciPy, Astropy

## Installation Options

.. tab-set::

    .. tab-item:: 🏁 Minimal
        :sync: minimal

        For users who only need core features (TimeSeriesMatrix, basic plotting, etc.):

        ```bash
        pip install git+https://github.com/tatsuki-washimi/gwexpy.git
        ```

    .. tab-item:: ✨ Recommended
        :sync: recommended

        Includes general signal processing, fitting, and mapping features:

        ```bash
        pip install "gwexpy[analysis,fitting,plotting] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
        ```

    .. tab-item:: 🌌 GW Analysis (LIGO/Virgo/KAGRA, etc.)
        :sync: gw

        For gravitational-wave data analysis (nds2, frames). **Binary dependencies must be installed via Conda first.**

        ```bash
        # 1. Install binary dependencies via Conda
        conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp

        # 2. Install GWexpy with extras
        pip install "gwexpy[gw] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
        ```

    .. tab-item:: 🛠️ For Developers
        :sync: developer

        For modifying code or running tests (editable mode):

        ```bash
        git clone https://github.com/tatsuki-washimi/gwexpy.git
        cd gwexpy
        pip install -e ".[dev,all]"
        ```

## OS-specific Notes

.. tab-set::

    .. tab-item:: Linux / WSL2

        Most features work with standard `pip`. Conda is recommended for `nds2` and frame file support.

    .. tab-item:: macOS (Intel/Apple Silicon)

        On Apple Silicon (M1/M2/M3), we strongly recommend installing complex binary packages via Conda (Miniforge/Mamba).

    .. tab-item:: Windows (Native)

        Basic analysis works, but advanced I/O (e.g., `nds2`) is recommended to be used via **WSL2**.

## PyPI / Conda Status

GWexpy is currently preparing for official registration on PyPI and conda-forge.
Until the official release, please **install directly from GitHub**.

* **PyPI**: Coming soon (`pip install gwexpy`)
* **Conda**: Coming soon (`conda install -c conda-forge gwexpy`)

## Troubleshooting

If you encounter errors during installation (especially regarding `nds2`, `minepy`, or `PyQt`), 
please refer to the :doc:`Troubleshooting Guide <troubleshooting>`.

## Next Steps

* :doc:`Quickstart <quickstart>` - Create your first plot in 5 minutes.
* :doc:`Getting Started <getting_started>` - Systematic learning roadmap.
