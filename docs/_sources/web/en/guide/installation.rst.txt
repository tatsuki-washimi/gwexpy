Installation
============

GWexpy requires Python 3.9+ and depends on GWpy, NumPy, SciPy, and Astropy.

Basic install
-------------

.. note::
   GWexpy is not published on PyPI yet. The recommended install method is via GitHub.

.. code-block:: bash

   # From GitHub (recommended)
   pip install git+https://github.com/tatsuki-washimi/gwexpy.git

   # From a local checkout
   pip install .

Development install
-------------------

.. code-block:: bash

   pip install -e ".[dev]"

Optional extras
---------------

GWexpy exposes optional extras for domain-specific features:

- ``.[gw]`` for GW data analysis (nds2, frames, noise models)
- ``.[stats]`` for stats & signal analysis (polars, ARIMA, ICA/PCA)
- ``.[fitting]`` for advanced fitting (iminuit, emcee, corner)
- ``.[astro]`` for astroparticle physics tools (specutils, pyspeckit)
- ``.[geophysics]`` for obspy, mth5, etc.
- ``.[audio]`` for librosa/pydub helpers
- ``.[bio]`` for mne/neo/elephant integrations
- ``.[interop]`` for high-level interoperability (torch, jax, dask, etc.)
- ``.[control]`` for python-control integration
- ``.[plot]`` for plotting & mapping (pygmt)
- ``.[analysis]`` for transforms and time-frequency tools
- ``.[gui]`` for the experimental Qt GUI
- ``.[all]`` to install all optional dependencies

Combine extras as needed, for example:

.. code-block:: bash

   pip install ".[gw,stats,plot]"

You can also install extras directly from GitHub:

.. code-block:: bash

   pip install "gwexpy[analysis] @ git+https://github.com/tatsuki-washimi/gwexpy.git"

.. note::
   The ``[gw]`` extra includes dependencies like ``nds2-client`` which are **not available on PyPI**.
   To use these features, you must install dependencies via **Conda** first:

   .. code-block:: bash

      conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp
      pip install ".[gw]"
