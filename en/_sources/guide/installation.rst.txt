Installation
============

GWExPy requires Python 3.9+ and depends on GWpy, NumPy, SciPy, and Astropy.

Basic install
-------------

.. code-block:: bash

   # From a local checkout
   pip install .

   # From GitHub (no PyPI release)
   pip install git+https://github.com/tatsuki-washimi/gwexpy.git

Development install
-------------------

.. code-block:: bash

   pip install -e ".[dev]"

Optional extras
---------------

GWExPy exposes optional extras for domain-specific features:

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
