Installation
============

GWExPy requires Python 3.9+ and depends on GWpy, NumPy, SciPy, and Astropy.

Basic install (from GitHub, no PyPI/conda release)
-------------

.. code-block:: bash

   pip install git+https://github.com/tatsuki-washimi/gwexpy.git

Development install
-------------------

.. code-block:: bash

   pip install -e ".[dev]"

Optional extras
---------------

GWExPy exposes optional extras for interoperability:

- ``.[data]`` for I/O helpers (xarray, h5py, netCDF4)
- ``.[geophysics]`` for obspy, mth5, etc.
- ``.[interop]`` for high-level interoperability (torch, jax, dask, etc.)
- ``.[control]`` for python-control integration
- ``.[audio]`` for librosa/pydub helpers
- ``.[fitting]`` for advanced fitting (iminuit, emcee, corner)

Combine extras as needed, for example:

.. code-block:: bash

   pip install ".[data,control]"
