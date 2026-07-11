I/O
===

.. currentmodule:: gwexpy.io

File format readers and writers registered in the GWpy I/O registry.
Supported formats include HDF5, GWF frames, NDS2, LigoDW XML, TDMS, and Zarr.

Zarr readers fail fast when per-array timing metadata is missing. Legacy
stores without ``sample_rate`` or ``dt`` must be recovered explicitly with
``sample_rate_override=...`` or ``dt_override=...``.

.. automodule:: gwexpy.io
   :members:
   :undoc-members:
   :show-inheritance:
