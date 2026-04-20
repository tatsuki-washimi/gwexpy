I/O
===

**Stability:** Stable

.. currentmodule:: gwexpy.io

File format readers and writers registered in the GWpy I/O registry.
Supported formats include HDF5, GWF frames, NDS2, LigoDW XML, TDMS, and Zarr.

Zarr readers fail fast when per-array timing metadata is missing. Legacy
stores without ``sample_rate`` or ``dt`` must be recovered explicitly with
``sample_rate_override=...`` or ``dt_override=...``. For the dedicated
TimeSeries Zarr module reference, see
:doc:`gwexpy.timeseries.io.zarr_ <gwexpy.timeseries.io.zarr_>`.

.. automodule:: gwexpy.io
   :members:
   :undoc-members:
   :show-inheritance:
