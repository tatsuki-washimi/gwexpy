gwexpy.timeseries.io.zarr\_
===========================

.. note::

   Current behavior follows the Phase 7 truth ledger and the ``#8-19``
   fail-fast fix:

   - Per-array ``sample_rate`` is the primary timing metadata.
   - ``dt`` is accepted as a fallback and converted to ``sample_rate = 1 / dt``.
   - If both are missing, reading raises ``ValueError`` instead of silently
     assuming ``1 Hz``.
   - Legacy stores can be recovered intentionally with exactly one of
     ``sample_rate_override=...`` or ``dt_override=...``.

.. automodule:: gwexpy.timeseries.io.zarr_

   .. rubric:: Functions

   .. autosummary::

      read_timeseriesdict_zarr
      read_timeseries_zarr
      read_timeseriesmatrix_zarr
      write_timeseriesdict_zarr
      write_timeseries_zarr

