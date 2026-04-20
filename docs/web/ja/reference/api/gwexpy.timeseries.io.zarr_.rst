gwexpy.timeseries.io.zarr\_
===========================

.. note::

   現在の挙動は Phase 7 Truth Ledger と ``#8-19`` fail-fast 修正に従います。

   - 配列ごとの timing metadata は ``sample_rate`` を優先します。
   - ``dt`` だけがある場合は ``sample_rate = 1 / dt`` として復元します。
   - 両方ない場合は、以前のように暗黙で ``1 Hz`` を仮定せず
     ``ValueError`` を送出します。
   - legacy store を意図的に読む場合だけ、
     ``sample_rate_override=...`` または ``dt_override=...`` のどちらか一方を
     明示してください。

.. automodule:: gwexpy.timeseries.io.zarr_

   .. rubric:: Functions

   .. autosummary::

      read_timeseriesdict_zarr
      read_timeseries_zarr
      read_timeseriesmatrix_zarr
      write_timeseriesdict_zarr
      write_timeseries_zarr
