入出力 (I/O)
============

**安定性:** 安定

.. currentmodule:: gwexpy.io

GWpy I/O レジストリに登録されたファイル形式の読み書きツール。
サポートされている形式には、HDF5、GWF フレーム、NDS2、LigoDW XML、TDMS、および Zarr が含まれます。

Zarr reader は配列ごとの timing metadata が欠落している場合に fail-fast します。
``sample_rate`` または ``dt`` を持たない legacy store を読む場合は、
``sample_rate_override=...`` か ``dt_override=...`` を明示的に渡してください。
専用の TimeSeries Zarr モジュール参照は
:doc:`gwexpy.timeseries.io.zarr_ <gwexpy.timeseries.io.zarr_>` を見てください。

.. automodule:: gwexpy.io
   :members:
   :undoc-members:
   :show-inheritance:
