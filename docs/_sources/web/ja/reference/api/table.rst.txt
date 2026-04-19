テーブル (Table)
================

**安定性:** 安定

.. currentmodule:: gwexpy.table

概要
----

.. autosummary::
   :toctree: _autosummary

   SegmentTable
   SegmentCell

SegmentTable クラス
-------------------

.. autoclass:: gwexpy.table.segment_table.SegmentTable
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

   .. rubric:: ファクトリメソッド

   .. autosummary::

      ~SegmentTable.from_segments
      ~SegmentTable.from_table

   .. rubric:: カラム管理

   .. autosummary::

      ~SegmentTable.add_column
      ~SegmentTable.add_series_column

   .. rubric:: 行単位の処理

   .. autosummary::

      ~SegmentTable.apply
      ~SegmentTable.map
      ~SegmentTable.crop
      ~SegmentTable.asd

   .. rubric:: 選択と変換

   .. autosummary::

      ~SegmentTable.select
      ~SegmentTable.fetch
      ~SegmentTable.materialize
      ~SegmentTable.to_pandas
      ~SegmentTable.copy

   .. rubric:: 描画 (代表的な API)

   .. autosummary::

      ~SegmentTable.segments
      ~SegmentTable.overlay_spectra
      ~SegmentTable.plot
      ~SegmentTable.scatter
      ~SegmentTable.hist
      ~SegmentTable.overlay

SegmentCell クラス
------------------

.. autoclass:: gwexpy.table.segment_cell.SegmentCell
   :members:
   :undoc-members:
   :show-inheritance:

モジュール
----------

.. automodule:: gwexpy.table.segment_table
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: SegmentTable, RowProxy
