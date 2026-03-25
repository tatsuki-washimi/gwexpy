Table
=====

.. currentmodule:: gwexpy.table

Overview
--------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   SegmentTable
   SegmentCell

SegmentTable Class
------------------

.. autoclass:: gwexpy.table.segment_table.SegmentTable
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

   .. rubric:: Factory Methods

   .. autosummary::
      :nosignatures:

      ~SegmentTable.from_segments
      ~SegmentTable.from_table

   .. rubric:: Column Management

   .. autosummary::
      :nosignatures:

      ~SegmentTable.add_column
      ~SegmentTable.add_series_column

   .. rubric:: Row-wise Processing

   .. autosummary::
      :nosignatures:

      ~SegmentTable.apply
      ~SegmentTable.map
      ~SegmentTable.crop
      ~SegmentTable.asd

   .. rubric:: Selection & Conversion

   .. autosummary::
      :nosignatures:

      ~SegmentTable.select
      ~SegmentTable.fetch
      ~SegmentTable.materialize
      ~SegmentTable.to_pandas
      ~SegmentTable.copy

   .. rubric:: Drawing (Representative APIs)

   .. autosummary::
      :nosignatures:

      ~SegmentTable.segments
      ~SegmentTable.overlay_spectra
      ~SegmentTable.plot
      ~SegmentTable.scatter
      ~SegmentTable.hist
      ~SegmentTable.overlay

SegmentCell Class
-----------------

.. autoclass:: gwexpy.table.segment_cell.SegmentCell
   :members:
   :undoc-members:
   :show-inheritance:

Module Contents
---------------

.. automodule:: gwexpy.table.segment_table
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: SegmentTable, RowProxy
