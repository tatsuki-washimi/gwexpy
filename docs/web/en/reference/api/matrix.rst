Matrix Containers
=================

.. note::
   Page role: Secondary API category

**Stability:** Stable

Multi-channel containers that group multiple :class:`~gwexpy.timeseries.TimeSeries`,
:class:`~gwexpy.frequencyseries.FrequencySeries`, or :class:`~gwexpy.spectrogram.Spectrogram`
objects and expose vectorized operations across all channels simultaneously.

.. note::
   Learning path:
   Start here after the matrix-oriented tutorials if you want class members and exact method signatures.

.. seealso::

   :doc:`../api/timeseries`
      Container-level time-series API that many matrix methods mirror.
   :doc:`../api/frequencyseries`
      Frequency-domain container API used by `FrequencySeriesMatrix`.
   :doc:`../api/spectrogram`
      Time-frequency container API used by `SpectrogramMatrix`.
   :doc:`../../user_guide/tutorials/case_signal_extraction`
      Multi-channel time-frequency workflow with matrix-style analysis.
   :doc:`../../user_guide/tutorials/index`
      Tutorial hub for matrix-first learning paths before API lookup.
   :doc:`../topics`
      Concept hub for conventions and advanced/theory notes tied to matrix workflows.

Time Series Matrix
------------------

.. currentmodule:: gwexpy.timeseries

.. autosummary::
   :toctree: _autosummary

   TimeSeriesMatrix

.. autoclass:: TimeSeriesMatrix
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Frequency Series Matrix
-----------------------

.. currentmodule:: gwexpy.frequencyseries

.. autosummary::
   :toctree: _autosummary

   FrequencySeriesMatrix

.. autoclass:: FrequencySeriesMatrix
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

Spectrogram Matrix
------------------

.. currentmodule:: gwexpy.spectrogram

.. autosummary::
   :toctree: _autosummary

   SpectrogramMatrix

.. autoclass:: SpectrogramMatrix
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
