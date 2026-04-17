Matrix Containers
=================

**Stability:** Stable

Multi-channel containers that group multiple :class:`~gwexpy.timeseries.TimeSeries`,
:class:`~gwexpy.frequencyseries.FrequencySeries`, or :class:`~gwexpy.spectrogram.Spectrogram`
objects and expose vectorized operations across all channels simultaneously.

.. seealso::

   :doc:`../../user_guide/validated_algorithms`
      Algorithm notes for matrix-wide FFT, PSD, coherence, and related estimators.
   :doc:`../../user_guide/tutorials/index`
      End-to-end tutorials using `TimeSeriesMatrix`, `FrequencySeriesMatrix`, and `SpectrogramMatrix`.

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
