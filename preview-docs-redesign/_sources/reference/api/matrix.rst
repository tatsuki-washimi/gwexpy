Matrix Containers
=================

Multi-channel containers that group multiple :class:`~gwexpy.timeseries.TimeSeries`,
:class:`~gwexpy.frequencyseries.FrequencySeries`, or :class:`~gwexpy.spectrogram.Spectrogram`
objects and expose vectorized operations across all channels simultaneously.

.. note::
   Learning path:
   Start here after the matrix-oriented tutorials if you want class members and exact method signatures.

Time Series Matrix
------------------

.. currentmodule:: gwexpy.timeseries

.. autosummary::
   :toctree:

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
   :toctree:

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
   :toctree:

   SpectrogramMatrix

.. autoclass:: SpectrogramMatrix
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
