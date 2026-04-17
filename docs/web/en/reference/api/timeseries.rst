Time Series
===========

**Stability:** Stable

.. currentmodule:: gwexpy.timeseries

Overview
--------

.. seealso::

   :doc:`../../user_guide/validated_algorithms`
      Validation notes for FFT, PSD, ASD, and coherence-related estimators.
   :doc:`../FFT_Conventions`
      Fourier normalization and axis conventions used by GWexpy.

.. autosummary::
   :toctree: _autosummary

   TimeSeries

TimeSeries Class
----------------

.. autoclass:: TimeSeries
   :no-index:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :member-order: bysource

   .. rubric:: Methods

   .. autosummary::

      ~TimeSeries.fft
      ~TimeSeries.rfft
      ~TimeSeries.psd
      ~TimeSeries.asd
      ~TimeSeries.csd
      ~TimeSeries.spectrogram
      ~TimeSeries.coherence
      ~TimeSeries.filter
      ~TimeSeries.resample
      ~TimeSeries.detrend
      ~TimeSeries.cepstrum
      ~TimeSeries.cwt
      ~TimeSeries.dct
      ~TimeSeries.emd
      ~TimeSeries.hht
      ~TimeSeries.hilbert_analysis

Module Contents
---------------

.. automodule:: gwexpy.timeseries
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: TimeSeries
