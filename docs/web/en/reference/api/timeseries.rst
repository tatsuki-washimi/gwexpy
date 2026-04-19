Time Series
===========

**Stability:** Stable

.. currentmodule:: gwexpy.timeseries

Overview
--------

.. seealso::

   :ref:`validated-en-transient-fft`
      Validation assumptions and evidence for transient-amplitude FFT behavior.
   :ref:`validated-en-arima-forecast`
      GPS timestamp assumptions for ``ArimaResult.forecast()``.
   :ref:`validated-en-mcmc-gls`
      Likelihood assumptions relevant when time-series data flow into GLS or MCMC fitting paths.
   :doc:`../FFT_Conventions`
      Fourier normalization and axis conventions used by GWexpy.
   :doc:`../../user_guide/tutorials/intro_timeseries`
      Introductory walkthrough for the core ``TimeSeries`` API before the reference details below.
   :doc:`../../user_guide/tutorials/matrix_timeseries`
      Multi-channel and matrix-style time-series tutorial that complements collection-oriented APIs.
   :doc:`../../user_guide/tutorials/case_signal_extraction`
      Example workflow that uses transient-style time-frequency analysis paths.
   :doc:`../../user_guide/tutorials/advanced_arima`
      Time-series forecasting tutorial that maps back to ``ArimaResult.forecast()``.

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
