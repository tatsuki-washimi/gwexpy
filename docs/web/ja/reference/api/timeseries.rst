時系列 (Time Series)
=====================

.. currentmodule:: gwexpy.timeseries

概要
----

.. seealso::

   :doc:`../../user_guide/validated_algorithms`
      FFT・PSD・ASD・コヒーレンス推定の検証ノート。
   :doc:`../FFT_Conventions`
      GWexpy が採用するフーリエ正規化と軸の規約。

.. autosummary::
   :toctree: _autosummary

   TimeSeries

TimeSeries クラス
-----------------

.. autoclass:: TimeSeries
   :no-index:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :member-order: bysource

   .. rubric:: メソッド

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

モジュール内容
--------------

.. automodule:: gwexpy.timeseries
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: TimeSeries
