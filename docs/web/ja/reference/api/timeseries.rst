時系列 (Time Series)
====================

**安定性:** Stable

.. currentmodule:: gwexpy.timeseries

概要
----

.. seealso::

   :ref:`validated-ja-transient-fft`
      トランジェント FFT の振幅規約と前提条件。
   :ref:`validated-ja-arima-forecast`
      ``ArimaResult.forecast()`` の GPS 時刻前提。
   :ref:`validated-ja-mcmc-gls`
      時系列データが GLS / MCMC フィットに渡る際の尤度前提。
   :doc:`../FFT_Conventions`
      GWexpy が採用するフーリエ正規化と軸の規約。
   :doc:`../../user_guide/tutorials/intro_timeseries`
      ``TimeSeries`` の基本操作を先に確認するための入門チュートリアル。
   :doc:`../../user_guide/tutorials/matrix_timeseries`
      複数チャネルや行列系の時系列処理を扱うチュートリアル。
   :doc:`../../user_guide/tutorials/case_signal_extraction`
      transient 系の解析導線を含む時系列チュートリアル。
   :doc:`../../user_guide/tutorials/advanced_arima`
      ``ArimaResult.forecast()`` に戻れる ARIMA チュートリアル。

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
