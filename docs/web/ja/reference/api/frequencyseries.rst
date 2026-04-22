周波数系列 (Frequency Series)
=============================

.. note::
   ページ種別: 二次 API カテゴリ

**安定性:** 安定

.. currentmodule:: gwexpy.frequencyseries

概要
----

.. note::
   学習ステップ:
   ``FrequencySeries`` の基礎チュートリアルを読んだあとや、フィッティング/スペクトル解析から正確な API に戻りたい場合に使ってください。

.. seealso::

   :doc:`../../user_guide/tutorials/index`
      機能別に学び始めるためのチュートリアル一覧。
   :doc:`../../user_guide/tutorials/intro_frequencyseries`
      API を引く前に確認したい ``FrequencySeries`` の基本例。
   :doc:`../../user_guide/physics_models`
      周波数領域モデリングとスペクトル解釈の背景。
   :doc:`../FFT_Conventions`
      GWexpy が採用するフーリエ正規化と軸の規約。
   :doc:`../Spectral`
      ``FrequencySeries`` を入出力に使う PSD / ASD 推定の概念整理。
   :doc:`../../user_guide/tutorials/case_bootstrap_gls_fitting`
      周波数領域フィッティングに戻れる具体的な事例。
   :doc:`../topics`
      検証前提や規約を概念別に確認するための入口。

.. autosummary::
   :toctree: _autosummary

   FrequencySeries

FrequencySeries クラス
----------------------

.. autoclass:: FrequencySeries
   :no-index:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :member-order: bysource

   .. rubric:: メソッド

   .. autosummary::

      ~FrequencySeries.ifft
      ~FrequencySeries.idct
      ~FrequencySeries.rms
      ~FrequencySeries.abs
      ~FrequencySeries.angle
      ~FrequencySeries.phase
      ~FrequencySeries.filter

モジュール内容
--------------

.. automodule:: gwexpy.frequencyseries
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: FrequencySeries
