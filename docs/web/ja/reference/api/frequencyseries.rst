周波数系列 (Frequency Series)
=============================

**安定性:** Stable

.. currentmodule:: gwexpy.frequencyseries

概要
----

.. seealso::

   :doc:`../../user_guide/physics_models`
      周波数領域モデリングとスペクトル解釈の背景。
   :doc:`../FFT_Conventions`
      GWexpy が採用するフーリエ正規化と軸の規約。
   :doc:`../Spectral`
      ``FrequencySeries`` を入出力に使う PSD / ASD 推定の概念整理。
   :doc:`../../user_guide/tutorials/case_bootstrap_gls_fitting`
      周波数領域フィッティングに戻れる具体的な事例。

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
