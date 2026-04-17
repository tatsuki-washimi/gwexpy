周波数系列 (Frequency Series)
=============================

.. currentmodule:: gwexpy.frequencyseries

概要
----

.. seealso::

   :doc:`../../user_guide/physics_models`
      周波数領域モデリングとスペクトル解釈の背景。
   :doc:`../FFT_Conventions`
      GWexpy が採用するフーリエ正規化と軸の規約。

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
