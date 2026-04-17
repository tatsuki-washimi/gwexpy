スペクトログラム (Spectrogram)
==============================

**安定性:** Stable

.. currentmodule:: gwexpy.spectrogram

概要
----

.. seealso::

   :doc:`../FFT_Conventions`
      GWexpy が採用するフーリエ正規化と軸の規約。
   :doc:`../../user_guide/tutorials/case_signal_extraction`
      ``Spectrogram`` 系 API に戻れる時間周波数事例。
   :doc:`../../user_guide/numerical_stability`
      FFT ベースの時間周波数解析で確認すべき安定化ノート。

.. autosummary::
   :toctree: _autosummary

   Spectrogram

Spectrogram クラス
------------------

.. autoclass:: Spectrogram
   :no-index:
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
   :member-order: bysource

   .. rubric:: メソッド

   .. autosummary::

      ~Spectrogram.plot
      ~Spectrogram.crop
      ~Spectrogram.percentile
      ~Spectrogram.ratio
      ~Spectrogram.filter

モジュール内容
--------------

.. automodule:: gwexpy.spectrogram
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: Spectrogram
