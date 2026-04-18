スペクトログラム (Spectrogram)
==============================

.. note::
   ページ種別: 二次 API カテゴリ

**安定性:** Stable

.. currentmodule:: gwexpy.spectrogram

概要
----

.. note::
   学習導線:
   入門チュートリアルの後や、時間周波数ワークフローから正確な API 詳細に戻りたい場合にこのページを使ってください。

.. seealso::

   :doc:`../../user_guide/tutorials/index`
      機能別に学び始めるためのチュートリアル一覧。
   :doc:`../../user_guide/tutorials/intro_spectrogram`
      API を引く前に確認したい ``Spectrogram`` の基本例。
   :doc:`../FFT_Conventions`
      GWexpy が採用するフーリエ正規化と軸の規約。
   :doc:`../../user_guide/tutorials/case_signal_extraction`
      ``Spectrogram`` 系 API に戻れる時間周波数事例。
   :doc:`../../user_guide/numerical_stability`
      FFT ベースの時間周波数解析で確認すべき安定化ノート。
   :doc:`../topics`
      規約や高度・理論系の導線を概念別にたどる入口。

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
