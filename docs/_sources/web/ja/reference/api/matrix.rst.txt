行列コンテナ (Matrix Containers)
================================

.. note::
   ページ種別: 二次 API カテゴリ

**安定性:** 安定

複数の :class:`~gwexpy.timeseries.TimeSeries`、:class:`~gwexpy.frequencyseries.FrequencySeries`、
または :class:`~gwexpy.spectrogram.Spectrogram` をグループ化し、
全チャンネルに対して同時にベクトル化された演算を実行する多チャンネルコンテナ。

.. note::
   学習ステップ:
   行列系チュートリアルを読んだあとに、メンバーや正確なシグネチャを確認したい場合はこのページを起点にしてください。

.. seealso::

   :doc:`../api/timeseries`
      行列 API と対になる時系列コンテナ API。
   :doc:`../api/frequencyseries`
      `FrequencySeriesMatrix` と対応する周波数領域コンテナ API。
   :doc:`../api/spectrogram`
      `SpectrogramMatrix` と対応する時間周波数コンテナ API。
   :doc:`../../user_guide/tutorials/case_signal_extraction`
      matrix 系の時間周波数解析へつながる具体例。
   :doc:`../../user_guide/tutorials/index`
      行列中心の学習を始めるチュートリアル一覧。
   :doc:`../topics`
      行列ワークフローに関わる規約や高度ノートの入口。

時系列行列
------------------

.. currentmodule:: gwexpy.timeseries

.. autosummary::
   :toctree: _autosummary

   TimeSeriesMatrix

.. autoclass:: TimeSeriesMatrix
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

周波数系列行列
------------------

.. currentmodule:: gwexpy.frequencyseries

.. autosummary::
   :toctree: _autosummary

   FrequencySeriesMatrix

.. autoclass:: FrequencySeriesMatrix
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource

スペクトログラム行列
--------------------

.. currentmodule:: gwexpy.spectrogram

.. autosummary::
   :toctree: _autosummary

   SpectrogramMatrix

.. autoclass:: SpectrogramMatrix
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
   :member-order: bysource
