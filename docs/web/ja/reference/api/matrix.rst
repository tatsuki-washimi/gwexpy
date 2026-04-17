行列コンテナ
============

複数の :class:`~gwexpy.timeseries.TimeSeries`、:class:`~gwexpy.frequencyseries.FrequencySeries`、
または :class:`~gwexpy.spectrogram.Spectrogram` をグループ化し、
全チャンネルに対して同時にベクトル化された演算を実行する多チャンネルコンテナ。

.. seealso::

   :doc:`../../user_guide/validated_algorithms`
      行列全体に対する FFT・PSD・コヒーレンスなどのアルゴリズム検証。
   :doc:`../../user_guide/tutorials/index`
      `TimeSeriesMatrix`・`FrequencySeriesMatrix`・`SpectrogramMatrix` を使うチュートリアル集。

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
