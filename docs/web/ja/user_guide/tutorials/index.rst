チュートリアル
==============

**GWexpy** の使い方を、対話的な例題（Jupyter Notebook）を通して学びます。

.. tip::
   初めての方は :doc:`../getting_started` から始めることを推奨します。

.. note::
   これらのチュートリアルは Jupyter Notebook から生成されています。ローカルで実行するには、各ページ右上の "Edit on GitHub" から `.ipynb` を取得できます。

I. 基本データ構造
-----------------
基本的なデータコンテナと操作方法について説明します。

.. toctree::
   :maxdepth: 1
   :caption: 基本データ構造

   intro_timeseries
   intro_frequencyseries
   intro_spectrogram
   intro_plotting
   intro_mapplotting
   intro_interop

II. 多チャンネル & 行列コンテナ
----------------------------------------
Matrixクラスを使用して、複数のチャンネルを効率的に扱う方法を説明します。

.. toctree::
   :maxdepth: 1
   :caption: 行列コンテナ

   matrix_timeseries
   matrix_frequencyseries
   matrix_spectrogram

III. 高次元フィールド (Field API)
---------------------------------
4次元時空におけるスカラ場、ベクトル場、テンソル場を扱うための次世代 API について説明します。

.. toctree::
   :maxdepth: 1
   :caption: フィールド API

   field_scalar_intro
   field_vector_intro
   field_tensor_intro
   field_scalar_signal

IV. 高度な信号処理
------------------
統計的分析や、高度な信号変換手法について説明します。

.. toctree::
   :maxdepth: 1
   :caption: 高度な解析

   advanced_fitting
   advanced_peak_detection
   advanced_hht
   時間-周波数解析の包括的比較 (Notebook) <time_frequency_analysis_comparison>
   時間-周波数解析手法の比較 <time_frequency_comparison>
   advanced_arima
   advanced_correlation
   ML前処理手法 <ml_preprocessing_methods>
   重力波解析のための線形代数 <advanced_linear_algebra>
   Field API × 高度な解析統合 <field_advanced_integration>

V. 特殊ツール
--------------
ノイズ源特定や診断タスクのための専用ツールについて説明します。

.. toctree::
   :maxdepth: 1
   :caption: 特殊ツール

   advanced_bruco

.. note::
   実践的なケーススタディと応用例は :doc:`../examples/index` に統合されています。
   ノイズバジェット解析、伝達関数計算、アクティブダンピングなどの実例集を参照してください。
