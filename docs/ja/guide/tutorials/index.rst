チュートリアル
==============

**GWExPy** の使い方を、対話的な例題（Jupyter Notebook）を通して学びます。

.. note::
   これらのチュートリアルは Jupyter Notebook から生成されています。ローカルで実行するには、各ページの右上からダウンロードできます。

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
------------------------------
Matrixクラスを使用して、複数のチャンネルを効率的に扱う方法を説明します。

.. toctree::
   :maxdepth: 1
   :caption: 行列コンテナ

   matrix_timeseries
   matrix_frequencyseries
   matrix_spectrogram

III. 高次元フィールド (Field API)
---------------------------------
スカラ場、ベクトル場、テンソル場を扱うための次世代 API について説明します。

.. toctree::
   :maxdepth: 1
   :caption: フィールド API

   field_scalar_intro
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
   advanced_arima
   advanced_correlation

V. 特殊ツール
--------------
ノイズ源特定や診断タスクのための専用ツールについて説明します。

.. toctree::
   :maxdepth: 1
   :caption: 特殊ツール

   advanced_bruco

VI. ケーススタディ (具体例)
---------------------------
重力波データ解析における実践的な応用例を紹介します。

.. toctree::
   :maxdepth: 1
   :caption: ケーススタディ

   case_noise_budget
   case_transfer_function
   case_active_damping
