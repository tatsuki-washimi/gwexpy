.. meta::
   :description: GWexpy のチュートリアル一覧です。データ構造、解析ワークフロー、難易度、目的から notebook を選べます。

.. _tutorials-ja-entry:

チュートリアル
==============

.. note::
   ページ種別: ガイド索引

**GWexpy** の使い方を、対話的な例題（Jupyter Notebook）を通して学びます。

**対象読者:** GWexpy をこれから学ぶ人、GWpy から移行する人、解析実務で手順を確認したい人。
**前提知識:** GWexpy の実行環境があり、:doc:`../getting_started` の基本事項を把握していること。
**このページの用途:** データ構造、ワークフロー、難易度から適切なチュートリアルを選ぶこと。
**検索のヒント:** notebook, Jupyter, チュートリアル一覧, 初学者向け, フィールド API, 信号処理, 相互運用

.. tip::
   初めての方は :doc:`../getting_started` から始めることを推奨します。

これらのチュートリアルは Jupyter Notebook から生成されています。ローカルで実行したい場合は、対応する ``.ipynb`` をリポジトリ内の ``docs/web/ja/user_guide/tutorials/`` から取得してください。

.. note::
   このページの見方:
   I-II は基礎、III は Field API による高次元フィールド、IV-VI は解析ワークフロー、VII はセグメント表ベースの解析をまとめています。

.. note::
   命名規則: 日本語タイトルは「機能名: 実施タスク」で統一します。
   API 固有名詞のみ英語表記を残し、一般語は日本語を優先します。

.. note::
   各項目には、難易度・所要時間の目安・主な対象読者を付けています。
   対象読者は `初学者`・`GWpyユーザー`・`解析実務者` の 3 分類で整理しています。

.. note::
   `Tutorials` はクラスや機能ごとの使い方を学ぶための例です。
   複数機能を組み合わせたテーマ別の実演は、正本一覧である :doc:`../../examples/index` の `Case Studies` に集約しています。

.. note::
   リファレンスへの橋渡し:
   チュートリアルの後で正確なクラス、メソッド、規約を確認したい場合は、まず :doc:`../../reference/index` に進み、必要に応じて :doc:`../../reference/api/index` または :doc:`../../reference/topics` を開いてください。

.. _tutorials-ja-core-entry:

I. 基本データ構造
-----------------
基本的なデータコンテナと操作方法について説明します。

- :doc:`TimeSeries: 基本 <intro_timeseries>` :bdg-primary:`初級` :bdg-secondary:`15分` :bdg-info:`初学者`
- :doc:`FrequencySeries: 基本 <intro_frequencyseries>` :bdg-primary:`初級` :bdg-secondary:`15分` :bdg-info:`初学者`
- :doc:`Spectrogram: 基本 <intro_spectrogram>` :bdg-primary:`初級` :bdg-secondary:`15分` :bdg-info:`初学者`
- :doc:`ノイズ生成: 基本 <intro_noise>` :bdg-primary:`初級` :bdg-secondary:`15分` :bdg-info:`初学者`
- :doc:`Plotting: 基本 <intro_plotting>` :bdg-primary:`初級` :bdg-secondary:`15分` :bdg-info:`初学者`
- :doc:`マッププロット: 基本 <intro_mapplotting>` :bdg-primary:`初級` :bdg-secondary:`20分` :bdg-info:`GWpyユーザー`
- :doc:`ヒストグラム: 基本 <intro_histogram>` :bdg-primary:`初級` :bdg-secondary:`15分` :bdg-info:`初学者`

II. 多チャンネル & 行列コンテナ
----------------------------------------
Matrixクラスを使用して、複数のチャンネルを効率的に扱う方法を説明します。

- :doc:`TimeSeriesMatrix: 行列処理の基本 <matrix_timeseries>` :bdg-primary:`初級` :bdg-secondary:`20分` :bdg-info:`GWpyユーザー`
- :doc:`FrequencySeriesMatrix: 行列処理の基本 <matrix_frequencyseries>` :bdg-primary:`初級` :bdg-secondary:`20分` :bdg-info:`GWpyユーザー`
- :doc:`SpectrogramMatrix: 行列処理の基本 <matrix_spectrogram>` :bdg-primary:`初級` :bdg-secondary:`20分` :bdg-info:`GWpyユーザー`

III. 高次元フィールド (Field API)
---------------------------------
4次元時空におけるスカラ場、ベクトル場、テンソル場を扱うための次世代 API について説明します。

- :doc:`Field API: ScalarField の基本 <field_scalar_intro>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`
- :doc:`Field API: VectorField の基本 <field_vector_intro>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`
- :doc:`Field API: TensorField の基本 <field_tensor_intro>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`
- :doc:`Field API: ScalarField の信号処理 <field_scalar_signal>` :bdg-primary:`中級` :bdg-secondary:`45分` :bdg-info:`解析実務者`
- :doc:`Field API: 高度解析ワークフロー <field_advanced_workflow>` :bdg-primary:`中級` :bdg-secondary:`45分` :bdg-info:`解析実務者`
- :doc:`Field API: 高度解析の統合 <field_advanced_integration>` :bdg-primary:`中級` :bdg-secondary:`45分` :bdg-info:`解析実務者`

IV. 高度な信号処理
------------------
統計的分析や、高度な信号変換手法について説明します。

- :doc:`フィッティング: 基本 <intro_fitting>` :bdg-primary:`初級` :bdg-secondary:`20分` :bdg-info:`GWpyユーザー`
- :doc:`フィッティング: スペクトル線解析 <advanced_fitting>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`
- :doc:`スペクトログラム: 正規化とクリーニング <advanced_spectrogram_processing>` :bdg-primary:`中級` :bdg-secondary:`45分` :bdg-info:`解析実務者`
- :doc:`ピーク検出: 基本 <advanced_peak_detection>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`
- :doc:`ピーク追跡: 時間変化の解析 <advanced_peak_tracking>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`
- :doc:`HHT: 解析 <advanced_hht>` :bdg-primary:`上級` :bdg-secondary:`45分` :bdg-info:`解析実務者`
- :doc:`時間-周波数解析: インタラクティブ比較 <time_frequency_analysis_comparison>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`
- :doc:`時間-周波数解析: 手法比較ガイド <time_frequency_comparison>` :bdg-primary:`中級` :bdg-secondary:`45分` :bdg-info:`解析実務者`
- :doc:`制御解析: 離散化の基礎 <advanced_control_discretization>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`
- :doc:`制御解析: 共振とフィードバックの基礎 <advanced_control_basics>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`
- :doc:`制御解析: 実測応答からのプラントモデリング <advanced_control_modeling>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`
- :doc:`ARIMA: 時系列予測 <advanced_arima>` :bdg-primary:`上級` :bdg-secondary:`45分` :bdg-info:`解析実務者`
- :doc:`相関解析: 統計的手法 <advanced_correlation>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`
- :doc:`ML 前処理: 個別手法 <ml_preprocessing_methods>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`
- :doc:`線形代数: 重力波解析への応用 <advanced_linear_algebra>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`
- :doc:`非ガウス雑音解析: Rayleigh と Gaussian-Chi <rayleigh_gauch_tutorial>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`
- :doc:`カップリング解析: マルチチャンネル結合 <advanced_coupling>` :bdg-primary:`上級` :bdg-secondary:`45分` :bdg-info:`解析実務者`
- :doc:`分解解析: PCA・ICA と固有モード <advanced_decomposition>` :bdg-primary:`上級` :bdg-secondary:`45分` :bdg-info:`解析実務者`

V. データ I/O と相互運用
-------------------------
ファイルの読み書き、外部ライブラリとの変換、観測データの取り込みを扱います。

- :doc:`相互運用: 基本 <intro_interop>` :bdg-primary:`初級` :bdg-secondary:`20分` :bdg-info:`GWpyユーザー`

VI. ノイズハンティングと特殊ツール
-----------------------------------
ノイズ源特定や診断タスクのための専用ツールについて説明します。

- :doc:`BruCo: 基本 <advanced_bruco>` :bdg-primary:`上級` :bdg-secondary:`45分` :bdg-info:`解析実務者`

.. _tutorials-ja-segment-entry:

VII. セグメント解析
-------------------
時間区間（セグメント）をベースとした表形式の解析手法について説明します。

- :doc:`SegmentTable: 基本 <intro_segment_table>` :bdg-primary:`初級` :bdg-secondary:`15分` :bdg-info:`初学者`
- :doc:`セグメント解析: 基本パイプライン <intro_table>` :bdg-primary:`初級` :bdg-secondary:`20分` :bdg-info:`GWpyユーザー`
- :doc:`ASD 解析: パイプライン <segment_asd_pipeline>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`
- :doc:`セグメント解析: 可視化 <segment_visualization>` :bdg-primary:`中級` :bdg-secondary:`30分` :bdg-info:`解析実務者`

.. note::
   テーマ別の実演は :doc:`../../examples/index` を参照してください。
   ノイズバジェット解析、伝達関数計測、ML 前処理パイプライン、イベント同期解析などをまとめています。
   次の一歩としては、そこにある `ML 前処理パイプライン` と
   `Bootstrap PSD と GLS フィッティング` を参照してください。

.. seealso::
   次に読むページ:

   - :doc:`../getting_started` でユーザーガイドの導入を確認する
   - :doc:`../../reference/index` で API・クラス・トピック別の詳細を調べる
   - :doc:`../../reference/api/index` でカテゴリ別の API 入口を開く
   - :doc:`../../reference/topics` で高度・理論、規約、橋渡しページを確認する

.. toctree::
   :hidden:
   :maxdepth: 1

   intro_timeseries
   intro_frequencyseries
   intro_spectrogram
   intro_noise
   intro_plotting
   intro_mapplotting
   intro_histogram
   matrix_timeseries
   matrix_frequencyseries
   matrix_spectrogram
   field_scalar_intro
   field_vector_intro
   field_tensor_intro
   field_scalar_signal
   field_advanced_workflow
   intro_fitting
   advanced_fitting
   advanced_spectrogram_processing
   advanced_peak_detection
   advanced_peak_tracking
   advanced_hht
   time_frequency_analysis_comparison
   time_frequency_comparison
   advanced_control_discretization
   advanced_control_basics
   advanced_control_modeling
   advanced_arima
   advanced_correlation
   ml_preprocessing_methods
   advanced_linear_algebra
   field_advanced_integration
   rayleigh_gauch_tutorial
   advanced_coupling
   advanced_decomposition
   intro_interop
   advanced_bruco
   intro_segment_table
   intro_table
   segment_asd_pipeline
   segment_visualization
