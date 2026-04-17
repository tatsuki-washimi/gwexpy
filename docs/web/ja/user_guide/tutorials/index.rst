チュートリアル
==============

**GWexpy** の使い方を、対話的な例題（Jupyter Notebook）を通して学びます。

.. tip::
   初めての方は :doc:`../getting_started` から始めることを推奨します。

これらのチュートリアルは Jupyter Notebook から生成されています。ローカルで実行したい場合は、対応する ``.ipynb`` をリポジトリ内の ``docs/web/ja/user_guide/tutorials/`` から取得してください。

.. note::
   命名規則: 日本語タイトルは「機能名: 実施タスク」で統一します。
   API 固有名詞のみ英語表記を残し、一般語は日本語を優先します。

I. 基本データ構造
-----------------
基本的なデータコンテナと操作方法について説明します。

II. 多チャンネル & 行列コンテナ
----------------------------------------
Matrixクラスを使用して、複数のチャンネルを効率的に扱う方法を説明します。

III. 高次元フィールド (Field API)
---------------------------------
4次元時空におけるスカラ場、ベクトル場、テンソル場を扱うための次世代 API について説明します。

IV. 高度な信号処理
------------------
統計的分析や、高度な信号変換手法について説明します。

V. 特殊ツール
--------------
ノイズ源特定や診断タスクのための専用ツールについて説明します。

VI. セグメント解析
------------------
時間区間（セグメント）をベースとした表形式の解析手法について説明します。

.. toctree::
   :maxdepth: 1

   TimeSeries: 基本 <intro_timeseries>
   FrequencySeries: 基本 <intro_frequencyseries>
   Spectrogram: 基本 <intro_spectrogram>
   ノイズ生成: 基本 <intro_noise>
   Plotting: 基本 <intro_plotting>
   マッププロット: 基本 <intro_mapplotting>
   相互運用: 基本 <intro_interop>
   ヒストグラム: 基本 <intro_histogram>
   TimeSeriesMatrix: 行列処理の基本 <matrix_timeseries>
   FrequencySeriesMatrix: 行列処理の基本 <matrix_frequencyseries>
   SpectrogramMatrix: 行列処理の基本 <matrix_spectrogram>
   Field API: ScalarField の基本 <field_scalar_intro>
   Field API: VectorField の基本 <field_vector_intro>
   Field API: TensorField の基本 <field_tensor_intro>
   Field API: ScalarField の信号処理 <field_scalar_signal>
   Field API: 高度解析ワークフロー <field_advanced_workflow>
   フィッティング: 基本 <intro_fitting>
   フィッティング: スペクトル線解析 <advanced_fitting>
   スペクトログラム: 正規化とクリーニング <advanced_spectrogram_processing>
   ケーススタディ: Bootstrap PSD と GLS フィッティング <case_bootstrap_gls_fitting>
   ピーク検出: 基本 <advanced_peak_detection>
   ピーク追跡: 時間変化の解析 <advanced_peak_tracking>
   HHT: 解析 <advanced_hht>
   時間-周波数解析: インタラクティブ比較 <time_frequency_analysis_comparison>
   時間-周波数解析: 手法比較ガイド <time_frequency_comparison>
   ARIMA: 時系列予測 <advanced_arima>
   相関解析: 統計的手法 <advanced_correlation>
   ML 前処理: 個別手法 <ml_preprocessing_methods>
   ケーススタディ: ML 前処理パイプライン <case_ml_preprocessing>
   線形代数: 重力波解析への応用 <advanced_linear_algebra>
   Field API: 高度解析の統合 <field_advanced_integration>
   非ガウス雑音解析: Rayleigh と Gaussian-Chi <rayleigh_gauch_tutorial>
   カップリング解析: マルチチャンネル結合 <advanced_coupling>
   分解解析: PCA・ICA と固有モード <advanced_decomposition>
   ケーススタディ: ObsPy 連携による地震データ解析 <case_seismic_obspy>
   ケーススタディ: GBD 形式 I/O <case_gbd_format>
   BruCo: 基本 <advanced_bruco>
   ケーススタディ: BruCo と ICA によるノイズ削減 <case_bruco_ica_denoising>
   BruCo: バイリニアカップリングと AM/FM 復調 <case_bruco_advanced>
   ケーススタディ: バイオリンモード解析 <case_violin_mode>
   ケーススタディ: シューマン共鳴解析 <case_schumann_resonance>
   SegmentTable: 基本 <intro_segment_table>
   セグメント解析: 基本パイプライン <intro_table>
   ASD 解析: パイプライン <segment_asd_pipeline>
   セグメント解析: 可視化 <segment_visualization>
   ケーススタディ: イベント同期解析 <case_segment_analysis>

.. note::
   実践的なケーススタディと応用例は :doc:`../../examples/index` に統合されています。
   ノイズバジェット解析、伝達関数計算、アクティブダンピングなどの実例集を参照してください。
