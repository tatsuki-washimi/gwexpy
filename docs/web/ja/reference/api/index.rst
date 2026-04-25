API リファレンス (API Reference)
================================

.. note::
   ページ種別: API カテゴリ索引

**安定性:** 安定

主要モジュールをカテゴリ別に並べています。迷った場合は `timeseries`, `matrix`, `fields`, `signal`, `fitting`, `io` から確認してください。

**対象読者:** どのサブシステムを見たいかが分かっており、カテゴリ単位で API をたどりたい利用者。
**このページの用途:** リファレンス総合入口やチュートリアルから、適切なモジュール群へ進むこと。

.. note::
   高度・理論解説:
   メソッドやシンボルではなく、検証前提、フーリエ規約、監査根拠を確認したい場合は :doc:`../topics` と :doc:`../../user_guide/validated_algorithms` を参照してください。

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - カテゴリ
     - 安定性
     - 対象
   * - 時系列 (Time Series)
     - 安定
     - 時間領域コンテナと関連 API
   * - 周波数系列 (Frequency Series)
     - 安定
     - 周波数領域コンテナと変換 API
   * - スペクトログラム (Spectrogram)
     - 安定
     - 時間周波数コンテナと描画 API
   * - 行列コンテナ (Matrix Containers)
     - 安定
     - バッチ処理向けコンテナ群
   * - フィールド (Fields)
     - 実験的
     - Field 系クラスと演算 API
   * - スペクトル解析 (Spectral)
     - 安定
     - PSD/ASD 推定と関連ヘルパ
   * - 解析ユーティリティ (Analysis)
     - 安定
     - 結合解析、統計、補助解析 API
   * - ノイズシミュレーション (Noise)
     - 安定
     - 合成ノイズモデルと補助関数
   * - 前処理 (Preprocessing)
     - 安定
     - クリーニング、正規化、前処理 API
   * - フィッティング (Fitting)
     - 安定
     - 回帰、GLS、フィッティング API
   * - 基本型 (Types)
     - 安定
     - 共通型と基底コンテナ
   * - テーブル (Table)
     - 安定
     - テーブル系データ構造
   * - ヒストグラム (Histogram)
     - 安定
     - ヒストグラム API
   * - セグメント (Segments)
     - 安定
     - セグメントとデータ品質 API
   * - 信号処理 (Signal Processing)
     - 安定
     - フィルタと信号処理ヘルパ
   * - 時刻・時間 (Time)
     - 安定
     - 時刻変換と GPS 補助 API
   * - 外部連携 (Interoperability)
     - 安定
     - 外部ライブラリ変換 API
   * - 天体物理ユーティリティ (Astrophysics)
     - 安定
     - 天体物理向け補助 API
   * - 検出器ユーティリティ (Detector)
     - 安定
     - 検出器・装置向け補助 API
   * - 描画 (Plotting)
     - 安定
     - 描画と Figure 補助 API
   * - 入出力 (I/O)
     - 安定
     - 読み書きとフォーマット統合
   * - 互換・補助 (Extra)
     - 安定
     - 互換入口と補助 API
   * - コマンドラインインターフェース (CLI)
     - 実験的
     - エントリポイント、バージョン表示、将来の CLI ワークフロー
   * - グラフィカルユーザーインターフェース (GUI)
     - 実験的
     - GUI と対話ツール

.. seealso::
   ハブ間の移動:

   - :doc:`../index` でリファレンス総合入口に戻る
   - :doc:`../topics` で理論、規約、橋渡しページから入る
   - :doc:`../../user_guide/tutorials/index` で機能別チュートリアルから学ぶ
   - :doc:`matrix`, :doc:`frequencyseries`, :doc:`spectrogram` でよく使う二次カテゴリを確認する

.. toctree::
   :maxdepth: 2

   時系列 (Time Series) <timeseries>
   周波数系列 (Frequency Series) <frequencyseries>
   スペクトログラム (Spectrogram) <spectrogram>
   行列コンテナ (Matrix Containers) <matrix>
   フィールド (Fields) <fields>
   スペクトル解析 (Spectral) <spectral>
   解析ユーティリティ (Analysis) <analysis>
   ノイズシミュレーション (Noise) <noise>
   前処理 (Preprocessing) <preprocessing>
   フィッティング (Fitting) <fitting>
   基本型 (Types) <types>
   テーブル (Table) <table>
   ヒストグラム (Histogram) <histogram>
   セグメント (Segments) <segments>
   信号処理 (Signal Processing) <signal>
   時刻・時間 (Time) <time>
   外部連携 (Interoperability) <interop>
   天体物理ユーティリティ (Astrophysics) <astro>
   検出器ユーティリティ (Detector) <detector>
   描画 (Plotting) <plot>
   入出力 (I/O) <io>
   互換・補助 (Extra) <extra>
   コマンドラインインターフェース (CLI) <cli>
   グラフィカルユーザーインターフェース (GUI) <gui>
