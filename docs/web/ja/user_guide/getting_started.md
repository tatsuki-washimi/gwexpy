# はじめに (GWexpy は、Python 3.11 以上で動作する重力波データ解析ライブラリです。
GWpy との高い互換性を維持しつつ、より直感的な API と豊富な解析機能を提供します。
あなたのバックグラウンドと目的に合わせて、最適なスタート地点を選択してください。

## 概要 (Quick Summary)

.. list-table::
   :widths: 25 75

   * - **対象読者**
     - 物理実験データの解析者、GWpy ユーザー、Python での信号処理に関心がある方
   * - **前提知識**
     - Python 3.11+ の基礎、NumPy 配列操作、(推奨) Matplotlib
   * - **所要時間**
     - 最短 5 分（クイックスタート）〜 30 分（基本ハンズオン）
   * - **到達点**
     - データの読み込み、可視化、基本的な周波数解析の実行

## あなたに最適なスタート地点 (Choose Your Path)

.. grid:: 3
    :gutter: 3

    .. grid-item-card:: 🚀 5分で最初のプロット
        :link: quickstart
        :link-type: doc

        対象: Python 3.11 以上の環境が手元にあり、すぐにコードを動かしたい方。
        内容: 最短 3 行でデータを取得・描画するコードと、Google Colab での実行環境を提供します。

    .. grid-item-card:: 📖 30分で基本操作
        :link: tutorials/index
        :link-type: doc

        データの構造や GWexpy 固有の行列演算を基礎から学びます。チュートリアル Notebook を順に進めることで、基本的な解析ワークフローを習得できます。

    .. grid-item-card:: 🔄 GWpy から移行
        :link: gwexpy_for_gwpy_users_ja
        :link-type: doc

        既に GWpy を使っている方向け。主な差分と、新機能を活用したコードの簡略化を学びます。

## 学習ロードマップ (Learning Path)

### 1. 準備

まず :doc:`インストールガイド <installation>` で環境を構築してください。

### 2. 基本データ構造の習得

主要なコンテナの使い方を以下の順序で学ぶことを推奨します：

1. [{doc}`tutorials/intro_timeseries <tutorials/intro_timeseries>`] - 時系列データの基本
2. [{doc}`tutorials/intro_frequencyseries <tutorials/intro_frequencyseries>`] - 周波数系列の基本
3. [{doc}`tutorials/intro_spectrogram <tutorials/intro_spectrogram>`] - スペクトログラムの基本
4. [{doc}`tutorials/intro_plotting <tutorials/intro_plotting>`] - プロット機能のカスタマイズ

### 3. 高度な解析機能

目的に応じて以下のガイドを参照してください：

* **多チャンネル・行列処理**: :doc:`行列コンテナ (Matrix) の活用 <tutorials/matrix_timeseries>`
* **高次元データ**: :doc:`Field API 入門 <tutorials/field_scalar_intro>` / :doc:`スライス操作ガイド <scalarfield_slicing>`
* **信号処理**: :doc:`フィッティング <tutorials/advanced_fitting>` / :doc:`HHT <tutorials/advanced_hht>` / :doc:`ARIMA <tutorials/advanced_arima>`

### 4. 実践的な活用

実際の解析ワークフローは、:doc:`ケーススタディ集 <../examples/index>` で確認できます。

## 次のステップ (Next Steps)

* [{doc}`実例集ギャラリー <../examples/index>`] - 視覚的な使用例とケーススタディ
* 全チュートリアル一覧: [{doc}`tutorials/index <tutorials/index>`]
* API リファレンス: [{doc}`リファレンス <../reference/index>`]
* [{doc}`検証済みアルゴリズム <validated_algorithms>`] - 数値的正確性の検証レポート
