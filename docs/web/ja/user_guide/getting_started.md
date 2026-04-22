---
myst:
  html_meta:
    description: "GWexpy の学習を始める入口として、クイックスタート、チュートリアル、GWpy からの移行、学習ロードマップを整理した案内ページです。"
---

# はじめに (Getting Started)

GWexpy は、Python 3.11 以上で動作する重力波データ解析ライブラリです。GWpy との高い互換性を維持しつつ、より直感的な API と豊富な解析機能を提供します。
あなたのバックグラウンドと目的に合わせて、最適なスタート地点を選択してください。

## 概要 (Quick Summary)

| 項目 | 内容 |
| --- | --- |
| **ページ種別** | ガイド |
| **対象読者** | 物理実験データの解析者、GWpy ユーザー、Python での信号処理に関心がある方 |
| **前提知識** | Python 3.11+ の基礎、NumPy 配列操作、(推奨) Matplotlib |
| **こんなときに読む** | 学習順序を決めたい、GWpy からの入口を探したい、最短ルートを知りたい |
| **所要時間** | 最短 5 分（クイックスタート）〜 30 分（基本ハンズオン） |
| **到達点** | データの読み込み、可視化、基本的な周波数解析の実行 |
| **検索キーワード** | getting started, 学習ロードマップ, GWpy 移行, チュートリアル入口 |

共通の前提条件と FFT・時刻系の規約を先に確認したい場合は、[前提条件と規約](prerequisites_and_conventions.md) を入口として使ってください。

## クイックリンク

- [あなたに最適なスタート地点](#あなたに最適なスタート地点-choose-your-path)
- [学習ロードマップ](#learning-path)
- [次のステップ](#next-to-read)

## あなたに最適なスタート地点 (Choose Your Path)

### 🚀 5分で最初のプロット

[クイックスタート](quickstart.md)

対象: Python 3.11 以上の環境が手元にあり、すぐにコードを動かしたい方。
内容: 最短 3 行でデータを取得・描画するコードと、Google Colab での実行環境を提供します。

### 📖 30分で基本操作

[チュートリアル一覧](tutorials/index.rst)

データの構造や GWexpy 固有の行列演算を基礎から学びます。チュートリアル Notebook を順に進めることで、基本的な解析ワークフローを習得できます。

### 🔄 GWpy から移行

[GWpy からの移行](gwexpy_for_gwpy_users_ja.md)

既に GWpy を使っている方向け。まずは移行レシピで基本を把握し、必要に応じて [GWpy 差分 API 一覧](gwpy_added_api_index_ja.md) で追加 API を参照してください。

<a id="learning-path"></a>

## 学習ロードマップ (Learning Path)

### 1. 準備

まず [インストールガイド](installation.md) で環境を構築してください。

### 2. 基本データ構造の習得

主要なコンテナの使い方を以下の順序で学ぶことを推奨します：

1. [時系列データの基本](tutorials/intro_timeseries.ipynb)
2. [周波数系列の基本](tutorials/intro_frequencyseries.ipynb)
3. [スペクトログラムの基本](tutorials/intro_spectrogram.ipynb)
4. [プロット機能のカスタマイズ](tutorials/intro_plotting.ipynb)

### 3. 高度な解析機能

目的に応じて以下のガイドを参照してください：

* **多チャンネル・行列処理**: [行列コンテナ (Matrix) の活用](tutorials/matrix_timeseries.ipynb)
* **高次元データ**: [Field API 入門](tutorials/field_scalar_intro.ipynb) / [ScalarField のスライス操作ガイド](scalarfield_slicing.md)
* **信号処理**: [フィッティング](tutorials/advanced_fitting.ipynb) / [HHT](tutorials/advanced_hht.ipynb) / [ARIMA](tutorials/advanced_arima.ipynb)

### 4. 実践的な活用

実際の解析ワークフローは、[ケーススタディ集](../examples/index.rst) で確認できます。

<a id="next-to-read"></a>
<a id="next-steps"></a>

## 次に読む (Next to Read)

* [ケーススタディ集](../examples/index.rst) - 視覚的な使用例と解析ワークフロー
* [全チュートリアル一覧](tutorials/index.rst)
* [GWpy ユーザー向け移行ガイド](gwexpy_for_gwpy_users_ja.md) - 移行レシピで基本を把握する
* [GWpy 差分 API 一覧](gwpy_added_api_index_ja.md) - 追加 API を差分観点で引く
* [前提条件と規約](prerequisites_and_conventions.md) - 環境前提、GPS 時刻、FFT 規約の入口
* [API リファレンス](../reference/index.rst)
* [検証済みアルゴリズム](validated_algorithms.md) - 数値的正確性の検証レポート
