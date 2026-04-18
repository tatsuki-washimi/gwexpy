---
myst:
  html_meta:
    description: "GWexpy を最短で始めるための導入ガイドです。GitHub からのインストール、3 行の最小例、次に読むチュートリアルへの入口をまとめています。"
---

# クイックスタート (Quickstart)

GWexpy を使って、最初の解析図を最短で作成しましょう。

## このページでわかること

| 項目 | 内容 |
| --- | --- |
| **ページ種別** | ガイド |
| **対象読者** | まず 1 枚プロットを出したい初回利用者、GWpy 互換の入口を見たい利用者 |
| **前提** | Python 3.11 以上、基本的な `pip` 操作、NumPy 配列の基礎 |
| **こんなときに読む** | まず import が通るか確認したい、3 行の最小例を見たい、次に何を読むか決めたい |
| **検索キーワード** | quickstart, 最初のプロット, `TimeSeries`, CSD, 最小例 |

## このページの近道

- [クイックインストール](#install-command)
- [3行で最初の図を出す](#quick-demo)
- [30分で学べるハンズオン](#30分で学べるハンズオン-interactive-tutorial)
- [複数チャンネルの解析例](#複数チャンネルの解析例)
- [次のステップ](#next-to-read)

<a id="install-command"></a>

## クイックインストール (Quick Install)

現在、開発版のため GitHub から直接インストールしてください：

- 目的: 開発版を最短でインストールする
- 入力: Python 3.11 以上と `pip`
- 出力: 最初のサンプルを実行できる GWexpy 環境

```bash
pip install git+https://github.com/tatsuki-washimi/gwexpy.git
```

<a id="quick-demo"></a>

## 3行で最初の図を出す (3-line Quickstart)

GWexpy の `TimeSeries` は NumPy 配列から直接作成でき、標準的なプロット機能を備えています。

- 目的: ランダムな時系列から最初のプロットを 1 枚表示する
- 入力: 4096 サンプルの NumPy 配列、サンプルレート 4096 Hz、開始時刻 `t0=0`
- 出力: `TimeSeries` オブジェクトと描画ウィンドウ

```python
import numpy as np
from gwexpy.timeseries import TimeSeries

ts = TimeSeries(np.random.randn(4096), sample_rate=4096.0, t0=0)
ts.plot().show()
```

## 30分で学べるハンズオン (Interactive Tutorial)

より実践的なワークフローを学びたい場合は、以下のチュートリアルを推奨します。Google Colab ですぐに実行可能です。

### 🧪 GWexpy 基本ハンズオン

[チュートリアル一覧を見る](tutorials/index.rst)

データの読み込みから、周波数解析（ASD/CSD）、最新のフィールド API による行列操作までを一通り体験します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsuki-washimi/gwexpy/blob/main/docs/web/ja/user_guide/tutorials/intro_timeseries.ipynb)

## 主要概念 (Core Concepts)

GWexpy を使いこなすための 2 つの重要な柱です。

* **TimeSeries / FrequencySeries**: 
  単一チャンネルのデータを扱う基本クラスです。GWpy と高い互換性があり、既存のコードをそのまま動かすことができます。
* **TimeSeriesMatrix / ScalarField**:
  複数チャンネル（行列形式）や多次元のフィールド系データを扱うための新しい API です。100 チャンネルを超えるような大規模な解析も、1 行のコードで安全に処理できます。

## 複数チャンネルの解析例

複数チャンネルの相関（CSD）を計算する例です。

- 目的: 複数チャンネルをまとめて `TimeSeriesMatrix` に変換し、相互相関スペクトルを計算する
- 入力: 2 チャンネルの `TimeSeriesDict` と `fftlength=1`
- 出力: `csd` オブジェクトと CSD プロット

```python
import numpy as np
from gwexpy.timeseries import TimeSeries, TimeSeriesDict

# データの作成
tsd = TimeSeriesDict({
    "H1:STRAIN": TimeSeries(np.random.randn(4096 * 4), sample_rate=4096, t0=0),
    "L1:STRAIN": TimeSeries(np.random.randn(4096 * 4), sample_rate=4096, t0=0),
})

# 行列に変換してクロススペクトル密度 (CSD) を計算
csd = tsd.to_matrix().csd(fftlength=1)
csd.plot().show()
```

ここでは `TimeSeriesDict` を `TimeSeriesMatrix` に変換し、チャンネル間のクロススペクトル密度を `csd` として得ています。

## 困ったときは

実行時にエラーが発生したり、プロットが表示されない場合は、[トラブルシューティング](troubleshooting.md) を確認してください。

<a id="next-to-read"></a>
<a id="next-steps"></a>

## 次に読む

* [インストールガイド](installation.md) - 環境の構築
* [はじめに](getting_started.md) - 体系的な学習ロードマップ
* [前提条件と規約](prerequisites_and_conventions.md) - GPS 時刻や FFT の前提を先に確認する
* [GWpy からの移行](gwexpy_for_gwpy_users_ja.md) - 既存ユーザー向け差分ガイド
