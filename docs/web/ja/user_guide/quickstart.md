# クイックスタート (Quickstart)

GWexpy を使って、最初の解析図を最短で作成しましょう。

## クイックインストール (Quick Install)

現在、開発版のため GitHub から直接インストールしてください：

```bash
pip install git+https://github.com/tatsuki-washimi/gwexpy.git
```

## 3行で最初の図を出す (3-line Quickstart)

GWexpy の `TimeSeries` は NumPy 配列から直接作成でき、標準的なプロット機能を備えています。

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

データの読み込みから、周波数解析（ASD/CSD）、最新の ScalarField API による行列操作までを一通り体験します。

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatsuki-washimi/gwexpy/blob/main/docs/web/ja/user_guide/tutorials/intro_timeseries.ipynb)

## 主要概念 (Core Concepts)

GWexpy を使いこなすための 2 つの重要な柱です。

* **TimeSeries / FrequencySeries**: 
  単一チャンネルのデータを扱う基本クラスです。GWpy と高い互換性があり、既存のコードをそのまま動かすことができます。
* **TimeSeriesMatrix / ScalarField**:
  複数チャンネル（行列形式）や多次元データを扱うための新しい API です。100 チャンネルを超えるような大規模な解析も、1 行のコードで安全に処理できます。

## 複数チャンネルの解析例

複数チャンネルの相関（CSD）を計算する例です。

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

## 次のステップ

* [インストールガイド](installation.md) - 環境の構築
* [はじめに](getting_started.md) - 体系的な学習ロードマップ
* [GWpy からの移行](gwexpy_for_gwpy_users_ja.md) - 既存ユーザー向け差分ガイド
