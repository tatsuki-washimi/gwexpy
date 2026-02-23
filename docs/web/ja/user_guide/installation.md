# インストール

GWexpy は Python 3.9+ を必要とし、GWpy, NumPy, SciPy, および Astropy に依存します。

## 基本インストール

```bash
pip install gwexpy
```

:::{note}
**GitHub から開発版をインストール (最新機能):**

```bash
pip install git+https://github.com/tatsuki-washimi/gwexpy.git
```

:::

:::{important}
**重力波データ解析 (LIGO/Virgo/KAGRA等) の機能を使用する場合**

`[gw]` エクストラに含まれる `nds2-client` や `python-framel` などの一部ライブラリは、システムの依存関係が複雑なため **PyPI では提供されていません**。
これらの機能を使用するには、まず **Conda (Miniforge/Anaconda)** を用いて依存関係をインストールすることを強く推奨します:

```bash
# 1. 外部依存関係のインストール
conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp

# 2. GWexpy のインストール (エクストラ指定)
pip install "gwexpy[gw] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
```

:::

## 開発用インストール

```bash
pip install -e ".[dev]"
```

## オプション機能の追加

GWexpy はドメイン固有の機能のためにオプションの依存関係を提供しています:

- `.[gw]` : 重力波データ解析 (nds2, frames, noise models)
- `.[stats]` : 統計解析 (polars, ARIMA, ICA/PCA)
- `.[fitting]` : 高度なフィッティング (iminuit, emcee, corner)
- `.[astro]` : 天体物理ツール (specutils, pyspeckit)
- `.[geophysics]` : 地球物理 (obspy, mth5, etc.)
- `.[audio]` : 音響解析 (librosa/pydub helpers)
- `.[bio]` : 生体信号 (mne/neo/elephant integrations)
- `.[interop]` : 高度な相互運用性 (torch, jax, dask, etc.)
- `.[control]` : 制御工学 (python-control integration)
- `.[plot]` : プロット・マッピング (pygmt)
- `.[analysis]` : 変換および時間周波数ツール
- `.[gui]` : 実験的 Qt GUI
- `.[all]` : 全ての依存関係をインストール

必要に応じて組み合わせてください:

```bash
pip install ".[gw,stats,plot]"
```

GitHub から直接エクストラを指定してインストールすることもできます:

```bash
pip install "gwexpy[analysis] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
```

:::{note}
`[gw]` エクストラを使用する場合は、上記の「基本インストール」セクションの important を参照してください。
:::

:::{note}
**Maximal Information Coefficient (MIC) の計算**

MIC の計算には `minepy` が必要です。Python 3.11+ では標準の `pip` や `conda` でのインストールに失敗することがあります。その場合は、リポジトリに含まれる以下の自動ビルドスクリプトを使用してください：

```bash
python scripts/install_minepy.py
```

:::

## 次のステップ

GWexpyのインストールが完了したら、基本的な使い方を学びましょう:

- [クイックスタート](quickstart.md) - 時系列データの生成とプロット
- [はじめに](getting_started.md) - 完全な学習パス
