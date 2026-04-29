---
myst:
  html_meta:
    description: "GWexpy のインストール方法を、最小構成、推奨構成、GW 解析向け構成、開発者向け構成ごとに整理し、extras と依存関係の注意点もまとめます。"
---

# インストールガイド (Installation)

## このページでわかること

| 項目 | 内容 |
| --- | --- |
| **ページ種別** | ガイド |
| **対象読者** | GWexpy を初めて導入する利用者、extras の選び方を知りたい利用者、開発環境を作るコントリビュータ |
| **前提** | Python 3.11 以上、`pip` の基本操作。GW 依存を使う場合は Conda の基本操作 |
| **こんなときに読む** | 最小構成で入れたい、Conda が必要か判断したい、extras の違いを知りたい |
| **検索キーワード** | `install`, `pip`, `conda`, extras, 開発版, NDS2, FrameLIB |

## このページの近道

- [インストールコマンド](#ja-install-command)
- [インストール手順](#1-インストール手順-installation-steps)
- [依存関係トラブルシューティング](#21-依存関係トラブルシューティング)
- [オプション依存関係 (Extras) の詳細](#3-オプション依存関係-extras-の詳細)
- [OS 別の注意点](#4-os-別の注意点)
- [セキュリティに関する注意 (Pickle)](#5-セキュリティに関する注意-pickle)
- [次に読む](#next-to-read)

:::{note}
GWexpy は現在 **開発版** であり、**PyPI / Conda にはまだ公開されていません**。現時点では、このページにある GitHub からのソースインストールを利用してください。
:::

GWexpy は **Python 3.11 以上** をサポートしています。解析の目的に合わせて、いくつかのインストールオプション（extras）を選択できます。

(ja-install-command)=
## インストールコマンド

| 目標 | インストールコマンド | 特徴 |
| --- | --- | --- |
| 最小構成 | `pip install git+https://github.com/tatsuki-washimi/gwexpy.git` | 数値コンテナと基本演算のみ。依存最小。 |
| **推奨構成** | `pip install "gwexpy[analysis,fitting,plotting] @ git+https://github.com/tatsuki-washimi/gwexpy.git"` | 高度な統計解析、最小二乗法、地図描画など。 |
| 重力波解析 | `pip install "gwexpy[gw,io] @ git+https://github.com/tatsuki-washimi/gwexpy.git"` | フレームファイル読み込み、NDS2、公式計算ツール等。 |
| 開発 / 全機能 | `pip install "gwexpy[all] @ git+https://github.com/tatsuki-washimi/gwexpy.git"` | すべてのオプション機能を有効化。 |

## 1. インストール手順 (Installation Steps)

### 最小構成 (Minimal)

依存パッケージを最小限に抑え、`ScalarField` などの基本フィールド系データ構造だけを利用する場合です。

- 目的: まず import と基本コンテナだけを試す
- 入力: Python 3.11 以上と `pip`
- 出力: 最小依存の GWexpy 環境

NDS2、FrameLIB、`pygmt` などのバイナリ依存を使う場合は、この節ではなく [コンダ環境 (Recommended / GW Analysis)](#コンダ環境-recommended--gw-analysis) から始めてください。

```bash
pip install git+https://github.com/tatsuki-washimi/gwexpy.git
```

### コンダ環境 (Recommended / GW Analysis)

重力波解析（NDS2 や FrameLIB 等）を行う場合は、Conda (Miniforge 等) を使用してバイナリ依存関係を先に解決することを強く推奨します。

:::{warning}
Conda 環境を使う場合は、`base` や別用途の既存環境に直接 `pip install` せず、**GWexpy 専用の新しい環境** を作成してから導入してください。Conda が管理するバイナリ依存関係と `pip` で追加する Python パッケージを同じ専用環境に閉じ込めることで、環境破損のリスクを下げられます。
:::

- 目的: NDS2 / FrameLIB などのバイナリ依存を含む GW 解析環境を作る
- 入力: Conda が使える端末環境
- 出力: `gw` extras を含む専用 `gwexpy` 環境

```bash
# 1. 仮想環境の作成とバイナリ依存の解決
conda create -n gwexpy python=3.11
conda activate gwexpy
conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp

# 2. GWexpy と解析用オプションをインストール
pip install "gwexpy[gw,analysis,fitting] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
```

`No module named nds2` や FrameLIB 関連の import error が出る場合は、まず上の `conda install -c conda-forge ...` を専用環境で再実行してください。NDS2 や FrameLIB を使わない場合は、`gw` extras を外した最小構成または推奨構成で十分です。

### 開発者モード (For Developers)

ソースコードから最新版をインストールし、テスト環境を構築する場合です。**Conda は必須ではありません。** `gw` 系のバイナリ依存関係まで検証する場合は上の Conda 環境を使い、それ以外のドキュメント修正や通常開発では `venv` 等の標準的な仮想環境でも構いません。

- 目的: ローカル clone から編集・テストできる状態にする
- 入力: Git と Python 仮想環境
- 出力: `pip install -e` による編集可能インストール

```bash
git clone https://github.com/tatsuki-washimi/gwexpy.git
cd gwexpy
pip install -e ".[dev,all]"
```

## 2.1. 依存関係トラブルシューティング

- `No module named nds2` が出る場合: `python-nds2-client` が未導入です。専用の Conda 環境を有効化し、`conda install -c conda-forge python-nds2-client` を実行してください。
- FrameLIB / `framecpp` 系のエラーが出る場合: `python-framel` と `ldas-tools-framecpp` を同じ Conda 環境に入れ直してください。
- 既存の Conda 環境で依存関係が混線した場合: その環境を修復するより、`conda create -n gwexpy python=3.11` で作り直す方が安全です。
- [クイックスタート](quickstart.md) から導入したあとで GW 系のバイナリ依存が必要になった場合: パッケージを場当たり的に足す前に [コンダ環境 (Recommended / GW Analysis)](#コンダ環境-recommended--gw-analysis) に戻ってください。

---

## 3. オプション依存関係 (Extras) の詳細

| エクストラ名 | 主要な導入パッケージ | 主な用途 |
| --- | --- | --- |
| `analysis` | `scikit-learn`, `statsmodels`, `pmdarima` | ノイズ除去、統計予測、機械学習。 |
| `fitting` | `iminuit`, `emcee`, `corner` | 最小二乗法、MCMC 解析。 |
| `gw` | `lalsuite`, `gwosc`, `gwinc`, `ligo.skymap` | 重力波データ検索、感度計算、スカイマップ描画。 |
| `io` | `nptdms` | LabVIEW TDMS 形式の読み込み。 |
| `netcdf4` | `netCDF4`, `xarray` | xarray 経由の NetCDF4 時系列ファイルの読み書き。 |
| `zarr` | `zarr` | Zarr array store の読み書き。 |
| `plotting` | `pygmt` | 高精度な地図投影（GeoMap）。 |
| `audio` | `pydub`, `tinytag` | 音声書き出し、オーディオ解析、任意のメタデータ抽出。 |
| `seismic` | `obspy`, `mth5`, `mtpy` | 地震・地磁気データ解析。 |
| `control` | `control` (python-control) | 制御系モデル・伝達関数解析。 |
| `gui` | `PyQt5`, `pyqtgraph` | ソース / 開発環境向けの試作 GUI 依存関係。GUI アプリは初回 PyPI のサポート対象には含めず、`all` にも含めません。 |

---

## 4. OS 別の注意点

* **Linux**: 標準的なビルドツール (`build-essential`) を事前に導入してください。
* **macOS (Apple Silicon)**: `conda-forge` チャンネルを利用することで、多くのバイナリが M1/M2/M3 ネイティブで動作します。
* **Windows (WSL2)**: Windows 本体ではなく、WSL2 上の Linux 環境へのインストールを推奨します。

## 5. セキュリティに関する注意 (Pickle)

GWexpy は解析結果の共有を容易にするため、`Pickle` を用いた保存・復元機能をサポートしています。これには **透過 Pickle** 技術が使われており、読み込み側に GWexpy がなくても GWpy オブジェクトとして復元できる利便性があります。

:::{caution}
**信頼できないソースからの Pickle ファイルをロードしないでください。**
Python の `pickle` モジュールは、ロード時に任意のコードを実行できる脆弱性（Arbitrary Code Execution）を持っています。解析データの授受は、常に信頼できる経路で行うか、`HDF5` などのより安全なシリアライズ形式を検討してください。

:::
<a id="next-to-read"></a>
<a id="next-steps"></a>

## 6. 次に読む

* [クイックスタート](quickstart.md) - 最小コードで import と描画を確認する
* [トラブルシューティング](troubleshooting.md) - インストールや描画トラブルを症状から逆引きする
* [はじめに](getting_started.md) - インストール後の学習順序を決める
* [前提条件と規約](prerequisites_and_conventions.md) - GPS 時刻や FFT の前提をまとめて確認する
