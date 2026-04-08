# インストールガイド (Installation)

GWexpy は **Python 3.11 以上** をサポートしています。解析の目的に合わせて、いくつかのインストールオプション（extras）を選択できます。

.. list-table:: 目的別インストールオプションの推奨
   :widths: 25 50 25
   :header-rows: 1

   * - 目標
     - インストールコマンド
     - 特徴
   * - 最小構成
     - `pip install gwexpy`
     - 数値コンテナと基本演算のみ。依存最小。
   * - **推奨構成**
     - `pip install "gwexpy[analysis,fitting,plotting]"`
     - 高度な統計解析、最小二乗法、地図描画など。
   * - 重力波解析
     - `pip install "gwexpy[gw,io]"`
     - フレームファイル読み込み、NDS2、公式計算ツール等。
   * - 開発 / 全機能
     - `pip install "gwexpy[all]"`
     - すべてのオプション機能を有効化。

## 1. インストール手順 (Installation Steps)

.. tab-set::

   .. tab-item:: 最小構成 (Minimal)

      依存パッケージを最小限に抑え、`ScalarField` などの基本コンテナのみを利用する場合です。

      ```bash
      pip install gwexpy
      ```

   .. tab-item:: コンダ環境 (Recommended / GW Analysis)

      重力波解析（NDS2 や FrameLIB 等）を行う場合は、Conda (Miniforge 等) を使用してバイナリ依存関係を先に解決することを強く推奨します。

      ```bash
      # 1. 仮想環境の作成とバイナリ依存の解決
      conda create -n gwexpy python=3.11
      conda activate gwexpy
      conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp

      # 2. GWexpy と解析用オプションをインストール
      pip install "gwexpy[gw,analysis,fitting]"
      ```

   .. tab-item:: 開発者モード (For Developers)

      ソースコードから最新版をインストールし、テスト環境を構築する場合です。

      ```bash
      git clone https://github.com/tatsuki-washimi/gwexpy.git
      cd gwexpy
      pip install -e ".[dev,all]"
      ```

---

## 2. オプション依存関係 (Extras) の詳細

.. list-table:: エクストラパッケージ一覧
   :widths: 20 40 40
   :header-rows: 1

   * - エクストラ名
     - 主要な導入パッケージ
     - 主な用途
   * - `analysis`
     - `scikit-learn`, `statsmodels`, `pmdarima`
     - ノイズ除去、統計予測、機械学習。
   * - `fitting`
     - `iminuit`, `emcee`, `corner`
     - 最小二乗法、MCMC 解析。
   * - `gw`
     - `lalsuite`, `gwosc`, `gwinc`, `ligo.skymap`
     - 重力波データ検索、感度計算、スカイマップ描画。
   * - `io`
     - `nptdms`
     - LabVIEW TDMS 形式の読み込み。
   * - `plotting`
     - `pygmt`
     - 高精度な地図投影（GeoMap）。
   * - `audio`
     - `pydub`
     - 音声書き出し、オーディオ解析。

---

## 3. OS 別の注意点

* **Linux**: 標準的なビルドツール (`build-essential`) を事前に導入してください。
* **macOS (Apple Silicon)**: `conda-forge` チャンネルを利用することで、多くのバイナリが M1/M2/M3 ネイティブで動作します。
* **Windows (WSL2)**: Windows 本体ではなく、WSL2 上の Linux 環境へのインストールを推奨します。

## 4. セキュリティに関する注意 (Pickle)

GWexpy は解析結果の共有を容易にするため、`Pickle` を用いた保存・復元機能をサポートしています。これには **透過 Pickle** 技術が使われており、読み込み側に GWexpy がなくても GWpy オブジェクトとして復元できる利便性があります。

.. caution::
   **信頼できないソースからの Pickle ファイルをロードしないでください。**
   Python の `pickle` モジュールは、ロード時に任意のコードを実行できる脆弱性（Arbitrary Code Execution）を持っています。解析データの授受は、常に信頼できる経路で行うか、`HDF5` などのより安全なシリアライズ形式を検討してください。

## 5. 次のステップ

* :doc:`quickstart` - 3行で始める解析
* :doc:`getting_started` - 学習ロードマップ
