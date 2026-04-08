# インストール (Installation)

GWexpy は **Python 3.9 以上** をサポートしています。
解析の目的に合わせて、以下のいずれかの方法でインストールしてください。

## サポート環境

.. list-table:: 
   :widths: 30 70

   * - **OS**
     - Linux, macOS, Windows (WSL2 推奨)
   * - **Python**
     - 3.9, 3.10, 3.11, 3.12
   * - **主要な依存関係**
     - GWpy, NumPy, SciPy, Astropy

## インストール方法の選択

.. tab-set::

    .. tab-item:: 🏁 最小インストール
        :sync: minimal

        GWexpy のコア機能（TimeSeriesMatrix, 基本プロット等）のみを使用する場合：

        ```bash
        pip install git+https://github.com/tatsuki-washimi/gwexpy.git
        ```

    .. tab-item:: ✨ 推奨インストール
        :sync: recommended

        一般的な信号処理、フィッティング、地図描画機能を含める場合：

        ```bash
        pip install "gwexpy[analysis,fitting,plotting] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
        ```

    .. tab-item:: 🌌 GW 解析 (LIGO/Virgo/KAGRA 等)
        :sync: gw

        重力波データ解析（nds2, frames）の機能を使用する場合。**Conda での事前準備が必須です。**

        ```bash
        # 1. Conda でバイナリ依存関係をインストール
        conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp

        # 2. GWexpy をインストール
        pip install "gwexpy[gw] @ git+https://github.com/tatsuki-washimi/gwexpy.git"
        ```

    .. tab-item:: 🛠️ 開発者向け
        :sync: developer

        コードの変更やテストを行う場合（編集可能モード）：

        ```bash
        git clone https://github.com/tatsuki-washimi/gwexpy.git
        cd gwexpy
        pip install -e ".[dev,all]"
        ```

## OS 別の注意事項

.. tab-set::

    .. tab-item:: Linux / WSL2

        ほとんどの機能は標準的な `pip` で動作します。`nds2` 等を利用する場合は Conda を推奨します。

    .. tab-item:: macOS (Intel/Apple Silicon)

        Apple Silicon (M1/M2/M3) 環境では、一部のバイナリパッケージを Conda (Miniforge/Mamba) からインストールすることを強く推奨します。

    .. tab-item:: Windows (Native)

        基本的な解析は動作しますが、高度な入出力機能（`nds2` 等）は **WSL2** 上での利用を推奨します。

## PyPI / Conda での提供状況

現在、GWexpy は PyPI および conda-forge への登録準備中です。
正式リリースまでは **GitHub からの直接インストール** をご利用ください。

* **PyPI**: 近日対応予定 (`pip install gwexpy`)
* **Conda**: 近日対応予定 (`conda install -c conda-forge gwexpy`)

## トラブルシューティング

インストール時にエラー（特に `nds2`, `minepy`, `PyQt` 関連）が発生した場合は、
:doc:`トラブルシューティングガイド <troubleshooting>` を参照してください。

## 次のステップ

* :doc:`クイックスタート <quickstart>` - 5分で最初のプロットを作成
* :doc:`はじめに <getting_started>` - 体系的な学習ロードマップ
