インストール
============

GWexpy は Python 3.9+ を必要とし、GWpy, NumPy, SciPy, および Astropy に依存します。

基本インストール
----------------

.. note::
   GWexpy は現在 **Pre-release (先行公開版)** です。近日中に PyPI への登録を予定していますが、現時点では最新の機能や修正が含まれる GitHub からの直接インストールを推奨します。

   .. code-block:: bash

      # GitHub から直接 (推奨)
      pip install git+https://github.com/tatsuki-washimi/gwexpy.git

.. important::
   **重力波データ解析 (LIGO/Virgo/KAGRA等) の機能を使用する場合**

   ``[gw]`` エクストラに含まれる ``nds2-client`` や ``python-framel`` などの一部ライブラリは、システムの依存関係が複雑なため **PyPI では提供されていません**。
   これらの機能を使用するには、まず **Conda (Miniforge/Anaconda)** を用いて依存関係をインストールすることを強く推奨します:

   .. code-block:: bash

      # 1. 外部依存関係のインストール
      conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp

      # 2. GWexpy のインストール (エクストラ指定)
      pip install "gwexpy[gw] @ git+https://github.com/tatsuki-washimi/gwexpy.git"

開発用インストール
------------------

.. code-block:: bash

   pip install -e ".[dev]"

オプション機能の追加
--------------------

GWexpy はドメイン固有の機能のためにオプションの依存関係を提供しています:

- ``.[gw]`` : 重力波データ解析 (nds2, frames, noise models)
- ``.[stats]`` : 統計解析 (polars, ARIMA, ICA/PCA)
- ``.[fitting]`` : 高度なフィッティング (iminuit, emcee, corner)
- ``.[astro]`` : 天体物理ツール (specutils, pyspeckit)
- ``.[geophysics]`` : 地球物理 (obspy, mth5, etc.)
- ``.[audio]`` : 音響解析 (librosa/pydub helpers)
- ``.[bio]`` : 生体信号 (mne/neo/elephant integrations)
- ``.[interop]`` : 高度な相互運用性 (torch, jax, dask, etc.)
- ``.[control]`` : 制御工学 (python-control integration)
- ``.[plot]`` : プロット・マッピング (pygmt)
- ``.[analysis]`` : 変換および時間周波数ツール
- ``.[gui]`` : 実験的 Qt GUI
- ``.[all]`` : 全ての依存関係をインストール

必要に応じて組み合わせてください:

.. code-block:: bash

   pip install ".[gw,stats,plot]"

GitHub から直接エクストラを指定してインストールすることもできます:

.. code-block:: bash

   pip install "gwexpy[analysis] @ git+https://github.com/tatsuki-washimi/gwexpy.git"

.. note::
   ``[gw]`` エクストラに含まれる ``nds2-client`` などの一部ライブラリは **PyPI では提供されていません**。
   これらの機能を使用するには、まず **Conda** を用いて依存関係をインストールする必要があります:

   .. code-block:: bash

      conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp
      pip install ".[gw]"
