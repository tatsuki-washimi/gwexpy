インストール
============

GWExPy は Python 3.9+ を必要とし、GWpy, NumPy, SciPy, および Astropy に依存します。

基本インストール
----------------

.. code-block:: bash

   # ローカルチェックアウトから
   pip install .

   # GitHub から直接 (PyPIリリースが無い場合)
   pip install git+https://github.com/tatsuki-washimi/gwexpy.git

開発用インストール
------------------

.. code-block:: bash

   pip install -e ".[dev]"

オプション機能の追加
--------------------

GWExPy はドメイン固有の機能のためにオプションの依存関係を提供しています:

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

.. note::
   ``[gw]`` エクストラに含まれる ``nds2-client`` などの一部ライブラリは **PyPI では提供されていません**。
   これらの機能を使用するには、まず **Conda** を用いて依存関係をインストールする必要があります:

   .. code-block:: bash

      conda install -c conda-forge python-nds2-client python-framel ldas-tools-framecpp
      pip install ".[gw]"
