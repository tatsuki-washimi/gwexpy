トピック別参照 (Topics)
=======================

**安定性:** Stable

このページは、クラス名やモジュール名ではなく、概念や用途からリファレンスを探したい場合の入口です。

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - トピック
     - 安定性
     - 入口
   * - 理論と規約
     - Stable
     - :doc:`FFT_Conventions`
   * - スペクトル解析とフィッティング
     - Stable
     - :doc:`Spectral`, :doc:`fitting`
   * - ノイズ生成ヘルパ
     - Stable
     - :doc:`Noise`
   * - 互換・補助 API
     - Stable
     - :doc:`api/extra`

概念別ガイド
------------

- :doc:`FFT_Conventions` ではフーリエ正規化、軸規約、対応 API を確認できます。
- :doc:`Spectral` では PSD、ASD、bootstrap 系推定の入口をまとめています。
- :doc:`fitting` では最小二乗、GLS、MCMC 系のフィッティング API をまとめています。
- :doc:`Noise` では合成ノイズや代理波形生成の入口をまとめています。

橋渡しページ
------------

- :doc:`api/extra` では互換入口と追加 API を確認できます。
- :doc:`../user_guide/gwexpy_for_gwpy_users_ja` では GWpy 移行ガイドを確認できます。
- :doc:`../user_guide/gwpy_added_api_index_ja` では GWpy 差分観点で API をたどれます。

.. toctree::
   :maxdepth: 1

   FFT_Conventions
   Spectral
   fitting
   Noise
   api/extra
