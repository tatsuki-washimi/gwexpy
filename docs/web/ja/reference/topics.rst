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
     - :doc:`FFT_Conventions`, :doc:`../user_guide/prerequisites_and_conventions`
   * - スペクトル解析とフィッティング
     - Stable
     - :doc:`Spectral`, :doc:`fitting`, :doc:`../user_guide/tutorials/case_bootstrap_gls_fitting`
   * - 検証・監査ノート
     - Stable
     - :doc:`../user_guide/validated_algorithms`, :doc:`../user_guide/numerical_stability`
   * - ノイズ生成ヘルパ
     - Stable
     - :doc:`Noise`
   * - 互換・補助 API
     - Stable
     - :doc:`api/extra`

概念別ガイド
------------

- :doc:`FFT_Conventions` ではフーリエ正規化、軸規約、対応 API を確認できます。
- :doc:`../user_guide/prerequisites_and_conventions` では時刻系、FFT 規約、物理解釈の共通前提を確認できます。
- :doc:`Spectral` では PSD、ASD、bootstrap 系推定の入口をまとめています。
- :doc:`fitting` では最小二乗、GLS、MCMC 系のフィッティング API をまとめています。
- :doc:`../user_guide/validated_algorithms` では監査根拠付きの前提条件と API 対応を確認できます。
- :doc:`../user_guide/numerical_stability` では適応ホワイトニングなどの安定化方針を確認できます。
- :doc:`Noise` では合成ノイズや代理波形生成の入口をまとめています。

橋渡しページ
------------

- :doc:`api/extra` では互換入口と追加 API を確認できます。
- :doc:`../user_guide/gwexpy_for_gwpy_users_ja` では GWpy 移行ガイドを確認できます。
- :doc:`../user_guide/gwpy_added_api_index_ja` では GWpy 差分観点で API をたどれます。
- :doc:`../user_guide/tutorials/field_scalar_intro` と :doc:`../user_guide/tutorials/advanced_arima` はリファレンスへ戻りやすい個別チュートリアル入口です。

.. toctree::
   :maxdepth: 1

   FFT_Conventions
   Spectral
   fitting
   Noise
   api/extra
