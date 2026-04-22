トピック別参照 (Topics)
=======================

.. note::
   ページ種別: 理論・概念の入口

**安定性:** 安定

このページは、クラス名やモジュール名ではなく、概念や用途からリファレンスを探したい場合の入口です。

**対象読者:** 分析上の問いは明確で、対応する規約、理論メモ、補助 API を探したい利用者。
**このページの用途:** フーリエ規約、検証前提、スペクトル推定、互換レイヤなどの概念から参照を始めること。

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - トピック
     - 安定性
     - 入口
   * - 理論と規約
     - 安定
     - :doc:`FFT_Conventions`, :doc:`../user_guide/prerequisites_and_conventions`
   * - スペクトル解析とフィッティング
     - 安定
     - :doc:`Spectral`, :doc:`fitting`, :doc:`../user_guide/tutorials/case_bootstrap_gls_fitting`
   * - 検証・監査ノート
     - 安定
     - :doc:`../user_guide/validated_algorithms`, :doc:`../user_guide/numerical_stability`
   * - ノイズ生成ヘルパ
     - 安定
     - :doc:`Noise`
   * - 互換・補助 API
     - 安定
     - :doc:`api/extra`

概念別ガイド
------------

- :doc:`FFT_Conventions` ではフーリエ正規化、軸規約、対応 API を確認できます。
- :doc:`../user_guide/prerequisites_and_conventions` では時刻系、FFT 規約、物理解釈の共通前提を確認できます。
- :doc:`Spectral` では PSD、ASD、bootstrap 系推定の入口をまとめています。
- :doc:`fitting` では最小二乗、GLS、MCMC 系のフィッティング API をまとめています。
- :doc:`../user_guide/validated_algorithms` では、高度・理論向けの入口として、監査根拠付きの前提条件と API 対応を確認できます。
- :doc:`../user_guide/numerical_stability` では適応ホワイトニングなどの安定化方針を確認できます。
- :doc:`Noise` では合成ノイズや代理波形生成の入口をまとめています。

橋渡しページ
------------

- :doc:`api/extra` では互換入口と追加 API を確認できます。
- :doc:`../user_guide/gwexpy_for_gwpy_users_ja` では GWpy 移行ガイドを確認できます。
- :doc:`../user_guide/gwpy_added_api_index_ja` では GWpy 差分観点で API をたどれます。
- :doc:`../user_guide/tutorials/field_scalar_intro` と :doc:`../user_guide/tutorials/advanced_arima` はリファレンスへ戻りやすい個別チュートリアル入口です。

.. seealso::
   ハブ間の移動:

   - :doc:`index` でリファレンス全体の入口に戻る
   - :doc:`api/index` でモジュール・カテゴリ別に API を探す
   - :doc:`../user_guide/tutorials/index` でチュートリアルから学習を始める
   - :doc:`../user_guide/validated_algorithms` で高度・理論寄りのリンクをたどる

.. toctree::
   :maxdepth: 1

   FFT_Conventions
   Spectral
   fitting
   Noise
   api/extra
