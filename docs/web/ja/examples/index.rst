.. meta::
   :description: GWexpy のケーススタディをテーマ別に探せるメインのギャラリーです。キャリブレーション、外部連携、ML、ノイズハンティングをまとめています。

.. _examples-ja-gallery-entry:

ケーススタディ（目的別） / Case Studies
=======================================

.. note::
   ページ種別: ガイド索引

複数の GWexpy 機能を解析目的に沿って組み合わせる、目的別の実践デモ集です。
クラスや機能ごとの使い方を順番に学びたい場合は、 :doc:`../user_guide/tutorials/index` を参照してください。

**対象読者:** 基本操作を理解したうえで、実務に近い一連の解析例を探したい利用者。
**前提知識:** :doc:`../user_guide/tutorials/index` や :doc:`../user_guide/getting_started` にある主要クラスの基本を理解していること。
**このページの用途:** GWexpy の目的別ケーススタディをテーマ別に探すためのメインの一覧として使うこと。
**検索のヒント:** ケーススタディ, ギャラリー, 実践ワークフロー, キャリブレーション, 相互運用, ノイズハンティング, ML

.. note::
   `Case Studies` は解析目的ごとの実演、`Tutorials` は機能やクラスごとの学習例です。
   このページを `case_*` ノートブックのメインの一覧として扱います。

.. note::
   このページの見方:
   I はキャリブレーションと制御、II は外部連携と再現性、III は統計・機械学習、IV はノイズハンティングと検出器診断を扱います。

.. note::
   例の読み方:
   目的: 自分の課題に近いワークフローを選ぶこと。
   入力: 関連する GWexpy オブジェクトの基本知識と notebook 実行環境。
   出力: 読むべき、またはローカルで実行すべきケーススタディ notebook の候補。

.. _examples-ja-featured-gallery:

注目ギャラリー
--------------

.. note::
   この注目カードは、ホームページの teaser で使っている 3 枚のサムネイルと同じ画像を再利用しています。
   ただしメインのはこのページです。ホームページは短い導入であり、以下のカテゴリ別一覧が正式なギャラリーです。

.. note::
   最小限のビジュアル索引:

   - `ノイズバジェット` のサムネイル -> :ref:`section-iv-noise-hunting-and-detector-diagnostics-ja` から探し始める
   - `伝達関数推定` のサムネイル -> :ref:`section-i-calibration-response-and-control-ja` から探し始める
   - `アクティブダンピング` のサムネイル -> :ref:`section-i-calibration-response-and-control-ja` から探し始める

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: ノイズバジェット
      :img-top: /_static/images/case_noise_budget_thumb.png
      :img-alt: ノイズバジェット例のサムネイル
      :link: ../user_guide/tutorials/case_noise_budget
      :link-type: doc
      :text-align: center

      検出器雑音の主要な結合経路を切り分け、どこから対策すべきかを比較します。

   .. grid-item-card:: 伝達関数推定
      :img-top: /_static/images/case_transfer_function_thumb.png
      :img-alt: 伝達関数推定例のサムネイル
      :link: ../user_guide/tutorials/case_transfer_function
      :link-type: doc
      :text-align: center

      コヒーレンス評価、応答のフィッティング、実測伝達経路の解釈をひとつの流れで確認します。

   .. grid-item-card:: アクティブダンピング
      :img-top: /_static/images/case_active_damping_thumb.png
      :img-alt: アクティブダンピング例のサムネイル
      :link: ../user_guide/tutorials/case_active_damping
      :link-type: doc
      :text-align: center

      6 自由度防振系に対する MIMO 制御ワークフローを具体例でたどれます。

.. _examples-ja-canonical-list:

.. _section-i-calibration-response-and-control-ja:

I. キャリブレーション・応答・制御
---------------------------------

- :doc:`アクティブダンピング: 6自由度防振系の MIMO 制御 <../user_guide/tutorials/case_active_damping>`
- :doc:`伝達関数計測: 実測・コヒーレンス・フィッティング <../user_guide/tutorials/case_transfer_function>`
- :doc:`キャリブレーションパイプライン: Counts から Strain へ <../user_guide/tutorials/case_calibration_pipeline>`
- :doc:`DTT XML 活用: 測定済み応答の読み込みと再利用 <../user_guide/tutorials/case_dttxml_calibration>`

.. _section-ii-interoperability-io-and-reproducibility-ja:

II. 外部連携・I/O・再現性
-------------------------

- :doc:`Finesse 3 連携: シミュレーションと測定の比較 <../user_guide/tutorials/case_finesse_optics>`
- :doc:`ObsPy 連携: 地震データの取り込みと解析 <../user_guide/tutorials/case_seismic_obspy>`
- :doc:`GBD 形式 I/O: 書き出しと再読込の実務フロー <../user_guide/tutorials/case_gbd_format>`
- :doc:`HDF5 provenance: 再現可能なメタデータ管理 <../user_guide/tutorials/case_hdf5_provenance>`
- :doc:`PyCBC 連携: gwexpy 前処理から探索まで <../user_guide/tutorials/case_pycbc_search>`

.. _section-iii-statistical-and-ml-workflows-ja:

III. 統計・機械学習ワークフロー
--------------------------------

- :doc:`Bootstrap PSD と GLS フィッティング <../user_guide/tutorials/case_bootstrap_gls_fitting>`
- :doc:`ML 前処理パイプライン: 特徴量整形と比較 <../user_guide/tutorials/case_ml_preprocessing>`
- :doc:`イベント同期解析: SegmentTable による窓選択 <../user_guide/tutorials/case_segment_analysis>`
- :doc:`物理妥当性検証: 単位・数値床・健全性テスト <../user_guide/tutorials/case_physics_validation>`
- :doc:`ARIMA ベースのバースト検出 <../user_guide/tutorials/case_arima_burst_search>`
- :doc:`信号抽出: 色付き雑音からの微弱信号回収 <../user_guide/tutorials/case_signal_extraction>`

.. _section-iv-noise-hunting-and-detector-diagnostics-ja:

IV. ノイズハンティングと検出器診断
----------------------------------

- :doc:`ノイズバジェット解析: 多チャンネル相関で寄与源を切り分ける <../user_guide/tutorials/case_noise_budget>`
- :doc:`ロックイン検出: 弱い AM/FM 構造の復元 <../user_guide/tutorials/case_lockin_detection>`
- :doc:`ウィーナーフィルタ: コヒーレント雑音の差し引き <../user_guide/tutorials/case_wiener_filter>`
- :doc:`カップリング解析: チャンネル間伝達経路の推定 <../user_guide/tutorials/case_coupling_analysis>`
- :doc:`BruCo と ICA: ウィットネス選定から差し引きまで <../user_guide/tutorials/case_bruco_ica_denoising>`
- :doc:`BruCo 応用: バイリニア結合と AM/FM の失敗モード <../user_guide/tutorials/case_bruco_advanced>`
- :doc:`バイオリンモード解析: 共振モードの同定と追跡 <../user_guide/tutorials/case_violin_mode>`
- :doc:`シューマン共鳴解析: 環境磁場モードの読み解き <../user_guide/tutorials/case_schumann_resonance>`
- :doc:`グリッチ詳細解析: Q 変換と Omega スキャン <../user_guide/tutorials/case_glitch_analysis>`

.. note::
   各 API の詳細（引数・戻り値・クラス一覧）は :doc:`../reference/index` を参照してください。

.. seealso::
   次に読むページ:

   - :doc:`../user_guide/tutorials/index` でクラス別・機能別の基礎を固める
   - :doc:`../reference/index` でケーススタディ内の API 詳細を確認する
   - :doc:`../user_guide/io_formats` で対応フォーマットと読み書き経路を調べる

.. toctree::
   :hidden:

   ../user_guide/tutorials/case_active_damping
   ../user_guide/tutorials/case_transfer_function
   ../user_guide/tutorials/case_calibration_pipeline
   ../user_guide/tutorials/case_dttxml_calibration
   ../user_guide/tutorials/case_finesse_optics
   ../user_guide/tutorials/case_seismic_obspy
   ../user_guide/tutorials/case_gbd_format
   ../user_guide/tutorials/case_hdf5_provenance
   ../user_guide/tutorials/case_pycbc_search
   ../user_guide/tutorials/case_bootstrap_gls_fitting
   ../user_guide/tutorials/case_ml_preprocessing
   ../user_guide/tutorials/case_segment_analysis
   ../user_guide/tutorials/case_physics_validation
   ../user_guide/tutorials/case_arima_burst_search
   ../user_guide/tutorials/case_signal_extraction
   ../user_guide/tutorials/case_noise_budget
   ../user_guide/tutorials/case_lockin_detection
   ../user_guide/tutorials/case_wiener_filter
   ../user_guide/tutorials/case_coupling_analysis
   ../user_guide/tutorials/case_bruco_ica_denoising
   ../user_guide/tutorials/case_bruco_advanced
   ../user_guide/tutorials/case_violin_mode
   ../user_guide/tutorials/case_schumann_resonance
   ../user_guide/tutorials/case_glitch_analysis
