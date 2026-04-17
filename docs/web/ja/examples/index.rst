ケーススタディ (Case Studies)
===============================

複数の GWexpy 機能をひとつのテーマに沿って組み合わせる、実践的なデモ集です。
クラスや機能ごとの使い方を順番に学びたい場合は、 :doc:`../user_guide/tutorials/index` を参照してください。

.. note::
   `Case Studies` はテーマ別の実演、`Tutorials` は class/feature examples です。
   このページを `case_*` ノートブックの正本一覧として扱います。

I. キャリブレーション・応答・制御
---------------------------------

- :doc:`アクティブダンピング: 6自由度防振系の MIMO 制御 <../user_guide/tutorials/case_active_damping>`
- :doc:`伝達関数計測: 実測・コヒーレンス・フィッティング <../user_guide/tutorials/case_transfer_function>`
- :doc:`キャリブレーションパイプライン: Counts から Strain へ <../user_guide/tutorials/case_calibration_pipeline>`
- :doc:`DTT XML 活用: 測定済み応答の読み込みと再利用 <../user_guide/tutorials/case_dttxml_calibration>`

II. 外部連携・I/O・再現性
-------------------------

- :doc:`Finesse 3 連携: シミュレーションと測定の比較 <../user_guide/tutorials/case_finesse_optics>`
- :doc:`ObsPy 連携: 地震データの取り込みと解析 <../user_guide/tutorials/case_seismic_obspy>`
- :doc:`GBD 形式 I/O: 書き出しと再読込の実務フロー <../user_guide/tutorials/case_gbd_format>`
- :doc:`HDF5 provenance: 再現可能なメタデータ管理 <../user_guide/tutorials/case_hdf5_provenance>`
- :doc:`PyCBC 連携: gwexpy 前処理から探索まで <../user_guide/tutorials/case_pycbc_search>`

III. 統計・機械学習ワークフロー
--------------------------------

- :doc:`Bootstrap PSD と GLS フィッティング <../user_guide/tutorials/case_bootstrap_gls_fitting>`
- :doc:`ML 前処理パイプライン: 特徴量整形と比較 <../user_guide/tutorials/case_ml_preprocessing>`
- :doc:`イベント同期解析: SegmentTable による窓選択 <../user_guide/tutorials/case_segment_analysis>`
- :doc:`物理妥当性検証: 単位・数値床・健全性テスト <../user_guide/tutorials/case_physics_validation>`

IV. ノイズハンティングと検出器診断
----------------------------------

- :doc:`ノイズバジェット解析: 多チャンネル相関で寄与源を切り分ける <../user_guide/tutorials/case_noise_budget>`
- :doc:`Bruco と ICA: ラインノイズ低減の統合フロー <../user_guide/tutorials/case_bruco_ica_denoising>`
- :doc:`Bruco 応用: バイリニア結合と AM/FM 復調 <../user_guide/tutorials/case_bruco_advanced>`
- :doc:`バイオリンモード解析: 共振モードの同定と追跡 <../user_guide/tutorials/case_violin_mode>`
- :doc:`シューマン共鳴解析: 環境磁場モードの読み解き <../user_guide/tutorials/case_schumann_resonance>`
- :doc:`グリッチ詳細解析: Q 変換と Omega スキャン <../user_guide/tutorials/case_glitch_analysis>`

.. note::
   各 API の詳細（引数・戻り値・クラス一覧）は :doc:`../reference/index` を参照してください。

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
   ../user_guide/tutorials/case_noise_budget
   ../user_guide/tutorials/case_bruco_ica_denoising
   ../user_guide/tutorials/case_bruco_advanced
   ../user_guide/tutorials/case_violin_mode
   ../user_guide/tutorials/case_schumann_resonance
   ../user_guide/tutorials/case_glitch_analysis
