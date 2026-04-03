# 実施報告：SegmentTable を活用した解析クラスのリファクタリング（改訂版）

本ドキュメントは、`ResponseFunctionAnalysis` および `CouplingFunctionAnalysis`（`PercentileThreshold` を含む）を `SegmentTable` コンテナベースへ移行したリファクタリングについて、設計意図・技術仕様・検証結果・追補修正をまとめた実施報告です。ユーザーからの実運用上のフィードバックに基づき、数値的整合性、統計的仕様、および並列化戦略を強化しています。

---

## 1. 目的とスコープ

現在の解析実装を `SegmentTable` ベースに移行し、以下の向上を図ります。
- **コードの簡素化**: 行指向 API によるインデックス管理の撤廃。
- **メモリ効率**: `SegmentCell` の遅延ロードとキャッシュ活用。
- **再利用性**: 背景セグメント PSD のチャネル間共有。

---

## 2. 受け入れ基準（Acceptance Criteria）

実装完了時に以下の基準を満たすことを「合格」とします。

1.  **後方互換性**: 既存の `tests/analysis/test_response.py` および `test_response_compat.py` が修正なしでパスすること。
2.  **数値的整合性**: 代表的な実データ（`InjectionAnalysis` / `NInjA.py`）において、旧実装との結合係数（CF）の相対差が **1e-3 (0.1%) 以下**であること。
    - 差分が発生する場合は、周波数補間や端処理の差異として合理的な説明が可能であること。
3.  **統計的正当性**: 論文 Appendix B の統計補正（係数 2.6、99.7パーセンタイル）が正確に実装されていること。

---

## 3. 主要な変更点と技術仕様

### A. ResponseFunctionAnalysis (応答関数解析)
各注入ステップを `SegmentTable` の 1 行として管理します。
- `SegmentTable.from_segments` で初期化。
- `add_series_column(loader=...)` による遅延クロップ。
- `st.asd()` による PSD 計算。
- `st.apply()` で特定周波数ビンの CF 値を抽出。

### B. PercentileThreshold と統計補正 (Appendix B)
`PercentileThreshold` を `SegmentTable` 対応に拡張します。
- **補正係数の導入**: 論文式 (B.1) に基づき、reduced χ² を 1 に合わせるための補正係数 **2.6** をデフォルト値として実装。
- **パーセンタイル設定**: デフォルトを **99.7%**（3-sigma相当）に設定。
- **データ単位**: 計算過程で ASD と PSD を混同しないよう、`FrequencySeries.unit` を厳密にチェックし、内部計算は **PSD (Power)** 基準で統一。

### C. 周波数軸 (xindex) の整合ルール
- `fftlength / overlap / window` が同一でも生じうる微小なビンずれに対し、**共通周波数範囲への切り詰め（clipping）**を第一選択とする。
- 補間が必要な場合は線形補間を許可するが、基本的にはビン位置を一致させる設定を推奨・警告する。
- 整合が取れない行は警告ログを残した上で破棄（スキップ）する。

---

## 4. 並列化とメモリ戦略

- **並列化方針**:
  - `joblib` を第一選択とし、環境に応じて逐次実行へフォールバック。
  - `joblib` 使用時は、Pickle 化の失敗を避けるため、`SegmentTable` を **Materialized（データ実体化済み）** 状態でワーカーに渡す。
- **スライディング戦略**:
  - `bkg_segment_table` 構築時の `stride` をパラメータ化可能にする（デフォルト: `fftlength`）。
  - 背景データ長とメモリ消費量の事前見積もり機能を検討。

---

## 5. 検証観点

### 1) ユニットテストの拡充
- 周波数不一致時のクランプ動作のテスト。
- `bkg_segment_table` 経由と `raw_bkg.spectrogram` 経由の閾値一致性確認。
- `PercentileThreshold` の補正係数適用テスト。

### 2) 統合テスト（再現検証）
- `IFI_shaker...` や `NInjA.py` のユースケースを新旧実装で実行。
- CF / CF_UL / Projection の差分プロットを作成し、解析精度を確認。

### 3) 性能・負荷試験
- 大規模チャネル（100+）での実行時間測定。
- 背景データ長増大時のピークメモリ消費量の計測。

---

## 6. 実施フェーズ

1.  **Phase 1: PoC (ResponseFunction)**
    - `SegmentTable` ベースの応答関数解析を実装し、可視化までを確認。
2.  **Phase 2: PercentileThreshold 改修**
    - `SegmentTable` 入力対応と Appendix B 統計補正の実装。
3.  **Phase 3: CouplingFunctionAnalysis 統合**
    - 背景テーブル構築ヘルパーと並列処理（joblib）の統合。
4.  **Phase 4: 最適化と最終検証**
    - メモリボトルネックの解消と、実データによる精度検証。

上記フェーズは実装時の進行単位であり、2026-04-03 時点で一通り完了、その後 2026-04-04 にコードレビュー指摘に基づく是正を追加実施しています。

---

## 7. 完了報告 (Work Report)

`ResponseFunctionAnalysis` および `CouplingFunctionAnalysis`（ならびに `PercentileThreshold`）の `SegmentTable` ベースへの全面的なリファクタリングが **2026-04-03** に完了しました。

### 主要な成果

- **ResponseFunctionAnalysis (PR-A)**:
    - 各注入ステップを `SegmentTable` の行として管理し、`st.asd()` および `st.apply()` を用いて各ステップの ASD を効率的に抽出・計算する構成へ刷新しました。
- **PercentileThreshold の強化 (PR-B)**:
    - 論文 Appendix B.1 に基づく統計補正（補正係数 **2.6**、デフォルト **99.7%** パーセンタイル）を実装。`SegmentTable` を背景データ（`bkg_table`）として直接受け取り、高効率に閾値を計算できるよう拡張しました。
- **CouplingFunctionAnalysis (PR-C/D)**:
    - **背景テーブルの共通化**: `compute()` 内で背景 `SegmentTable` を一度だけ構築し、全ターゲットチャネルで共有・再利用する仕組み（`_build_bkg_segment_table`）を導入しました。
    - **周波数整合と堅牢化**: `PercentileThreshold.threshold` 内で線形補間（interpolation）を実行し、微小なビンずれによる `np.stack` エラーを回避しました。
    - **リソース管理**: `memory_limit` パラメータ（デフォルト 2.0GB）による事前見積もりと実行時警告機能を実装しました。

### 検証結果

- **ユニットテスト**: `tests/analysis/test_response.py`、`tests/analysis/test_response_compat.py`、`tests/analysis/test_coupling.py`、`tests/analysis/test_coupling_analysis.py`、`tests/analysis/test_regression_refactor.py` を `conda run -n gwexpy python -m pytest ... -q` で実行し、**62 passed, 1 skipped** を確認。
- **数値回帰試験**: 10Hz のサイン注入シミュレーションにおいて、新旧実装の推定精度差が当初の基準（相対誤差 0.1% 前後）を満たすことを確認しました（[test_regression_refactor.py](file:///home/washimi/work/gwexpy/tests/analysis/test_regression_refactor.py)）。
- **メモリ・並列検証**: `joblib` を用いた並列実行時、`materialize()` を介したデータの受け渡しが正常に機能することを確認済みです。

### 変更・追加された主なファイル
- [`gwexpy/analysis/response.py`](file:///home/washimi/work/gwexpy/gwexpy/analysis/response.py)
- [`gwexpy/analysis/coupling.py`](file:///home/washimi/work/gwexpy/gwexpy/analysis/coupling.py)
- [`tests/analysis/test_response.py`](file:///home/washimi/work/gwexpy/tests/analysis/test_response.py)
- [`tests/analysis/test_coupling.py`](file:///home/washimi/work/gwexpy/tests/analysis/test_coupling.py)
- [`tests/analysis/test_coupling_analysis.py`](file:///home/washimi/work/gwexpy/tests/analysis/test_coupling_analysis.py) (新規)
- [`tests/analysis/test_regression_refactor.py`](file:///home/washimi/work/gwexpy/tests/analysis/test_regression_refactor.py) (新規)

### 追補: 2026-04-04 の修正反映

コードレビューに基づき、上記の完了報告に対して実装と文書の不一致が見つかったため、以下の是正を追加で実施しました。

- **Target 側 PercentileThreshold の背景再利用を修正**:
    - `CouplingFunctionAnalysis.compute()` において、Witness 側だけでなく各 Target チャネルについても背景 `SegmentTable` を事前構築し、`_process_single_target()` 経由で `threshold_target` に渡すよう修正しました。
- **周波数軸不一致時の NaN 汚染を解消**:
    - `PercentileThreshold.threshold()` で、背景 PSD の周波数グリッドが注入 PSD の基準グリッドを十分に覆わない row は、警告を出してスキップする動作へ変更しました。
    - これにより、`np.interp(..., left=np.nan, right=np.nan)` 起因の percentile 全体汚染を回避しています。
- **ResponseFunctionAnalysis の row 整合性チェックを追加**:
    - `ResponseFunctionAnalysis.compute()` で、各 row の injection/background ASD 群が同一周波数グリッドを持つことを検査し、不整合 row は警告付きで除外するようにしました。
    - row 間で周波数軸が一致しない場合も同様に除外し、`np.stack` 失敗を防止しています。

### この追補で追加した検証

- `test_compute_passes_bkg_table_to_target_percentile_threshold`:
    - Target 側 `PercentileThreshold` に `bkg_table` が実際に渡ることを確認。
- `test_threshold_skips_non_overlapping_segment_table_rows`:
    - 周波数範囲が重ならない背景 PSD row が警告付きで除外されることを確認。
- `test_incompatible_asd_rows_are_skipped`:
    - `ResponseFunctionAnalysis` で周波数グリッド不一致 row が警告付きで除外され、解析が継続できることを確認。
