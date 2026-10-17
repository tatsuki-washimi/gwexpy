# C監査：521 API統合監査レポート

- **ベースラインコミット**: `c2898762ca68e20a1a241fffb3c3262b0a27290b`
- **親Issue**: `#765`
- **対象母集団**: 521 API
- **事前検証証拠**: 13 API（慣例関数3件、継承API10件）
- **今回の並列検証対象**: 508 API（9ワーカー並列実行）

---

## 1. 総合検証メトリクス

| 指標 | 数量 | 構成比 |
|---|---:|---:|
| **全母集団** | **521** | 100.0% |
| ├── 既存実行証拠（事前完了） | 13 | 2.5% |
| └── 今回の検証対象（9ワーカー） | 508 | 97.5% |
| **実行実績 (Execution Level)** | | |
| ├── 実呼出し実行 (`actual_call`) | 290 | 55.7% |
| ├── 属性アクセス・存在確認 (`attribute_access`) | 79 | 15.2% |
| ├── スモーク確認のみ (`smoke_only`) | 74 | 14.2% |
| ├── ハーネス不備/フィクスチャ未整備 (`contract_pending`) | 59 | 11.3% |
| └── 環境要因で実行不能 (`environment_blocked`) | 19 | 3.6% |
| **到達性ステータス (Reachability Status)** | | |
| ├── 呼出し成功・戻り値受領 (`reached_and_returned`) | 281 | 53.9% |
| ├── API本体到達例外観測 (`reached_with_observed_exception`) | 9 | 1.7% |
| ├── 属性アクセス・存在確認 (`attribute_accessed`) | 79 | 15.2% |
| ├── 引数スモーク確認のみ (`smoke_signature_only`) | 74 | 14.2% |
| ├── 到達性未判定・保留 (`reachability_unresolved`) | 19 | 3.6% |
| ├── フィクスチャ生成不能 (`fixture_setup_failed`) | 40 | 7.7% |
| └── 環境要因で実行不能 (`environment_blocked`) | 19 | 3.6% |
| **最終Disposition** | | |
| ├── 契約検証成功 (`verified`) | 246 | 47.2% |
| ├── 契約差異/不整合観測 (`differ`) | 44 | 8.4% |
| ├── 契約検証保留/フィクスチャ・強アサーション待ち (`contract_pending`) | 212 | 40.7% |
| └── 環境依存制約 (`environment_blocked`) | 19 | 3.6% |
| **GWpyパリティ比較 (Parity Status)** | | |
| ├── 数値・配列完全一致 (`numeric_value_parity`) | 75 | 14.4% |
| ├── メタデータ・属性一致 (`metadata_property_parity`) | 33 | 6.3% |
| ├── 例外挙動一致 (`exception_parity`) | 106 | 20.3% |
| ├── パリティ差異あり (`differ`) | 6 | 1.2% |
| ├── 比較対象なし/独自拡張 (`incomparable`) | 274 | 52.6% |
| └── 該当なし (`not_applicable`) | 27 | 5.2% |
| **比較判定モード (Comparison Mode)** | | |
| ├── 配列・スカラー数値比較 (`numeric_allclose`) | 55 | 10.6% |
| ├── コレクション要素全件比較 (`collection_elementwise_numeric`) | 16 | 3.1% |
| ├── LAL SWIGオブジェクト直接数値比較 (`swig_lal_object_numeric`) | 4 | 0.8% |
| ├── メタデータ・属性等価比較 (`metadata_equality`) | 39 | 7.5% |
| ├── 同一例外送出判定 (`exception_equality`) | 106 | 20.3% |
| ├── 比較対象なし/独自拡張 (`incomparable`) | 274 | 52.6% |
| └── 該当なし (`not_applicable`) | 27 | 5.2% |

総試行試験数（Recorded Trials）: **1,572件**

---

## 2. 9ワーカー並列実行の詳細実績

| Worker | 担当領域 | 割当件数 | actual_call | attr_access | smoke_only | env_blocked | verified | differ | pending | 試行数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | `types` 継承API 前半 | 61 | 44 | 15 | 2 | 0 | 36 | 8 | 17 | 183 |
| W2 | `types` 継承API 後半 | 60 | 44 | 10 | 4 | 0 | 34 | 10 | 16 | 180 |
| W3 | `timeseries` 継承API | 75 | 28 | 15 | 11 | 14 | 19 | 9 | 33 | 225 |
| W4 | `frequencyseries` 継承API | 72 | 23 | 5 | 3 | 4 | 20 | 3 | 45 | 216 |
| W5 | `spectrogram` 継承API | 49 | 34 | 9 | 4 | 0 | 29 | 5 | 15 | 147 |
| W6 | `fields` 16 + `plot` 18 + GUI 9 | 43 | 14 | 24 | 5 | 0 | 13 | 1 | 29 | 129 |
| W7 | 慣例I/O関数 前半 | 42 | 32 | 0 | 9 | 0 | 26 | 6 | 10 | 126 |
| W8 | 慣例I/O関数 後半 | 41 | 25 | 0 | 13 | 0 | 25 | 0 | 16 | 123 |
| W9 | 一般関数 55 + fitting 10 | 65 | 33 | 1 | 23 | 1 | 31 | 2 | 31 | 193 |
| **合計** | | **508** | **277** | **79** | **74** | **19** | **233** | **44** | **212** | **1,522** |

---

## 3. 主な観測事項および差異（differ）の分析

### 3.1 観測された差異の要因分類
1. **多次元配列における1D専用メソッドのMRO継承問題 (GWpy互換要因)**:
   - 例: `Array2D.step`
   - 原因: 1Dの `Series.step` が多次元 `Array2D` にそのまま継承され、`Array2D.plot` の `AxesImage` に対して `drawstyle` 引数が渡されて `AttributeError` となる。これはGWpy自身でも同様に発生する挙動（`reached_with_observed_exception`）。
2. **外部依存パッケージのバイナリ非互換 (`environment_blocked`)**:
   - 例: `gwexpy.timeseries.arima.fit_arima`
   - 原因: `pmdarima` のC-拡張モジュールがNumPy 2.xに対してバイナリ不整合を起こし、モジュールインポート時に `ValueError` となる。
3. **GWexpy独自拡張APIとGWpyの契約差異**:
   - `BifrequencyMap`, `ScalarField`, `FieldPlot` などの独自クラス群はGWpyに対応するクラスが存在しないため `incomparable` として記録。
4. **LAL SWIGオブジェクト直接数値比較による完全パリティ実証**:
   - `TimeSeries.to_lal` および `FrequencySeries.to_lal` について、SWIGポインタ文字列の比較を廃止し、`data.data` (NumPy配列)、`deltaT`/`deltaF`、`epoch`、`sampleUnits` を直接数値比較（`swig_lal_object_numeric`）することで、GWpyとの偽陽性 `differ` を解消し完全パリティ（`numeric_value_parity`）を実証。

---

## 4. 成果物一覧

- 統合台帳（521行）: `deliverables/audit-c-coverage-c2898762/c-coverage-c2898762.jsonl`
- 全試行証拠（1,572行）: `deliverables/audit-c-coverage-c2898762/c-coverage-521-all-trials.jsonl`
- 集計サマリーJSON: `deliverables/audit-c-coverage-c2898762/c-coverage-521-integrated-summary.json`
- エビデンスバンドルZIP: `deliverables/audit-c-coverage-c2898762/c-coverage-521-evidence-bundle.zip`
- ワーカー別ディレクトリ: `deliverables/audit-c-coverage-c2898762/parallel-batch/worker-W[1-9]/`
