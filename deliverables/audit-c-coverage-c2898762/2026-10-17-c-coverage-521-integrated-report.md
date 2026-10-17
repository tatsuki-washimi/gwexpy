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
| ├── 実呼出し実行 (`actual_call`) | 356 | 68.3% |
| ├── 属性アクセス・存在確認 (`attribute_access`) | 72 | 13.8% |
| ├── スモーク確認のみ (`smoke_only`) | 74 | 14.2% |
| └── 環境要因で実行不能 (`environment_blocked`) | 19 | 3.6% |
| **最終Disposition** | | |
| ├── 契約検証成功 (`verified`) | 219 | 42.0% |
| ├── 契約差異/不整合観測 (`differ`) | 137 | 26.3% |
| ├── 契約検証保留/フィクスチャ・強アサーション待ち (`contract_pending`) | 146 | 28.0% |
| └── 環境依存制約 (`environment_blocked`) | 19 | 3.6% |
| **GWpyパリティ比較** | | |
| ├── 数値・単位・形状一致 (`value_parity`) | 112 | 21.5% |
| ├── 例外挙動一致 (`exception_parity`) | 106 | 20.3% |
| ├── パリティ差異あり (`differ`) | 2 | 0.4% |
| ├── 試行エラー/例外観測 (`trial_error`) | 26 | 5.0% |
| ├── 比較対象なし/独自拡張 (`incomparable`) | 274 | 52.6% |
| └── 該当なし (`not_applicable`) | 1 | 0.2% |

総試行試験数（Recorded Trials）: **1572件**

---

## 2. 9ワーカー並列実行の詳細実績

| Worker | 担当領域 | 割当件数 | actual_call | attr_access | smoke_only | env_blocked | verified | differ | pending | 試行数 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| W1 | `types` 継承API 前半 | 61 | 44 | 15 | 2 | 0 | 33 | 11 | 17 | 183 |
| W2 | `types` 継承API 後半 | 60 | 46 | 10 | 4 | 0 | 31 | 15 | 14 | 180 |
| W3 | `timeseries` 継承API | 75 | 35 | 15 | 11 | 14 | 18 | 17 | 26 | 225 |
| W4 | `frequencyseries` 継承API | 72 | 60 | 5 | 3 | 4 | 18 | 42 | 8 | 216 |
| W5 | `spectrogram` 継承API | 49 | 37 | 8 | 4 | 0 | 26 | 11 | 12 | 147 |
| W6 | `fields` 16 + `plot` 18 + GUI 9 | 43 | 20 | 18 | 5 | 0 | 13 | 7 | 23 | 129 |
| W7 | 慣例I/O関数 前半 | 42 | 33 | 0 | 9 | 0 | 24 | 9 | 9 | 126 |
| W8 | 慣例I/O関数 後半 | 41 | 28 | 0 | 13 | 0 | 23 | 5 | 13 | 123 |
| W9 | 一般関数 55 + fitting 10 | 65 | 40 | 1 | 23 | 1 | 20 | 20 | 24 | 193 |
| **合計** | | **508** | **343** | **72** | **74** | **19** | **206** | **137** | **146** | **1,522** |

---

## 3. 主な観測事項および差異（differ）の分析

### 3.1 観測された差異の要因分類
1. **多次元配列における1D専用メソッドのMRO継承問題 (GWpy互換要因)**:
   - 例: `Array2D.step`
   - 原因: 1Dの `Series.step` が多次元 `Array2D` にそのまま継承され、`Array2D.plot` の `AxesImage` に対して `drawstyle` 引数が渡されて `AttributeError` となる。これはGWpy自身でも同様に発生する挙動。
2. **外部依存パッケージのバイナリ非互換 (`environment_blocked`)**:
   - 例: `gwexpy.timeseries.arima.fit_arima`
   - 原因: `pmdarima` のC-拡張モジュールがNumPy 2.xに対してバイナリ不整合を起こし、モジュールインポート時に `ValueError` となる。
3. **GWexpy独自拡張APIとGWpyの契約差異**:
   - `BifrequencyMap`, `ScalarField`, `FieldPlot` などの独自クラス群はGWpyに対応するクラスが存在しないため `incomparable` として記録。

---

## 4. 成果物一覧

- 統合台帳（521行）: `deliverables/audit-c-coverage-c2898762/c-coverage-c2898762.jsonl`
- 全試行証拠（1572行）: `deliverables/audit-c-coverage-c2898762/c-coverage-521-all-trials.jsonl`
- 集計サマリーJSON: `deliverables/audit-c-coverage-c2898762/c-coverage-521-integrated-summary.json`
- ワーカー別ディレクトリ: `deliverables/audit-c-coverage-c2898762/parallel-batch/worker-W[1-9]/`
