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
| **実行実績** | | |
| ├── 実際に実行されたAPI (`executed`) | 520 | 99.8% |
| ├── 環境要因で実行不能 (`environment_blocked`) | 1 | 0.2% |
| └── 未実行 (`not_run`) | 0 | 0.0% |
| **最終Disposition** | | |
| ├── 契約検証成功 (`verified`) | 395 | 75.8% |
| ├── 契約差異/不整合観測 (`differ`) | 125 | 24.0% |
| └── 環境依存制約 (`environment_blocked`) | 1 | 0.2% |
| **GWpyパリティ比較** | | |
| ├── 完全一致 (`match`) | 276 | 53.0% |
| ├── 差異あり (`differ`) | 3 | 0.6% |
| └── 比較対象なし/拡張機能 (`incomparable`) | 240 | 46.1% |
| └── 該当なし (`not_applicable`) | 1 | 0.2% |

総試行試験数（Recorded Trials）: **1572件**

---

## 2. 9ワーカー並列実行の詳細実績

| Worker | 担当領域 | 割当件数 | 実行数 | 検証成功 | 差異/不整合 | 環境制約 | 試行数 |
|---|---|---:|---:|---:|---:|---:|---:|
| W1 | `types` 継承API 前半 | 61 | 61 | 56 | 5 | 0 | 183 |
| W2 | `types` 継承API 後半 | 60 | 60 | 54 | 6 | 0 | 180 |
| W3 | `timeseries` 継承API | 75 | 75 | 52 | 23 | 0 | 225 |
| W4 | `frequencyseries` 継承API | 72 | 72 | 30 | 42 | 0 | 216 |
| W5 | `spectrogram` 継承API | 49 | 49 | 41 | 8 | 0 | 147 |
| W6 | `fields` 16 + `plot` 18 + GUI 9 | 43 | 43 | 39 | 4 | 0 | 129 |
| W7 | 慣例I/O関数 前半 | 42 | 42 | 42 | 0 | 0 | 126 |
| W8 | 慣例I/O関数 後半 | 41 | 41 | 41 | 0 | 0 | 123 |
| W9 | 一般関数 55 + fitting 10 | 65 | 64 | 27 | 37 | 1 | 193 |
| **合計** | | **508** | **507** | **382** | **125** | **1** | **1,522** |

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
