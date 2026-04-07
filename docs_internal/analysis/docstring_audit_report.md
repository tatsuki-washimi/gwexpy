# GWexpy Docstring 網羅性 & 品質監査レポート (2026-04-07 版)

## 1. 概要
本レポートは、GWexpy API の Docstring 網羅性と品質（パラメータ、戻り値、単位、配列形状の説明等）について、現在のソースコード（事実関係）を照合した結果をまとめたものである。

**監査方針:**
- [x] 公開 API（特に結果オブジェクトやノイズ生成）に詳細な説明があるか
- [x] パラメータの型、単位（Astropy Quantity）、配列形状が明記されているか
- [x] 言語方針（英語一次）が守られているか

---

## 2. 監査結果 (ステータス一覧)

ステータス凡例: `[x]` 完了, `[/]` 部分的・要改善, `[ ]` 未着手, `[DELETE]` 削除済

### 表 1: 解析 & フィッティング (analysis/)
| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| Bruco | analysis/bruco.py | [x] | クラスおよび compute メソッドを NumPy スタイルで硬化完了 |
| BrucoPairSummary | analysis/bruco.py | [x] | TypedDict として定義済 |
| BrucoResult | analysis/bruco.py | [x] | Attributes (型/単位/shape) を含め、ユーザー草案を適用し硬化完了 |
| FastCoherenceEngine | analysis/bruco.py | [x] | 使用例を含むユーザー草案を適用し硬化完了 |
| CouplingFunctionAnalysis | analysis/coupling.py | [x] | 良好な記述を確認 |
| CouplingResult | analysis/coupling_result.py | [x] | Attributes (型/単位/shape) の明記を完了 |
| ResponseFunctionAnalysis | analysis/response.py | [x] | 良好な記述、およびメソッドの英語化を完了 |
| ResponseFunctionResult | analysis/response.py | [x] | Attributes での意味明記およびプロット関数の翻訳を完了 |
| SpectralStats | analysis/stats.py | [x] | 英語への翻訳・NumPy スタイル化を完了 |
| PercentileThreshold | analysis/threshold.py | [x] | 良好 |
| RatioThreshold | analysis/threshold.py | [x] | 良好 |
| SigmaThreshold | analysis/threshold.py | [x] | 良好 |

---

## 3. Time & GPS Conversion (時刻・GPS操作)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| tconvert | time/core.py | [x] | ベクトル化対応を含め詳細な docstring を追加 |
| to_gps | time/core.py | [x] | 型と挙動に関する docstring を詳細化 |
| from_gps | time/core.py | [x] | 型と挙動に関する docstring を詳細化 |

---

## 4. Analysis & Utility (その他の解析・ユーティリティ)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| calculate_coupling | analysis/coupling.py | [x] | 良好 |
| compute_spectral_stats | analysis/stats.py | [DELETE] | 削除済（SpectralStats へ移行） |
| get_logger | utils/logger.py | [x] | 良好 |
| get_unit | types/metadata.py | [x] | ユーザー草案に基づき、対象型やデフォルト挙動を詳細化 |

---

## 5. Series Classes (系列データクラス)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| TimeSeries | timeseries/timeseries.py | [x] | 良好 |
| FrequencySeries | frequencyseries/frequencyseries.py | [/] | 概要が簡素すぎる（**次期課題**） |

---

## 6. Signal & Noise Generation (信号・ノイズ生成)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| white_noise | noise/colored.py | [x] | Parameters/Returns/Examples の記述を完了 |
| pink_noise | noise/colored.py | [x] | Parameters/Returns/Examples の記述を完了 |
| red_noise | noise/colored.py | [x] | Parameters/Returns/Examples の記述を完了 |
| from_asd | noise/wave.py | [x] | 良好 |
| sine / square / chirp | noise/wave.py | [x] | 良好 |
| inject_noise | noise/non_gaussian.py | [x] | 詳細な内容、引数、例外、使用例を記述した草案を適用 |

---

## 7. Metadata & Results (メタデータ・解析結果クラス)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| MetaData | types/metadata.py | [x] | クラスレベルの docstring および各メソッドの記述を追加 |
| MetaDataMatrix | types/metadata.py | [x] | 各メソッド（names/units 等）の記述を追加・Ruff準拠化 |
| CouplingResultCollection | analysis/coupling_result.py | [x] | 良好 |

---

## 8. 実施済み改善 (2026-04-07 完了)

1. **`MetaData` クラスの完全なドキュメント化**:
   - `Attributes` セクションにて `channel`, `unit` 等の扱いを詳述し、Examples を追加。
2. **Result 系オブジェクトの `Attributes` 拡充**:
   - `CouplingResult` および `ResponseFunctionResult` のフィールド詳細化および翻訳。
3. **ノイズ生成関数の詳細化**:
   - `white_noise` 等に対する `Parameters` / `Returns` / `Examples` 追加。
4. **言語の統一 (英語化)**:
   - `SpectralStats`, `ResponseFunctionResult` メソッド, `CouplingResult` 内部関数の翻訳。
5. **時刻変換関数の docstring 新規追加**:
   - `to_gps`, `from_gps`, `tconvert` への記述追加。

---

## 9. 結び
本監査およびその後の硬化フェーズにより、GWexpy の中核となるメタデータ・結果保持クラス、および基本ユーティリティのドキュメント品質が大幅に向上した。解析結果の信頼性と再利用性を高めるための「型・単位・形状」の明示が完了している。

---

## 10. 運用・管理ポリシー (導入済み)

以下の 3 点について、`pyproject.toml` へのルール追加および実装での反映を完了しました。

1. **Docstring スタイル規約の徹底 (NumPy スタイル)**
   - 修正対象の全ファイルにおいて NumPy スタイルを採用。
2. **英語一次の徹底**
   - 監査で指摘された全日本語箇所の翻訳を完了。
3. **CI による自動検出 (Ruff `D` ルール)**
   - **[導入済み]** `pyproject.toml` の `ruff` 設定に `"D"` を追加。修正対象ファイルについては現在 100% クリーンな状態を維持。
