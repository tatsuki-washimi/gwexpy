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
| :--- | :--- | :---: | :--- |
| Bruco | analysis/bruco.py | [/] | 概要はあるが、パラメータの詳細が不足 |
| BrucoPairSummary | analysis/bruco.py | [x] | TypedDict として定義済 |
| BrucoResult | analysis/bruco.py | [/] | 概要はあるが `Attributes` セクションが不足 |
| FastCoherenceEngine | analysis/bruco.py | [/] | 1行の概要のみ。内部実装寄り |
| CouplingFunctionAnalysis | analysis/coupling.py | [x] | 良好な記述を確認 |
| CouplingResult | analysis/coupling_result.py | [/] | クラス docstring に `Attributes` (型/単位/shape) が不足 |
| ResponseFunctionAnalysis | analysis/response.py | [x] | 良好な記述を確認 |
| ResponseFunctionResult | analysis/response.py | [/] | Dataclass だが `Attributes` での意味明記が不足 |
| SpectralStats | analysis/stats.py | [/] | 実装済だが**日本語**。英語への翻訳が必要 |
| PercentileThreshold | analysis/threshold.py | [x] | 良好 |
| RatioThreshold | analysis/threshold.py | [x] | 良好 |
| SigmaThreshold | analysis/threshold.py | [x] | 良好 |

---

## 3. Time & GPS Conversion (時刻・GPS操作)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| tconvert | time/core.py | [ ] | 実装内に docstring が存在しない |
| to_gps | time/core.py | [ ] | 実装内に docstring が存在しない |
| from_gps | time/core.py | [ ] | 実装内に docstring が存在しない |

---

## 4. Analysis & Utility (その他の解析・ユーティリティ)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| calculate_coupling | analysis/coupling.py | [x] | 良好 |
| compute_spectral_stats | analysis/stats.py | [DELETE] | 削除済（SpectralStats へ移行） |
| get_logger | utils/logger.py | [x] | 良好 |
| get_unit | types/metadata.py | [/] | 戻り値の型や挙動の説明が曖昧 |

---

## 5. Series Classes (系列データクラス)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| TimeSeries | timeseries/timeseries.py | [x] | 良好 |
| FrequencySeries | frequencyseries/frequencyseries.py | [/] | 概要が簡素すぎる |

---

## 6. Signal & Noise Generation (信号・ノイズ生成)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| white_noise | noise/colored.py | [/] | 1行のみ。Parameters/Returns の記述なし |
| pink_noise | noise/colored.py | [/] | 1行のみ。Parameters/Returns の記述なし |
| red_noise | noise/colored.py | [/] | 1行のみ。Parameters/Returns の記述なし |
| from_asd | noise/wave.py | [x] | 良好 |
| sine / square / chirp | noise/wave.py | [x] | 良好 |
| inject_noise | noise/non_gaussian.py | [/] | 1行のみ。概要が不足 |

---

## 7. Metadata & Results (メタデータ・解析結果クラス)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| MetaData | types/metadata.py | [ ] | クラスレベルの docstring が欠落（最優先課題） |
| MetaDataMatrix | types/metadata.py | [x] | 良好 |
| CouplingResultCollection | analysis/coupling_result.py | [x] | 良好 |

---

## 8. 今後の改善アクション (最優先)

1. **`MetaData` クラスの完全なドキュメント化**:
   - `Attributes` セクションにて `channel`, `unit`, `t0`, `dt` 等の扱いを詳述。
2. **Result 系オブジェクトの `Attributes` 拡充**:
   - `CouplingResult` および `ResponseFunctionResult` のフィールド詳細化。
3. **ノイズ生成関数のパラメータ詳細化**:
   - `white_noise` 等に対する `Parameters` / `Returns` 追加。
4. **言語の統一 (英語化)**:
   - `SpectralStats` の翻訳。
5. **時刻変換関数の docstring 新規追加**:
   - `to_gps`, `from_gps`, `tconvert` への記述追加。

---

## 9. 結び
本監査により、GWexpy は「主要なフロー」については良好なドキュメントを持つが、「結果の保持（Result classes）」および「基盤となるメタデータ (MetaData)」において情報の断片化や欠落があることが判明した。

---

## 10. 今後の運用・管理ポリシーへの提言

ドキュメントの品質向上とそれを維持する仕組みとして、以下の3点を導入・明文化することを推奨します。

1. **Docstring のスタイル規約の徹底**
   - NumPy スタイルを標準とし、`Parameters`, `Returns`, `Attributes`, `Examples`, `Notes` などのセクションを適切に設ける。
   - 特に `Attributes` や `Returns` では、変数の型、単位（Astropy Quantity など）、配列形状（shape）などの物理的・実装上の詳細を漏れなく記述する。

2. **英語一次の明文化**
   - 新規開発・既存コードに関わらず、Docstring はすべて英語を一次言語とする。
   - 日本語のコメントや docstring が残存している場合は、発見次第英語に翻訳・置換する。

3. **CI による自動検出**
   - `ruff` (pydocstyle 相当のルール `D` 系) や `mypy` などを活用し、Docstring の欠落や型・説明の不整合を CI パイプラインで機械的に検出・ブロックする仕組みを導入する。
