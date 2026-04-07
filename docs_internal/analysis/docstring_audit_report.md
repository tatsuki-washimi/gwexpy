# GWexpy Docstring Audit Report

このドキュメントは、`gwexpy` リポジトリ内のすべての公開クラスおよび関数の Docstring 網羅性と品質を監査した結果を記録する資産です。

## 監査ステータス凡例

- `[x]`: 良好（概要、Parameters, Returns が適切に記述されている）
- `[/]`: 不完全（一部の記述が不足、または形式が不適切）
- `[ ]`: 未記述（Docstring が存在しない）

---

## 1. Analysis & Fitting (解析・フィッティング)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| Bruco | analysis/bruco.py | [/] | 概要はあるが、パラメータの詳細が不足 |
| BrucoPairSummary | analysis/bruco.py | [x] | 良好 |
| BrucoResult | analysis/bruco.py | [x] | 良好 |
| FastCoherenceEngine | analysis/bruco.py | [/] | 内部クラス寄りの記述だが、Docstring は簡素 |
| CouplingFunctionAnalysis | analysis/coupling.py | [x] | 良好。背景データ管理についても言及あり |
| CouplingResult | analysis/coupling_result.py | [x] | 良好 |
| ResponseFunctionAnalysis | analysis/response.py | [x] | 良好。ラッパー関数も適切に記述 |
| ResponseFunctionResult | analysis/response.py | [x] | 良好 |
| SpectralStats | analysis/stats.py | [/] | 日本語。内容自体は Attributes 含め詳細 |
| PercentileThreshold | analysis/threshold.py | [x] | 良好 |
| RatioThreshold | analysis/threshold.py | [x] | 良好。統計的仮定の記述あり |
| SigmaThreshold | analysis/threshold.py | [x] | 良好 |

---

## 2. Time & GPS Conversion (時刻・GPS操作)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| tconvert | time/core.py | [x] | 良好。多様な入力形式への対応が明記 |
| to_gps | time/core.py | [x] | 良好 |
| from_gps | time/core.py | [x] | 良好 |

---

## 3. Analysis & Utility (その他の解析・ユーティリティ)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| calculate_coupling | analysis/coupling.py | [x] | 良好なラッパー関数記述 |
| compute_spectral_stats | analysis/stats.py | [ ] | Docstring 欠落、または極めて簡素 |
| get_logger | utils/logger.py | [x] | 良好 |
| get_unit | types/metadata.py | [/] | Returns の記述が曖昧 |

---

## 4. Series Classes (系列データクラス)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| TimeSeries | timeseries/timeseries.py | [x] | 良好。機能一覧や互換性に関する詳細な記述あり |
| FrequencySeries | frequencyseries/frequencyseries.py | [/] | 概要が簡素すぎる（Light wrapper の記述のみ） |

---

## 5. Signal & Noise Generation (信号・ノイズ生成)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| white_noise | noise/colored.py | [/] | 1行のみ。Parameters/Returns の記述なし |
| pink_noise | noise/colored.py | [/] | 1行のみ。Parameters/Returns の記述なし |
| red_noise | noise/colored.py | [/] | 1行のみ。Parameters/Returns の記述なし |
| from_asd | noise/wave.py | [x] | 良好。アルゴリズムの解説あり |
| sine / square / chirp | noise/wave.py | [x] | 良好。引数、単位、チャンネル、名称など詳細 |
| inject_noise | noise/non_gaussian.py | [x] | 良好 |

---

## 6. Preprocessing & Decomposition (前処理・要因分析)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| align_timeseries_collection | timeseries/preprocess.py | [x] | 良好。処理フローと Notes が充実 |
| impute_timeseries | timeseries/preprocess.py | [x] | 良好。各手法や制限（limit/max_gap）の説明あり |
| standardize_timeseries | timeseries/preprocess.py | [x] | 優秀。利用例 (Examples) あり |
| standardize_matrix | timeseries/preprocess.py | [x] | 優秀。axis の挙動が詳細に解説されている |
| whiten_matrix | timeseries/preprocess.py | [x] | 優秀。PCA/ZCA の違いと Examples あり |
| pca_fit / ica_fit | timeseries/decomposition.py | [/] | 概要はあるが、Preprocessing 系に比べ Example 等が不足 |

---

## 7. Field Analysis & Visualization (フィールド解析・描画)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| compute_psd / coherence_map | fields/signal.py | [x] | 良好 |
| make_demo_scalar_field | fields/demo.py | [x] | 良好。デモ用パラメータの詳細あり |
| make_propagating_gaussian | fields/demo.py | [/] | 他の関数への参照のみで、自身の引数説明がない |
| plot_mmm / plot_summary | plot/plot.py | [x] | 良好。NumPy スタイルでパラメータを詳述 |
| plot_gauch_dashboard | plot/gauch_dashboard.py | [x] | 良好 |

---

## 8. Metadata & Results (メタデータ・解析結果クラス)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| MetaData | types/metadata.py | [ ] | 実質的な記述なし |
| MetaDataMatrix | types/metadata.py | [x] | 良好 |
| CouplingResult | analysis/coupling_result.py | [/] | 概要のみ。Attributes の説明が不足 |
| CouplingResultCollection | analysis/coupling_result.py | [x] | 良好。日本語で Examples あり |
| ResponseFunctionResult | analysis/response.py | [/] | 概要のみ。Attributes の説明が不足 |

---

## 9. 内部実装・共通ツール (Internal & Utils)

| 要素名 | ファイルパス | Docstring 状態 | 判定・コメント |
| :--- | :--- | :---: | :--- |
| read_timeseries_* | io/registry.py (等) | [/] | 概要のみ。内部実装のため詳細は省略傾向 |
| parse_fftlength_or_overlap | utils/fft_args.py | [x] | 良好 |
| pint_unit_to_astropy_unit | utils/units.py | [x] | 良好 |
| validate_* | (各所) | [/] | 簡素な記述が多い |
